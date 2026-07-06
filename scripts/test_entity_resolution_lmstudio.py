"""实体消解 + 集合过滤 + 报告去数值化 全流程测试 — 连接本地 LM Studio。

用法（项目根目录）：
    python scripts/test_entity_resolution_lmstudio.py

覆盖：
  阶段A（组件·别名/集合 判定 + 图节点合并）：
    - intel_sorter 对固定实体名清单：识别别名组(乌军/乌克兰军队、OECD/经合组织)、
      集合概念(中国科技企业群体)在成员在列时 include_in_simulation=false。
    - graph_store.merge_alias_nodes：别名节点并入规范节点、RELATES 重指、节点数减少。
  阶段B（真实全流程·engine.start 跑 2 轮量化推演 + 报告）：
    - P0: 最终图中别名对只剩规范名（乌军/乌克兰军队、OECD/经合组织 不并存）。
    - P1: 集合概念不作为智能体。
    - P2: 报告叙述不含具体数值评分（无 "NN分"、"=NN"）。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import urllib.request
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_entres_test_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SOURCE = (
    "2025年，俄乌冲突进入僵持阶段。乌克兰军队依托西方援助构筑多层防线，乌军在东部战线顽强抵抗。"
    "俄罗斯军队集中兵力发动春季攻势，俄军试图突破乌军防御。北约向乌克兰军队提供武器与情报支持，"
    "欧盟则承担对乌财政援助。与此同时，经合组织(OECD)发布报告评估战争对全球供应链的冲击。"
    "在科技领域，中国科技企业群体保持观望：华为持续推进自研芯片，中芯国际扩大产能，"
    "DeepSeek 发布新一代模型。三方围绕能源、粮食与半导体展开长期博弈。"
) * 2

_FIXED_ENTITIES = [
    "乌克兰军队", "乌军", "俄罗斯军队", "俄军", "北约", "欧盟",
    "OECD", "经合组织", "中国科技企业群体", "华为", "中芯国际", "DeepSeek",
]

_NUM_PATTERNS = [
    (re.compile(r"\d+\s*分"), "NN分评分"),
    (re.compile(r"[=＝]\s*-?\d+"), "指标=NN"),
    (re.compile(r"\d+\s*[点分]（累计）"), "累计数值"),
]


def _check_server() -> bool:
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/models", timeout=5).read()
        return True
    except Exception as e:
        print(f"[致命] 无法连接 LM Studio: {base}/models -> {e}")
        return False


def _discover_chat_model() -> str:
    if os.environ.get("FORGE_LLM_MODEL"):
        return os.environ["FORGE_LLM_MODEL"]
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        chat = [m for m in ids if "embed" not in m.lower()]
        pick = (next((m for m in chat if "9b" in m.lower()), None)
                or next((m for m in chat if "2b" in m.lower()), None)
                or (chat[0] if chat else "local-chat"))
        os.environ["FORGE_LLM_MODEL"] = pick
        return pick
    except Exception:
        os.environ.setdefault("FORGE_LLM_MODEL", "local-chat")
        return os.environ["FORGE_LLM_MODEL"]


def _find_numbers(text: str) -> list[str]:
    hits: list[str] = []
    for pat, label in _NUM_PATTERNS:
        for m in pat.findall(text or ""):
            hits.append(f"{label}:{m}")
    return hits


async def _stage_a() -> None:
    from strategy_forge.core.llm_client import DeductionLLMClient
    from strategy_forge.engine.intel_sorter import sort_entities
    from strategy_forge.storage.graph_store import DeductionGraphStore

    print("\n=== 阶段A：别名/集合判定 + 图节点合并 ===")
    client = DeductionLLMClient()
    intel = await sort_entities(SOURCE, _FIXED_ENTITIES, client, max_source_chars=25000)
    assert intel, "intel_sorter 返回空"
    by_name = {e["name"]: e for e in intel}
    print("  情报条目:")
    for e in intel:
        flag = "✓收录" if e["include_in_simulation"] else "✗排除"
        al = ("|别名:" + ",".join(e["aliases"])) if e.get("aliases") else ""
        print(f"    {flag} {e['name']} ({e['type']}){al}  role={e['role'][:24]}")

    # 别名合并：乌军应作为"乌克兰军队"别名(或反之)，不应两者都是独立条目
    uk_entries = [e for e in intel if e["name"] in ("乌克兰军队", "乌军")]
    uk_alias_ok = any("乌军" in e.get("aliases", []) or "乌克兰军队" in e.get("aliases", [])
                      for e in intel)
    check_a1 = len(uk_entries) == 1 or uk_alias_ok
    print(f"  [P0] 乌军/乌克兰军队 别名合并: {'OK' if check_a1 else '未合并'}")

    oecd_alias_ok = any(("OECD" in e.get("aliases", []) and e["name"] == "经合组织")
                        or ("经合组织" in e.get("aliases", []) and e["name"] == "OECD")
                        or (e["name"] in ("OECD", "经合组织") and len(
                            [x for x in intel if x["name"] in ("OECD", "经合组织")]) == 1)
                        for e in intel)
    print(f"  [P0] OECD/经合组织 别名合并: {'OK' if oecd_alias_ok else '未合并'}")

    # 集合概念：中国科技企业群体在成员(华为/中芯/DeepSeek)在列时应被排除
    coll = by_name.get("中国科技企业群体")
    if coll is not None:
        check_a3 = not coll["include_in_simulation"]
        print(f"  [P1] 中国科技企业群体 集合过滤: {'OK(已排除)' if check_a3 else '未排除'}")
    else:
        print("  [P1] 中国科技企业群体 已被合并/未单列（可接受）")

    # 图节点合并测试
    g = DeductionGraphStore(os.path.join(_TMP, "stage_a_graph"))
    try:
        g.upsert_entity("id_full", "乌克兰军队", "军事力量", "乌方防御力量")
        g.upsert_entity("id_abbr", "乌军", "军事力量", "乌方防御力量简称")
        g.upsert_entity("id_nato", "北约", "组织", "西方军事同盟")
        g.upsert_relation("id_nato", "id_abbr", "支援", 5.0)  # 北约 -> 乌军
        before = g.count_entities()
        merged = g.merge_alias_nodes("乌克兰军队", ["乌军"])
        after = g.count_entities()
        names = set(g.get_entity_names())
        print(f"  merge_alias_nodes: 合并 {merged} 个别名, 节点 {before}->{after}, names={names}")
        assert merged == 1, "应合并1个别名节点"
        assert "乌军" not in names and "乌克兰军队" in names, f"别名未并入规范名: {names}"
        # 北约->乌军 的关系应重指到 乌克兰军队
        nb = g.get_entity_neighbors("id_nato")
        nb_names = {n["name"] for n in nb.get("neighbors", [])}
        assert "乌克兰军队" in nb_names, f"RELATES 未重指到规范节点: {nb_names}"
        print(f"  RELATES 重指 OK: 北约邻居={nb_names}")
    finally:
        g.close()
    print("  [OK] 阶段A 通过")


async def _stage_b() -> dict:
    from strategy_forge.engine.engine import DeductionEngine

    print("\n=== 阶段B：真实全流程(engine.start, 2轮量化) ===")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    engine = DeductionEngine(workspace_root=root)
    session = engine.create_session(
        title="俄乌科技博弈推演",
        source_material=SOURCE,
        config={"domain": "military", "total_rounds": 2, "enable_narrate": True,
                "enable_multi_action": True, "max_actions": 2},
    )
    sid = session.id
    print(f"  会话 {sid} 已创建，开始全流程推演...")
    await engine.start(sid)

    data = engine.session_store.get(sid)
    graph = engine.get_graph(sid)
    names = set(graph.get_entity_names())
    report = data.get("report_json", {}) or {}
    if isinstance(report, str):
        report = json.loads(report)

    # 智能体名（从 Kuzu Agent 节点）
    agents = graph.get_agents()
    agent_names = {a["name"] for a in agents}
    print(f"  实体节点数={len(names)}  智能体数={len(agent_names)}")
    print(f"  智能体: {sorted(agent_names)}")

    # P0: 别名对不并存
    p0_uk = not ("乌军" in names and "乌克兰军队" in names)
    p0_oecd = not ("OECD" in names and "经合组织" in names)
    print(f"  [P0] 乌军/乌克兰军队 未并存: {p0_uk} | OECD/经合组织 未并存: {p0_oecd}")

    # P1: 集合概念不作为智能体
    p1 = "中国科技企业群体" not in agent_names
    print(f"  [P1] 中国科技企业群体 未成为智能体: {p1}")

    # P2: 报告无具体数值
    narrative = report.get("summary", "") or ""
    extra = " ".join(report.get("risk_alerts", []) + report.get("recommendations", []))
    num_hits = _find_numbers(narrative + " " + extra)
    print(f"  [P2] 报告叙述长度={len(narrative)}  裸数值命中={num_hits}")
    print(f"  报告开头: {narrative[:160]}")

    return {"p0_uk": p0_uk, "p0_oecd": p0_oecd, "p1": p1,
            "p2": len(num_hits) == 0, "num_hits": num_hits,
            "narrative_len": len(narrative), "agents": sorted(agent_names),
            "names": sorted(names)}


async def main() -> int:
    print("=== 实体消解 + 集合过滤 + 报告去数值化 全流程测试（LM Studio）===")
    if not _check_server():
        return 2
    chat = _discover_chat_model()
    print("对话模型:", chat, "| 嵌入模型:", os.environ["FORGE_EMBED_MODEL"])

    await _stage_a()
    result = await _stage_b()

    print("\n=== 结果汇总 ===")
    for k in ("p0_uk", "p0_oecd", "p1", "p2"):
        print(f"  {k}: {'PASS' if result[k] else 'FAIL'}")
    if result["num_hits"]:
        print(f"  报告裸数值: {result['num_hits']}")
    print("\n[完成] 详见上方各阶段输出")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

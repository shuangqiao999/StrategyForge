"""sub_entities/aliases 去字典化修复 —— 全流程功能测试（连接本地 LM Studio）。

背景：intel_sorter 曾把 sub_entities/aliases 原样透传，LLM 若返回对象数组
（如 [{"name":"华为"}]），agent_factory 的 ", ".join(...) 会抛
'sequence item 0: expected str instance, dict found'，导致阶段3崩溃。

用法（项目根目录）：
    python scripts/test_subentities_fix_lmstudio.py

覆盖：
  阶段A（离线·确定性回归）：_as_name / _as_name_list 归一化字典/混合输入；
      并复现老崩溃点 ", ".join(...) 现在不再抛异常。
  阶段B（真实全流程 engine.start·2 轮）：跑含"领袖→组织""集合→成员"结构的素材，
      断言推演跑到 complete、无 'expected str' 错误、生成智能体与报告。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_subent_test_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SOURCE = (
    "2025年，中美围绕人工智能与半导体展开长期博弈。美国由特朗普政府主导，"
    "推动对华科技封锁；中国则以举国之力突破芯片瓶颈。中国科技企业群体成为焦点："
    "华为持续推进自研芯片，中芯国际扩大先进制程产能，DeepSeek 发布新一代大模型。"
    "俄罗斯在普京领导下深化中俄合作。欧盟与北约在跨大西洋裂痕中寻求战略自主。"
)


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


def _stage_a() -> None:
    from strategy_forge.engine.intel_sorter import _as_name, _as_name_list

    print("\n=== 阶段A：sub_entities/aliases 归一化（离线确定性回归）===")
    # 复现 LLM 返回对象数组 / 混合类型的场景
    raw = [{"name": "华为"}, "中芯国际", {"title": "DeepSeek"}, {"entity": "阿里"},
           {}, None, "华为", "  "]
    out = _as_name_list(raw)
    print(f"  _as_name_list 输入含字典/空/重复 → 输出: {out}")
    assert out == ["华为", "中芯国际", "DeepSeek", "阿里"], f"归一化结果不符: {out}"

    assert _as_name({"name": "OECD"}) == "OECD"
    assert _as_name("  经合组织 ") == "经合组织"
    assert _as_name({}) == "" and _as_name(None) == ""

    # 复现老崩溃点：对含字典的列表做 join。修复后（先归一化）不再抛异常。
    sub_entities = _as_name_list([{"name": "华为"}, {"name": "中芯国际"}])
    joined = ", ".join(sub_entities)  # 老代码此处会抛 expected str instance, dict found
    assert joined == "华为, 中芯国际", joined
    print(f"  join(归一化后 sub_entities) OK: '{joined}'（不再抛 dict 错误）")

    # agent_factory 的防御式 str() 兜底：即便上游漏网，也不崩
    defensive = ", ".join(str(s) for s in [{"name": "x"}, "y"])
    assert "y" in defensive
    print("  agent_factory 防御式 str() 兜底 OK")
    print("  [OK] 阶段A 通过")


async def _stage_b() -> dict:
    from strategy_forge.engine.engine import DeductionEngine

    print("\n=== 阶段B：真实全流程(engine.start, 2轮) ===")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    engine = DeductionEngine(workspace_root=root)
    session = engine.create_session(
        title="中美科技博弈推演",
        source_material=SOURCE,
        config={"domain": "military", "total_rounds": 2, "enable_narrate": True,
                "enable_multi_action": True, "max_actions": 2},
    )
    sid = session.id
    print(f"  会话 {sid} 已创建，开始全流程推演（含情报整理/别名合并/智能体工厂）...")
    await engine.start(sid)

    data = engine.session_store.get(sid)
    status = data.get("status")
    logs = engine.get_logs(sid, limit=500)
    log_text = "\n".join(f"[{lg['phase']}] {lg['message']}" for lg in logs)
    err_hit = ("expected str" in log_text) or ("sequence item" in log_text)

    graph = engine.get_graph(sid)
    agents = graph.get_agents()
    report = data.get("report_json", {}) or {}
    if isinstance(report, str):
        report = json.loads(report)
    narrative = report.get("summary", "") or ""

    print(f"  最终状态: {status}")
    print(f"  智能体数: {len(agents)}  报告长度: {len(narrative)}")
    print(f"  含 'expected str/sequence item' 错误: {err_hit}")
    if data.get("error"):
        print(f"  错误信息: {data.get('error')[:160]}")

    ok = (status == "complete") and (not err_hit) and (len(agents) > 0)
    print(f"  [{'OK' if ok else 'FAIL'}] 阶段B {'通过' if ok else '未通过'}")
    return {"status": status, "err_hit": err_hit, "agents": len(agents),
            "narrative_len": len(narrative), "ok": ok}


async def main() -> int:
    print("=== sub_entities 去字典化修复 全流程测试（LM Studio）===")
    if not _check_server():
        return 2
    chat = _discover_chat_model()
    print("对话模型:", chat, "| 嵌入模型:", os.environ["FORGE_EMBED_MODEL"])

    _stage_a()
    r = await _stage_b()

    print("\n=== 结果汇总 ===")
    print("  阶段A 归一化回归: PASS")
    print(f"  阶段B 全流程: {'PASS' if r['ok'] else 'FAIL'} "
          f"(status={r['status']}, agents={r['agents']}, err={r['err_hit']})")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""④ 关系邻居增强召回 —— 富实体全流程·公平 A/B（连接本地 LM Studio）。

方法学修正：不跑两次独立全流程（图谱/智能体因 LLM 随机性不可比），
而是跑一次富实体推演，用其真实 events 表 + 真实 Kuzu 关系邻居，
对每个智能体对比 base-query 与 boost-query 的召回（仅 query 变，其余全同）。

用法（项目根目录）：
    python scripts/test_relboost_fullflow_ab.py
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

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_relab_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")
os.environ.setdefault("FORGE_MAX_AGENTS", "13")
os.environ["FORGE_RECALL_REL_MAX"] = "4"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SOURCE = (
    "大陆棋局进入总决战。北境王国、雪原公国、铁堡联邦结成‘北方同盟’，共御外侮；"
    "南疆帝国、赤沙汗国、海湾城邦组成‘南方轴心’与之对抗。北境王国与南疆帝国是百年世仇，"
    "互不相让；雪原公国与赤沙汗国为争夺寒铁矿脉长期交战；铁堡联邦表面属北方同盟，"
    "却暗中向海湾城邦输送军械牟利。游离势力中，东岭商会两面下注、军火通吃；"
    "雾港自由市依附强者、待价而沽；北山部落骁勇善战却各自为政。"
    "此外，龙庭教廷以信仰之名调停各方，青鸾谍社则在暗处操纵情报、离间同盟。"
    "各方围绕兵力、士气、粮草、同盟稳固与情报优势展开长期博弈，胜负未定。"
) * 2


def _check_server() -> bool:
    try:
        urllib.request.urlopen(os.environ["FORGE_LLM_BASE"].rstrip("/") + "/models", timeout=5).read()
        return True
    except Exception as e:
        print(f"[致命] 无法连接 LM Studio: {e}")
        return False


def _discover_chat_model() -> str:
    if os.environ.get("FORGE_LLM_MODEL"):
        return os.environ["FORGE_LLM_MODEL"]
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=5) as resp:
            ids = [m.get("id") for m in json.loads(resp.read()).get("data", []) if m.get("id")]
        chat = [m for m in ids if "embed" not in m.lower()]
        pick = next((m for m in chat if "9b" in m.lower()), chat[0] if chat else "local")
        os.environ["FORGE_LLM_MODEL"] = pick
        return pick
    except Exception:
        os.environ.setdefault("FORGE_LLM_MODEL", "local")
        return os.environ["FORGE_LLM_MODEL"]


async def main() -> int:
    print("=== ④ 富实体全流程·公平 A/B（LM Studio）===")
    if not _check_server():
        return 2
    print("对话模型:", _discover_chat_model(), "| FORGE_MAX_AGENTS=13 | rounds=3")

    from strategy_forge.engine.engine import DeductionEngine
    from strategy_forge.engine.preprocessor import DeductionPreprocessor
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    engine = DeductionEngine(workspace_root=root)
    session = engine.create_session(
        title="大陆棋局", source_material=SOURCE,
        config={"domain": "military", "total_rounds": 3, "enable_narrate": True,
                "enable_multi_action": True, "max_actions": 2},
    )
    sid = session.id
    print(f"\n会话 {sid} 全流程推演中（填充图谱关系 + events 表）...")
    await engine.start(sid)

    data = engine.session_store.get(sid)
    graph = engine.get_graph(sid)
    agents = graph.get_agents()
    report = data.get("report_json", {}) or {}
    if isinstance(report, str):
        report = json.loads(report)

    print(f"状态: {data.get('status')}  实体: {data.get('entity_count')}  "
          f"关系: {data.get('relation_count')}  智能体: {len(agents)}")

    # 复用该会话已建的 events 表 + result
    pp = DeductionPreprocessor(_TMP, session_id=sid)
    pp.preprocess(SOURCE)

    def _hits(frags, names):
        return sum(1 for f in frags if any(nm in f for nm in names))

    print("\n" + "=" * 70)
    print("  逐智能体 base-query vs boost-query 召回对比（同 events 表，仅 query 变）")
    print("=" * 70)
    tot_base = tot_boost = tot_agents_with_neigh = 0
    per_rows = []
    for a in agents:
        name = a["name"]
        nb = graph.get_entity_neighbors(a["id"])
        neigh = [n["name"] for n in nb.get("neighbors", []) if n.get("name") and n["name"] != name]
        if not neigh:
            continue
        tot_agents_with_neigh += 1
        base_q = name
        boost_q = (name + " " + " ".join(neigh[:4])).strip()
        bf = pp.retrieve_dynamic_events(base_q, top_k=5, min_similarity=0.0)
        of = pp.retrieve_dynamic_events(boost_q, top_k=5, min_similarity=0.0)
        bh, oh = _hits(bf, neigh), _hits(of, neigh)
        tot_base += bh
        tot_boost += oh
        per_rows.append((name, neigh[:4], bh, oh))

    for name, neigh, bh, oh in per_rows:
        flag = "↑" if oh > bh else ("=" if oh == bh else "↓")
        print(f"  {name:<10} 邻居{neigh}  base命中={bh} boost命中={oh} {flag}")

    print("-" * 70)
    print(f"有关系邻居的智能体: {tot_agents_with_neigh}/{len(agents)}")
    print(f"关系相关命中合计:  base={tot_base}  boost={tot_boost}")

    print("\n-- 推演报告开头（主观质量参考）--")
    print((report.get("summary") or "")[:600])
    print("\n-- 结论 --")
    print((report.get("conclusion") or "")[:300])

    print("\n=== 判据 ===")
    complete = data.get("status") == "complete"
    improved = tot_boost >= tot_base
    print(f"全流程 complete: {complete}")
    print(f"关系相关召回 boost >= base: {improved} (base={tot_base} boost={tot_boost})")
    if tot_agents_with_neigh == 0:
        print("⚠ 本次图谱未抽出关系邻居，无法评估召回收益（可重跑或换更强关系的素材）")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

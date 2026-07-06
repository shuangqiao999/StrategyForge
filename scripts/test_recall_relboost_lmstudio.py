"""④ 关系邻居增强召回 A/B 验证（连接本地 LM Studio）。

用法（项目根目录）：
    python scripts/test_recall_relboost_lmstudio.py

做法：
  1) 全流程跑一次量化推演(persist_events=True)填充 LanceDB events 表 + 建立 _rel_context；
  2) 对目标 agent，用 base query 与 rel-boost query 分别召回动态事件，
     统计"命中盟友/对手名"的片段数，断言 boost >= base 且都非空；
  3) 全流程健壮性：FORGE_RECALL_REL_BOOST=0 与 =1 各跑一次 engine.start，均需 complete。
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

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_relboost_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")
os.environ.setdefault("FORGE_MAX_AGENTS", "10")
os.environ["FORGE_RECALL_REL_MAX"] = "4"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SOURCE = (
    "北境三方对峙：甲军团与乙军团结为盟友，共同对抗宿敌丙军团。甲军团勇猛主攻，"
    "乙军团据守山地为甲军团提供侧翼掩护；丙军团机动灵活，屡次突袭甲军团补给线并离间甲乙同盟。"
    "三方围绕雪原要塞展开长期拉锯，胜负取决于兵力、士气、粮草与同盟稳固程度。"
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


async def _run(boost: str, sid_title: str):
    """跑一次全流程，返回 (engine, sid)。"""
    os.environ["FORGE_RECALL_REL_BOOST"] = boost
    from strategy_forge.engine.engine import DeductionEngine
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    engine = DeductionEngine(workspace_root=root)
    session = engine.create_session(
        title=sid_title, source_material=SOURCE,
        config={"domain": "military", "total_rounds": 3, "enable_narrate": False,
                "enable_multi_action": True, "max_actions": 2},
    )
    await engine.start(session.id)
    return engine, session.id


async def main() -> int:
    print("=== ④ 关系邻居增强召回 A/B（LM Studio）===")
    if not _check_server():
        return 2
    print("对话模型:", _discover_chat_model())
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  \u2713 {name} {detail}")
        else:
            failed += 1
            print(f"  \u2717 {name} FAILED {detail}")

    # ── 全流程健壮性：boost 关/开 都要 complete ──
    eng0, sid0 = await _run("0", "relboost-off")
    st0 = eng0.session_store.get(sid0).get("status")
    check("boost=0 全流程 complete", st0 == "complete", f"status={st0}")

    eng1, sid1 = await _run("1", "relboost-on")
    st1 = eng1.session_store.get(sid1).get("status")
    check("boost=1 全流程 complete", st1 == "complete", f"status={st1}")

    # ── 召回命中对比：复用 boost=1 那次的 events 表 + _rel_context 无法直接拿，
    #    改为直接用其 preprocessor 对目标 agent 做 base vs boost query 召回对比 ──
    from strategy_forge.engine.preprocessor import DeductionPreprocessor
    pp = DeductionPreprocessor(_TMP, session_id=sid1)  # 打开该会话已建的 events 表
    pp.preprocess(SOURCE)  # 需要 result 才能召回；events 表按 session_id 复用

    # ── 召回机制对比（受控·确定性）：注入事件，部分只提及"关系邻居"名，
    #    对比 base('甲军团') vs boost('甲军团 乙军团 丙军团') 的关系相关命中 ──
    from strategy_forge.engine.preprocessor import DeductionPreprocessor
    pp = DeductionPreprocessor(_TMP, session_id="relctl_" + uuid.uuid4().hex[:4])
    pp.preprocess(SOURCE)  # 建 events 表 + 探测维度

    target_name = "甲军团"
    neighbors = ["乙军团", "丙军团"]  # 甲的盟友/宿敌
    injected = [
        # 只提"关系邻居"、不提甲的事件（base 用甲查询难 surface，boost 应拉入）
        "乙军团独自据守山地防线严阵以待",
        "丙军团趁夜色单独机动迂回渗透",
        # 大量以甲军团为主的干扰事件（base 用甲查询会优先占满 top_k，挤出邻居事件）
        "甲军团加固前沿营盘工事",
        "甲军团补充粮草辎重物资",
        "甲军团士气高涨全线待命",
        "甲军团调整炮兵阵地部署",
        "甲军团侦察兵前出勘察地形",
        "甲军团主力在黎明发动总攻",
        "甲军团设立野战指挥所",
        "甲军团抢修受损的攻城器械",
        # 无关噪声
        "粮草运输队沿雪原缓慢前行",
        "北境突降大雪严重影响视野",
    ]
    for c in injected:
        pp.add_event_memory(c, "A", 1, event_type="attack")

    def _rel_hits(frags):
        return sum(1 for f in frags if any(nm in f for nm in neighbors))

    topk = 3
    base_frags = pp.retrieve_dynamic_events(target_name, top_k=topk, min_similarity=0.0)
    boost_frags = pp.retrieve_dynamic_events(
        (target_name + " " + " ".join(neighbors)).strip(), top_k=topk, min_similarity=0.0)
    bh, oh = _rel_hits(base_frags), _rel_hits(boost_frags)
    print(f"  base('{target_name}', top{topk}) 命中邻居 {bh}: {[x[:14] for x in base_frags]}")
    print(f"  boost('{target_name} {' '.join(neighbors)}', top{topk}) 命中邻居 {oh}: {[x[:14] for x in boost_frags]}")
    check("boost 召回非空", len(boost_frags) > 0)
    check("boost 关系相关命中 >= base", oh >= bh, f"base={bh} boost={oh}")
    check("boost 确实 surface 了邻居事件(>=1)", oh >= 1, f"boost={oh}")

    print(f"\n结果: {passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""LanceDB 全链路充分利用 功能测试 — 连接本地 LM Studio (9B 对话 + embeddinggemma 嵌入)。

用法（项目根目录）：
    python scripts/test_lancedb_fulllink_lmstudio.py

覆盖：
  A. Preprocessor 直接链路：preprocess→chunks 索引 + FTS(混合检索) + 维度探测；
     查询嵌入缓存 / 实体召回结果缓存命中；events 写入；
     retrieve_dynamic_events 的 .where() 过滤(排除目标/干预)；
     retrieve_latest_intervention 的 where 下推。
  B. 量化推演全链路(persist_events=True)：量化轮 R1/R2 语义召回 + W4 写入 LanceDB events 表。
  C. 优化器隔离(persist_events=False)：量化轮不写 events 表(隔离保持)。
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

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_lancedb_test_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SOURCE = (
    "北境三大军团对峙于雪原。甲军团由勇猛激进的统帅指挥，倾向集中兵力主动进攻，"
    "但粮草补给线漫长。乙军团老成持重，依托山地构筑防线，擅长防守反击与消耗对手士气。"
    "丙军团机动灵活，常以迂回与外交手段分化对手，避免正面决战。三方围绕雪原要塞的"
    "控制权展开长期博弈，胜负取决于兵力、士气、粮草与统帅决断的综合较量。"
) * 3


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
        pick = next((m for m in chat if "9b" in m.lower()), None) or (chat[0] if chat else "local-9b")
        os.environ["FORGE_LLM_MODEL"] = pick
        return pick
    except Exception:
        os.environ.setdefault("FORGE_LLM_MODEL", "local-9b")
        return os.environ["FORGE_LLM_MODEL"]


def _mk_agent(name: str, persona: str):
    from strategy_forge.engine.models import DeductionAgentProfile
    return DeductionAgentProfile(entity_id=uuid.uuid4().hex[:8], name=name,
                                 persona=persona, background="", goals=["击败对手，保全本部"])


def _stage_a() -> None:
    from strategy_forge.engine.preprocessor import DeductionPreprocessor
    print("\n=== 阶段A：Preprocessor 直接链路（缓存 / FTS / where 过滤）===")
    pp = DeductionPreprocessor(_TMP, session_id="lancetest_a")
    res = pp.preprocess(SOURCE)
    assert res.total_chunks > 0, "未切出任何 chunk"
    assert pp._dim > 0, "嵌入维度探测失败（embeddinggemma 未响应？）"
    print(f"  preprocess: {res.total_chunks} chunks, dim={pp._dim}, fts_ready={pp._fts_ready}")
    assert pp._fts_ready, "FTS 索引未建立（混合检索不可用）"

    # 实体召回 + 结果缓存命中
    r1 = pp.retrieve_for_entity("甲军团", top_k=3, must_contain={"甲军团"})
    r2 = pp.retrieve_for_entity("甲军团", top_k=3, must_contain={"甲军团"})
    assert r1 == r2, "两次召回结果应一致"
    assert pp.recall_cache_hits >= 1, "实体召回结果缓存未命中"
    print(f"  retrieve_for_entity(混合检索) 命中 {len(r1)} 段；recall_cache_hits={pp.recall_cache_hits}")

    # 查询嵌入缓存命中
    pp._sync_embed_single("缓存探针文本")
    pp._sync_embed_single("缓存探针文本")
    assert pp.embed_cache_hits >= 1, "查询嵌入缓存未命中"
    print(f"  embed_cache_hits={pp.embed_cache_hits}")

    # 写入事件 + 干预/目标
    pp.add_event_memory("甲军团对乙军团发动猛攻", "A", 1, event_type="attack", priority=0.5)
    pp.add_event_memory("乙军团据守山地防线", "B", 1, event_type="defend", priority=0.5)
    pp.add_event_memory("最高指令：务必保全甲军团主力", "system_user", 1,
                        event_type="user_intervention", priority=1.0)
    pp.add_event_memory("不可变目标：夺取雪原要塞", "system_user", 1,
                        event_type="immutable_goal", priority=0.9)

    # 动态召回应排除 干预/目标（.where 过滤）
    dyn = pp.retrieve_dynamic_events("进攻 防守 战斗", top_k=5, min_similarity=0.0)
    joined = " || ".join(dyn)
    assert "最高指令" not in joined and "不可变目标" not in joined, \
        f"动态召回未排除目标/干预: {dyn}"
    print(f"  retrieve_dynamic_events(.where 排除目标/干预) 返回 {len(dyn)} 条，已正确过滤")

    # 干预召回(where 下推)
    iv = pp.retrieve_latest_intervention()
    assert iv is not None and iv["priority"] >= 0.9, f"干预召回异常: {iv}"
    print(f"  retrieve_latest_intervention(where 下推): priority={iv['priority']} \"{iv['content'][:20]}…\"")
    print("  [OK] 阶段A 通过")


async def _run_quant(session_id: str, persist: bool, rounds: int):
    from strategy_forge.engine.preprocessor import DeductionPreprocessor
    from strategy_forge.engine.rule_engine import RuleEngine
    from strategy_forge.engine.simulator import SimulationEngine
    pp = DeductionPreprocessor(_TMP, session_id=session_id)
    pp.preprocess(SOURCE)
    re_engine = RuleEngine.from_domain("military")
    agents = [
        _mk_agent("甲军团", "勇猛激进，集中兵力进攻，兼顾补给"),
        _mk_agent("乙军团", "老成持重，攻守兼备，依托山地"),
        _mk_agent("丙军团", "机动灵活，迂回外交，多线施压"),
    ]
    states = {a.entity_id: re_engine.init_state(a.entity_id, a.name) for a in agents}
    eng = SimulationEngine(
        agents=agents, graph=None, total_rounds=rounds, log_fn=lambda p, m: None,
        preprocessor=pp, pre_goals=["攻守兼顾，速战速决"],
        seed=7, temperature=0.6, persist_events=persist, max_concurrent=2,
        rule_engine=re_engine, states=states, enable_narrate=False,
        enable_multi_action=True, max_actions=3,
    )
    for rnd in range(1, rounds + 1):
        await eng.run_round(rnd)
    return pp


async def main() -> int:
    print("=== LanceDB 全链路充分利用 测试（LM Studio）===")
    if not _check_server():
        return 2
    chat = _discover_chat_model()
    print("对话模型:", chat, "| 嵌入模型:", os.environ["FORGE_EMBED_MODEL"])

    _stage_a()

    print("\n=== 阶段B：量化推演全链路(persist_events=True) — R1/R2 召回 + W4 写入 ===")
    pp_b = await _run_quant("lancetest_b", persist=True, rounds=3)
    ev_count = pp_b._event_table.count_rows()
    assert ev_count > 0, "量化轮未向 LanceDB events 表写入(W4 失效)"
    print(f"  W4 写入生效：events 表行数={ev_count}")
    dyn_b = pp_b.retrieve_dynamic_events("进攻 防守 迂回", top_k=5, min_similarity=0.0)
    assert len(dyn_b) > 0, "量化推演后动态语义召回为空(R2 无内容)"
    print(f"  R2 动态召回生效：召回 {len(dyn_b)} 条模拟事件")
    print("  [OK] 阶段B 通过")

    print("\n=== 阶段C：优化器隔离(persist_events=False) — 量化轮不写 events 表 ===")
    pp_c = await _run_quant("lancetest_c", persist=False, rounds=2)
    ev_count_c = pp_c._event_table.count_rows()
    assert ev_count_c == 0, f"隔离模式不应写 events 表，但行数={ev_count_c}"
    print(f"  隔离保持：events 表行数={ev_count_c}（未写入）")
    print("  [OK] 阶段C 通过")

    print("\n[全部通过] LanceDB 全链路测试 OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

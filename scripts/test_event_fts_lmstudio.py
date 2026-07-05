"""事件表混合检索(⑤) 功能测试 — 连接本地 LM Studio (对话 + embeddinggemma 嵌入)。

用法（项目根目录）：
    python scripts/test_event_fts_lmstudio.py

覆盖三处修复 + 全流程：
  阶段A（受控，确定性）：针对 retrieve_dynamic_events 混合检索三处修复
    1. 打分字段：hybrid 返回 _relevance_score，召回非空（防"全被 _distance 滤空"回归）；
    2. where 泄漏：immutable_goal / user_intervention 永不进入召回（即便字面命中查询词）；
    3. 关键词命中：字面含实体名的事件被 hybrid 召回且不劣于纯向量。
  阶段B（真实全流程）：FORGE_EVENT_HYBRID=1 下跑真实量化推演 3 轮，
    每轮真实 LLM 决策 + 事件写入 LanceDB + 每轮混合召回，断言无异常、召回生效、无系统事件泄漏。
  阶段C（回归）：默认关闭(纯向量)路径仍正常。
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

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_eventfts_test_" + uuid.uuid4().hex[:6])
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
        pick = (next((m for m in chat if "9b" in m.lower()), None)
                or next((m for m in chat if "2b" in m.lower()), None)
                or (chat[0] if chat else "local-chat"))
        os.environ["FORGE_LLM_MODEL"] = pick
        return pick
    except Exception:
        os.environ.setdefault("FORGE_LLM_MODEL", "local-chat")
        return os.environ["FORGE_LLM_MODEL"]


def _set_hybrid(enabled: bool) -> None:
    """运行时翻转事件混合检索开关（config 为单例，检索时读取其属性）。"""
    from strategy_forge.core.config import config
    config.deduction_event_hybrid = enabled


def _mk_agent(name: str, persona: str):
    from strategy_forge.engine.models import DeductionAgentProfile
    return DeductionAgentProfile(entity_id=uuid.uuid4().hex[:8], name=name,
                                 persona=persona, background="", goals=["击败对手，保全本部"])


def _contains_count(frags: list[str], token: str) -> int:
    return sum(1 for f in frags if token in f)


def _stage_a() -> None:
    from strategy_forge.engine.preprocessor import DeductionPreprocessor
    print("\n=== 阶段A：事件混合检索三处修复（受控确定性）===")
    pp = DeductionPreprocessor(_TMP, session_id="eventfts_a")
    pp.preprocess(SOURCE)
    assert pp._dim > 0, "嵌入维度探测失败（embeddinggemma 未响应？）"

    # 受控事件：e1/e2/e3 字面含各自军团名；e4 语义相关但不含"甲军团"字面；
    # e5/e6 为系统事件（e6 甚至字面含"甲军团"，用于验证排除优先于关键词命中）
    pp.add_event_memory("甲军团在黎明发动突袭，凿穿乙军团前哨", "A", 1, event_type="attack")
    pp.add_event_memory("乙军团加固山地防线，稳住阵脚", "B", 1, event_type="defend")
    pp.add_event_memory("丙军团派遣使者展开外交斡旋，分化联盟", "C", 1, event_type="diplomacy")
    pp.add_event_memory("补给车队沿漫长粮道缓慢推进，尘土蔽日", "A", 1, event_type="logistics")
    pp.add_event_memory("不可变目标：夺取雪原要塞", "system_user", 1,
                        event_type="immutable_goal", priority=0.9)
    pp.add_event_memory("最高指令：务必保全甲军团主力", "system_user", 1,
                        event_type="user_intervention", priority=1.0)

    query = "甲军团"
    top_k = 3

    # 纯向量（默认关闭）
    _set_hybrid(False)
    pp.clear_round_cache()
    vec = pp.retrieve_dynamic_events(query, top_k=top_k, min_similarity=0.0)
    print(f"  [vector] 召回 {len(vec)} 条: {[v[:24] for v in vec]}")

    # 混合检索（开启）
    _set_hybrid(True)
    pp.clear_round_cache()
    hyb = pp.retrieve_dynamic_events(query, top_k=top_k, min_similarity=0.0)
    print(f"  [hybrid] 召回 {len(hyb)} 条: {[h[:24] for h in hyb]}  fts_ready={pp._event_fts_ready}")

    # 修复1：hybrid 索引就绪且召回非空（防打分字段导致全空）
    assert pp._event_fts_ready, "事件 FTS 索引未建立，混合检索未真正启用"
    assert len(hyb) > 0, "修复1 失败：hybrid 召回为空（疑似 _distance/_relevance 打分未处理）"

    # 修复2：目标/干预事件绝不泄漏（两种模式都不能出现）
    for mode, frags in (("vector", vec), ("hybrid", hyb)):
        joined = " || ".join(frags)
        assert "不可变目标" not in joined, f"修复2 失败[{mode}]：immutable_goal 泄漏: {frags}"
        assert "最高指令" not in joined, f"修复2 失败[{mode}]：user_intervention 泄漏(即便字面含'甲军团'): {frags}"
    print("  修复2 OK：immutable_goal / user_intervention 均未泄漏（含'甲军团'的系统指令也被排除）")

    # 修复3：字面含实体名的事件被 hybrid 命中，且命中数不劣于纯向量
    assert _contains_count(hyb, "甲军团") >= 1, f"修复3 失败：hybrid 未召回字面含'甲军团'的事件: {hyb}"
    assert _contains_count(hyb, "甲军团") >= _contains_count(vec, "甲军团"), \
        f"修复3 失败：hybrid 关键词命中数({_contains_count(hyb, '甲军团')}) < vector({_contains_count(vec, '甲军团')})"
    print(f"  修复1/3 OK：hybrid 非空且关键词命中 {_contains_count(hyb, '甲军团')} 条 (vector {_contains_count(vec, '甲军团')} 条)")
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


async def _stage_b() -> None:
    print("\n=== 阶段B：真实全流程（FORGE_EVENT_HYBRID=1，量化推演 3 轮）===")
    _set_hybrid(True)
    pp = await _run_quant("eventfts_b", persist=True, rounds=3)
    ev_count = pp._event_table.count_rows()
    assert ev_count > 0, "量化推演未向 LanceDB events 表写入事件"
    assert pp._event_fts_ready, "推演过程中事件 FTS 未建立（混合检索未生效）"
    print(f"  events 表行数={ev_count}，fts_ready={pp._event_fts_ready}")
    dyn = pp.retrieve_dynamic_events("甲军团 进攻", top_k=5, min_similarity=0.0)
    assert len(dyn) > 0, "全流程后混合动态召回为空"
    joined = " || ".join(dyn)
    assert "不可变目标" not in joined and "最高指令" not in joined, \
        f"全流程召回泄漏系统事件: {dyn}"
    print(f"  混合动态召回生效：{len(dyn)} 条，无系统事件泄漏")
    print("  [OK] 阶段B 通过")


async def _stage_c() -> None:
    print("\n=== 阶段C：回归（默认关闭=纯向量）全流程 ===")
    _set_hybrid(False)
    pp = await _run_quant("eventfts_c", persist=True, rounds=2)
    ev_count = pp._event_table.count_rows()
    assert ev_count > 0, "纯向量全流程未写入事件"
    dyn = pp.retrieve_dynamic_events("乙军团 防守", top_k=5, min_similarity=0.0)
    assert len(dyn) > 0, "纯向量动态召回为空"
    assert not pp._event_fts_ready, "默认关闭时不应触发事件 FTS 构建"
    print(f"  纯向量路径正常：events={ev_count}，召回 {len(dyn)} 条，event_fts_ready={pp._event_fts_ready}(应为False)")
    print("  [OK] 阶段C 通过")


async def main() -> int:
    print("=== 事件表混合检索(⑤) 测试（LM Studio）===")
    if not _check_server():
        return 2
    chat = _discover_chat_model()
    print("对话模型:", chat, "| 嵌入模型:", os.environ["FORGE_EMBED_MODEL"])

    _stage_a()
    await _stage_b()
    await _stage_c()

    print("\n[全部通过] 事件表混合检索 ⑤ 测试 OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

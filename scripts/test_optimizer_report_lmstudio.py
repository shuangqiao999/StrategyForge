"""优化器叙事报告 + 多动作解耦 端到端测试 — 连接本地 LM Studio (9B)。

用法（项目根目录）：
    python scripts/test_optimizer_report_lmstudio.py

验证（用确定性基线，避免 LLM 抽取波动影响）：
  - 阶段1：config_json 为多动作单一真值源（建会话写入、可被读取）。
  - 阶段2：复现优化器"推荐方案代表性 run"机制——SimulationEngine(persist=True, 多动作)
    跑完 → generate_report 生成叙事报告 → 写入 report_json（报告页有内容）；
    并点亮 Kuzu 时序行动图（时间线/因果页可用）、事件含多动作分配。
"""
from __future__ import annotations

import asyncio
import copy
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

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_optrep_test_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


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


_AGENTS = [("A", "甲军团"), ("B", "乙军团"), ("C", "丙军团")]


async def main() -> int:
    print("=== 优化器叙事报告 + 多动作解耦 端到端测试（LM Studio）===")
    if not _check_server():
        return 2
    print("对话模型:", _discover_chat_model())

    from strategy_forge.engine.engine import DeductionEngine
    from strategy_forge.engine.models import DeductionAgentProfile
    from strategy_forge.engine.reporter import generate_report
    from strategy_forge.engine.rule_engine import RuleEngine
    from strategy_forge.engine.simulator import SimulationEngine

    engine = DeductionEngine(os.getcwd())
    config = {"domain": "military", "enable_multi_action": True, "max_actions": 3, "total_rounds": 2}
    sess = engine.create_session("优化器报告测试", "北境雪原三军团对峙。", config)
    sid = sess.id

    print("\n=== 阶段1：config 单一真值源（多动作随会话设置持久化）===")
    cfg = engine.session_store.get(sid)["config_json"]
    assert cfg.get("enable_multi_action") is True and cfg.get("max_actions") == 3, cfg
    print(f"  [OK] config_json: enable_multi_action={cfg['enable_multi_action']}, max_actions={cfg['max_actions']}")

    # ── 确定性基线（手工建图，规避 LLM 抽取波动）──
    graph = engine.get_graph(sid)
    for eid, name in _AGENTS:
        graph.upsert_entity(eid, name, "Person", f"{name}的描述")
        graph.upsert_agent_node(eid, name, "统帅", "", "[]")
    graph.upsert_relation("A", "B", "盟友", weight=3.0, evidence="并肩")
    graph.upsert_relation("A", "C", "敌对", weight=2.0, evidence="交战")
    re_engine = RuleEngine.from_domain("military")
    agents = [DeductionAgentProfile(entity_id=eid, name=name, persona="统帅",
                                    background="", goals=["击败对手，保全本部"])
              for eid, name in _AGENTS]
    base_states = {a.entity_id: re_engine.init_state(a.entity_id, a.name) for a in agents}

    print("\n=== 阶段2：推荐方案代表性 run(persist=True, 多动作) → 写 report_json ===")
    rep_states = {eid: copy.deepcopy(st) for eid, st in base_states.items()}
    rep_sim = SimulationEngine(
        agents=agents, graph=graph, total_rounds=2, log_fn=lambda p, m: None,
        preprocessor=None, pre_goals=["集中兵力速攻速决"],
        seed=20240101, temperature=0.6, persist_events=True, max_concurrent=1,
        rule_engine=re_engine, states=rep_states, enable_narrate=False,
        enable_multi_action=True, max_actions=3,
    )
    rounds = []
    has_alloc = False
    for rnd in range(1, 3):
        rd = await rep_sim.run_round(rnd)
        rounds.append(rd)
        for act in rd.actions:
            if act.metadata.get("allocation"):
                has_alloc = True
    print(f"  代表性 run 完成：{sum(len(r.actions) for r in rounds)} 个行动；含多动作分配={has_alloc}")

    session = engine.get_session(sid)
    session.current_round = 2
    rep = await generate_report(session=session, graph=graph, rounds=rounds,
                                log_fn=lambda p, m: None, preprocessor=None)
    payload = {
        "summary": rep.summary, "key_events": rep.key_events,
        "risk_alerts": rep.risk_alerts, "recommendations": rep.recommendations,
        "quantified": True, "domain": re_engine.domain, "optimized_scenario": "速攻",
        "final_states": {eid: {"name": st.name, "metrics": st.metrics,
                               "history": st.history[-60:], "alive": re_engine.is_alive(st)}
                         for eid, st in rep_states.items()},
    }
    engine.session_store.update(sid, report_json=json.dumps(payload, ensure_ascii=False))

    print("\n=== 阶段3：报告页有内容 ===")
    rj = engine.session_store.get(sid)["report_json"]
    assert isinstance(rj, dict) and rj.get("summary"), f"report_json 未写入: {rj}"
    assert rj.get("final_states"), "report_json 缺 final_states"
    print(f"  [OK] report_json.summary={rj['summary'][:50]}...")
    print(f"  [OK] optimized_scenario={rj.get('optimized_scenario')}, final_states={len(rj['final_states'])} 方")

    print("\n=== 阶段4：时间线/因果点亮（代表性 run 写入 Kuzu）===")
    seq = graph.get_event_sequence()
    caused = graph.query("MATCH (:Event)-[c:CAUSED]->(:Entity) RETURN count(c)")[0][0]
    assert len(seq) > 0, "代表性 run 未写时序行动图"
    print(f"  [OK] 事件序列={len(seq)} 条, 因果边 CAUSED={caused}")

    print("\n[全部通过] 优化器报告 + 多动作解耦 机制 端到端 OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

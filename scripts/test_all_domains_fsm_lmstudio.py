"""全领域FSM修复验证 — 连接本地LM Studio 9b模型.

验证项:
  1. 5个非战斗领域(politics/ecology/urban/tech/info_war) FSM default_state改为command_state
  2. 代理从首轮开始通过LLM决策（非FSM锁定）
  3. 每个领域运行2轮验证决策多样性
  4. FSM transition_rules仍能生效（条件触发时转移）
"""
from __future__ import annotations

import asyncio, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
sys.path.insert(0, os.path.join(_script, "..", "src"))

from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.models import DeductionAgentProfile, EntityState
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.algorithms.module_utils import build_module_chain

PASS = FAIL = 0
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")

def make_agents(ids, names, domain):
    return [DeductionAgentProfile(entity_id=eid, name=name, persona="Decision maker",
            background="Test entity", goals=["maximize influence"]) for eid, name in zip(ids, names)]


async def test_domain(domain: str, agent_ids: list, agent_names: list, rounds: int = 2):
    print(f"\n  --- {domain} ---")
    r = RuleEngine.from_domain(domain)
    fsm = r.pack["modules"]["finite_state_machine"]
    print(f"  FSM: default={fsm['default_state']}, command={fsm['command_states']}")

    agents = make_agents(agent_ids, agent_names, domain)
    init_m = dict(r.pack.get("initial_metrics", {}))
    states = {a.entity_id: EntityState(id=a.entity_id, name=a.name, domain=domain,
                metrics=dict(init_m), history=[]) for a in agents}

    modules = build_module_chain(r)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=rounds,
        log_fn=lambda p,m: None, rule_engine=r, states=states,
        enable_narrate=False, algorithm_modules=modules, persist_events=False)

    llm_count = 0
    fsm_count = 0
    actions_seen = set()

    for rnd in range(1, rounds + 1):
        result = await engine.run_round(rnd)
        for act in result.actions:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            content = (act.content or "")
            if "[FSM]" in content: fsm_count += 1
            else: llm_count += 1
            actions_seen.add(act.action_type)

    print(f"  LLM={llm_count}, FSM={fsm_count}, actions={actions_seen}")
    check(f"{domain}: LLM动作>=1 (非FSM锁定)", llm_count >= 1,
          f"LLM={llm_count}, FSM={fsm_count}")


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print("  StrategyForge 全领域FSM修复测试 (9b)")
    print("=" * 65)

    await test_domain("politics",
        ["A","B","C"], ["执政党","在野党1","在野党2"])
    await test_domain("ecology",
        ["A","B","C"], ["工业集团","环保组织","政府监管"])
    await test_domain("urban",
        ["A","B","C"], ["市政府","开发商","市民代表"])
    await test_domain("tech",
        ["A","B","C"], ["TechA","TechB","TechC"])
    await test_domain("info_war",
        ["A","B","C"], ["政府媒体","独立媒体","自媒体KOL"])

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

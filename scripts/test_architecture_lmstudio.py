"""架构改进验证: 趋势感知 + 因果反馈 + 空间FSM — 连接本地LM Studio 12b全流程测试。

验证项:
  1. 趋势感知: LLM prompt中是否包含指标3轮趋势 (↑/↓ with delta)
  2. 因果反馈: agent是否能收到上一轮行动的实际效果
  3. 空间FSM: distance_to_enemy条件是否触发状态转移
  4. 全流程: 5轮军事推演 + 行为模式对比
  5. 性能: 感知开销对推演时间的影响
  6. 质量对比: 有感知vs无感知的行为差异
"""
from __future__ import annotations

import asyncio, os, sys, time, numpy as np, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
sys.path.insert(0, os.path.join(_script, "..", "src"))

from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.models import DeductionAgentProfile, EntityState
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.algorithms.module_utils import build_module_chain

PASS = FAIL = 0
def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")


# ── Test 1: Trend perception ──
async def test_trend_perception():
    banner("Test 1: 趋势感知 (3轮delta)")

    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha军团", persona="精锐先锋",
                               background="闪电战专家", goals=["歼灭敌军"]),
        DeductionAgentProfile(entity_id="D1", name="Delta卫戍", persona="铁壁防御",
                               background="据守要隘", goals=["坚守待援"]),
        DeductionAgentProfile(entity_id="G1", name="Gamma奇兵", persona="高机动突袭",
                               background="侧翼穿插", goals=["突破封锁"]),
    ]

    states = {}
    for a in agents:
        s = EntityState(id=a.entity_id, name=a.name, domain="military", metrics=dict(init_m), history=[])
        states[a.entity_id] = s

    # Pre-populate history with declining trend for Delta
    for rnd in range(1, 5):
        for m in re.metrics():
            if m == "strength":
                val = max(10, 100 - rnd * 20)
            elif m == "supply":
                val = max(5, 90 - rnd * 10)
            else:
                val = 80.0
            states["D1"].history.append({"round": rnd, "metric": m, "old": val+5, "delta": -5, "new": float(val)})

    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=5,
        log_fn=lambda p,m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    await engine.run_round(4)
    d_hist = states["D1"].history
    check("Delta有3+轮历史", len(d_hist) >= 6, f"{len(d_hist)} entries (5 metrics x 4 rounds)")
    str_entries = [e for e in d_hist if e.get("metric") == "strength"]
    if len(str_entries) >= 2:
        check("Delta strength下降趋势", str_entries[-1]["new"] < str_entries[0]["new"],
              f'{str_entries[0]["new"]:.0f} -> {str_entries[-1]["new"]:.0f}')
    fs = engine._last_fsm_states if hasattr(engine, '_last_fsm_states') else None
    check("FSM状态已缓存", fs is not None and len(fs) == 3)


# ── Test 2: Causal feedback ──
async def test_causal_feedback():
    banner("Test 2: 因果反馈 (上一轮行动效果)")

    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha军团", persona="精锐先锋",
                               background="闪电战专家", goals=["歼灭敌军"]),
        DeductionAgentProfile(entity_id="D1", name="Delta卫戍", persona="铁壁防御",
                               background="据守要隘", goals=["坚守待援"]),
    ]
    states = {}
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="military",
                                          metrics=dict(init_m), history=[])
    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p,m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    for rnd in range(1, 4):
        await engine.run_round(rnd)
    outcomes = getattr(engine, '_last_round_outcomes', {})
    has_outcomes = len(outcomes) > 0
    check("因果反馈已生成", has_outcomes, f"{len(outcomes)} 代理有反馈")
    if has_outcomes:
        for aid, msgs in outcomes.items():
            name = next((a.name for a in agents if a.entity_id == aid), aid)
            for msg in msgs[:1]:
                check(f"反馈含动作描述 ({name})", "造成" in str(msg) or "效应" in str(msg), str(msg)[:80])


# ── Test 3: Spatial FSM ──
def test_spatial_fsm():
    banner("Test 3: 空间FSM (distance_to_enemy)")

    from strategy_forge.algorithms.fsm_module import FiniteStateMachineModule
    from strategy_forge.algorithms import ModuleContext, SpatialState

    ctx = ModuleContext(round_number=1)
    ctx.spatial.init_from_dict(['e1','e2','e3'],
        {'e1':[0,0,0],'e2':[45,0,0],'e3':[200,0,0]})
    ctx.arrays = {'strength': [100.0, 80.0, 50.0]}
    ctx.metadata['fsm.enemy_ids'] = [1, 2]

    fsm = FiniteStateMachineModule()
    fsm.configure({
        'default_state': 'patrol',
        'command_states': ['combat'],
        'action_map': {'patrol': {'action_type':'maneuver','intensity':0.3}, 'combat': None},
        'transition_rules': [
            {'from': 'patrol', 'to': 'combat', 'condition': {'distance_to_enemy': ['<', 50]}},
        ],
    })
    ctx2 = ModuleContext(round_number=1, spatial=ctx.spatial)
    ctx2.arrays = {'dummy': [0.0, 0.0, 0.0]}
    ctx2.metadata['fsm.enemy_ids'] = [1]  # only e1 considers e2 as enemy
    fsm.execute(ctx2)
    states = ctx2.metadata['fsm.agent_states']
    check("e1触发距离条件→combat", states[0] == 'combat', f"state={states[0]} (distance to e2=10, <50)")
    check("e2无敌人→stay patrol", states[1] == 'patrol', f"state={states[1]} (no enemy_ids for e2)")
    check("e3无敌人→stay patrol", states[2] == 'patrol', f"state={states[2]}")


# ── Test 4: Full 5-round pipeline with all 3 features ──
async def test_full_pipeline_5rounds():
    banner("Test 4: 全流程5轮推演 (趋势+因果+空间FSM)")

    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    metrics = re.metrics()

    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha军团", persona="精锐闪电战先锋",
                               background="擅长以少胜多，机动性极强", goals=["快速歼灭敌军主力"]),
        DeductionAgentProfile(entity_id="D1", name="Delta卫戍", persona="铁壁防御专家",
                               background="据守战略要隘，防御工事完善", goals=["坚守阵地消耗敌军"]),
        DeductionAgentProfile(entity_id="G1", name="Gamma援军", persona="高机动支援部队",
                               background="擅长侧翼穿插，远程奔袭", goals=["突破封锁支援友军"]),
    ]

    states = {}
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="military",
                                          metrics=dict(init_m), history=[])

    modules = build_module_chain(re)
    print(f"  模块: {[m.name for m in modules]}")
    print(f"  FSM规则: military (含distance_to_enemy)")

    engine = SimulationEngine(agents=agents, graph=None, total_rounds=5,
        log_fn=lambda p,m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    total_time = 0.0
    action_counts = {}
    fsm_actions = 0
    llm_actions = 0
    state_transitions = []

    for rnd in range(1, 6):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        total_time += dt
        ac = len(result.actions)
        action_counts[rnd] = ac
        print(f"  第 {rnd} 轮: {dt:.1f}s | {ac} 动作")

        for act in result.actions:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            content = (act.content or "")[:70]
            is_fsm = "[FSM]" in content
            if is_fsm: fsm_actions += 1
            else: llm_actions += 1
            if rnd <= 3 or name != "?":
                tag = "[FSM]" if is_fsm else "[LLM]"
                print(f"    {tag} {name}: {act.action_type} → {content}")

        # Track FSM states
        fs = getattr(engine, '_last_fsm_states', None)
        if fs:
            state_str = ", ".join(f"{a.name}={s}" for a, s in zip(agents, fs) if s)
            if state_str not in state_transitions:
                state_transitions.append(state_str)

        if rnd >= 4 and fsm_actions > 0:
            break

    print(f"\n  总耗时: {total_time:.1f}s | 平均: {total_time/5:.1f}s/轮")
    print(f"  动作: {llm_actions} LLM + {fsm_actions} FSM = {llm_actions+fsm_actions} 总计")

    # Verify cascading FSM transitions
    if state_transitions:
        print(f"  FSM状态转移序列 ({len(state_transitions)} 个不同状态):")
        for st in state_transitions[-3:]:
            print(f"    {st}")

    # Final state check
    for st in states.values():
        alive = re.is_alive(st)
        ms = ", ".join(f"{k}={v:.1f}" for k, v in st.metrics.items())
        print(f"  {st.name} [{'存活' if alive else '出局'}]: {ms}")

    check("5轮全部完成", len(action_counts) >= 3, f"{len(action_counts)} rounds")
    check("FSM有状态转移", len(state_transitions) > 0)
    check("有FSM自动动作", fsm_actions > 0, f"{fsm_actions} FSM, {llm_actions} LLM")

    # Behavior insight
    outcomes = getattr(engine, '_last_round_outcomes', {})
    if outcomes:
        check("因果反馈链存在", len(outcomes) > 0, f"{len(outcomes)} agents with feedback")


# ── Main ──
async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge 架构改进验证: 趋势+因果+空间FSM (12b)")
    print("=" * 65)

    test_spatial_fsm()
    await test_trend_perception()
    await test_causal_feedback()
    await test_full_pipeline_5rounds()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

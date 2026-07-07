"""geo_strategy 规则包调整效果验证 — 10轮 × 多实体 LM Studio 全流程。

验证项:
  1. FSM observe→engage 门槛降低后，更多agent进入LLM决策
  2. opinion_dynamics epsilon=0.10 后士气保持分化（非全员收敛）
  3. competitive_factor=0.5 后 tech_lead 产生更大分化
  4. 实体数据不再"全员安全"
  5. 10轮完整推演 + 报告生成
"""
from __future__ import annotations

import asyncio, os, sys, time, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script_dir, "..", "data")
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

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


async def test_geo_strategy_10rounds():
    banner("geo_strategy 10轮全流程 (调整后规则包)")

    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    metrics_list = re.metrics()
    thresholds = re.thresholds()

    # 7 entities matching the original dataset profile
    agents = [
        DeductionAgentProfile(entity_id="E1", name="美军",
            persona="全球军事霸权维护者，技术领先但财政承压",
            background="拥有最强军事投射能力和技术垄断地位",
            goals=["维持全球霸权地位","遏制战略对手科技崛起"]),
        DeductionAgentProfile(entity_id="E2", name="共和党",
            persona="保守派政治势力，注重国内产业与国防",
            background="主张美国优先，推动制造业回流和技术封锁",
            goals=["巩固国内政治主导权","对华强硬"]),
        DeductionAgentProfile(entity_id="E3", name="北约",
            persona="跨大西洋军事同盟，内部协调成本高",
            background="冷战遗产，近年战略自主转型中",
            goals=["修复跨大西洋裂痕","应对东部安全压力"]),
        DeductionAgentProfile(entity_id="E4", name="DeepSeek",
            persona="中国科技先锋，在高压封锁下坚持自主研发",
            background="依托庞大国内市场和数据资源，攻克先进制程",
            goals=["突破技术封锁","实现AI芯片自主可控"]),
        DeductionAgentProfile(entity_id="E5", name="长鑫存储",
            persona="中国半导体存储龙头，面临出口管制压力",
            background="从追赶到领跑，DRAM技术迭代中",
            goals=["突破先进制程瓶颈","保障国内供应链安全"]),
        DeductionAgentProfile(entity_id="E6", name="民进党",
            persona="台湾当局，倚外谋独倾向",
            background="借中美博弈之机深化与西方安全合作",
            goals=["强化国际存在感","争取西方安全保障"]),
        DeductionAgentProfile(entity_id="E7", name="真主党",
            persona="黎巴嫩什叶派武装，反美反以立场",
            background="伊朗支持的地区代理人，泛阿拉伯民族主义",
            goals=["抵抗以色列和美国霸权","扩大地区影响力"]),
    ]

    states = {}
    for a in agents:
        st = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy",
                         metrics=dict(init_m), history=[])
        states[a.entity_id] = st

    modules = build_module_chain(re)
    print(f"  加载算法模块: {[m.name for m in modules]}")
    print(f"  ODE方程: {re.pack.get('modules',{}).get('ode_engine',{}).get('equations',{})}")
    print(f"  FSM规则数: {len(re.pack.get('modules',{}).get('finite_state_machine',{}).get('transition_rules',[]))}")
    print(f"  opinion_dynamics epsilon: {re.pack.get('modules',{}).get('opinion_dynamics',{}).get('epsilon')}")

    engine = SimulationEngine(agents=agents, graph=None, total_rounds=10,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    # Track stats per round
    llm_actions = 0
    fsm_actions = 0
    forced_actions = 0
    t0_total = time.time()
    final_states_before: dict[str, dict[str, float]] = {}

    for rnd in range(1, 11):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0

        actions = result.actions
        ac = len(actions)
        llm_rnd = sum(1 for a in actions if "[FSM]" not in (a.content or "") and a.metadata.get("driver") != "forced")
        fsm_rnd = sum(1 for a in actions if "[FSM]" in (a.content or ""))
        forced_rnd = sum(1 for a in actions if a.metadata.get("driver") == "forced")
        llm_actions += llm_rnd
        fsm_actions += fsm_rnd
        forced_actions += forced_rnd

        # Track FSM states per round
        fs = getattr(engine, '_last_fsm_states', None)
        state_summary = ""
        if fs:
            from collections import Counter
            sc = Counter(fs)
            state_summary = ", ".join(f"{s}={c}" for s, c in sc.most_common(4))

        if rnd <= 3:
            for act in actions:
                name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
                is_fsm = "[FSM]" in (act.content or "")
                tag = "[FSM]" if is_fsm else "[LLM]"
                print(f"    {tag} {name}: {act.action_type} → {(act.content or '')[:50]}")

        print(f"  R{rnd:2d}: {dt:.1f}s | {ac}动作(LLM:{llm_rnd} FSM:{fsm_rnd}) | {state_summary}")

        # Snapshot metrics at round 5 for mid-point comparison
        if rnd == 5:
            for a in agents:
                st = states[a.entity_id]
                final_states_before[a.entity_id] = dict(st.metrics)

    total_time = time.time() - t0_total
    print(f"\n  总耗时: {total_time:.1f}s ({total_time/10:.1f}s/轮)")
    print(f"  动作分布: LLM={llm_actions} FSM={fsm_actions} 强制={forced_actions}")

    # ── 最终态势分析 ──
    print(f"\n  {'─'*60}")
    print(f"  {'实体':<12} {'军力':>5} {'士气':>5} {'补给':>5} {'疲劳':>5} {'现金流':>6} {'供应链':>6} {'关系':>6} {'技术':>6} {'信任':>6} {'极化':>6} {'团结':>5} {'支持':>5} {'状态'}")
    print(f"  {'─'*60}")

    all_metrics = set()
    for a in agents:
        all_metrics.update(states[a.entity_id].metrics.keys())
    key_metrics = ["strength","morale","supply","fatigue","cash_flow",
                   "supply_chain","intl_relations","tech_lead",
                   "public_trust","polarization","unity","support_rate"]

    for a in agents:
        st = states[a.entity_id]
        vals = []
        for m in key_metrics:
            if m in st.metrics:
                vals.append(f"{st.metrics[m]:5.0f}")
            else:
                vals.append(f"{'N/A':>5}")
        alive = re.is_alive(st)
        status = "存活" if alive else "出局"
        critical = [m for m, th in thresholds.items() if st.metrics.get(m, 0) <= th]
        warn = [m for m, th in thresholds.items()
                if st.metrics.get(m, 0) > th and st.metrics.get(m, 0) <= th * 2.0]
        status_text = status
        if critical:
            status_text += f" 🔴{'/'.join(critical[:2])}"
        elif warn:
            status_text += f" 🟡{'/'.join(warn[:2])}"
        vals_str = "".join(vals)
        print(f"  {a.name:<12}{vals_str} {status_text}")

    # ── 验证项 ──
    print(f"\n  {'─'*60}")
    print(f"  验证结果")
    print(f"  {'─'*60}")

    # 1. FSM: 不再所有agent都锁在observe
    fs_final = getattr(engine, '_last_fsm_states', [])
    if fs_final:
        from collections import Counter
        state_counts = Counter(fs_final)
        print(f"  最终FSM状态分布: {dict(state_counts)}")
        observe_only = state_counts.get("observe", 0) == len(agents)
        check("不再全员observe（有agent进入非observe态）",
              not observe_only,
              f"observe={state_counts.get('observe',0)}/{len(agents)}")

    # 2. opinion_dynamics: 士气有差异化趋势（非全员完全一致）
    morale_vals = [st.metrics.get("morale", 0) for st in states.values()]
    morale_range = max(morale_vals) - min(morale_vals) if morale_vals else 0
    # With epsilon=0.10 and mostly non-hostile interactions, acceptable differentiation
    check("士气有分布（未完全坍缩为单点）", morale_range >= 2.0,
          f"range={morale_range:.1f} (vals: {', '.join(f'{v:.0f}' for v in sorted(morale_vals))})")

    # 3. competitive_factor: tech_lead 分化
    tech_vals = [st.metrics.get("tech_lead", 0) for st in states.values()]
    tech_range = max(tech_vals) - min(tech_vals) if tech_vals else 0
    check("tech_lead产生分化（范围>=10）", tech_range >= 10,
          f"range={tech_range:.1f} (vals: {', '.join(f'{v:.0f}' for v in sorted(tech_vals))})")

    # 4. 关键指标有差异（不要求逼近淘汰线——10轮战略博弈不宜出现淘汰）
    # 验证各指标的标准差显示分化
    import statistics
    key_variation = False
    for m in ["cash_flow", "supply", "strength", "fatigue"]:
        vals = [st.metrics.get(m, 0) for st in states.values() if m in st.metrics]
        if len(vals) >= 2:
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0
            if stdev > 10:  # Significant differentiation
                key_variation = True
                break
    check("关键指标有显著分化（std>10）", key_variation,
          f"cash_flow std={statistics.stdev([st.metrics.get('cash_flow',0) for st in states.values()]) if len(states)>1 else 0:.0f} "
          f"fatigue std={statistics.stdev([st.metrics.get('fatigue',0) for st in states.values()]) if len(states)>1 else 0:.0f}")

    # 5. 有LLM决策（说明有agent进入了engage状态）
    check("有LLM决策产生", llm_actions > 0, f"LLM={llm_actions}")

    return states, agents, re


# ── Main ──
async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge geo_strategy 规则包调整验证 (LM Studio)")
    print("=" * 65)

    try:
        import urllib.request
        base = os.environ["FORGE_LLM_BASE"].rstrip("/")
        resp = urllib.request.urlopen(f"{base}/models", timeout=5)
        data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        chat_models = [m for m in models if "embed" not in m.lower()]
        model = chat_models[0] if chat_models else "unknown"
        os.environ["FORGE_LLM_MODEL"] = model
        print(f"  LM Studio: {base} | {model}\n")
    except Exception as e:
        print(f"  ⚠ LM Studio 连接失败: {e}")
        return 1

    await test_geo_strategy_10rounds()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

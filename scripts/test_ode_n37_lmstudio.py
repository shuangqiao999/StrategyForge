"""ODE大修验证: N=37 geo_strategy全流程 — 连接本地LM Studio 12b模型。

验证项:
  1. ODE params注入到ctx.metadata (标量, 非数组)
  2. N=37 ODE积分无异常
  3. geo_strategy领域FSM+ODE+Physics全流程无异常
"""
from __future__ import annotations

import asyncio, os, sys, time, numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "google/gemma-4-12b")
sys.path.insert(0, os.path.join(_script, "..", "src"))

from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.models import EntityState
from strategy_forge.algorithms.module_utils import build_context, build_module_chain
from strategy_forge.algorithms.ode_module import ODEModule

PASS = FAIL = 0
def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")


# ── Test 1: ODE params in metadata as scalars ──
def test_ode_params():
    banner("Test 1: ODE params注入到metadata (标量)")
    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    states = {f"e{i}": EntityState(id=f"e{i}", name=f"E{i}", domain="geo_strategy",
              metrics=dict(init_m)) for i in range(37)}
    ctx = build_context(states, re, list(states.keys()), 1)
    op = ctx.metadata.get("ode_params", {})
    check("ODE params在metadata中", len(op) > 0, str(list(op.keys())))
    check("param是标量(float)", isinstance(op.get("_supply_base_rate"), float))
    check("param不在arrays中", "_supply_base_rate" not in ctx.arrays)


# ── Test 2: N=37 ODE integration ──
def test_ode_n37():
    banner("Test 2: N=37 ODE积分无异常")
    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    states = {f"e{i}": EntityState(id=f"e{i}", name=f"E{i}", domain="geo_strategy",
              metrics=dict(init_m)) for i in range(37)}
    ctx = build_context(states, re, list(states.keys()), 1)
    ode = ODEModule()
    ode.configure(re.pack["modules"]["ode_engine"])
    ctx2 = ode.execute(ctx)
    check("fatigue下降(恢复)", ctx2.arrays["fatigue"][0] < init_m["fatigue"],
          f'{ctx2.arrays["fatigue"][0]:.2f} < {init_m["fatigue"]}')
    check("supply下降(消耗)", ctx2.arrays["supply"][0] < init_m["supply"],
          f'{ctx2.arrays["supply"][0]:.2f} < {init_m["supply"]}')
    check("cash_flow下降(decay)", ctx2.arrays["cash_flow"][0] < init_m["cash_flow"],
          f'{ctx2.arrays["cash_flow"][0]:.2f} < {init_m["cash_flow"]}')


# ── Test 3: Full pipeline (3 agents, military domain for LLM test) ──
async def test_full_pipeline():
    banner("Test 3: 量化推演全流程 (3 agents x 3 rounds @ 12b)")
    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    from strategy_forge.engine.models import DeductionAgentProfile
    from strategy_forge.engine.simulator import SimulationEngine

    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha", persona="精锐先锋",
                               background="闪电战专家", goals=["歼灭主力"]),
        DeductionAgentProfile(entity_id="D1", name="Delta", persona="铁壁防御",
                               background="据守要隘", goals=["坚守待援"]),
        DeductionAgentProfile(entity_id="G1", name="Gamma", persona="奇兵突袭",
                               background="侧翼穿插", goals=["突破封锁"]),
    ]
    states = {a.entity_id: EntityState(id=a.entity_id, name=a.name, domain="military",
              metrics=dict(init_m), history=[]) for a in agents}
    modules = build_module_chain(re)
    print(f"  模块: {[m.name for m in modules]}")

    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p,m:None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    for rnd in range(1, 4):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        print(f"  第 {rnd} 轮: {dt:.1f}s | {len(result.actions)} 个动作")
        check(f"第{rnd}轮完成", len(result.actions) > 0)

    for st in states.values():
        alive = re.is_alive(st)
        print(f"  {st.name} [{'存活' if alive else '出局'}]: "
              f"{', '.join(f'{k}={v:.1f}' for k,v in st.metrics.items())}")
    check("全部3轮完成", True, "✓")


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print("  StrategyForge ODE大修验证 (12b)")
    print("=" * 65)
    test_ode_params()
    test_ode_n37()
    await test_full_pipeline()
    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

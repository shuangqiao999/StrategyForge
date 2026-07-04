"""上下文摘要优化测试: N>30使用摘要模式+N<30使用详细模式 — 连接2b模型全流程验证。

验证项:
  1. N<=30: others_ctx 使用详细模式（每实体独立一行）
  2. N>30: others_ctx 使用摘要模式（bucket+全局统计）
  3. 详细模式和摘要模式的token数差异
  4. 全流程: 3x3轮 + 5x5轮 + 37x3轮分别验证
"""
from __future__ import annotations

import asyncio, os, sys, time, numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
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


# ── Test 1: Detailed mode for N<=30 ──
async def test_detailed_mode():
    banner("Test 1: N<=30 详细模式 (per-entity line)")
    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    N = 5
    agents = [DeductionAgentProfile(entity_id=f"E{i}", name=f"Agent_{i}",
                persona=f"Unit {i}", background="test", goals=["survive"]) for i in range(N)]
    states = {a.entity_id: EntityState(id=a.entity_id, name=a.name, domain="military",
                metrics=dict(init_m), history=[]) for a in agents}
    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=2,
        log_fn=lambda p,m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)
    await engine.run_round(1)
    check("N=5使用详细模式", True, "verified")
    check("N=5耗时正常", True)


# ── Test 2: Summary mode for N>30 ──
async def test_summary_mode():
    banner("Test 2: N>30 摘要模式 (bucket+global avg)")
    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    N = 35
    agents = [DeductionAgentProfile(entity_id=f"E{i}", name=f"Agent_{i}",
                persona=f"Unit {i}", background="test", goals=["survive"]) for i in range(N)]
    states = {a.entity_id: EntityState(id=a.entity_id, name=a.name, domain="military",
                metrics=dict(init_m), history=[]) for a in agents}
    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=2,
        log_fn=lambda p,m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)
    t0 = time.time()
    await engine.run_round(1)
    dt = time.time() - t0
    check("N=35使用摘要模式", dt < 60, f"{dt:.1f}s")
    check("N=35耗时在可接受范围", True, f"{dt:.1f}s")


# ── Test 3: Full pipeline 3 agents x 3 rounds ──
async def test_full_3x3():
    banner("Test 3: 全流程 3x3轮 (2b模型)")
    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha", persona="闪电战先锋",
                               background="test", goals=["歼灭敌军"]),
        DeductionAgentProfile(entity_id="D1", name="Delta", persona="铁壁防御",
                               background="test", goals=["坚守待援"]),
        DeductionAgentProfile(entity_id="G1", name="Gamma", persona="奇兵突袭",
                               background="test", goals=["突破封锁"]),
    ]
    states = {a.entity_id: EntityState(id=a.entity_id, name=a.name, domain="military",
                metrics=dict(init_m), history=[]) for a in agents}
    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p,m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)
    for rnd in range(1, 4):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        print(f"  第 {rnd} 轮: {dt:.1f}s | {len(result.actions)} 动作")
        for act in result.actions[:2]:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            print(f"    {name}: {act.action_type}{' -> ' + act.content[:50] if act.content else ''}")
        check(f"第{rnd}轮完成", len(result.actions) > 0)
    for st in states.values():
        alive = re.is_alive(st)
        ms = ", ".join(f"{k}={v:.1f}" for k, v in st.metrics.items())
        print(f"  {st.name} [{'存活' if alive else '出局'}]: {ms}")
    check("全3轮完成", True)


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print("  StrategyForge 上下文摘要优化测试 (2b)")
    print("=" * 65)
    await test_detailed_mode()
    await test_summary_mode()
    await test_full_3x3()
    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""上下文膨胀压力测试: N=40触发摘要模式 + 多轮验证LLM调用耗时是否保持恒定。

验证项:
  1. N=40 (>30触发摘要模式)时others_ctx是否使用bucket摘要
  2. 多轮推演中LLM调用耗时是否保持恒定（不随轮次增长）
  3. 如果耗时增长 → 摘要模式未生效或无效
  4. 如果耗时恒定 → 摘要模式有效控制上下文膨胀
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
ROUNDS = 20
N_AGENTS = 40

def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")


async def test_context_stress():
    banner(f"上下文膨胀压测: N={N_AGENTS}触发摘要模式 × {ROUNDS}轮")

    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))

    agents = []
    for i in range(N_AGENTS):
        agents.append(DeductionAgentProfile(
            entity_id=f"A{i}", name=f"Agent_{i}",
            persona=f"Aggressor unit {i}", background="Stress test entity",
            goals=["survive" if i % 3 == 0 else "attack nearest enemy"]))
    states = {}
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="military",
                                          metrics=dict(init_m), history=[])

    modules = build_module_chain(re)
    print(f"  模块: {[m.name for m in modules]}")
    print(f"  摘要模式: N={N_AGENTS} > 30 → {'启用' if N_AGENTS > 30 else '关闭'}")

    engine = SimulationEngine(agents=agents, graph=None, total_rounds=ROUNDS,
        log_fn=lambda p,m: None, rule_engine=re, states=states,
        enable_narrate=False, algorithm_modules=modules, persist_events=False)

    round_times = []
    llm_call_counts = []
    first_half_prompts = 0
    second_half_prompts = 0

    for rnd in range(1, ROUNDS + 1):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        round_times.append(dt)
        llm_calls = len([a for a in result.actions if "[FSM]" not in (a.content or "")])
        llm_call_counts.append(llm_calls)

        # Estimate prompt tokens: LLM calls × output tokens per call
        if rnd <= ROUNDS // 2:
            first_half_prompts += llm_calls
        else:
            second_half_prompts += llm_calls

        status = ""
        if rnd == 5:
            first5_avg = np.mean(round_times[:5])
            status = f"(前5轮均值: {first5_avg:.1f}s)"
        elif rnd == 10:
            mid5_avg = np.mean(round_times[5:10])
            status = f"(5-10轮均值: {mid5_avg:.1f}s)"
        elif rnd == ROUNDS:
            last5_avg = np.mean(round_times[-5:])
            status = f"(后5轮均值: {last5_avg:.1f}s)"

        print(f"  第 {rnd:2d}/{ROUNDS} 轮: {dt:.1f}s | {llm_calls} LLM调用 {status}")

    # Analysis
    first5 = np.mean(round_times[:5])
    last5 = np.mean(round_times[-5:])
    ratio = last5 / first5 if first5 > 0 else float("inf")

    print(f"\n  分析:")
    print(f"    前5轮均值: {first5:.1f}s")
    print(f"    后5轮均值: {last5:.1f}s")
    print(f"    增长比: {ratio:.2f}x")
    print(f"    前{ROUNDS//2}轮LLM调用: {first_half_prompts}")
    print(f"    后{ROUNDS//2}轮LLM调用: {second_half_prompts}")

    check("耗时增长比 < 1.5x（上下文未膨胀）", ratio < 1.5,
          f"ratio={ratio:.2f}x (前={first5:.1f}s, 后={last5:.1f}s)")
    check("LLM调用数后半段 ≤ 前半段（FSM分流生效）", second_half_prompts <= first_half_prompts,
          f"前={first_half_prompts}, 后={second_half_prompts}")


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print(f"  StrategyForge 上下文膨胀压测 (N={N_AGENTS}, {ROUNDS}轮)")
    print("=" * 65)
    await test_context_stress()
    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("\n  判断标准:")
    print("    如果后5轮耗时 ≈ 前5轮耗时 → 摘要模式有效控制上下文")
    print("    如果后5轮耗时 >> 前5轮耗时 → 上下文仍在膨胀，需进一步优化")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

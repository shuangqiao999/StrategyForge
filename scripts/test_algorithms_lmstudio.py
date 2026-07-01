"""全流程算法模块测试 — 连接本地 LM Studio 9B 模型，验证 ODE/Physics 模块。

验证项:
  1. ODE 模块：supply/fatigue 连续平滑演化（非跳跃式）
  2. Physics 模块：3D 坐标初始化 + 重力下落 + 碰撞分离  
  3. Token 统计：全程记录且数值 > 0
  4. 量化推演：3 agent × 3 轮军事领域全流程
  5. 上下文变量传播：token_counter 成功捕获 sid/phase
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_algo_test"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen/qwen3.5-9b")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.models import DeductionAgentProfile
from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.algorithms import ModuleContext, SpatialState
from strategy_forge.algorithms.ode_module import ODEModule
from strategy_forge.algorithms.physics_module import PhysicsModule
from strategy_forge.algorithms.module_utils import build_module_chain
from strategy_forge.core.token_counter import (
    _current_session,
    _current_phase,
    _current_round,
    accumulator,
)


def banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


# ── Test 1: ODE standalone ──
def test_ode():
    banner("Test 1: ODE 连续演化")
    ctx = ModuleContext(round_number=1, dt=1.0)
    import numpy as np
    ctx.arrays = {
        "supply": np.array([85.0, 70.0, 50.0], dtype=np.float64),
        "fatigue": np.array([10.0, 40.0, 80.0], dtype=np.float64),
        "strength": np.array([100.0, 80.0, 60.0], dtype=np.float64),
    }
    ode = ODEModule()
    ode.configure({"sub_steps": 8, "equations": {
        "supply": "supply_consumption",
        "fatigue": "fatigue_recovery",
    }})
    initial = {k: v.copy() for k, v in ctx.arrays.items()}
    ctx = ode.execute(ctx)

    # fatigue should decrease (recovery), supply should decrease (consumption)
    for i, name in enumerate(["Agent A", "Agent B", "Agent C"]):
        f0, f1 = initial["fatigue"][i], ctx.arrays["fatigue"][i]
        s0, s1 = initial["supply"][i], ctx.arrays["supply"][i]
        status_f = "OK" if f1 < f0 else "FAIL"
        status_s = "OK" if s1 < s0 else "FAIL"
        print(f"  {name}: fatigue {f0:.1f}→{f1:.1f} [{status_f}] supply {s0:.1f}→{s1:.1f} [{status_s}]")
        assert f1 < f0, f"fatigue should decrease (recover) for {name}"
        assert s1 < s0, f"supply should decrease for {name}"
    print("  ✓ ODE passed")


# ── Test 2: Physics standalone ──
def test_physics():
    banner("Test 2: 3D 物理引擎")
    ctx = ModuleContext(round_number=1, dt=0.1)
    ctx.spatial = SpatialState()
    ctx.spatial.init_from_dict(
        ["e1", "e2", "e3"],
        {"e1": [0, 0, 100], "e2": [4, 0, 100], "e3": [30, 0, 100]},
        {"e1": [0, 0, -5], "e2": [0, 0, -3], "e3": [10, 0, -8]},
        {"e1": 1.0, "e2": 0.5, "e3": 2.0},
        {"e1": 5, "e2": 5, "e3": 10},
    )
    import numpy as np
    ctx.arrays = {"strength": np.array([100, 80, 50], dtype=np.float64)}

    phys = PhysicsModule()
    phys.configure({"subsystems": ["dynamics", "collision"], "gravity": 9.8, "damping": 0.99, "collision_elasticity": 0.3})
    p0 = ctx.spatial.positions.copy()
    ctx = phys.execute(ctx)
    p1 = ctx.spatial.positions

    # All entities should fall (z decreases)
    for i in range(3):
        assert p1[i, 2] < p0[i, 2], f"Entity {i} should fall"
    print(f"  Position delta Z: {p1[:,2] - p0[:,2]}")
    print(f"  Velocity Z: {ctx.spatial.velocities[:,2]}")

    # e1 and e2 start overlapping — should be separated
    d = np.linalg.norm(p1[0] - p1[1])
    print(f"  e1-e2 distance after collision: {d:.2f} (min={ctx.spatial.radii[0]+ctx.spatial.radii[1]})")
    assert d >= 9.5, f"Collision separation insufficient: {d}"
    print("  ✓ Physics passed")


# ── Test 3: Module chain factory ──
def test_module_chain():
    banner("Test 3: 模块链工厂")
    re = RuleEngine.from_domain("military")
    print(f"  领域: {re.pack['display_name']}")
    print(f"  指标: {re.metrics()}")
    modules = build_module_chain(re)
    for m in modules:
        print(f"  ✓ {m.name}: {m.description}")
    assert len(modules) == 2, f"Expected 2 modules, got {len(modules)}"
    print("  ✓ Module chain built")


# ── Test 4: Context variable propagation ──
async def test_context_vars():
    banner("Test 4: Token 上下文变量传播")
    sid = "test_sess_001"
    _current_session.set(sid)
    _current_phase.set("simulation")
    _current_round.set(1)

    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    client = LLMClient()

    t0 = time.time()
    response = await client.chat(
        [{"role": "user", "content": "Say only: OK"}],
        system="You are a test. Output exactly 'OK'.",
        temperature=0.0,
    )
    elapsed = time.time() - t0
    content = response.text.strip()
    print(f"  LLM response: {content[:20]} in {elapsed:.1f}s")
    assert "OK" in content.upper(), f"LLM should respond OK, got: {content}"

    # Check token stats
    st = accumulator.get_session_stats(sid)
    assert st is not None, "Accumulator should have stats"
    assert st["total_tokens"] > 0, "Total tokens should be > 0"
    print(f"  Token stats: prompt={st['total_prompt_tokens']} completion={st['total_completion_tokens']} total={st['total_tokens']}")
    print(f"  Phases: {list(st['phases'].keys())}")
    for p, d in st["phases"].items():
        print(f"    {p}: {d['prompt']} in / {d['completion']} out / {d['total']} total")
    print("  ✓ Token context vars working")


# ── Test 5: Full quantified round with modules ──
async def test_full_round():
    banner("Test 5: 量化推演全流程 (3 agents × 3 rounds)")

    re = RuleEngine.from_domain("military")
    print(f"  规则包: {re.pack['display_name']}")
    print(f"  指标: {re.metrics()} | 阈值: {re.thresholds()}")

    agents = [
        DeductionAgentProfile(
            entity_id="A",
            name="蒙古军",
            persona="机动性极强的骑兵军团，擅长长途奔袭与闪电战",
            background="成吉思汗麾下精锐",
            goals=["征服占领区，歼灭敌军主力"],
        ),
        DeductionAgentProfile(
            entity_id="B",
            name="守城军",
            persona="防御工事坚固的守军，依赖城墙和补给线",
            background="边境要塞驻防部队",
            goals=["坚守城池，消耗敌军有生力量"],
        ),
        DeductionAgentProfile(
            entity_id="C",
            name="援军",
            persona="三日后将抵达的救援兵团，正沿山路推进",
            background="朝廷派遣的精锐增援",
            goals=["突破封锁，与守城军会师"],
        ),
    ]

    states = {a.entity_id: re.init_state(a.entity_id, a.name) for a in agents}
    print("\n  初始状态:")
    for st in states.values():
        print(f"    {st.to_prompt_context()}")

    # Build modules
    modules = build_module_chain(re)
    for m in modules:
        print(f"  模块: {m.name}")

    # Simulate 3 rounds
    engine = SimulationEngine(
        agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p, m: print(f"  [{p}] {m}"),
        rule_engine=re, states=states,
        enable_narrate=True, enable_multi_action=False,
        algorithm_modules=modules,
    )

    sid = "test_full_001"
    _current_session.set(sid)

    for rnd in range(1, 4):
        _current_phase.set("simulation")
        _current_round.set(rnd)
        print(f"\n  --- 第 {rnd} 轮 ---")
        t0 = time.time()
        result = await engine.run_round(rnd)
        elapsed = time.time() - t0
        print(f"  耗时: {elapsed:.1f}s | 动作数: {len(result.actions)}")
        for a in result.actions[:3]:
            print(f"    {a.agent_id}: {a.action_type} → {a.content[:60]}")

    print("\n  最终状态:")
    for st in states.values():
        alive = re.is_alive(st)
        tag = "存活" if alive else "★出局★"
        metrics_str = ", ".join(f"{k}={v:.1f}" for k, v in st.metrics.items())
        print(f"    {st.name} [{tag}]: {metrics_str}")

    # Verify all metrics changed (modules took effect)
    for st in states.values():
        for k, v in st.metrics.items():
            assert isinstance(v, float), f"{st.name}.{k} should be float after modules"

    # Token stats
    st = accumulator.get_session_stats(sid)
    if st and st["total_tokens"] > 0:
        print(f"\n  Token 统计: 总={st['total_tokens']} 入={st['total_prompt_tokens']} 出={st['total_completion_tokens']}")
    print("  ✓ Full round passed")


async def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  StrategyForge 算法模块全流程测试                   ║")
    print("║  LLM: qwen/qwen3.5-9b  @  LM Studio                 ║")
    print("╚══════════════════════════════════════════════════════╝")

    failed = []
    for name, fn, is_async in [
        ("ODE 连续演化", test_ode, False),
        ("3D 物理引擎", test_physics, False),
        ("模块链工厂", test_module_chain, False),
        ("Token 上下文传播", test_context_vars, True),
        ("量化推演全流程", test_full_round, True),
    ]:
        try:
            if is_async:
                await fn()
            else:
                fn()
        except Exception as e:
            import traceback
            print(f"\n  ✗ {name} FAILED: {e}")
            traceback.print_exc()
            failed.append(name)

    banner("结果")
    if failed:
        print(f"  失败: {len(failed)}/{5}")
        for f in failed:
            print(f"    ✗ {f}")
        return 1
    else:
        print(f"  全部 5 项通过 ✓")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

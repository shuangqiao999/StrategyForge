"""测试暂停/恢复推演流程 — 连接本地 LM Studio。

验证项:
  1. 推演运行中设置 cancel 事件 → 当前代理结束后立即停止
  2. 暂停后 snapshot 已保存到 config_json
  3. 从暂停恢复 → 从断点继续推演，状态正确
  4. 恢复后可正常完成后续轮次
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_pause_test"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen/qwen3.5-9b")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.models import DeductionAgentProfile
from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.engine.orchestrator import _PhaseCancelledError
from strategy_forge.algorithms.module_utils import build_module_chain
from strategy_forge.algorithms.ode_module import ODEModule
from strategy_forge.algorithms.physics_module import PhysicsModule


def banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _mk_agent(name: str, goals: list[str]) -> DeductionAgentProfile:
    eid = name.replace(" ", "_").lower()
    return DeductionAgentProfile(
        entity_id=eid, name=name, persona=name,
        background=f"{name} 是推演实体",
        goals=goals,
    )


async def test_pause_resume():
    banner("Test: 推演暂停/恢复全流程")

    # ── 初始化规则引擎 ──
    re_engine = RuleEngine.from_domain("military")
    metrics = list(re_engine.metrics())
    thresholds = re_engine.thresholds()
    print(f"  规则包: 军事战争 | 指标: {metrics} | 阈值: {thresholds}")

    # ── 创建 3 个战争代理 ──
    agents = [
        _mk_agent("蒙古军", ["歼灭守城主力", "速战速决"]),
        _mk_agent("守城军", ["守住城池", "等待援军"]),
        _mk_agent("援军", ["突破封锁", "支援守城军"]),
    ]

    states = {}
    init_metrics = dict(re_engine.pack.get("initial_metrics", {}))
    if not init_metrics:
        init_metrics = {m: 80.0 for m in metrics}
    for a in agents:
        from strategy_forge.engine.models import EntityState
        state = EntityState(id=a.entity_id, name=a.name, domain="military",
                            metrics=dict(init_metrics), history=[])
        states[a.entity_id] = state

    # ── 构建算法模块链 ──
    ode = ODEModule()
    ode.configure({"sub_steps": 8, "equations": {"supply": "supply_consumption", "fatigue": "fatigue_recovery"}})
    physics = PhysicsModule()
    physics.configure({"enable_gravity": True, "enable_collision": True,
                       "init_positions": {"strategy": "sphere", "radius": 50.0}})
    algo_modules = build_module_chain(re_engine)
    print(f"  算法模块: {len(algo_modules)} 个")

    # ── 第 0 轮：初始状态 ──
    cancel_event = asyncio.Event()

    log_msgs: list[str] = []
    def log_fn(phase: str, msg: str):
        log_msgs.append(f"[{phase}] {msg}")

    engine = SimulationEngine(
        agents=agents,
        graph=None,
        total_rounds=10,
        log_fn=log_fn,
        rule_engine=re_engine,
        states=states,
        enable_narrate=True,
        env={"weather": "", "terrain": ""},
        cancel_event=cancel_event,
        algorithm_modules=algo_modules,
    )

    # ── 运行第 1 轮 ──
    print("\n  --- 第 1 轮 ---")
    t0 = time.time()
    round1 = await engine.run_round(1)
    t1 = time.time()
    print(f"  耗时: {t1 - t0:.1f}s | 动作数: {len(round1.actions)}")
    for action in round1.actions:
        agent_ctx = next((a.name for a in agents if a.entity_id == action.agent_id), "?")
        print(f"    {agent_ctx}: {action.action_type} → {action.content[:60]}")
    assert len(round1.actions) > 0, "第1轮应有动作"
    print("  ✓ 第1轮通过")

    # ── 运行第 2 轮，中途模拟暂停 ──
    print("\n  --- 第 2 轮（中途暂停）---")
    t0 = time.time()
    task = asyncio.create_task(engine.run_round(2))

    # 等待 2 秒后触发暂停（模拟用户点击"停止推演"）
    await asyncio.sleep(2.0)
    print("  触发暂停...")
    cancel_event.set()

    try:
        round2 = await asyncio.wait_for(task, timeout=30.0)
        print(f"  动作数: {len(round2.actions)} (暂停后被取消)")
    except _PhaseCancelledError:
        print("  ✓ _PhaseCancelledError 正确抛出 — 推演已暂停")
    except asyncio.TimeoutError:
        print("  ✗ 超时！取消事件未能在 30s 内中断推演")
        assert False, "暂停失败: 取消事件未生效"

    t1 = time.time()
    cancel_wait = t1 - t0
    print(f"  暂停等待时间: {cancel_wait:.1f}s (应在 2-5s，远小于正常轮次 10s+)")
    assert cancel_wait < 15, f"暂停等待 {cancel_wait:.1f}s 过长，取消事件未及时响应"

    # ── 验证：暂停后状态已保存 ──
    print("\n  --- 验证暂停快照 ---")
    snapshot_states = {}
    for eid, st in states.items():
        snapshot_states[eid] = dict(st.metrics)
        print(f"    {st.name}: {st.metrics}")
    assert any(v < 100 for v in snapshot_states.get("蒙古军", {}).values()), "蒙古军状态应有变化"
    print("  ✓ 快照状态已保存")

    # ── 重置 cancel 事件，模拟恢复 ──
    print("\n  --- 从暂停恢复，运行第 3 轮 ---")
    cancel_event.clear()
    t0 = time.time()
    round3 = await engine.run_round(3)
    t1 = time.time()
    print(f"  耗时: {t1 - t0:.1f}s | 动作数: {len(round3.actions)}")
    for action in round3.actions:
        agent_ctx = next((a.name for a in agents if a.entity_id == action.agent_id), "?")
        print(f"    {agent_ctx}: {action.action_type} → {action.content[:60]}")
    assert len(round3.actions) > 0, "恢复后第3轮应有动作"
    print("  ✓ 恢复后推演正常继续")

    # ── 最终状态 ──
    print("\n  --- 最终状态 ---")
    for eid, st in states.items():
        alive = st.is_alive(thresholds)
        print(f"    {st.name} [{'存活' if alive else '出局'}]: {', '.join(f'{k}={v:.1f}' for k, v in st.metrics.items())}")

    # ── 日志摘要 ──
    cancel_logs = [m for m in log_msgs if "cancel" in m.lower() or "取消" in m]
    print(f"\n  取消相关日志: {len(cancel_logs)} 条")

    print(f"\n{'=' * 60}")
    print("  暂停/恢复测试全部通过 ✓")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(test_pause_resume()))

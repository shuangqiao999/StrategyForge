"""P0-2: observe 态实体在指标恶化时交还 LLM（observe_llm_guard），不再锁定 observe。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from strategy_forge.algorithms import ModuleContext, SpatialState
from strategy_forge.algorithms.fsm_module import FiniteStateMachineModule


def _build(guard):
    ctx = ModuleContext(round_number=1)
    ctx.arrays = {"strength": np.array([20.0, 70.0], dtype=np.float64)}
    ctx.spatial = SpatialState()
    ctx.spatial.init_from_dict(["e0", "e1"])
    fsm = FiniteStateMachineModule()
    fsm.configure({
        "default_state": "observe",
        "command_states": ["engage"],
        "observe_llm_guard": guard,
        "transition_rules": [],
        "action_map": {"observe": {"action_type": "observe", "intensity": 0.3, "target": ""}},
    })
    return ctx, fsm


def test_degraded_observe_hands_to_llm():
    ctx, fsm = _build([{"metric": "strength", "op": "<", "threshold": 40}])
    fsm.execute(ctx)
    actions = ctx.metadata["fsm.agent_actions"]
    assert actions[0] is None, "degraded observe entity must hand off to LLM"
    assert actions[1] is not None and actions[1]["action_type"] == "observe"


def test_no_guard_keeps_deterministic_observe():
    ctx, fsm = _build([])
    fsm.execute(ctx)
    actions = ctx.metadata["fsm.agent_actions"]
    assert actions[0] is not None and actions[0]["action_type"] == "observe"
    assert actions[1] is not None and actions[1]["action_type"] == "observe"


def test_observe_stale_rounds_hands_to_llm_across_rounds():
    """后续项：连续 observe 超过 observe_stale_rounds 后交还 LLM（跨轮计数持久化）。"""
    fsm = FiniteStateMachineModule()
    fsm.configure({
        "default_state": "observe",
        "command_states": ["engage"],
        "observe_llm_guard": [],
        "observe_stale_rounds": 3,
        "transition_rules": [],
        "action_map": {"observe": {"action_type": "observe", "intensity": 0.3, "target": ""}},
    })
    blob = None
    for _round in range(3):
        ctx = ModuleContext(round_number=_round + 1)
        ctx.arrays = {"strength": np.array([80.0], dtype=np.float64)}
        ctx.spatial = SpatialState()
        ctx.spatial.init_from_dict(["e0"])
        if blob is not None:
            ctx.metadata["fsm.state_blob"] = blob
        fsm.execute(ctx)
        blob = ctx.metadata["fsm.state_blob"]
    # 连续 observe 3 轮后，第 3 轮 action 应交还 LLM（None）
    assert blob["observe_streak"][0] == 3
    assert blob["agent_states"][0] == "observe"
    # 第 3 轮（observe_streak==3）应触发 stale 交还
    ctx = ModuleContext(round_number=4)
    ctx.arrays = {"strength": np.array([80.0], dtype=np.float64)}
    ctx.spatial = SpatialState()
    ctx.spatial.init_from_dict(["e0"])
    ctx.metadata["fsm.state_blob"] = blob
    fsm.execute(ctx)
    assert ctx.metadata["fsm.agent_actions"][0] is None


def test_observe_streak_resets_when_leaving_observe():
    fsm = FiniteStateMachineModule()
    fsm.configure({
        "default_state": "observe",
        "command_states": ["engage"],
        "observe_stale_rounds": 3,
        "transition_rules": [
            {"from": "observe", "to": "engage", "condition": {"strength": [">", 50]}},
        ],
        "action_map": {"observe": {"action_type": "observe", "intensity": 0.3, "target": ""}},
    })
    ctx = ModuleContext(round_number=1)
    ctx.arrays = {"strength": np.array([80.0], dtype=np.float64)}
    ctx.spatial = SpatialState()
    ctx.spatial.init_from_dict(["e0"])
    ctx.metadata["fsm.state_blob"] = {"agent_states": ["observe"], "streaks": {}, "observe_streak": [5]}
    fsm.execute(ctx)
    blob = ctx.metadata["fsm.state_blob"]
    assert blob["agent_states"][0] == "engage"
    assert blob["observe_streak"][0] == 0, "离开 observe 后计数应清零"

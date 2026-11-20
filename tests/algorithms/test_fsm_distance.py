"""P0-1: _resolve_metric 的 distance_to_enemy/ally 分支必须返回最近目标的欧氏距离。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from strategy_forge.algorithms import ModuleContext, SpatialState
from strategy_forge.algorithms.fsm_module import FiniteStateMachineModule


def _make_ctx(positions, enemy_ids=None, ally_ids=None):
    ctx = ModuleContext(round_number=1)
    ctx.spatial = SpatialState()
    eids = [f"e{i}" for i in range(len(positions))]
    ctx.spatial.init_from_dict(eids, {f"e{i}": pos for i, pos in enumerate(positions)})
    ctx.arrays = {"strength": np.array([50.0] * len(positions), dtype=np.float64)}
    if enemy_ids is not None:
        ctx.metadata["fsm.enemy_ids"] = enemy_ids
    if ally_ids is not None:
        ctx.metadata["fsm.ally_ids"] = ally_ids
    return ctx


def test_distance_to_enemy_below_threshold():
    ctx = _make_ctx([[0, 0, 0], [10, 0, 0]], enemy_ids=[1])
    assert FiniteStateMachineModule._resolve_metric(ctx, 0, "distance_to_enemy") == pytest.approx(10.0)
    assert FiniteStateMachineModule._match_condition(ctx, 0, {"distance_to_enemy": ["<", 200]}) is True


def test_distance_to_enemy_above_threshold():
    ctx = _make_ctx([[0, 0, 0], [10, 0, 0]], enemy_ids=[1])
    assert FiniteStateMachineModule._match_condition(ctx, 0, {"distance_to_enemy": ["<", 5]}) is False


def test_distance_to_enemy_nearest_target():
    ctx = _make_ctx([[0, 0, 0], [100, 0, 0], [30, 0, 0]], enemy_ids=[1, 2])
    assert FiniteStateMachineModule._resolve_metric(ctx, 0, "distance_to_enemy") == pytest.approx(30.0)


def test_distance_to_enemy_no_targets_returns_none():
    ctx = _make_ctx([[0, 0, 0], [10, 0, 0]], enemy_ids=[])
    assert FiniteStateMachineModule._resolve_metric(ctx, 0, "distance_to_enemy") is None


def test_distance_out_of_bounds_returns_none():
    ctx = _make_ctx([[0, 0, 0], [10, 0, 0]], enemy_ids=[5])
    assert FiniteStateMachineModule._resolve_metric(ctx, 0, "distance_to_enemy") is None


def test_distance_to_enemy_uniform_polarization_falls_back_to_all_others():
    # enemy_ids/ally_ids 未配置且 polarization 均匀 → 兜底把所有其他实体视为对手
    ctx = _make_ctx([[0, 0, 0], [10, 0, 0], [30, 0, 0]])
    ctx.arrays["polarization"] = np.array([50.0, 50.0, 50.0])
    assert FiniteStateMachineModule._resolve_metric(ctx, 0, "distance_to_enemy") == pytest.approx(10.0)
    assert FiniteStateMachineModule._match_condition(ctx, 0, {"distance_to_enemy": ["<", 200]}) is True


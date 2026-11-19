"""P1-1: build_context 从规则包写入 _competition_groups；_competitive_logistic 分组独立。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from strategy_forge.algorithms.module_utils import build_context, _resolve_competition_groups
from strategy_forge.algorithms.ode_module import _competitive_logistic


class _State:
    def __init__(self, metrics):
        self.metrics = metrics


class _RE:
    def __init__(self, pack):
        self.pack = pack


def _base_pack():
    return {
        "metrics": ["tech_lead"],
        "modules": {"ode_engine": {"params": {}}},
    }


def test_build_context_writes_groups_from_dict():
    pack = _base_pack()
    pack["modules"]["ode_engine"]["params"]["competition_groups"] = {
        "e0": 0, "e1": 0, "e2": 1,
    }
    states = {"e0": _State({"tech_lead": 50}), "e1": _State({"tech_lead": 60}),
              "e2": _State({"tech_lead": 70})}
    ctx = build_context(states, _RE(pack), ["e0", "e1", "e2"], 1)
    groups = ctx.metadata["_competition_groups"]
    assert np.array_equal(groups, np.array([0, 0, 1]))


def test_build_context_writes_groups_from_list():
    pack = _base_pack()
    pack["modules"]["ode_engine"]["params"]["competition_groups"] = [1, 1, 2]
    states = {"e0": _State({"tech_lead": 50}), "e1": _State({"tech_lead": 60}),
              "e2": _State({"tech_lead": 70})}
    ctx = build_context(states, _RE(pack), ["e0", "e1", "e2"], 1)
    groups = ctx.metadata["_competition_groups"]
    assert np.array_equal(groups, np.array([1, 1, 2]))


def test_no_groups_leaves_metadata_unset():
    states = {"e0": _State({"tech_lead": 50}), "e1": _State({"tech_lead": 60})}
    ctx = build_context(states, _RE(_base_pack()), ["e0", "e1"], 1)
    assert "_competition_groups" not in ctx.metadata


def test_resolve_invalid_returns_none():
    assert _resolve_competition_groups([1, 2], ["e0", "e1", "e2"]) is None
    assert _resolve_competition_groups("bad", ["e0"]) is None


def test_competitive_logistic_groups_are_independent():
    params = {
        "_carrying_capacity": 100, "_logistic_rate": 0.06,
        "_diffusion_rate": 0.015, "_competition_factor": 0.3,
    }
    full = np.array([10.0, 90.0, 30.0, 30.0])
    groups = np.array([0, 0, 1, 1])
    dy_full = _competitive_logistic(full, {**params, "_competition_groups": groups})

    sub = np.array([30.0, 30.0])
    dy_sub = _competitive_logistic(sub, params)

    # 组 1 的两个实体应与"仅含组 1"独立演化完全一致
    assert np.allclose(dy_full[2:], dy_sub)

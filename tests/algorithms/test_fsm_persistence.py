"""后续项：FSM 状态跨轮持久化（build_context 回注 ↔ simulator rekey）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from strategy_forge.algorithms.module_utils import build_context, rekey_fsm_blob


class _State:
    def __init__(self, metrics=None):
        self.metrics = metrics or {}


class _RE:
    def __init__(self, pack):
        self.pack = pack


def _pack():
    return {
        "metrics": [],
        "modules": {
            "finite_state_machine": {"default_state": "observe"},
            "ode_engine": {"params": {}},
            "physics_engine": {},
        },
    }


def test_roundtrip_align_and_rekey():
    entity_ids = ["a", "b", "c"]
    prev = {
        "states": {"a": "engage", "b": "observe", "c": "retreat"},
        "streaks": {"x->y.abcd12": {"a": 1, "b": 2, "c": 3}},
        "observe_streak": {"a": 0, "b": 4, "c": 0},
    }
    states = {eid: _State() for eid in entity_ids}
    ctx = build_context(states, _RE(_pack()), entity_ids, 2, prev_fsm_state=prev)

    blob = ctx.metadata["fsm.state_blob"]
    assert blob["agent_states"] == ["engage", "observe", "retreat"]
    assert blob["observe_streak"] == [0, 4, 0]
    assert blob["streaks"]["x->y.abcd12"] == [1, 2, 3]

    rekeyed = rekey_fsm_blob(blob, entity_ids)
    assert rekeyed["states"] == prev["states"]
    assert rekeyed["observe_streak"] == prev["observe_streak"]
    assert rekeyed["streaks"]["x->y.abcd12"] == prev["streaks"]["x->y.abcd12"]


def test_missing_entity_defaults_to_observe_zero():
    entity_ids = ["a", "b", "c"]
    prev = {"states": {"a": "engage"}, "streaks": {}, "observe_streak": {"a": 2}}
    states = {eid: _State() for eid in entity_ids}
    ctx = build_context(states, _RE(_pack()), entity_ids, 2, prev_fsm_state=prev)
    blob = ctx.metadata["fsm.state_blob"]
    assert blob["agent_states"] == ["engage", "observe", "observe"]
    assert blob["observe_streak"] == [2, 0, 0]

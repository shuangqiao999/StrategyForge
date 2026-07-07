"""Unit tests for OpinionDynamicsModule, PhysicsModule, PipelineEngine, FSM."""
import sys
import numpy as np

sys.path.insert(0, "src")

from strategy_forge.algorithms.opinion_dynamics import OpinionDynamicsModule
from strategy_forge.algorithms.physics_module import PhysicsModule
from strategy_forge.algorithms.pipeline_engine import PipelineEngine
from strategy_forge.algorithms.fsm_module import FiniteStateMachineModule
from strategy_forge.algorithms.base import ModuleContext, SpatialState, AlgorithmModule


def _make_ctx(n: int = 4, **extra_arrays) -> ModuleContext:
    arrays = {"strength": np.linspace(30, 90, n, dtype=np.float64)}
    arrays.update({k: np.array(v, dtype=np.float64) for k, v in extra_arrays.items()})
    sp = SpatialState()
    sp.positions = np.random.default_rng(42).uniform(0, 100, (n, 3)).astype(np.float64)
    sp.radii = np.ones(n, dtype=np.float64) * 5.0
    sp.velocities = np.zeros((n, 3), dtype=np.float64)
    sp.forces = np.zeros((n, 3), dtype=np.float64)
    sp.masses = np.ones(n, dtype=np.float64)
    return ModuleContext(dt=1.0, arrays=arrays, spatial=sp)


# ── OpinionDynamicsModule ──

class TestOpinionDynamics:
    def test_graph_spatial_default(self):
        m = OpinionDynamicsModule()
        m.configure({"target_metrics": ["strength"], "epsilon": 0.5})
        ctx = _make_ctx(4)
        result = m.execute(ctx)
        assert "opinion_dynamics.updated_metrics" in result.metadata

    def test_graph_complete(self):
        m = OpinionDynamicsModule()
        m.configure({"target_metrics": ["strength"], "graph_type": "complete", "epsilon": 0.5})
        ctx = _make_ctx(4)
        result = m.execute(ctx)
        assert result.arrays["strength"] is not None

    def test_graph_social(self):
        m = OpinionDynamicsModule()
        m.configure({"target_metrics": ["strength"], "graph_type": "social", "epsilon": 0.5})
        n = 4
        ctx = _make_ctx(n)
        ctx.metadata["social_graph"] = np.eye(n, dtype=np.float64)
        result = m.execute(ctx)
        assert result.arrays["strength"] is not None

    def test_graph_from_metadata(self):
        m = OpinionDynamicsModule()
        m.configure({"target_metrics": ["strength"], "graph_type": "from_metadata", "epsilon": 0.5})
        n = 4
        ctx = _make_ctx(n)
        ctx.metadata["relation_graph"] = np.ones((n, n), dtype=np.float64) * 0.5
        np.fill_diagonal(ctx.metadata["relation_graph"], 0.0)
        result = m.execute(ctx)
        assert result.arrays["strength"] is not None

    def test_weighted_hk(self):
        m = OpinionDynamicsModule()
        m.configure({"target_metrics": ["strength"], "weighted": True, "graph_type": "complete", "epsilon": 0.5})
        n = 4
        ctx = _make_ctx(n)
        ctx.metadata["relation_weights"] = np.random.default_rng(1).uniform(0.1, 1.0, (n, n))
        result = m.execute(ctx)
        assert result.arrays["strength"] is not None

    def test_fixed_norm_range(self):
        m = OpinionDynamicsModule()
        m.configure({"target_metrics": ["strength"], "norm_range": [0, 100], "graph_type": "complete", "epsilon": 0.5})
        ctx = _make_ctx(4)
        result = m.execute(ctx)
        assert result.arrays["strength"] is not None
        # Values should stay within [0, 100] range
        assert np.all(result.arrays["strength"] >= 0)
        assert np.all(result.arrays["strength"] <= 100)

    def test_empty_metrics_skipped(self):
        m = OpinionDynamicsModule()
        m.configure({"target_metrics": [], "epsilon": 0.5})
        ctx = _make_ctx(4)
        result = m.execute(ctx)
        assert result is ctx

    def test_single_entity_skipped(self):
        m = OpinionDynamicsModule()
        m.configure({"target_metrics": ["strength"], "epsilon": 0.5})
        arrays = {"strength": np.array([50.0], dtype=np.float64)}
        sp = SpatialState()
        sp.positions = np.array([[0, 0, 0]], dtype=np.float64)
        sp.radii = np.array([1.0])
        sp.velocities = np.zeros((1, 3))
        sp.forces = np.zeros((1, 3))
        sp.masses = np.array([1.0])
        ctx = ModuleContext(dt=1.0, arrays=arrays, spatial=sp)
        result = m.execute(ctx)
        assert result is ctx


# ── PhysicsModule ──

class TestPhysicsSpeedClamp:
    def test_max_speed_applied(self):
        m = PhysicsModule()
        m.configure({"max_speed": 10.0, "subsystems": ["dynamics"]})
        n = 3
        sp = SpatialState()
        sp.positions = np.zeros((n, 3), dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64)
        sp.velocities = np.array([[100, 0, 0], [0, 200, 0], [0, 0, 50]], dtype=np.float64)
        sp.forces = np.zeros((n, 3), dtype=np.float64)
        sp.masses = np.ones(n, dtype=np.float64)
        ctx = ModuleContext(dt=0.1, spatial=sp)
        m._step_dynamics(sp, 0.1)
        speeds = np.linalg.norm(sp.velocities, axis=1)
        assert np.all(speeds <= 10.0 + 1e-2)

    def test_no_max_speed_when_zero(self):
        m = PhysicsModule()
        m.configure({"max_speed": 0, "subsystems": ["dynamics"]})
        n = 3
        sp = SpatialState()
        sp.positions = np.zeros((n, 3), dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64)
        sp.velocities = np.array([[100, 0, 0]], dtype=np.float64)
        sp.forces = np.zeros((1, 3), dtype=np.float64)
        sp.masses = np.array([1.0])
        m._step_dynamics(sp, 0.1)
        assert sp.velocities[0, 0] > 50  # Should not be capped

    def test_diffusion_boundary_reflect(self):
        m = PhysicsModule()
        m.configure({"diffusion_boundary": "reflect", "diffusion_rate": 0.1, "subsystems": ["diffusion"]})
        n = 3
        sp = SpatialState()
        sp.positions = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64) * 5.0
        sp.velocities = np.zeros((n, 3))
        sp.forces = np.zeros((n, 3))
        sp.masses = np.ones(n, dtype=np.float64)
        arrays = {"pollution": np.array([-5.0, 10.0, 50.0], dtype=np.float64)}
        ctx = ModuleContext(dt=1.0, spatial=sp, arrays=arrays, diffusion_fields=["pollution"])
        result = m.execute(ctx)
        assert np.all(result.arrays["pollution"] >= 0.0)  # reflect clips to >=0


# ── PipelineEngine ──

class DummyModule(AlgorithmModule):
    def __init__(self, name, raise_on_execute=False):
        self._name = name
        self._raise = raise_on_execute

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Dummy {self._name}"

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        if self._raise:
            raise RuntimeError("test error")
        ctx.metadata[self._name] = True
        return ctx


class DummyFinalizer(DummyModule):
    IS_FINALIZER = True


class TestPipelineEngine:
    def test_order_execution(self):
        engine = PipelineEngine()
        a = DummyModule("a")
        b = DummyModule("b")
        engine.register(a)
        engine.register(b)
        ctx = _make_ctx(2)
        result = engine.run(ctx, {"order": ["a", "b"]})
        assert result.metadata["a"] is True
        assert result.metadata["b"] is True

    def test_finalizer_sorted_last(self):
        engine = PipelineEngine()
        fin = DummyFinalizer("finalizer")
        engine.register(DummyModule("normal"))
        engine.register(fin)
        ctx = _make_ctx(2)
        result = engine.run(ctx, {"order": ["finalizer", "normal"]})
        assert result.metadata.get("normal") is True
        assert result.metadata.get("finalizer") is True

    def test_conditional_and_compound(self):
        engine = PipelineEngine()
        engine.register(DummyModule("mod_a"))
        engine.register(DummyModule("mod_b"))
        ctx = _make_ctx(2)
        ctx.metadata["flag_x"] = 1.0
        ctx.metadata["flag_y"] = 0.0
        result = engine.run(ctx, {
            "order": ["mod_a", "mod_b"],
            "conditionals": {
                "execute_mod_a": "flag_x > 0 and flag_y < 1",
                "execute_mod_b": "flag_x > 10 or flag_y > 10",
            }
        })
        assert result.metadata.get("mod_a") is True
        assert result.metadata.get("mod_b") is None  # neither condition met → skipped

    def test_conditional_or(self):
        engine = PipelineEngine()
        engine.register(DummyModule("mod_a"))
        ctx = _make_ctx(2)
        ctx.metadata["flag_x"] = 0.0
        ctx.metadata["flag_y"] = 1.0
        result = engine.run(ctx, {
            "order": ["mod_a"],
            "conditionals": {"execute_mod_a": "flag_x > 0 or flag_y > 0"}
        })
        assert result.metadata.get("mod_a") is True

    def test_strict_mode_raises(self):
        engine = PipelineEngine()
        engine.register(DummyModule("bad", raise_on_execute=True))
        ctx = _make_ctx(2)
        import pytest
        with pytest.raises(RuntimeError, match="test error"):
            engine.run(ctx, {"order": ["bad"]}, strict_mode=True)

    def test_strict_mode_false_suppresses(self):
        engine = PipelineEngine()
        engine.register(DummyModule("bad", raise_on_execute=True))
        ctx = _make_ctx(2)
        result = engine.run(ctx, {"order": ["bad"]}, strict_mode=False)
        assert result is ctx  # should continue

    def test_unregistered_module_skipped(self):
        engine = PipelineEngine()
        engine.register(DummyModule("a"))
        ctx = _make_ctx(2)
        result = engine.run(ctx, {"order": ["a", "ghost"]})
        assert result.metadata.get("a") is True

    def test_empty_pipeline(self):
        engine = PipelineEngine()
        ctx = _make_ctx(2)
        result = engine.run(ctx)
        assert result is ctx


# ── FiniteStateMachineModule ──

class TestFSM:
    def test_basic_transition(self):
        m = FiniteStateMachineModule()
        m.configure({
            "transition_rules": [
                {"from": "idle", "to": "active", "condition": {"strength": [">", 50]}}
            ],
            "default_state": "idle",
        })
        n = 4
        arrays = {"strength": np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float64)}
        sp = SpatialState()
        sp.positions = np.zeros((n, 3), dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64)
        sp.velocities = np.zeros((n, 3))
        sp.forces = np.zeros((n, 3))
        sp.masses = np.ones(n, dtype=np.float64)
        ctx = ModuleContext(dt=1.0, arrays=arrays, spatial=sp)
        result = m.execute(ctx)
        states = result.metadata["fsm.agent_states"]
        assert states[0] == "idle"
        assert states[1] == "idle"
        assert states[2] == "active"
        assert states[3] == "active"

    def test_action_mapping_non_command(self):
        m = FiniteStateMachineModule()
        m.configure({
            "transition_rules": [],
            "default_state": "defend",
            "action_map": {"defend": {"action_type": "fortify", "intensity": 0.8}},
            "command_states": ["combat"],
        })
        n = 2
        arrays = {"strength": np.array([50.0, 50.0], dtype=np.float64)}
        sp = SpatialState()
        sp.positions = np.zeros((n, 3), dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64)
        sp.velocities = np.zeros((n, 3))
        sp.forces = np.zeros((n, 3))
        sp.masses = np.ones(n, dtype=np.float64)
        ctx = ModuleContext(dt=1.0, arrays=arrays, spatial=sp)
        result = m.execute(ctx)
        actions = result.metadata["fsm.agent_actions"]
        assert actions[0]["action_type"] == "fortify"
        assert actions[0]["intensity"] == 0.8

    def test_command_state_returns_none_action(self):
        m = FiniteStateMachineModule()
        m.configure({
            "transition_rules": [],
            "default_state": "combat",
            "command_states": ["combat"],
        })
        n = 2
        arrays = {"strength": np.array([50.0, 50.0], dtype=np.float64)}
        sp = SpatialState()
        sp.positions = np.zeros((n, 3), dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64)
        sp.velocities = np.zeros((n, 3))
        sp.forces = np.zeros((n, 3))
        sp.masses = np.ones(n, dtype=np.float64)
        ctx = ModuleContext(dt=1.0, arrays=arrays, spatial=sp)
        result = m.execute(ctx)
        actions = result.metadata["fsm.agent_actions"]
        assert actions[0] is None
        assert actions[1] is None

    def test_streak_persists_via_metadata(self):
        m = FiniteStateMachineModule()
        m.configure({
            "transition_rules": [
                {"from": "idle", "to": "panic", "condition": {"strength": ["<", 40], "streak": 2}}
            ],
            "default_state": "idle",
        })
        n = 2
        arrays = {"strength": np.array([35.0, 35.0], dtype=np.float64)}
        sp = SpatialState()
        sp.positions = np.zeros((n, 3), dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64)
        sp.velocities = np.zeros((n, 3))
        sp.forces = np.zeros((n, 3))
        sp.masses = np.ones(n, dtype=np.float64)
        ctx = ModuleContext(dt=1.0, arrays=arrays, spatial=sp)
        # Round 1: condition met but streak=1 (need 2)
        result = m.execute(ctx)
        assert result.metadata["fsm.agent_states"][0] == "idle"
        # Round 2: use persisted streak via metadata — should transition
        result2 = m.execute(result)
        assert result2.metadata["fsm.agent_states"][0] == "panic"

    def test_auto_enemy_by_polarization(self):
        m = FiniteStateMachineModule()
        m.configure({
            "transition_rules": [
                {"from": "idle", "to": "engage", "condition": {"distance_to_enemy": ["<", 100]}}
            ],
            "default_state": "idle",
        })
        n = 4
        arrays = {
            "strength": np.ones(n, dtype=np.float64) * 50,
            "polarization": np.array([5.0, 5.0, -5.0, -5.0], dtype=np.float64),
        }
        sp = SpatialState()
        sp.positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64) * 1.0
        sp.velocities = np.zeros((n, 3))
        sp.forces = np.zeros((n, 3))
        sp.masses = np.ones(n, dtype=np.float64)
        ctx = ModuleContext(dt=1.0, arrays=arrays, spatial=sp)
        result = m.execute(ctx)
        assert "fsm.agent_states" in result.metadata

    def test_auto_enemy_fallback_all_enemies(self):
        m = FiniteStateMachineModule()
        m.configure({
            "transition_rules": [
                {"from": "idle", "to": "engage", "condition": {"distance_to_enemy": ["<", 100]}}
            ],
            "default_state": "idle",
        })
        n = 3
        arrays = {"strength": np.ones(n, dtype=np.float64) * 50}
        sp = SpatialState()
        sp.positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        sp.radii = np.ones(n, dtype=np.float64) * 1.0
        sp.velocities = np.zeros((n, 3))
        sp.forces = np.zeros((n, 3))
        sp.masses = np.ones(n, dtype=np.float64)
        ctx = ModuleContext(dt=1.0, arrays=arrays, spatial=sp)
        result = m.execute(ctx)
        assert "fsm.agent_states" in result.metadata

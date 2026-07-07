"""Unit tests for ODE module presets, Euler integration, and competition groups."""
import numpy as np
import sys

sys.path.insert(0, "src")

from strategy_forge.algorithms.ode_module import (
    _competitive_logistic,
    _cash_flow_dynamics,
    _logistic,
    _decay,
    _fatigue_recovery,
    _supply_consumption,
    _pollution_spread,
    _resource_depletion,
    ODE_PRESETS,
    ODEModule,
)
from strategy_forge.algorithms.base import ModuleContext


# ── Preset Function Tests ──

class TestCompetitiveLogistic:
    def test_no_groups_global_pool(self):
        n = 4
        values = np.array([80.0, 20.0, 50.0, 60.0], dtype=np.float64)
        ctx = {
            "_carrying_capacity": 100.0,
            "_logistic_rate": 0.06,
            "_diffusion_rate": 0.015,
            "_competition_factor": 0.3,
        }
        result = _competitive_logistic(values, ctx)
        assert result.shape == (n,)
        # Leaders (> global_mean=52.5) should have negative diffusion
        assert np.all(np.isfinite(result))

    def test_with_groups(self):
        values = np.array([80.0, 20.0, 90.0, 10.0], dtype=np.float64)
        groups = np.array([0, 0, 1, 1], dtype=np.int32)
        ctx = {
            "_carrying_capacity": 100.0,
            "_logistic_rate": 0.06,
            "_diffusion_rate": 0.015,
            "_competition_factor": 0.3,
            "_competition_groups": groups,
        }
        result = _competitive_logistic(values, ctx)
        assert result.shape == (4,)
        # Group 0: [80, 20] mean=50, leader (80) gets negative diffusion
        # Group 1: [90, 10] mean=50, leader (90) gets negative diffusion
        assert result[0] != result[2]
        assert np.all(np.isfinite(result))

    def test_group_single_entity(self):
        values = np.array([50.0, 80.0], dtype=np.float64)
        groups = np.array([0, 1], dtype=np.int32)
        ctx = {
            "_carrying_capacity": 100.0,
            "_logistic_rate": 0.06,
            "_diffusion_rate": 0.015,
            "_competition_factor": 0.3,
            "_competition_groups": groups,
        }
        result = _competitive_logistic(values, ctx)
        # Single-entity groups: mean = self, diffusion = 0
        assert result[0] != result[1]
        assert np.all(np.isfinite(result))


class TestCashFlowDynamics:
    def test_protection_clamp_handles_negative(self):
        values = np.array([50.0, 50.0], dtype=np.float64)
        ctx = {
            "_decay_rate": 0.01,
            "_base_recovery": 5.0,
            "_scale_protection": 10.0,
            "supply_chain": np.array([-10.0, 150.0], dtype=np.float64),
            "tech_lead": np.array([-5.0, 120.0], dtype=np.float64),
        }
        result = _cash_flow_dynamics(values, ctx)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        # e0: supply_chain clipped to 0, tech_lead clipped to 0 → protection=5.0
        # e1: supply_chain clipped to 100, tech_lead clipped to 100 → protection=15.0
        # decay = -0.01*50 = -0.5
        assert abs(result[0] - 4.5) < 1e-6
        assert abs(result[1] - 14.5) < 1e-6

    def test_cash_flow_positive_protection(self):
        values = np.array([100.0], dtype=np.float64)
        ctx = {
            "_decay_rate": 0.01,
            "_base_recovery": 0.0,
            "_scale_protection": 5.0,
            "supply_chain": np.array([100.0], dtype=np.float64),
            "tech_lead": np.array([100.0], dtype=np.float64),
        }
        result = _cash_flow_dynamics(values, ctx)
        # decay: -0.01 * 100 = -1.0, protection: 5.0
        # net: 4.0
        assert abs(result[0] - 4.0) < 0.01


class TestLogisticPresets:
    def test_decay(self):
        values = np.array([100.0, 50.0], dtype=np.float64)
        result = _decay(values, {"_decay_rate": 0.02})
        assert result[0] == -2.0
        assert result[1] == -1.0

    def test_logistic_at_capacity(self):
        values = np.array([100.0], dtype=np.float64)
        result = _logistic(values, {"_carrying_capacity": 100.0, "_logistic_rate": 0.03})
        # rate * 100 * (1 - 100/100) = 0
        assert abs(result[0]) < 1e-10

    def test_logistic_below_capacity(self):
        values = np.array([50.0], dtype=np.float64)
        result = _logistic(values, {"_carrying_capacity": 100.0, "_logistic_rate": 0.03})
        # 0.03 * 50 * (1 - 0.5) = 0.75
        assert result[0] > 0

    def test_fatigue_recovery_positive_values(self):
        values = np.array([100.0], dtype=np.float64)
        result = _fatigue_recovery(values, {"_fatigue_rate": 0.05})
        assert result[0] < 0

    def test_fatigue_recovery_zero_values(self):
        values = np.array([0.0], dtype=np.float64)
        result = _fatigue_recovery(values, {"_fatigue_rate": 0.05})
        assert result[0] == 0.0

    def test_supply_consumption_clamp(self):
        values = np.array([1.0], dtype=np.float64)
        ctx = {
            "strength": np.array([1000.0], dtype=np.float64),
            "_supply_base_rate": 0.3,
            "_supply_strength_factor": 0.01,
            "_dt": 1.0,
        }
        result = _supply_consumption(values, ctx)
        assert result[0] >= -1.0  # clamp prevents going below 0 instantly

    def test_pollution_spread(self):
        values = np.array([50.0, 0.0], dtype=np.float64)
        ctx = {
            "factory_output": np.array([100.0, 0.0], dtype=np.float64),
            "green_coverage": np.array([10.0, 100.0], dtype=np.float64),
        }
        result = _pollution_spread(values, ctx)
        assert result.shape == (2,)

    def test_resource_depletion(self):
        values = np.array([100.0], dtype=np.float64)
        ctx = {"population": np.array([1000.0], dtype=np.float64)}
        result = _resource_depletion(values, ctx)
        assert result[0] < 0


# ── ODEModule Configure / Euler Tests ──

class TestODEModule:
    def test_configure_default_sub_steps(self):
        ode = ODEModule()
        ode.configure({"equations": {"metric_a": "decay"}})
        assert ode._sub_steps == 8

    def test_configure_custom_sub_steps(self):
        ode = ODEModule()
        ode.configure({"sub_steps": 6, "equations": {"metric_a": "decay"}})
        assert ode._sub_steps == 6

    def test_euler_frozen_two_step(self):
        ode = ODEModule()
        ode.configure({"equations": {"a": "decay", "b": "decay"}})
        n = 3
        arrays = {"a": np.array([100.0, 80.0, 60.0], dtype=np.float64),
                   "b": np.array([50.0, 40.0, 30.0], dtype=np.float64)}
        # Snapshot original values (ModuleContext stores ref, not copy)
        orig_a = arrays["a"].copy()
        orig_b = arrays["b"].copy()
        ctx = ModuleContext(dt=1.0, arrays=arrays)
        result = ode._execute_euler(ctx)
        assert "a" in result.arrays
        assert "b" in result.arrays
        # Both metrics should have decayed from originals
        assert np.all(result.arrays["a"] < orig_a)
        assert np.all(result.arrays["b"] < orig_b)

    def test_euler_snapshot_recovery(self):
        ode = ODEModule()
        # Use a bad equation name to trigger KeyError in preset lookup
        ode.configure({"equations": {"a": "nonexistent_preset"}})
        arrays = {"a": np.array([100.0], dtype=np.float64)}
        ctx = ModuleContext(dt=1.0, arrays=arrays)
        result = ode._execute_euler(ctx)
        # Should NOT raise; should recover
        assert np.array_equal(result.arrays["a"], np.array([100.0]))

    def test_euler_empty_arrays(self):
        ode = ODEModule()
        ode.configure({"equations": {}})
        ctx = ModuleContext(dt=1.0, arrays={})
        result = ode.execute(ctx)
        assert result.arrays == {}


# ── Competition Groups Passthrough Tests ──

class TestCompetitionGroupPassthrough:
    def test_euler_passes_groups(self):
        ode = ODEModule()
        ode.configure({"equations": {"tech": "competitive_logistic"}})
        n = 4
        arrays = {"tech": np.array([80.0, 20.0, 90.0, 10.0], dtype=np.float64)}
        groups = np.array([0, 0, 1, 1], dtype=np.int32)
        metadata = {"_competition_groups": groups}
        ctx = ModuleContext(dt=1.0, arrays=arrays, metadata=metadata)
        result = ode._execute_euler(ctx)
        assert "tech" in result.arrays
        # Values should still be valid
        assert np.all(np.isfinite(result.arrays["tech"]))

    def test_euler_no_groups_still_works(self):
        ode = ODEModule()
        ode.configure({"equations": {"tech": "competitive_logistic"}})
        n = 3
        arrays = {"tech": np.array([50.0, 60.0, 40.0], dtype=np.float64)}
        ctx = ModuleContext(dt=1.0, arrays=arrays, metadata={})
        result = ode._execute_euler(ctx)
        assert "tech" in result.arrays
        assert np.all(np.isfinite(result.arrays["tech"]))


# ── ODE_PRESETS Registry Tests ──

class TestODEPresets:
    def test_all_presets_callable(self):
        for name, fn in ODE_PRESETS.items():
            assert callable(fn), f"Preset {name} is not callable"
            n = 3
            values = np.ones(n, dtype=np.float64)
            ctx = {
                "strength": np.ones(n, dtype=np.float64),
                "factory_output": np.ones(n, dtype=np.float64),
                "green_coverage": np.ones(n, dtype=np.float64),
                "population": np.ones(n, dtype=np.float64),
                "supply_chain": np.ones(n, dtype=np.float64) * 50,
                "tech_lead": np.ones(n, dtype=np.float64) * 50,
            }
            result = fn(values, ctx)
            assert result.shape == (n,), f"Preset {name} returned wrong shape"
            assert np.all(np.isfinite(result)), f"Preset {name} returned NaN/Inf"

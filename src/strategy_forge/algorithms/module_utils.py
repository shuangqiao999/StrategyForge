"""Module chain factory — creates and configures algorithm modules from rule pack data."""
from __future__ import annotations

from typing import Any

from .base import AlgorithmModule, ModuleContext, SpatialState, arrays_to_states, states_to_arrays
from .ode_module import ODEModule
from .physics_module import PhysicsModule


def build_module_chain(rule_engine: Any) -> list[AlgorithmModule]:
    """Create the default algorithm module chain from a RuleEngine instance.

    All modules are enabled by default. Configuration is derived from the
    rule pack's existing fields — no new fields are required.
    """
    modules: list[AlgorithmModule] = []
    pack: dict[str, Any] = getattr(rule_engine, "pack", {})

    # ── ODE module ──
    ode_cfg: dict[str, Any] = {"sub_steps": 4, "equations": {}}
    # Map metric names to built-in ODE presets by naming convention
    ode_map = {
        "fatigue": "fatigue_recovery",
        "supply": "supply_consumption",
        "pollution": "pollution_spread",
        "resources": "resource_depletion",
        "population": "logistic",
        "economy": "logistic",
    }
    for metric in pack.get("metrics", []):
        for pattern, preset in ode_map.items():
            if pattern in metric:
                ode_cfg["equations"][metric] = preset
                break
    ode = ODEModule()
    ode.configure(ode_cfg)
    modules.append(ode)

    # ── Physics module ──
    phys_cfg: dict[str, Any] = {
        "subsystems": ["dynamics", "collision", "diffusion", "explosion"],
        "gravity": 9.8,
        "damping": 0.98,
        "collision_elasticity": 0.5,
        "diffusion_rate": 0.05,
    }
    phys = PhysicsModule()
    phys.configure(phys_cfg)
    modules.append(phys)

    return modules


def build_context(
    states: dict[str, Any],
    rule_engine: Any,
    entity_ids: list[str],
    round_number: int,
) -> ModuleContext:
    """Build a ModuleContext from current EntityState dicts and rule pack config."""
    pack: dict[str, Any] = getattr(rule_engine, "pack", {})
    metric_names: list[str] = pack.get("metrics", [])

    ctx = ModuleContext(round_number=round_number)
    ctx.arrays = states_to_arrays(states, metric_names, entity_ids)

    # Spatial initialization
    init_pos = pack.get("initial_positions")
    init_vel = pack.get("initial_velocities")
    init_mass = pack.get("initial_masses")
    init_radius = pack.get("initial_radii")
    ctx.spatial.init_from_dict(entity_ids, init_pos, init_vel, init_mass, init_radius)

    return ctx


def apply_context_results(
    ctx: ModuleContext,
    states: dict[str, Any],
    entity_ids: list[str],
    rule_engine: Any,
) -> None:
    """Write module outputs back into EntityState objects."""
    pack: dict[str, Any] = getattr(rule_engine, "pack", {})
    metric_ranges_raw: dict = pack.get("metric_ranges", {})
    metric_ranges: dict[str, tuple[float, float]] = {}
    for k, v in metric_ranges_raw.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            metric_ranges[k] = (float(v[0]), float(v[1]))
    arrays_to_states(ctx, states, entity_ids, metric_ranges)

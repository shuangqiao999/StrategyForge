"""Module chain factory — creates and configures algorithm modules from rule pack data."""
from __future__ import annotations

from typing import Any

from .base import AlgorithmModule, ModuleContext, SpatialState, arrays_to_states, states_to_arrays
from .ode_module import ODEModule
from .physics_module import PhysicsModule


def build_module_chain(rule_engine: Any) -> list[AlgorithmModule]:
    """Create the default algorithm module chain from a RuleEngine instance.

    Configuration priority: rules.json ``modules`` section > built-in presets.
    If a rule pack has no ``modules`` key, behaviour is identical to before.
    """
    modules: list[AlgorithmModule] = []
    pack: dict[str, Any] = getattr(rule_engine, "pack", {})
    pack_modules: dict[str, Any] = pack.get("modules", {})

    # ── ODE module ──
    ode_user_eqs: dict[str, str] = pack_modules.get("ode_engine", {}).get("equations", {})
    # Built-in name→preset mapping (fallback when no user definition exists)
    ode_preset_map = {
        "fatigue": "fatigue_recovery",
        "supply": "supply_consumption",
        "pollution": "pollution_spread",
        "resources": "resource_depletion",
        "population": "logistic",
        "economy": "logistic",
        "market_share": "logistic",
        "cash_flow": "decay",
        "brand": "logistic",
    }
    final_eqs: dict[str, str] = {}
    for metric in pack.get("metrics", []):
        if metric in ode_user_eqs:
            final_eqs[metric] = ode_user_eqs[metric]       # user override
        else:
            for pattern, preset in ode_preset_map.items():   # fallback preset
                if pattern in metric:
                    final_eqs[metric] = preset
                    break
    ode_cfg: dict[str, Any] = {
        "sub_steps": int(pack_modules.get("ode_engine", {}).get("sub_steps", 4)),
        "equations": final_eqs,
    }
    ode = ODEModule()
    ode.configure(ode_cfg)
    modules.append(ode)

    # ── Physics module ──
    phys_user: dict[str, Any] = pack_modules.get("physics_engine", {})
    phys_cfg: dict[str, Any] = {
        "subsystems": phys_user.get("subsystems", ["dynamics", "collision", "diffusion", "explosion"]),
        "gravity": float(phys_user.get("gravity", 9.8)),
        "damping": float(phys_user.get("damping", 0.98)),
        "collision_elasticity": float(phys_user.get("collision_elasticity", 0.5)),
        "diffusion_rate": float(phys_user.get("diffusion_rate", 0.05)),
        "explosion_sources": phys_user.get("explosion_sources", []),
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

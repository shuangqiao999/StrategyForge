"""Module chain factory — creates and configures algorithm modules from rule pack data."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import AlgorithmModule, ModuleContext, SpatialState, arrays_to_states, states_to_arrays
from .fsm_module import FiniteStateMachineModule
from .ode_module import ODEModule
from .opinion_dynamics import OpinionDynamicsModule
from .physics_module import PhysicsModule
from .pipeline_engine import PipelineEngine

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when rules.json configuration is invalid (fail-fast before deduction starts)."""


# ── Module class registry (maps config keys to concrete classes) ──
_MODULE_CLASSES: dict[str, type[AlgorithmModule]] = {
    "ode_engine": ODEModule,
    "physics_engine": PhysicsModule,
    "opinion_dynamics": OpinionDynamicsModule,
    "finite_state_machine": FiniteStateMachineModule,
}


def build_pipeline(rule_engine: Any) -> PipelineEngine:
    """Build a PipelineEngine with all modules configured from rule pack.

    Reads rules.json modules.pipeline.order for execution order.
    Falls back to [ode_engine, physics_engine] if no pipeline config exists.
    """
    pack: dict[str, Any] = getattr(rule_engine, "pack", {})
    pack_modules: dict[str, Any] = pack.get("modules", {})
    pipeline_cfg: dict[str, Any] = pack_modules.get("pipeline", {})
    metrics: list[str] = pack.get("metrics", [])

    # ODE preset auto-matching (fallback when no equations defined)
    ode_preset_map = {
        "fatigue": "fatigue_recovery", "supply": "supply_consumption",
        "pollution": "pollution_spread", "resources": "resource_depletion",
        "population": "logistic", "economy": "logistic",
        "market_share": "logistic", "cash_flow": "decay", "brand": "logistic",
    }

    engine = PipelineEngine()

    order = pipeline_cfg.get("order", ["ode_engine", "physics_engine"])

    for name in order:
        cls = _MODULE_CLASSES.get(name)
        if cls is None:
            continue
        cfg = dict(pack_modules.get(name, {}))

        # ODE auto-config: fill equations from preset map if empty
        if name == "ode_engine" and not cfg.get("equations"):
            final_eqs: dict[str, str] = {}
            for metric in metrics:
                for pattern, preset in ode_preset_map.items():
                    if pattern in metric:
                        final_eqs[metric] = preset
                        break
            cfg["equations"] = final_eqs

        module = cls()
        module.configure(cfg)

        # FSM config validation: fail-fast on invalid condition metrics
        if name == "finite_state_machine":
            _validate_fsm_config(cfg, metrics)

        engine.register(module)

    return engine


def _validate_fsm_config(cfg: dict, metrics: list[str]) -> None:
    """Validate FSM transition_rules condition metrics are in the rule pack."""
    # Virtual / control keys that don't need to be in the rule pack metrics list
    _VIRTUAL_METRICS = frozenset({
        "distance_to_enemy", "distance_to_ally", "distance_to_nearest_entity",
        "streak",  # FSM historical condition (not a metric)
    })
    metrics_set = set(metrics) | _VIRTUAL_METRICS
    for rule in cfg.get("transition_rules", []):
        condition = rule.get("condition", {})
        for metric in condition:
            if metric not in metrics_set:
                raise ConfigValidationError(
                    f"FSM transition '{rule.get('from','?')} → {rule.get('to','?')}' "
                    f"references metric '{metric}' which is not in the rule pack metrics "
                    f"{metrics}. Check rules.json transition_rules."
                )


def build_module_chain(rule_engine: Any) -> list[AlgorithmModule]:
    """Backward-compat: returns module list from build_pipeline."""
    engine = build_pipeline(rule_engine)
    result: list[AlgorithmModule] = []
    # Get order from pipeline config
    pack: dict[str, Any] = getattr(rule_engine, "pack", {})
    order = pack.get("modules", {}).get("pipeline", {}).get("order", ["ode_engine", "physics_engine"])
    for name in order:
        mod = engine.get(name)
        if mod is not None:
            result.append(mod)
    return result


def build_context(
    states: dict[str, Any],
    rule_engine: Any,
    entity_ids: list[str],
    round_number: int,
    prev_spatial: Any = None,
) -> ModuleContext:
    """Build a ModuleContext from current EntityState dicts and rule pack config.

    If prev_spatial is provided, spatial state (positions/velocities/forces)
    carries over from the previous round instead of re-initializing.
    """
    pack: dict[str, Any] = getattr(rule_engine, "pack", {})
    metric_names: list[str] = pack.get("metrics", [])

    ctx = ModuleContext(round_number=round_number)
    ctx.arrays = states_to_arrays(states, metric_names, entity_ids)

    # Diffusion fields (which metrics are spatial and should be blurred)
    phys_cfg = pack.get("modules", {}).get("physics_engine", {})
    ctx.diffusion_fields = list(phys_cfg.get("diffusion_fields", []))

    # Spatial initialization: carry over from previous round if available
    if prev_spatial is not None:
        ctx.spatial = prev_spatial
    else:
        init_pos = pack.get("initial_positions")
        init_vel = pack.get("initial_velocities")
        init_mass = pack.get("initial_masses")
        init_radius = pack.get("initial_radii")
        ctx.spatial.init_from_dict(entity_ids, init_pos, init_vel, init_mass, init_radius)

    # ── Inject ODE params from rules.json (modules.ode_engine.params) + extractor fallback ──
    ode_cfg = pack.get("modules", {}).get("ode_engine", {})
    ode_params: dict[str, Any] = dict(ode_cfg.get("params", {}))
    extracted = pack.get("ode_params", {})
    if isinstance(extracted, dict) and "params" in extracted:
        ode_params.update(extracted["params"])
    if ode_params:
        ctx.metadata.setdefault("ode_params", {})
        for metric, param_dict in ode_params.items():
            if isinstance(param_dict, dict):
                for key, val in param_dict.items():
                    ctx_key = key if key.startswith("_") else f"_{key}"
                    ctx.metadata["ode_params"][ctx_key] = float(val)

    # Inject physics explosion sources from rules.json + extractor fallback
    phys_cfg = pack.get("modules", {}).get("physics_engine", {})
    phys_extracted = pack.get("physics_params", {})
    all_sources = list(phys_cfg.get("explosion_sources", []))
    all_sources.extend(phys_extracted.get("explosion_sources", []))
    if all_sources:
        ctx.metadata.setdefault("trigger_explosion", [])
        for src in all_sources:
            ctx.metadata["trigger_explosion"].append({
                "center": src.get("center", [0, 0, 0]),
                "power": float(src.get("power", 100.0)),
                "radius": float(src.get("radius", 50.0)),
            })

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

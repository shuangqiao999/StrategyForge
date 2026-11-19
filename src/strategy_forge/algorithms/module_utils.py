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


# ODE preset 子串匹配黑名单：这些 pattern 是常见子串，禁止参与前缀/后缀匹配，
# 避免 supply→supply_chain、resource→energy_resource 之类的巧合命中。
_ODE_SUBSTRING_BLACKLIST = frozenset({"supply", "resource", "resources"})


def _match_ode_preset(metric: str, ode_preset_map: dict[str, str]) -> str | None:
    """精确优先，其次前缀/后缀匹配（pattern 长度≥4 且非黑名单），否则返回 None。"""
    if metric in ode_preset_map:
        return ode_preset_map[metric]
    for pattern, preset in ode_preset_map.items():
        if len(pattern) < 4 or pattern in _ODE_SUBSTRING_BLACKLIST:
            continue
        if metric.startswith(pattern) or metric.endswith(pattern):
            return preset
    return None


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
                preset = _match_ode_preset(metric, ode_preset_map)
                if preset is not None:
                    final_eqs[metric] = preset
                else:
                    logger.debug("[ModuleUtils] metric '%s' 无 ODE preset，导数=0", metric)
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
    prev_fsm_state: Any = None,
) -> ModuleContext:
    """Build a ModuleContext from current EntityState dicts and rule pack config.

    If prev_spatial is provided, spatial state (positions/velocities/forces)
    carries over from the previous round instead of re-initializing.
    If prev_fsm_state is provided (entity_id-keyed dict), seed FSM 跨轮状态.
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

    # ── Inject ODE competition groups (modules.ode_engine.params.competition_groups) ──
    competition_groups = (pack.get("modules", {}).get("ode_engine", {}).get("params", {})
                          .get("competition_groups"))
    if competition_groups is None:
        competition_groups = pack.get("competition_groups")
    if competition_groups is not None:
        groups = _resolve_competition_groups(competition_groups, entity_ids)
        if groups is not None:
            ctx.metadata["_competition_groups"] = groups
        else:
            logger.debug("[ModuleUtils] competition_groups 配置格式无效，已忽略")
    else:
        logger.debug("[ModuleUtils] 未配置 competition_groups，competitive_logistic 使用全局池")

    # ── Seed FSM 跨轮状态（entity_id-keyed → 按 entity_ids 对齐为 index 列表）──
    if prev_fsm_state is not None and isinstance(prev_fsm_state, dict):
        fsm_cfg = pack.get("modules", {}).get("finite_state_machine", {})
        default_state = str(fsm_cfg.get("default_state", "idle"))
        states_map = prev_fsm_state.get("states", {}) or {}
        streaks_map = prev_fsm_state.get("streaks", {}) or {}
        obs_map = prev_fsm_state.get("observe_streak", {}) or {}
        ctx.metadata["fsm.state_blob"] = {
            "agent_states": [states_map.get(eid, default_state) for eid in entity_ids],
            "streaks": {
                rf: [m.get(eid, 0) for eid in entity_ids]
                for rf, m in streaks_map.items() if isinstance(m, dict)
            },
            "observe_streak": [obs_map.get(eid, 0) for eid in entity_ids],
        }

    return ctx


def _resolve_competition_groups(
    competition_groups: Any, entity_ids: list[str],
) -> np.ndarray | None:
    """将 competition_groups 配置转换为与 entity_ids 对齐的 int ndarray。

    支持两种格式：
      - dict: {entity_id: group_id(int)}
      - list: 与 entity_ids 等长的 int 列表
    未列出的实体默认归入 0 组。格式非法返回 None。
    """
    n = len(entity_ids)
    groups = np.zeros(n, dtype=np.int64)
    if isinstance(competition_groups, dict):
        for i, eid in enumerate(entity_ids):
            g = competition_groups.get(eid, 0)
            try:
                groups[i] = int(g)
            except (TypeError, ValueError):
                return None
        return groups
    if isinstance(competition_groups, (list, tuple)):
        if len(competition_groups) != n:
            return None
        try:
            groups = np.array([int(g) for g in competition_groups], dtype=np.int64)
        except (TypeError, ValueError):
            return None
        return groups
    return None


def rekey_fsm_blob(blob: Any, entity_ids: list[str]) -> dict[str, Any]:
    """把 FSM 模块输出的 index-aligned state_blob 转成 entity_id-keyed，便于跨轮回注。

    返回 {"states": {eid: state}, "streaks": {rf: {eid: int}}, "observe_streak": {eid: int}}。
    """
    out: dict[str, Any] = {"states": {}, "streaks": {}, "observe_streak": {}}
    if not isinstance(blob, dict):
        return out
    states = blob.get("agent_states", [])
    obs = blob.get("observe_streak", [])
    streaks = blob.get("streaks", {}) or {}
    for i, eid in enumerate(entity_ids):
        if i < len(states):
            out["states"][eid] = states[i]
        if i < len(obs):
            out["observe_streak"][eid] = obs[i]
    for rf, arr in streaks.items():
        d: dict[str, int] = {}
        if isinstance(arr, (list, tuple)):
            for i, eid in enumerate(entity_ids):
                if i < len(arr):
                    d[eid] = int(arr[i])
        out["streaks"][rf] = d
    return out


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

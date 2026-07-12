"""ODE module — continuous evolution of entity metrics within a round.

Uses scipy.integrate.solve_ivp (RK45 adaptive) when available.
Falls back to numpy Euler method if scipy is not installed.

All preset functions are pure: receive (values: np.ndarray, ctx_arrays: dict) →
return dy/dt as np.ndarray. No side effects, safe for vectorized integration.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp as _scipy_solve_ivp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .base import AlgorithmModule, ModuleContext

logger = logging.getLogger(__name__)

# ── Built-in ODE presets (pure functions: values, ctx_arrays → dy/dt) ──


def _decay(values: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Exponential decay: dy/dt = -rate * y. Rate configurable via ctx['_decay_rate']. Default 0.02."""
    rate = float(ctx.get("_decay_rate", 0.02))
    return -rate * values


def _logistic(values: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Logistic growth w/ configurable carrying capacity: dy/dt = rate*y*(1-y/K)
    
    Reads K from ctx['_carrying_capacity'] or defaults to ctx['_metric_range_hi'].
    Falls back to 100.0 as absolute default.
    """
    K = float(ctx.get("_carrying_capacity",
              ctx.get("_metric_range_hi", 100.0)))
    if K <= 0:
        K = 100.0
    rate = float(ctx.get("_logistic_rate", 0.03))
    return rate * values * (1.0 - values / K)


def _fatigue_recovery(values: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Fatigue recovery with saturation: dy/dt = -rate * sqrt(y) * (1 - y/K).
    Recovery rate peaks at moderate fatigue (~33) and tapers at both extremes,
    eliminating oscillation in the 50-70 range.
    Rate configurable via ctx['_fatigue_rate']; K via ctx['_carrying_capacity']."""
    rate = float(ctx.get("_fatigue_rate", 0.05))
    K = float(ctx.get("_carrying_capacity", 100.0))
    dy = np.zeros_like(values)
    mask = values > 0
    dy[mask] = -rate * np.sqrt(values[mask]) * (1.0 - values[mask] / K)
    return dy


def _supply_consumption(values: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Constant rate drain with strength-scaled consumption. Clamped so supply never goes negative."""
    strength = ctx.get("strength", np.zeros_like(values))
    rate = float(ctx.get("_supply_base_rate", 0.3))
    strength_factor = float(ctx.get("_supply_strength_factor", 0.01))
    raw = -rate - strength_factor * np.abs(strength) / 100.0
    # Clamp: derivative cannot drive value below 0 faster than its current value
    dt = float(ctx.get("_dt", 1.0))
    return np.maximum(raw, -values / max(dt, 0.01))


def _pollution_spread(values: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Pollution: generation from factories, dissipation over time"""
    factories = ctx.get("factory_output", np.zeros_like(values))
    greens = ctx.get("green_coverage", np.zeros_like(values))
    return 0.001 * factories - 0.05 * greens - 0.01 * values


def _resource_depletion(values: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Resource consumption proportional to population"""
    population = ctx.get("population", np.ones_like(values))
    return -0.005 * np.abs(population)


def _competitive_logistic(values: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Tech/talent competition: logistic growth + diffusion + crowding.
    
    - Leaders (>group_mean) lose advantage through knowledge spillover
    - Laggards (<group_mean) gain from absorption of frontier tech
    - Crowding effect prevents all entities converging to 100
    - Supports _competition_groups: int array (n,) partitioning entities into
      independent competition pools. When absent, all entities compete in one global pool.
    """
    K = float(ctx.get("_carrying_capacity", 100))
    rate = float(ctx.get("_logistic_rate", 0.06))
    diffusion_rate = float(ctx.get("_diffusion_rate", 0.015))
    comp = float(ctx.get("_competition_factor", 0.3))
    groups = ctx.get("_competition_groups", None)

    if groups is None:
        global_mean = float(ctx.get("_global_mean", np.mean(values)))
        growth = rate * values * (1.0 - values / K)
        diffusion = diffusion_rate * (global_mean - values)
        crowding = comp * values * np.abs(values - global_mean) / K
        return growth + diffusion - crowding

    result = np.zeros_like(values)
    for gid in np.unique(groups):
        mask = groups == gid
        gv = values[mask]
        if len(gv) == 0:
            continue
        gK = float(ctx.get("_carrying_capacity", K))
        actual_mean = np.mean(gv)
        growth = rate * gv * (1.0 - gv / gK)
        diffusion = diffusion_rate * (actual_mean - gv)
        crowding = comp * gv * np.abs(gv - actual_mean) / gK
        result[mask] = growth + diffusion - crowding
    return result


def _cash_flow_dynamics(values: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Cash flow with base recovery + scale protection from supply chain & tech + R&D monetization.
    
    _rnd_monetization models the revenue feedback from high R&D (licensing, tech premiums).
    Defaults to 0 for backward compatibility."""
    rate = float(ctx.get("_decay_rate", 0.01))
    base = float(ctx.get("_base_recovery", 0.0))
    scale = float(ctx.get("_scale_protection", 0.0))
    rnd_mon = float(ctx.get("_rnd_monetization", 0.0))
    supply_chain = ctx.get("supply_chain", np.zeros_like(values))
    tech_lead = ctx.get("tech_lead", np.ones_like(values) * 50)
    rnd = ctx.get("rnd", np.ones_like(values) * 50)
    protection = base + scale * (np.clip(supply_chain, 0, 100) / 100.0) * (np.clip(tech_lead, 0, 100) / 100.0)
    rnd_bonus = rnd_mon * (np.clip(rnd, 0, 100) / 100.0)
    return -rate * values + protection + rnd_bonus


ODE_PRESETS: dict[str, Any] = {
    "decay": _decay,
    "logistic": _logistic,
    "fatigue_recovery": _fatigue_recovery,
    "supply_consumption": _supply_consumption,
    "pollution_spread": _pollution_spread,
    "resource_depletion": _resource_depletion,
    "competitive_logistic": _competitive_logistic,
    "cash_flow_dynamics": _cash_flow_dynamics,
}

# Cross-metric dependencies for logging warnings
_ODE_DEPS: dict[str, list[str]] = {
    "supply_consumption": ["strength"],
    "pollution_spread": ["factory_output", "green_coverage"],
    "resource_depletion": ["population"],
    "cash_flow_dynamics": ["supply_chain", "tech_lead"],
}


class ODEModule(AlgorithmModule):
    """Continuous-time metric evolution (vectorized RK45 or fallback Euler)."""

    IS_FINALIZER = True

    @property
    def name(self) -> str:
        return "ode_engine"

    @property
    def description(self) -> str:
        return "常微分方程连续演化（N 实体 × M 指标" + (
            "，scipy RK45 自适应积分）" if _HAS_SCIPY else "，numpy Euler 法）")

    def configure(self, params: dict[str, Any]) -> None:
        self._ode_defs: dict[str, str] = params.get("equations", {})
        self._sub_steps: int = int(params.get("sub_steps", 8))
        if self._sub_steps < 4:
            logger.warning("[ODE] sub_steps=%d 较低，刚性方程可能精度不足", self._sub_steps)

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        if not ctx.arrays:
            return ctx
        if not hasattr(self, '_ode_defs'):
            return ctx
        if _HAS_SCIPY:
            return self._execute_scipy(ctx)
        return self._execute_euler(ctx)

    def _execute_scipy(self, ctx: ModuleContext) -> ModuleContext:
        """Vectorized RK45: one solve_ivp call for all entities × all metrics.

        All state is flattened into a single vector [metric0_e0, metric0_e1, ...,
        metric1_e0, metric1_e1, ...].  The ODE function slices it back for
        per-metric preset evaluation, then flattens the result.
        """
        keys = list(ctx.arrays.keys())
        n = len(ctx.arrays[keys[0]])
        y0 = np.concatenate([ctx.arrays[k] for k in keys])

        # Warn about missing dependency metrics (once per round)
        self._check_deps(ctx, keys)

        # Pre-compute slice intervals and views dict for reuse across integration steps
        _slices: list[tuple[int, int]] = []
        _start = 0
        for _k in keys:
            _slices.append((_start, _start + n))
            _start += n
        _views: dict[str, np.ndarray] = {}

        def ode_system(_t: float, y: np.ndarray) -> np.ndarray:
            _views.clear()
            for _idx, _k in enumerate(keys):
                s, e = _slices[_idx]
                _views[_k] = y[s:e]
            # Merge scalar ODE params from ctx.metadata as actual scalars
            ode_params = ctx.metadata.get("ode_params", {})
            for k, v in ode_params.items():
                _views[k] = float(v)
            competition_groups = ctx.metadata.get("_competition_groups", None)
            if competition_groups is not None:
                _views["_competition_groups"] = competition_groups
            deriv_parts: list[np.ndarray] = []
            for k in keys:
                eq_name = self._ode_defs.get(k, "")
                eq_fn = ODE_PRESETS.get(eq_name) if eq_name else None
                if eq_fn is not None:
                    deriv_parts.append(eq_fn(_views[k], _views))
                else:
                    deriv_parts.append(np.zeros(n, dtype=np.float64))
            return np.concatenate(deriv_parts)

        # Save pre-integration snapshot for recovery on failure
        arrays_snapshot = {k: v.copy() for k, v in ctx.arrays.items()}
        try:
            sol = _scipy_solve_ivp(
                ode_system, (0.0, ctx.dt), y0,
                method="RK45", rtol=1e-3, atol=1e-4,
                max_step=ctx.dt / 4,
            )
            if sol.success:
                final = sol.y[:, -1]
                start = 0
                for k in keys:
                    ctx.arrays[k] = final[start:start + n]
                    start += n
            else:
                logger.warning("[ODE] RK45 integration failed: %s | equations=%s — restoring pre-integration state",
                              sol.message, self._ode_defs)
                ctx.arrays = arrays_snapshot
        except Exception as e:
            logger.warning("[ODE] scipy solve_ivp failed: %s | equations=%s — restoring pre-integration state",
                          e, self._ode_defs)
            ctx.arrays = arrays_snapshot
        return ctx

    def _execute_euler(self, ctx: ModuleContext) -> ModuleContext:
        """Euler integration with frozen-state multi-step and snapshot recovery."""
        keys = list(ctx.arrays.keys())
        self._check_deps(ctx, keys)
        arrays_snapshot = {k: v.copy() for k, v in ctx.arrays.items()}
        try:
            sub_dt = ctx.dt / max(self._sub_steps, 1)
            for _ in range(self._sub_steps):
                augmented = dict(ctx.arrays)
                ode_params = ctx.metadata.get("ode_params", {})
                for k, v in ode_params.items():
                    augmented[k] = float(v)
                competition_groups = ctx.metadata.get("_competition_groups", None)
                if competition_groups is not None:
                    augmented["_competition_groups"] = competition_groups
                # Phase 1: compute all derivatives from frozen state
                dydt: dict[str, np.ndarray] = {}
                for key in keys:
                    eq_name = self._ode_defs.get(key, "")
                    eq_fn = ODE_PRESETS.get(eq_name) if eq_name else None
                    dydt[key] = eq_fn(ctx.arrays[key], augmented) if eq_fn else np.zeros_like(ctx.arrays[key])
                # Phase 2: apply all derivatives simultaneously
                for key in keys:
                    ctx.arrays[key] = ctx.arrays[key] + dydt[key] * sub_dt
            return ctx
        except Exception as e:
            logger.warning("[ODE] Euler 积分失败: %s | equations=%s — 恢复至积分前状态", e, self._ode_defs)
            ctx.arrays = arrays_snapshot
            return ctx

    def _check_deps(self, ctx: ModuleContext, keys: list[str]) -> None:
        """Log warnings when a preset's required metric is missing from ctx.arrays."""
        warned: set[str] = set()
        for key, eq_name in self._ode_defs.items():
            if eq_name in warned:
                continue
            needed = _ODE_DEPS.get(eq_name, [])
            missing = [n for n in needed if n not in keys]
            if missing:
                logger.warning(
                    "[ODE] preset '%s' (used by metric '%s') needs %s, not in ctx",
                    eq_name, key, missing,
                )
                warned.add(eq_name)


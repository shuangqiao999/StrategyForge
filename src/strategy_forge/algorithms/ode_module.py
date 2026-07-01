"""ODE module — continuous evolution of entity metrics within a round.

Uses per-metric differential equation dy/dt = f(y, t) defined in rule pack.
Falls back to exponential decay/growth if no equations defined.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import AlgorithmModule, ModuleContext


class ODEModule(AlgorithmModule):
    """Continuous-time metric evolution solver (Euler method)."""

    @property
    def name(self) -> str:
        return "ode_engine"

    @property
    def description(self) -> str:
        return "常微分方程连续演化（N 实体 × M 指标的平滑变化）"

    def configure(self, params: dict[str, Any]) -> None:
        self._sub_steps: int = int(params.get("sub_steps", 4))
        self._ode_defs: dict[str, str] = params.get("equations", {})

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        sub_dt = ctx.dt / max(self._sub_steps, 1)
        for _ in range(self._sub_steps):
            derivatives = self._compute_derivatives(ctx)
            for key, dy in derivatives.items():
                if key in ctx.arrays:
                    ctx.arrays[key] = ctx.arrays[key] + dy * sub_dt
        return ctx

    def _compute_derivatives(self, ctx: ModuleContext) -> dict[str, np.ndarray]:
        """Compute dy/dt for each metric, using rule-pack equations or defaults."""
        result: dict[str, np.ndarray] = {}
        for key, arr in ctx.arrays.items():
            n = len(arr)
            dy = np.zeros(n, dtype=np.float64)
            # Check if a custom equation is defined for this metric
            eq = self._ode_defs.get(key, "")
            if eq and eq in ODE_PRESETS:
                dy = ODE_PRESETS[eq](ctx, key)
            else:
                # Default: gentle regression toward a baseline (1% per sub_step)
                # Preserves non-NaN values, leaves NaN (dead entities) unchanged
                mask = ~np.isnan(arr)
                if mask.any():
                    dy[mask] = -0.002 * arr[mask]
            result[key] = dy
        return result


# ── Built-in ODE presets (equations referenceable by name in rule pack) ──


def _decay(ctx: ModuleContext, key: str) -> np.ndarray:
    """Exponential decay: dy/dt = -rate * y"""
    arr = ctx.arrays.get(key, np.array([]))
    if len(arr) == 0:
        return np.array([])
    return -0.02 * arr


def _logistic(ctx: ModuleContext, key: str) -> np.ndarray:
    """Logistic growth w/ carrying capacity 100: dy/dt = r*y*(1-y/K)"""
    arr = ctx.arrays.get(key, np.array([]))
    if len(arr) == 0:
        return np.array([])
    return 0.03 * arr * (1.0 - arr / 100.0)


def _fatigue_recovery(ctx: ModuleContext, key: str) -> np.ndarray:
    """Fast recovery when high, slow when low: dy/dt = -0.05 * sqrt(y)"""
    arr = ctx.arrays.get(key, np.array([]))
    if len(arr) == 0:
        return np.array([])
    dy = np.zeros(len(arr), dtype=np.float64)
    mask = arr > 0
    dy[mask] = -0.05 * np.sqrt(arr[mask])
    return dy


def _supply_consumption(ctx: ModuleContext, key: str) -> np.ndarray:
    """Constant rate drain with strength-scaled consumption"""
    arr = ctx.arrays.get(key, np.array([]))
    strength = ctx.arrays.get("strength", np.zeros(len(arr)))
    if len(arr) == 0:
        return np.array([])
    return -0.3 - 0.01 * np.abs(strength) / 100.0


def _pollution_spread(ctx: ModuleContext, key: str) -> np.ndarray:
    """Pollution: generation from factories, dissipation over time"""
    arr = ctx.arrays.get(key, np.array([]))
    factories = ctx.arrays.get("factory_output", np.zeros(len(arr)))
    greens = ctx.arrays.get("green_coverage", np.zeros(len(arr)))
    if len(arr) == 0:
        return np.array([])
    return 0.001 * factories - 0.05 * greens - 0.01 * arr


def _resource_depletion(ctx: ModuleContext, key: str) -> np.ndarray:
    """Resource consumption proportional to population"""
    arr = ctx.arrays.get(key, np.array([]))
    population = ctx.arrays.get("population", np.ones(len(arr)))
    if len(arr) == 0:
        return np.array([])
    return -0.005 * np.abs(population)


ODE_PRESETS: dict[str, Any] = {
    "decay": _decay,
    "logistic": _logistic,
    "fatigue_recovery": _fatigue_recovery,
    "supply_consumption": _supply_consumption,
    "pollution_spread": _pollution_spread,
    "resource_depletion": _resource_depletion,
}

"""ODE module — continuous evolution of entity metrics within a round.

Uses scipy.integrate.solve_ivp (RK45 adaptive) when available.
Falls back to numpy Euler method if scipy is not installed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp as _scipy_solve_ivp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .base import AlgorithmModule, ModuleContext


class ODEModule(AlgorithmModule):
    """Continuous-time metric evolution solver (adaptive RK45 or fallback Euler)."""

    @property
    def name(self) -> str:
        return "ode_engine"

    @property
    def description(self) -> str:
        return "常微分方程连续演化（N 实体 × M 指标" + (
            "，scipy RK45 自适应积分）" if _HAS_SCIPY else "，numpy Euler 法）")

    def configure(self, params: dict[str, Any]) -> None:
        self._ode_defs: dict[str, str] = params.get("equations", {})
        self._sub_steps: int = int(params.get("sub_steps", 4))

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        if _HAS_SCIPY:
            return self._execute_scipy(ctx)
        return self._execute_euler(ctx)

    def _execute_scipy(self, ctx: ModuleContext) -> ModuleContext:
        """Adaptive RK45 integration via scipy (higher precision)."""
        t_span = (0.0, ctx.dt)
        for key, arr in ctx.arrays.items():
            eq_name = self._ode_defs.get(key, "")
            eq_fn = ODE_PRESETS.get(eq_name) if eq_name else None

            for i in range(len(arr)):
                if np.isnan(arr[i]):
                    continue
                y0 = arr[i]
                if eq_fn is not None:
                    def ode_func(t: float, y: list[float], _i: int = i) -> list[float]:
                        backup = ctx.arrays[key][_i]
                        ctx.arrays[key][_i] = float(y[0])
                        dydt = eq_fn(ctx, key)
                        ctx.arrays[key][_i] = backup
                        return [float(dydt[_i])]
                else:
                    def ode_func(t: float, y: list[float]) -> list[float]:
                        return [-0.002 * float(y[0])]
                try:
                    sol = _scipy_solve_ivp(ode_func, t_span, [float(y0)],
                                           method='RK45', rtol=1e-4, atol=1e-6)
                    arr[i] = float(sol.y[0, -1])
                except Exception:
                    arr[i] = y0
        return ctx

    def _execute_euler(self, ctx: ModuleContext) -> ModuleContext:
        """Simple Euler integration (no scipy dependency)."""
        sub_dt = ctx.dt / max(self._sub_steps, 1)
        for _ in range(self._sub_steps):
            derivatives = self._compute_derivatives(ctx)
            for key, dy in derivatives.items():
                if key in ctx.arrays:
                    ctx.arrays[key] = ctx.arrays[key] + dy * sub_dt
        return ctx

    def _compute_derivatives(self, ctx: ModuleContext) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key, arr in ctx.arrays.items():
            n = len(arr)
            dy = np.zeros(n, dtype=np.float64)
            eq = self._ode_defs.get(key, "")
            if eq and eq in ODE_PRESETS:
                dy = ODE_PRESETS[eq](ctx, key)
            else:
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

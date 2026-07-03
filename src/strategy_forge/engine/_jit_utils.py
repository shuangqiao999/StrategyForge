"""JIT-accelerated batch operations for rule engine.

Provides Numba-accelerated variants of hotspot functions with pure-Python fallback.
Controlled by FORGE_DISABLE_NUMBA env var for environments where numba is unavailable.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_USE_NUMBA = os.getenv("FORGE_DISABLE_NUMBA", "").strip().lower() not in ("1", "true", "yes")

if _USE_NUMBA:
    try:
        from numba import jit, prange
        _HAS_NUMBA = True
    except ImportError:
        _HAS_NUMBA = False
else:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    logger.info("[JIT] batch delta acceleration: numba JIT enabled")
else:
    logger.info("[JIT] batch delta acceleration: numpy fallback (numba not available)")


def batch_apply_deltas(
    metrics_arr: np.ndarray,
    deltas_arr: np.ndarray,
    lo_arr: np.ndarray,
    hi_arr: np.ndarray,
) -> None:
    """Apply delta values to metric arrays with range clamping.

    Args:
        metrics_arr: (N, M) float64 — N entities × M metrics current values.
        deltas_arr: (N, M) float64 — delta to apply for each entity/metric.
        lo_arr: (M,) float64 — lower bounds per metric.
        hi_arr: (M,) float64 — upper bounds per metric.

    Modifies metrics_arr in-place.
    """
    if _HAS_NUMBA:
        _batch_apply_deltas_jit(metrics_arr, deltas_arr, lo_arr, hi_arr)
    else:
        _batch_apply_deltas_py(metrics_arr, deltas_arr, lo_arr, hi_arr)


if _HAS_NUMBA:
    @jit(nopython=True, parallel=True)
    def _batch_apply_deltas_jit(
        metrics: np.ndarray, deltas: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> None:
        N, M = metrics.shape
        for i in prange(N):
            for m in range(M):
                val = metrics[i, m] + deltas[i, m]
                if val < lo[m]:
                    val = lo[m]
                if val > hi[m]:
                    val = hi[m]
                metrics[i, m] = val
else:
    _batch_apply_deltas_jit = None  # type: ignore[assignment]


def _batch_apply_deltas_py(
    metrics: np.ndarray, deltas: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> None:
    """Pure Python fallback — still vectorized via numpy."""
    metrics[:] = np.clip(metrics + deltas, lo[np.newaxis, :], hi[np.newaxis, :])


def batch_eval_conditions(
    metric_arrays: dict[str, np.ndarray],
    conditions: list[tuple[str, str, float]],
) -> np.ndarray:
    """Evaluate condition expression against metric arrays. Returns (N,) bool mask.

    Args:
        metric_arrays: {metric_name: (N,) float64 array}.
        conditions: list of (metric, operator, threshold) tuples.
    Returns:
        (N,) bool array — True where all conditions are satisfied.
    """
    if not conditions:
        return np.ones(1, dtype=bool)
    mask = np.ones(len(next(iter(metric_arrays.values()))), dtype=bool)
    for metric, op, threshold in conditions:
        if metric not in metric_arrays:
            return np.zeros_like(mask)
        vals = metric_arrays[metric]
        if op == "<":
            mask &= vals < threshold
        elif op == ">":
            mask &= vals > threshold
        elif op == "<=":
            mask &= vals <= threshold
        elif op == ">=":
            mask &= vals >= threshold
        elif op == "==":
            mask &= np.abs(vals - threshold) < 1e-9
        elif op == "!=":
            mask &= np.abs(vals - threshold) >= 1e-9
    return mask


def _check_numba() -> bool:
    return _HAS_NUMBA

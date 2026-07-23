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

"""Algorithm module base — domain-agnostic compute units for the deduction engine.

All modules operate on generic numeric/spatial data via ModuleContext.
No domain knowledge is embedded in this layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SpatialState:
    """3D spatial state for N entities. All arrays are float32, shape (N,3) or (N,)."""

    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    velocities: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    masses: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    radii: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    forces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))

    def init_from_dict(
        self,
        entity_ids: list[str],
        init_pos: dict[str, list[float]] | None = None,
        init_vel: dict[str, list[float]] | None = None,
        init_mass: dict[str, float] | None = None,
        init_radius: dict[str, float] | None = None,
        box_size: float = 100.0,
    ) -> None:
        """Initialize from dict maps or auto-scatter entities in a 3D box."""
        n = len(entity_ids)
        rng = np.random.RandomState(42)
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.velocities = np.zeros((n, 3), dtype=np.float64)
        self.masses = np.ones(n, dtype=np.float64)
        self.radii = np.full(n, 5.0, dtype=np.float64)
        self.forces = np.zeros((n, 3), dtype=np.float64)

        for i, eid in enumerate(entity_ids):
            if init_pos and eid in init_pos:
                v = init_pos[eid]
                self.positions[i] = [float(v[j]) if j < len(v) else 0.0 for j in range(3)]
            else:
                self.positions[i] = rng.uniform(-box_size / 2, box_size / 2, 3)
            if init_vel and eid in init_vel:
                v = init_vel[eid]
                self.velocities[i] = [float(v[j]) if j < len(v) else 0.0 for j in range(3)]
            if init_mass and eid in init_mass:
                self.masses[i] = float(init_mass[eid])
            if init_radius and eid in init_radius:
                self.radii[i] = float(init_radius[eid])


@dataclass
class ModuleContext:
    """Shared data context passed through the algorithm module chain.

    --- Generic numeric layer ---
    arrays: dict[str, np.ndarray]
        Keyed by metric name, each value is shape (N,) float64 array.
        Modules can read/modify/add arrays freely.

    --- Spatial layer ---
    spatial: SpatialState
        3D state for entities with spatial properties.

    --- Control ---
    dt: float
        Integration step size in seconds (default 1.0 for one round).
    round_number: int
        Current simulation round.

    --- Passthrough ---
    metadata: dict
        Arbitrary key-value store for module-to-module communication.
    """

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    spatial: SpatialState = field(default_factory=SpatialState)
    dt: float = 1.0
    round_number: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── batch access helpers ──

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.metadata[key] = value


class AlgorithmModule(ABC):
    """Abstract base for all algorithm modules — domain-agnostic."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable summary of what this module does."""
        ...

    def configure(self, params: dict[str, Any]) -> None:
        """Optional module-specific configuration. Called before execute()."""
        pass

    @abstractmethod
    def execute(self, ctx: ModuleContext) -> ModuleContext:
        """Run the algorithm on ctx. Return the (possibly mutated) context."""
        ...

    def shutdown(self) -> None:
        """Optional cleanup."""
        pass


# ── bridge: EntityState ↔ ModuleContext ──


def states_to_arrays(
    states: dict[str, Any],
    metric_names: list[str],
    entity_ids: list[str],
) -> dict[str, np.ndarray]:
    """Convert EntityState metrics dicts → ModuleContext.arrays.

    Returns NaN for missing entities/metrics so modules can detect absence.
    """
    result: dict[str, np.ndarray] = {}
    for metric in metric_names:
        arr = np.full(len(entity_ids), np.nan, dtype=np.float64)
        for i, eid in enumerate(entity_ids):
            st = states.get(eid)
            if st is not None:
                arr[i] = float(st.metrics.get(metric, np.nan))
        result[metric] = arr
    return result


def arrays_to_states(
    ctx: ModuleContext,
    states: dict[str, Any],
    entity_ids: list[str],
    metric_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Write ctx.arrays back into EntityState.metrics, clamping to ranges."""
    ranges = metric_ranges or {}
    for metric, arr in ctx.arrays.items():
        lo, hi = ranges.get(metric, (-1e12, 1e12))
        for i, eid in enumerate(entity_ids):
            val = float(arr[i])
            if np.isnan(val):
                continue
            val = max(lo, min(hi, val))
            st = states.get(eid)
            if st is not None:
                st.metrics[metric] = val

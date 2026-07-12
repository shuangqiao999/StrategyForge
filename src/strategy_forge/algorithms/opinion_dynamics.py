"""Opinion Dynamics module — Hegselmann-Krause bounded confidence model.

Models how opinions/beliefs spread across entities in a social graph.
Each entity updates its opinion to the average of neighbors whose opinions
fall within epsilon distance.

Use cases: morale propagation, public trust erosion, alliance cohesion.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import AlgorithmModule, ModuleContext


class OpinionDynamicsModule(AlgorithmModule):
    """Hegselmann-Krause bounded confidence opinion dynamics.

    Reads: ctx.arrays[metric] for each metric specified in config,
           ctx.metadata["social_graph"] optional adjacency matrix override.

    Writes: ctx.arrays[metric] with updated opinion values.
    """

    REQUIRED_SIGNALS: list[str] = []
    OUTPUT_SIGNALS: list[str] = ["opinion_dynamics.updated_metrics"]

    def __init__(self) -> None:
        self._target_metrics: list[str] = []
        self._epsilon: float = 0.3  # bounded confidence radius (normalized 0-1)
        self._graph_type: str = "spatial"
        self._weighted: bool = False
        self._norm_range: tuple[float, float] | None = None
        self._alpha: float = 0.5  # convergence rate: 0=no change, 1=full neighbor influence

    @property
    def name(self) -> str:
        return "opinion_dynamics"

    @property
    def description(self) -> str:
        return "观点动力学（HK bounded confidence 模型）——民心/信任/士气的相互影响传播"

    def configure(self, params: dict[str, Any]) -> None:
        self._target_metrics = list(params.get("target_metrics", []))
        self._epsilon = float(params.get("epsilon", 0.3))
        self._graph_type = str(params.get("graph_type", "spatial"))
        self._weighted = bool(params.get("weighted", False))
        self._alpha = float(params.get("alpha", 0.5))
        nr = params.get("norm_range", None)
        if nr is not None and len(nr) == 2:
            self._norm_range = (float(nr[0]), float(nr[1]))
        else:
            self._norm_range = None

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        if not self._target_metrics:
            return ctx

        n = len(next(iter(ctx.arrays.values()))) if ctx.arrays else 0
        if n < 2:
            return ctx

        # Build adjacency / similarity matrix
        if self._graph_type == "social" and "social_graph" in ctx.metadata:
            adj = np.array(ctx.metadata["social_graph"], dtype=np.float64)
        elif self._graph_type == "complete":
            adj = np.ones((n, n), dtype=np.float64)
            np.fill_diagonal(adj, 0.0)
        elif self._graph_type == "from_metadata" and "relation_graph" in ctx.metadata:
            adj = np.array(ctx.metadata["relation_graph"], dtype=np.float64)
        else:
            # Default "spatial": spatial-proximity based graph
            sp = ctx.spatial
            adj = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                dvec = sp.positions - sp.positions[i]
                dist = np.linalg.norm(dvec, axis=1)
                threshold = (sp.radii[i] + sp.radii) * 2.0
                adj[i] = (dist < threshold).astype(np.float64)
            np.fill_diagonal(adj, 0.0)

        updated_metrics: list[str] = []
        for metric in self._target_metrics:
            if metric not in ctx.arrays:
                continue
            opinions = ctx.arrays[metric].copy()
            # Normalize to [0, 1] for epsilon comparison
            if self._norm_range is not None:
                lo, hi = self._norm_range
                scale = hi - lo
                if scale > 0:
                    norm = (opinions - lo) / scale
                else:
                    max_val = np.max(opinions) if len(opinions) > 0 else 1.0
                    max_val = max(max_val, 1.0)
                    norm = opinions / max_val
            else:
                max_val = np.max(opinions) if len(opinions) > 0 else 1.0
                if max_val <= 0:
                    max_val = 1.0
                norm = opinions / max_val
            new_opinions = opinions.copy()

            for i in range(n):
                diff = np.abs(norm - norm[i])
                mask = (diff <= self._epsilon) & (adj[i] > 0)
                mask[i] = False  # exclude self
                if np.any(mask):
                    if self._weighted and "relation_weights" in ctx.metadata:
                        w = np.array(ctx.metadata["relation_weights"])[i][mask]
                        w_sum = w.sum()
                        if w_sum > 0:
                            w = w / w_sum
                            new_opinions[i] = opinions[i] * (1 - self._alpha) + np.average(opinions[mask], weights=w) * self._alpha
                        else:
                            new_opinions[i] = opinions[i] * (1 - self._alpha) + np.mean(opinions[mask]) * self._alpha
                    else:
                        new_opinions[i] = opinions[i] * (1 - self._alpha) + np.mean(opinions[mask]) * self._alpha

            # Clamp back to original range to prevent numerical drift
            if self._norm_range is not None:
                lo, hi = self._norm_range
                new_opinions = np.clip(new_opinions, lo, hi)
            ctx.arrays[metric] = new_opinions
            updated_metrics.append(metric)

        ctx.metadata["opinion_dynamics.updated_metrics"] = updated_metrics
        return ctx

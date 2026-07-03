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
        self._use_social_graph: bool = False

    @property
    def name(self) -> str:
        return "opinion_dynamics"

    @property
    def description(self) -> str:
        return "观点动力学（HK bounded confidence 模型）——民心/信任/士气的相互影响传播"

    def configure(self, params: dict[str, Any]) -> None:
        self._target_metrics = list(params.get("target_metrics", []))
        self._epsilon = float(params.get("epsilon", 0.3))
        self._use_social_graph = bool(params.get("use_social_graph", False))

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        if not self._target_metrics:
            return ctx

        n = len(next(iter(ctx.arrays.values()))) if ctx.arrays else 0
        if n < 2:
            return ctx

        # Build adjacency / similarity matrix
        if self._use_social_graph and "social_graph" in ctx.metadata:
            adj = np.array(ctx.metadata["social_graph"], dtype=np.float64)
        else:
            # Default: spatial-proximity based graph (1 if distance < 2*radius, else 0)
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
            max_val = np.max(opinions) if len(opinions) > 0 else 1.0
            if max_val <= 0:
                max_val = 1.0
            # Normalize to [0, 1] for epsilon comparison
            norm = opinions / max_val
            new_opinions = opinions.copy()

            for i in range(n):
                diff = np.abs(norm - norm[i])
                mask = (diff <= self._epsilon) & (adj[i] > 0)
                mask[i] = False  # exclude self
                if np.any(mask):
                    neighbors = opinions[mask]
                    new_opinions[i] = opinions[i] * 0.5 + np.mean(neighbors) * 0.5

            ctx.arrays[metric] = new_opinions
            updated_metrics.append(metric)

        ctx.metadata["opinion_dynamics.updated_metrics"] = updated_metrics
        return ctx

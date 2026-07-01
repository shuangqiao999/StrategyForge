"""3D Physics module — rigid-body dynamics, collision detection, diffusion,
and radial explosion/shockwave fields. All subsystems are parameter-driven.

Collision switches to spatial hash when N > 150 for near-linear scaling.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import AlgorithmModule, ModuleContext


# ── Spatial hash helpers (lightweight, no extra deps) ──


def _build_hash(p: np.ndarray, r: np.ndarray) -> tuple[dict, float]:
    """Build cell→entity_indices map. Returns (hash_map, cell_size)."""
    cell_size = float(np.max(r) * 2.0 + 1e-3)
    hmap: dict[tuple[int, int, int], list[int]] = {}
    for i in range(len(p)):
        c = (int(np.floor(p[i, 0] / cell_size)),
             int(np.floor(p[i, 1] / cell_size)),
             int(np.floor(p[i, 2] / cell_size)))
        hmap.setdefault(c, []).append(i)
    return hmap, cell_size


def _hash_neighbor_candidates(c: tuple, hmap: dict, cell: tuple[int, int, int]) -> None:
    """Yield all entity indices in 3x3x3 adjacent cells."""
    seen: set[int] = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                for idx in hmap.get(neighbor, []):
                    if idx not in seen:
                        seen.add(idx)
                        c.add(idx)


class PhysicsModule(AlgorithmModule):
    """3D physics engine with selectable subsystems."""

    def __init__(self) -> None:
        self._enabled: dict[str, bool] = {
            "dynamics": True,
            "collision": True,
            "diffusion": True,
            "explosion": True,
        }
        self._params: dict[str, Any] = {
            "gravity": 9.8,
            "damping": 0.98,
            "collision_elasticity": 0.5,
            "diffusion_rate": 0.05,
            "diffusion_sigma_scale": 3.0,
            "explosion_sources": [],
        }

    @property
    def name(self) -> str:
        return "physics_engine"

    @property
    def description(self) -> str:
        return "3D 物理引擎（刚体动力学/碰撞(N>150空间哈希)/扩散/冲击波）"

    def configure(self, params: dict[str, Any]) -> None:
        if "subsystems" in params:
            self._enabled = {k: k in params["subsystems"] for k in self._enabled}
        for k in ("gravity", "damping", "collision_elasticity", "diffusion_rate",
                   "diffusion_sigma_scale"):
            if k in params:
                self._params[k] = float(params[k])
        if "explosion_sources" in params:
            self._params["explosion_sources"] = params["explosion_sources"]

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        sp = ctx.spatial
        n = len(sp.positions)
        if n == 0:
            return ctx

        if self._enabled["dynamics"]:
            self._step_dynamics(sp, ctx.dt)
        if self._enabled["collision"]:
            self._resolve_collisions(sp, ctx)
        if self._enabled["diffusion"]:
            self._step_diffusion(ctx)
        if self._enabled["explosion"]:
            self._apply_explosions(sp, ctx)

        return ctx

    # ── Subsystems ──

    def _step_dynamics(self, sp: Any, dt: float) -> None:
        g = self._params["gravity"]
        dam = self._params["damping"]
        dt_clamped = min(dt, 0.5)
        acc = sp.forces / np.maximum(sp.masses.reshape(-1, 1), 0.001)
        acc[:, 2] -= g
        sp.velocities = (sp.velocities + acc * dt_clamped) * dam
        sp.positions = sp.positions + sp.velocities * dt_clamped
        sp.forces.fill(0.0)

    def _resolve_collisions(self, sp: Any, ctx: ModuleContext) -> None:
        n = len(sp.positions)
        if n < 2:
            return
        if n > 150:
            self._resolve_collisions_hash(sp, self._params["collision_elasticity"])
        else:
            self._resolve_collisions_brute(sp, self._params["collision_elasticity"])

    def _resolve_collisions_brute(self, sp: Any, e: float) -> None:
        n = len(sp.positions)
        p, v, r, m = sp.positions, sp.velocities, sp.radii, sp.masses
        for i in range(n):
            for j in range(i + 1, n):
                dvec = p[i] - p[j]
                dist = float(np.linalg.norm(dvec))
                min_dist = r[i] + r[j]
                if dist >= min_dist or dist < 1e-9:
                    continue
                self._collision_response(p, v, m, i, j, dvec, dist, min_dist, e)

    def _resolve_collisions_hash(self, sp: Any, e: float) -> None:
        hmap, _ = _build_hash(sp.positions, sp.radii)
        p, v, r, m = sp.positions, sp.velocities, sp.radii, sp.masses
        for i in range(len(p)):
            cell = (int(np.floor(p[i, 0] / (np.max(r) * 2.0 + 1e-3))),
                    int(np.floor(p[i, 1] / (np.max(r) * 2.0 + 1e-3))),
                    int(np.floor(p[i, 2] / (np.max(r) * 2.0 + 1e-3))))
            candidates: set[int] = set()
            _hash_neighbor_candidates(candidates, hmap, cell)
            for j in candidates:
                if j <= i:
                    continue
                dvec = p[i] - p[j]
                dist = float(np.linalg.norm(dvec))
                min_dist = r[i] + r[j]
                if dist >= min_dist or dist < 1e-9:
                    continue
                self._collision_response(p, v, m, i, j, dvec, dist, min_dist, e)

    @staticmethod
    def _collision_response(p: np.ndarray, v: np.ndarray, m: np.ndarray,
                            i: int, j: int, dvec: np.ndarray, dist: float,
                            min_dist: float, e: float) -> None:
        norm = dvec / dist
        overlap = min_dist - dist
        total_m = m[i] + m[j]
        if total_m > 0:
            p[i] += norm * overlap * (m[j] / total_m)
            p[j] -= norm * overlap * (m[i] / total_m)
        rel_v_n = float(np.dot(v[i] - v[j], norm))
        if rel_v_n > 0:
            impulse = -(1.0 + e) * rel_v_n / max(total_m, 0.001)
            v[i] += impulse * m[j] * norm
            v[j] -= impulse * m[i] * norm

    def _step_diffusion(self, ctx: ModuleContext) -> None:
        sp = ctx.spatial
        n = len(sp.positions)
        if n < 2:
            return
        rate = self._params["diffusion_rate"]
        sigma_scale = float(self._params.get("diffusion_sigma_scale", 3.0))
        p = sp.positions
        target_keys = ctx.diffusion_fields if ctx.diffusion_fields else []
        for key in target_keys:
            if key not in ctx.arrays:
                continue
            arr = ctx.arrays[key]
            valid = ~np.isnan(arr)
            if valid.sum() < 2:
                continue
            new_arr = arr.copy()
            for i in range(n):
                if not valid[i]:
                    continue
                dvec = p - p[i]
                sqdist = np.sum(dvec * dvec, axis=1)
                sigma2 = (sp.radii[i] * sigma_scale) ** 2
                weights = np.exp(-sqdist / (sigma2 + 1e-6))
                weights[i] = 0.0
                w_sum = weights.sum()
                if w_sum > 1e-9:
                    new_arr[i] += rate * np.sum((arr - arr[i]) * weights) / w_sum
            ctx.arrays[key] = new_arr

    def _apply_explosions(self, sp: Any, ctx: ModuleContext) -> None:
        # Static sources from rule pack
        sources: list[dict] = list(self._params.get("explosion_sources", []))
        # Dynamic triggers from ctx.metadata
        for trigger in ctx.metadata.get("trigger_explosion", []):
            sources.append({
                "center": trigger.get("center", [0, 0, 0]),
                "power": float(trigger.get("power", 100.0)),
                "radius": float(trigger.get("radius", 50.0)),
            })
        if not sources:
            return
        p = sp.positions
        events: list[dict] = ctx.metadata.setdefault("explosion_events", [])
        for src in sources:
            center = np.array(src.get("center", [0, 0, 0]), dtype=np.float64)
            power = float(src.get("power", 100.0))
            radius = float(src.get("radius", 50.0))
            dvec = p - center
            dists = np.linalg.norm(dvec, axis=1)
            mask = dists < radius
            if not mask.any():
                continue
            frac = 1.0 - dists[mask] / (radius + 1e-6)
            directions = dvec[mask] / (dists[mask, np.newaxis] + 1e-6)
            sp.forces[mask] += directions * (power * frac[:, np.newaxis])
            events.append({
                "center": center.tolist(),
                "power": power,
                "radius": radius,
                "affected_count": int(mask.sum()),
                "max_damage_ratio": float(frac.max()),
            })

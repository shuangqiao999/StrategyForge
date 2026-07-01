"""3D Physics module — rigid-body dynamics, collision detection, diffusion,
and radial explosion/shockwave fields. All subsystems are parameter-driven.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import AlgorithmModule, ModuleContext


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
            "explosion_sources": [],
        }

    @property
    def name(self) -> str:
        return "physics_engine"

    @property
    def description(self) -> str:
        return "3D 物理引擎（刚体动力学/碰撞/扩散/冲击波）"

    def configure(self, params: dict[str, Any]) -> None:
        if "subsystems" in params:
            self._enabled = {k: k in params["subsystems"] for k in self._enabled}
        for k in ("gravity", "damping", "collision_elasticity", "diffusion_rate"):
            if k in params:
                self._params[k] = float(params[k])
        if "explosion_sources" in params:
            self._params["explosion_sources"] = params["explosion_sources"]

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        sp = ctx.spatial
        n = len(sp.positions)
        if n == 0:
            return ctx

        # ── 1. Rigid-body dynamics (Euler integration) ──
        if self._enabled["dynamics"]:
            self._step_dynamics(sp, ctx.dt)

        # ── 2. Collision detection + response ──
        if self._enabled["collision"]:
            self._resolve_collisions(sp, ctx)

        # ── 3. Diffusion (scalar fields over spatial adjacency) ──
        if self._enabled["diffusion"]:
            self._step_diffusion(ctx)

        # ── 4. Explosion / shockwave radial field ──
        if self._enabled["explosion"]:
            self._apply_explosions(sp, ctx)

        return ctx

    # ── Subsystems ──

    def _step_dynamics(self, sp: Any, dt: float) -> None:
        """Semi-implicit Euler: v += a*dt, p += v*dt. Gravity on -Z axis."""
        n = len(sp.positions)
        g = self._params["gravity"]
        dam = self._params["damping"]
        dt_clamped = min(dt, 0.5)  # limit step for stability
        # acceleration from forces + gravity
        acc = sp.forces / np.maximum(sp.masses.reshape(-1, 1), 0.001)
        acc[:, 2] -= g  # gravity along Z-down
        sp.velocities = (sp.velocities + acc * dt_clamped) * dam
        sp.positions = sp.positions + sp.velocities * dt_clamped
        sp.forces.fill(0.0)

    def _resolve_collisions(self, sp: Any, ctx: ModuleContext) -> None:
        """AABB broad-phase + sphere collision response."""
        n = len(sp.positions)
        if n < 2:
            return
        e = self._params["collision_elasticity"]

        # Simple O(n²) for moderate N — adequate for ≤ 200 entities
        p = sp.positions
        v = sp.velocities
        r = sp.radii
        m = sp.masses

        for i in range(n):
            for j in range(i + 1, n):
                dvec = p[i] - p[j]
                dist = np.linalg.norm(dvec)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-9:
                    # Separate overlapping entities
                    overlap = min_dist - dist
                    norm = dvec / dist
                    total_m = m[i] + m[j]
                    if total_m > 0:
                        p[i] += norm * overlap * (m[j] / total_m)
                        p[j] -= norm * overlap * (m[i] / total_m)
                    # Impulse-based velocity response
                    rel_v = np.dot(v[i] - v[j], norm)
                    if rel_v > 0:  # approaching
                        impulse = -(1.0 + e) * rel_v / max(total_m, 0.001)
                        v[i] += impulse * m[j] * norm
                        v[j] -= impulse * m[i] * norm

    def _step_diffusion(self, ctx: ModuleContext) -> None:
        """Diffuse scalar fields among spatially-adjacent entities (Gaussian kernel)."""
        sp = ctx.spatial
        n = len(sp.positions)
        if n < 2:
            return
        rate = self._params["diffusion_rate"]
        p = sp.positions
        # Only diffuse metrics explicitly declared as spatial diffusion fields
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
                sigma2 = (sp.radii[i] * 3.0) ** 2
                weights = np.exp(-sqdist / (sigma2 + 1e-6))
                weights[i] = 0.0
                w_sum = weights.sum()
                if w_sum > 1e-9:
                    new_arr[i] += rate * np.sum((arr - arr[i]) * weights) / w_sum
            ctx.arrays[key] = new_arr

    def _apply_explosions(self, sp: Any, ctx: ModuleContext) -> None:
        """Apply radial explosion forces and record damage events (no direct metric mutation)."""
        sources = self._params.get("explosion_sources", [])
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

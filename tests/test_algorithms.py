"""Smoke test for algorithm modules."""
import numpy as np
from strategy_forge.algorithms import ModuleContext, SpatialState
from strategy_forge.algorithms.physics_module import PhysicsModule
from strategy_forge.algorithms.ode_module import ODEModule

# ── Test collision resolution ──
ctx = ModuleContext(round_number=1)
ctx.spatial = SpatialState()
ctx.spatial.init_from_dict(
    ["e1", "e2", "e3"],
    {"e1": [0, 0, 0], "e2": [8, 0, 0], "e3": [50, 0, 0]},
    {"e1": [0, 0, 0], "e2": [0, 0, 0], "e3": [0, 0, 0]},
    {"e1": 1.0, "e2": 0.5, "e3": 2.0},
    {"e1": 5, "e2": 6, "e3": 10},
)
ctx.arrays = {"strength": np.array([100.0, 80.0, 50.0], dtype=np.float64)}

phys = PhysicsModule()
phys.configure({"subsystems": ["collision"], "collision_elasticity": 0.5})
ctx = phys.execute(ctx)

d12 = np.linalg.norm(ctx.spatial.positions[0] - ctx.spatial.positions[1])
assert d12 >= 10.5, f"Collision separation failed: {d12}"
print(f"Collision: e1-e2 distance = {d12:.2f} (>=11)")

# ── Test explosion ──
ctx2 = ModuleContext(round_number=2)
ctx2.spatial = ctx.spatial
ctx2.arrays = {"strength": np.array([100.0, 80.0, 50.0], dtype=np.float64)}
phys2 = PhysicsModule()
phys2.configure({
    "subsystems": ["explosion"],
    "explosion_sources": [{"center": [0, 0, 0], "power": 200, "radius": 100}],
})
ctx2 = phys2.execute(ctx2)
# Explosion applies forces to spatial state (not direct metric damage)
f = ctx2.spatial.forces
events = ctx2.metadata.get("explosion_events", [])
print(f"Explosion forces (e1): {f[0]}, affected={events[0]['affected_count'] if events else 0}")
assert np.linalg.norm(f[0]) > 10, "Explosion should apply significant force to nearby entity"
assert len(events) > 0, "Explosion should record events"

# ── Test ODE ──
ctx3 = ModuleContext(round_number=3)
ctx3.arrays = {
    "fatigue": np.array([80.0, 30.0], dtype=np.float64),
    "supply": np.array([50.0, 10.0], dtype=np.float64),
}
ode = ODEModule()
ode.configure({"sub_steps": 4, "equations": {"fatigue": "fatigue_recovery", "supply": "supply_consumption"}})
ctx3 = ode.execute(ctx3)
print(f"ODE: fatigue {ctx3.arrays['fatigue']} (should decrease)")
print(f"ODE: supply {ctx3.arrays['supply']} (should decrease)")
assert ctx3.arrays["fatigue"][0] < 80, "fatigue should recover (decrease)"

print("\nALL ALGORITHM TESTS PASSED")

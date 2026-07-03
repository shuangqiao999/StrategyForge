"""Algorithm modules for StrategyForge deduction engine.

Module registry: all available algorithm modules keyed by their name property.
New modules: import and register here to make them available in rules.json pipeline.
"""
from .base import AlgorithmModule, ModuleContext, SpatialState, arrays_to_states, states_to_arrays
from .fsm_module import FiniteStateMachineModule
from .ode_module import ODEModule
from .opinion_dynamics import OpinionDynamicsModule
from .physics_module import PhysicsModule
from .pipeline_engine import PipelineEngine
from .module_utils import apply_context_results, build_context, build_pipeline

# Module registry: {module_name: module_class}
MODULE_REGISTRY: dict[str, type[AlgorithmModule]] = {
    "ode_engine": ODEModule,
    "physics_engine": PhysicsModule,
    "opinion_dynamics": OpinionDynamicsModule,
    "finite_state_machine": FiniteStateMachineModule,
}

__all__ = [
    "AlgorithmModule",
    "FiniteStateMachineModule",
    "MODULE_REGISTRY",
    "ModuleContext",
    "ODEModule",
    "OpinionDynamicsModule",
    "PhysicsModule",
    "PipelineEngine",
    "SpatialState",
    "arrays_to_states",
    "apply_context_results",
    "build_context",
    "build_pipeline",
    "states_to_arrays",
]

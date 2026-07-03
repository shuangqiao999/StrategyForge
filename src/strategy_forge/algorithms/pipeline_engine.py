"""Pipeline Engine — configurable module execution from rules.json pipeline config.

Replaces hardcoded ODE→Physics ordering with a rules.json-driven dispatch.
No pipeline config = backward compatible with existing module chain behavior.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import AlgorithmModule, ModuleContext

logger = logging.getLogger(__name__)


class PipelineEngine:
    """Dispatches algorithm modules in the order specified by rules.json pipeline section."""

    def __init__(self) -> None:
        self._modules: dict[str, AlgorithmModule] = {}

    def register(self, module: AlgorithmModule) -> None:
        """Register a module instance by its name."""
        self._modules[module.name] = module

    def get(self, name: str) -> AlgorithmModule | None:
        """Get a registered module by name."""
        return self._modules.get(name)

    def run(
        self,
        ctx: ModuleContext,
        pipeline_cfg: dict[str, Any] | None = None,
        log_fn: Any = None,
    ) -> ModuleContext:
        """Execute modules in pipeline order.

        Args:
            ctx: The module context (mutated in-place by each module).
            pipeline_cfg: rules.json modules.pipeline dict. If None, runs all registered
                          modules in registration order (backward compat).
            log_fn: Optional logging callback (phase, msg).

        Returns:
            The (possibly mutated) ModuleContext.
        """
        if not self._modules:
            return ctx

        order: list[str] = []
        conditionals: dict[str, str] = {}
        context_mapping: dict[str, dict] = {}

        if pipeline_cfg:
            order = pipeline_cfg.get("order", list(self._modules.keys()))
            conditionals = pipeline_cfg.get("conditionals", {})

        if not order:
            order = list(self._modules.keys())

        # Auto-sort: IS_FINALIZER modules always run last.
        # This ensures analysis modules (opinion_dynamics, fsm) write metadata
        # before finalizer modules (ode_engine, physics_engine) overwrite arrays.
        original = list(order)
        finalizers = [n for n in order
                      if self._modules.get(n) and getattr(self._modules[n], "IS_FINALIZER", False)]
        non_final = [n for n in order if n not in finalizers]
        order = non_final + finalizers
        if order != original:
            logger.debug("[Pipeline] finalizers sorted to end: %s", [n for n in order if n in finalizers])

        for name in order:
            mod = self._modules.get(name)
            if mod is None:
                msg = f"模块 {name} 未注册，跳过"
                logger.warning("[Pipeline] %s", msg)
                if log_fn:
                    log_fn("pipeline", msg)
                continue

            # Check conditional execution
            cond_key = f"execute_{name}"
            if cond_key in conditionals and not self._eval_cond(
                conditionals[cond_key], ctx
            ):
                if log_fn:
                    log_fn("pipeline", f"条件跳过: {name}")
                continue

            # Signal validation
            if hasattr(mod, "REQUIRED_SIGNALS") and mod.REQUIRED_SIGNALS:
                if not mod._validate(ctx):
                    msg = f"模块 {name} 缺少前置信号 {mod.REQUIRED_SIGNALS}，跳过"
                    logger.warning("[Pipeline] %s", msg)
                    if log_fn:
                        log_fn("pipeline", msg)
                    continue

            try:
                ctx = mod.execute(ctx)
            except Exception as e:
                msg = f"模块 {name} 执行异常: {e}"
                logger.warning("[Pipeline] %s", msg)
                if log_fn:
                    log_fn("pipeline", msg)

        return ctx

    @staticmethod
    def _eval_cond(condition: str, ctx: ModuleContext) -> bool:
        """Evaluate a simple condition string against ctx.metadata.

        Supported: 'key == value', 'key != value', 'key < value', 'key > value'.
        """
        if not condition:
            return True
        parts = condition.split()
        if len(parts) < 3:
            return True
        key, op, val = parts[0], parts[1], parts[2]
        actual = ctx.metadata.get(key)
        if actual is None:
            return False
        if val == "true":
            val = True
        elif val == "false":
            val = False
        else:
            try:
                val = float(val)
            except (ValueError, TypeError):
                pass
        try:
            if op == "==":
                return actual == val
            if op == "!=":
                return actual != val
            if op == "<":
                return actual < val
            if op == ">":
                return actual > val
        except (TypeError, ValueError):
            return False
        return True

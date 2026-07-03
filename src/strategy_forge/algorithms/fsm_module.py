"""Finite State Machine module — discrete state transitions for NPC autonomy.

Reduces LLM call overhead by letting agents switch between predefined states
(patrol → alert → attack) based on entity metric thresholds.

Use cases: large-scale military unit behavior, resource allocation triggers.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import AlgorithmModule, ModuleContext


class FiniteStateMachineModule(AlgorithmModule):
    """Discrete state transition engine for entity behavior autonomy.

    Reads: ctx.arrays (entity metrics) for threshold evaluation.
    Config: transition_rules list of {from, to, condition}.
    Writes: ctx.metadata["fsm.agent_states"] = {entity_idx: state_name}.
    """

    REQUIRED_SIGNALS: list[str] = []
    OUTPUT_SIGNALS: list[str] = ["fsm.agent_states"]

    def __init__(self) -> None:
        self._rules: list[dict[str, Any]] = []
        self._default_state: str = "idle"

    @property
    def name(self) -> str:
        return "finite_state_machine"

    @property
    def description(self) -> str:
        return "有限状态机（离散状态转移）——降低 NPC 行为决策的 LLM 调用开销"

    def configure(self, params: dict[str, Any]) -> None:
        self._rules = list(params.get("transition_rules", []))
        self._default_state = str(params.get("default_state", "idle"))

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        n = len(next(iter(ctx.arrays.values()))) if ctx.arrays else 0
        if n == 0:
            return ctx

        # Load previous states
        prev_states: list[str] = ctx.metadata.get("fsm.agent_states", [self._default_state] * n)
        if isinstance(prev_states, dict):
            prev_states = [prev_states.get(i, self._default_state) for i in range(n)]
        if len(prev_states) != n:
            prev_states = [self._default_state] * n

        new_states = list(prev_states)

        for rule in self._rules:
            from_state = rule.get("from", "")
            to_state = rule.get("to", "")
            condition = rule.get("condition", {})
            if not from_state or not to_state:
                continue

            for i in range(n):
                if new_states[i] != from_state:
                    continue
                if self._match_condition(ctx, i, condition):
                    new_states[i] = to_state

        ctx.metadata["fsm.agent_states"] = new_states
        return ctx

    @staticmethod
    def _match_condition(ctx: ModuleContext, idx: int, condition: dict) -> bool:
        """Check if entity idx satisfies all condition thresholds."""
        for metric, (op, threshold) in condition.items():
            if metric not in ctx.arrays:
                return False
            val = float(ctx.arrays[metric][idx])
            if op == "<" and not (val < threshold):
                return False
            if op == ">" and not (val > threshold):
                return False
            if op == "<=" and not (val <= threshold):
                return False
            if op == ">=" and not (val >= threshold):
                return False
            if op == "==" and not (val == threshold):
                return False
        return True

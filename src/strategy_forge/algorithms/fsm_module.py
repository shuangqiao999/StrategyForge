"""Finite State Machine module — discrete state transitions for NPC autonomy.

Reduces LLM call overhead by letting agents switch between predefined states
(patrol → alert → attack) based on entity metric thresholds.

States marked as command_states are handed off to LLM for final decision.
All other states produce deterministic actions via action_map, bypassing LLM.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import numpy as np

from .base import AlgorithmModule, ModuleContext


class FiniteStateMachineModule(AlgorithmModule):
    """Discrete state transition engine for entity behavior autonomy.

    Reads: ctx.arrays (entity metrics) for threshold evaluation.
    Config: transition_rules, action_map, command_states, default_state.
    Writes: ctx.metadata["fsm.agent_states"], ctx.metadata["fsm.agent_actions"],
            ctx.metadata["fsm.command_states"].
    """

    REQUIRED_SIGNALS: list[str] = []
    OUTPUT_SIGNALS: list[str] = [
        "fsm.agent_states",
        "fsm.agent_actions",
        "fsm.command_states",
    ]

    def __init__(self) -> None:
        self._rules: list[dict[str, Any]] = []
        self._default_state: str = "idle"
        self._action_map: dict[str, dict[str, Any] | None] = {}
        self._command_states: set[str] = {"combat"}

    @property
    def name(self) -> str:
        return "finite_state_machine"

    @property
    def description(self) -> str:
        return "有限状态机（离散状态转移+动作映射）——降低 NPC 行为决策的 LLM 调用开销"

    def configure(self, params: dict[str, Any]) -> None:
        self._rules = list(params.get("transition_rules", []))
        self._default_state = str(params.get("default_state", "idle"))
        self._action_map = dict(params.get("action_map", {}))
        command = params.get("command_states", ["combat"])
        self._command_states = set(command if isinstance(command, list) else [command])

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        n = len(next(iter(ctx.arrays.values()))) if ctx.arrays else 0
        if n == 0:
            return ctx

        prev_states: list[str] = ctx.metadata.get(
            "fsm.agent_states", [self._default_state] * n
        )
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
        ctx.metadata["fsm.command_states"] = list(self._command_states)

        # Build deterministic actions for non-command agents
        agent_actions: list[dict[str, Any] | None] = []
        for i in range(n):
            state = new_states[i]
            if state in self._command_states:
                agent_actions.append(None)  # handed to LLM
            else:
                mapped = self._action_map.get(state)
                if mapped is None:
                    agent_actions.append({
                        "action_type": "observe", "intensity": 0.3,
                        "target": "", "rationale": f"[FSM] {state}",
                    })
                elif isinstance(mapped, dict):
                    agent_actions.append({
                        "action_type": mapped.get("action_type", "observe"),
                        "intensity": float(mapped.get("intensity", 0.5)),
                        "target": str(mapped.get("target", "") or ""),
                        "rationale": f"[FSM] {state}",
                    })
                else:
                    agent_actions.append(None)
        ctx.metadata["fsm.agent_actions"] = agent_actions

        return ctx

    @staticmethod
    def _match_condition(ctx: ModuleContext, idx: int, condition: dict) -> bool:
        """Check if entity idx satisfies all condition thresholds.
        
        Supports virtual spatial metrics:
          - distance_to_enemy / distance_to_ally: computed from ctx.spatial + metadata.
        """
        for metric, (op, threshold) in condition.items():
            val = FiniteStateMachineModule._resolve_metric(ctx, idx, metric)
            if val is None:
                return False
            if op == "<" and not (val < float(threshold)):
                return False
            if op == ">" and not (val > float(threshold)):
                return False
            if op == "<=" and not (val <= float(threshold)):
                return False
            if op == ">=" and not (val >= float(threshold)):
                return False
            if op == "==" and not (abs(val - float(threshold)) < 1e-9):
                return False
        return True

    @staticmethod
    def _resolve_metric(ctx: ModuleContext, idx: int, metric: str) -> float | None:
        """Resolve a metric value: real arrays first, then virtual spatial metrics."""
        # Real metric from arrays
        if metric in ctx.arrays:
            arr = ctx.arrays[metric]
            if idx < len(arr):
                return float(arr[idx])
        # Virtual spatial metrics
        sp = ctx.spatial
        n = len(sp.positions)
        if idx >= n:
            return None
        if metric in ("distance_to_enemy", "distance_to_ally"):
            enemy_ids = ctx.metadata.get("fsm.enemy_ids", [])
            ally_ids = ctx.metadata.get("fsm.ally_ids", [])
            targets = enemy_ids if "enemy" in metric else ally_ids
            if not targets:
                return None
            min_dist = float("inf")
            for tidx in targets:
                if not isinstance(tidx, int) or tidx >= n or tidx == idx:
                    continue
                    continue
                d = float(np.linalg.norm(sp.positions[idx] - sp.positions[tidx]))
                if d < min_dist:
                    min_dist = d
            return min_dist if min_dist != float("inf") else None
        if metric == "distance_to_nearest_entity":
            min_dist = float("inf")
            for j in range(n):
                if j == idx:
                    continue
                d = float(np.linalg.norm(sp.positions[idx] - sp.positions[j]))
                if d < min_dist:
                    min_dist = d
            return min_dist if min_dist != float("inf") else None
        return None

"""Finite State Machine module — discrete state transitions for NPC autonomy.

Reduces LLM call overhead by letting agents switch between predefined states
(patrol → alert → attack) based on entity metric thresholds.

States marked as command_states are handed off to LLM for final decision.
All other states produce deterministic actions via action_map, bypassing LLM.
"""
from __future__ import annotations

from typing import Any
import hashlib

import numpy as np

from .base import AlgorithmModule, ModuleContext


def _normalize_counters(raw: Any, n: int) -> list[int]:
    """把（可能来自上一轮的）计数器数据归一化为长度 n 的 int 列表。"""
    if isinstance(raw, dict):
        raw = [raw.get(i, 0) for i in range(n)]
    if isinstance(raw, (list, tuple)) and len(raw) == n:
        return [int(x) for x in raw]
    return [0] * n


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
        self._observe_llm_guard: list[dict[str, Any]] = []
        self._observe_stale_rounds: int = 0

    def configure(self, params: dict[str, Any]) -> None:
        self._rules = list(params.get("transition_rules", []))
        self._default_state = str(params.get("default_state", "idle"))
        self._action_map = dict(params.get("action_map", {}))
        command = params.get("command_states", ["combat"])
        self._command_states = set(command if isinstance(command, list) else [command])
        self._observe_llm_guard = list(params.get("observe_llm_guard", []))
        try:
            self._observe_stale_rounds = int(params.get("observe_stale_rounds", 0))
        except (TypeError, ValueError):
            self._observe_stale_rounds = 0

    @property
    def name(self) -> str:
        return "finite_state_machine"

    @property
    def description(self) -> str:
        return "有限状态机（离散状态转移+动作映射）——降低 NPC 行为决策的 LLM 调用开销"

    def execute(self, ctx: ModuleContext) -> ModuleContext:
        n = len(next(iter(ctx.arrays.values()))) if ctx.arrays else 0
        if n == 0:
            return ctx

        # 从上一轮持久化的状态 blob 恢复（跨轮 streak / observe 计数）
        blob = ctx.metadata.get("fsm.state_blob")
        prev_streaks: dict[str, list[int]] = {}
        observe_streak: list[int] = [0] * n
        if isinstance(blob, dict):
            prev_states = blob.get("agent_states", [self._default_state] * n)
            prev_streaks = blob.get("streaks") or {}
            observe_streak = _normalize_counters(blob.get("observe_streak"), n)
        else:
            prev_states = ctx.metadata.get("fsm.agent_states", [self._default_state] * n)

        if isinstance(prev_states, dict):
            prev_states = [prev_states.get(i, self._default_state) for i in range(n)]
        if len(prev_states) != n:
            prev_states = [self._default_state] * n

        new_states = list(prev_states)

        # Maintain per-entity streak counters for historical conditions
        streak_counters: dict[str, list[int]] = {}
        streak_keys: list[str] = []

        for rule in self._rules:
            from_state = rule.get("from", "")
            to_state = rule.get("to", "")
            condition = rule.get("condition", {})
            if not from_state or not to_state:
                continue
            streak_req = condition.get("streak", 1) if isinstance(condition, dict) else 1
            # Include condition fingerprint in streak key to avoid counter collision
            # when multiple rules share the same from→to transition but different conditions
            cond_fprint = hashlib.md5(
                str(sorted(condition.items())).encode()
            ).hexdigest()[:6] if isinstance(condition, dict) else "noop"
            rule_fprint = f"{from_state}->{to_state}.{cond_fprint}"
            if rule_fprint not in streak_counters:
                streak_counters[rule_fprint] = _normalize_counters(prev_streaks.get(rule_fprint), n)
                streak_keys.append(rule_fprint)
            counters = streak_counters[rule_fprint]
            for i in range(n):
                if new_states[i] != from_state:
                    counters[i] = 0
                    continue
                if self._match_condition(ctx, i, condition):
                    counters[i] += 1
                    if counters[i] >= streak_req:
                        new_states[i] = to_state
                        counters[i] = 0
                else:
                    counters[i] = 0

        # Persist streak counters in metadata
        for rf in streak_keys:
            ctx.metadata[f"fsm.streak.{rf}"] = streak_counters[rf]

        # 跨轮 observe 计数：连续停留在 observe 则递增，否则清零
        for i in range(n):
            if new_states[i] == "observe":
                observe_streak[i] += 1
            else:
                observe_streak[i] = 0

        ctx.metadata["fsm.agent_states"] = new_states
        ctx.metadata["fsm.command_states"] = list(self._command_states)

        # Build deterministic actions for non-command agents
        agent_actions: list[dict[str, Any] | None] = []
        for i in range(n):
            state = new_states[i]
            if state in self._command_states:
                agent_actions.append(None)  # handed to LLM
            elif state == "observe" and self._observe_handoff(ctx, i, observe_streak[i]):
                # 指标恶化，或连续 observe 超过阈值时，交还 LLM 重新决策
                agent_actions.append(None)
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

        # 持久化状态 blob（供下一轮 build_context 回注）
        ctx.metadata["fsm.state_blob"] = {
            "agent_states": new_states,
            "streaks": {rf: streak_counters[rf] for rf in streak_keys},
            "observe_streak": observe_streak,
        }

        return ctx

    def _observe_handoff(self, ctx: ModuleContext, idx: int, observe_rounds: int) -> bool:
        """observe 态实体是否应交还 LLM：指标恶化守卫，或连续 observe 超过阈值。"""
        return self._guard_triggered(ctx, idx) or (
            self._observe_stale_rounds > 0 and observe_rounds >= self._observe_stale_rounds
        )

    def _guard_triggered(self, ctx: ModuleContext, idx: int) -> bool:
        """observe 态实体命中任一 observe_llm_guard 条件即交还 LLM。"""
        for g in self._observe_llm_guard:
            if not isinstance(g, dict):
                continue
            metric = g.get("metric")
            op = g.get("op")
            threshold = g.get("threshold")
            if metric is None or op is None or threshold is None:
                continue
            if self._match_condition(ctx, idx, {metric: [op, threshold]}):
                return True
        return False

    @staticmethod
    def _match_condition(ctx: ModuleContext, idx: int, condition: dict) -> bool:
        """Check if entity idx satisfies all condition thresholds.
        
        Supports virtual spatial metrics:
          - distance_to_enemy / distance_to_ally: computed from ctx.spatial + metadata.
        """
        # Strip streak before iterating — it's an int, not a (op, threshold) tuple
        cond_pairs = [(m, op_th) for m, op_th in condition.items() if m != "streak"]
        for metric, (op, threshold) in cond_pairs:
            # opponent.<metric>：读取最强对手的指标值
            if metric.startswith("opponent."):
                real_metric = metric[len("opponent."):]
                val = FiniteStateMachineModule._resolve_opponent_metric(ctx, idx, real_metric)
            else:
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
            enemy_ids = ctx.metadata.get("fsm.enemy_ids")
            ally_ids = ctx.metadata.get("fsm.ally_ids")
            # Auto-divide when neither is explicitly configured
            if enemy_ids is None and ally_ids is None:
                polar = ctx.arrays.get("polarization")
                if polar is not None and len(polar) == n:
                    own_pol = float(polar[idx])
                    others = [j for j in range(n) if j != idx]
                    enemy_ids = [j for j in others if polar[j] * own_pol < 0 or abs(float(polar[j]) - own_pol) > 3.0]
                    ally_ids = [j for j in others if abs(float(polar[j]) - own_pol) <= 3.0]
                    # polarization 均匀导致无法划分出对手 → 兜底把其他实体都视为对手
                    if not enemy_ids:
                        enemy_ids = [j for j in range(n) if j != idx]
                        ally_ids = []
                else:
                    # Fallback: treat all other entities as enemies
                    enemy_ids = [j for j in range(n) if j != idx]
                    ally_ids = []
            # Use explicit values when provided
            enemy_ids = enemy_ids if enemy_ids is not None else []
            ally_ids = ally_ids if ally_ids is not None else []
            targets = enemy_ids if "enemy" in metric else ally_ids
            if not targets:
                return None
            valid = [t for t in targets if isinstance(t, int) and 0 <= t < n]
            if not valid:
                return None
            return float(min(np.linalg.norm(sp.positions[idx] - sp.positions[t]) for t in valid))

    @staticmethod
    def _resolve_opponent_metric(ctx: ModuleContext, idx: int, metric: str) -> float | None:
        """读取最强对手的指标值。从 enemy_ids 中找到距离最近的敌人，读取其 metric。"""
        sp = ctx.spatial
        n = len(sp.positions)
        if idx >= n or metric not in ctx.arrays:
            return None
        enemy_ids = ctx.metadata.get("fsm.enemy_ids")
        if enemy_ids is None:
            polar = ctx.arrays.get("polarization")
            if polar is not None and len(polar) == n:
                own_pol = float(polar[idx])
                enemy_ids = [j for j in range(n) if j != idx and
                            (polar[j] * own_pol < 0 or abs(float(polar[j]) - own_pol) > 3.0)]
                if not enemy_ids:
                    enemy_ids = [j for j in range(n) if j != idx]
            else:
                enemy_ids = [j for j in range(n) if j != idx]
        if not enemy_ids:
            return None
        # 找距离最近的对手
        min_dist = float("inf")
        nearest = None
        for tidx in enemy_ids:
            if not isinstance(tidx, int) or tidx >= n:
                continue
            d = float(np.linalg.norm(sp.positions[idx] - sp.positions[tidx]))
            if d < min_dist:
                min_dist = d
                nearest = tidx
        if nearest is not None and nearest < len(ctx.arrays.get(metric, [])):
            return float(ctx.arrays[metric][nearest])
        return None

"""Token statistics accumulator with contextvars.

Captures per-session, per-phase, per-round token usage from LLM API responses
without requiring any caller signature changes.
"""
from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field


@dataclass
class TokenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    duration_ms: int = 0


_current_session: contextvars.ContextVar[str] = contextvars.ContextVar(
    "token_session", default=""
)
_current_phase: contextvars.ContextVar[str] = contextvars.ContextVar(
    "token_phase", default="unknown"
)
_current_round: contextvars.ContextVar[int] = contextvars.ContextVar(
    "token_round", default=0
)


class TokenAccumulator:
    """Module-level singleton that accumulates token stats per session."""

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, dict[str, TokenStats | list]]] = {}

    def record(self, session_id: str, phase: str, round_num: int,
               stats: TokenStats) -> None:
        if session_id not in self._stats:
            self._stats[session_id] = {}
        if phase not in self._stats[session_id]:
            self._stats[session_id][phase] = {"total": TokenStats(), "rounds": {}}
        entry = self._stats[session_id][phase]
        tot: TokenStats = entry["total"]
        tot.prompt_tokens += stats.prompt_tokens
        tot.completion_tokens += stats.completion_tokens
        tot.total_tokens += stats.total_tokens
        tot.duration_ms += stats.duration_ms
        if phase == "simulation" and round_num >= 0:
            rounds: dict = entry["rounds"]  # type: ignore[assignment]
            if round_num not in rounds:
                rounds[round_num] = TokenStats()
            rs: TokenStats = rounds[round_num]
            rs.prompt_tokens += stats.prompt_tokens
            rs.completion_tokens += stats.completion_tokens
            rs.total_tokens += stats.total_tokens
            rs.duration_ms += stats.duration_ms

    def get_session_stats(self, session_id: str) -> dict | None:
        phases = self._stats.get(session_id)
        if not phases:
            return None
        total_p = sum(s.prompt_tokens for p in phases.values() for s in [p["total"]])  # type: ignore[union-attr]
        total_c = sum(s.completion_tokens for p in phases.values() for s in [p["total"]])  # type: ignore[union-attr]
        result: dict = {
            "total_prompt_tokens": total_p,
            "total_completion_tokens": total_c,
            "total_tokens": total_p + total_c,
            "phases": {},
            "rounds": {},
        }
        for ph_name, ph_data in phases.items():
            pt: TokenStats = ph_data["total"]  # type: ignore[assignment]
            result["phases"][ph_name] = {
                "prompt": pt.prompt_tokens,
                "completion": pt.completion_tokens,
                "total": pt.total_tokens,
            }
            if ph_name == "simulation":
                rounds_dict: dict = ph_data["rounds"]  # type: ignore[assignment]
                for rnd, rs in sorted(rounds_dict.items()):
                    rts: TokenStats = rs  # type: ignore[assignment]
                    result["rounds"][str(rnd)] = {
                        "prompt": rts.prompt_tokens,
                        "completion": rts.completion_tokens,
                        "total": rts.total_tokens,
                    }
        return result

    def remove_session(self, session_id: str) -> None:
        self._stats.pop(session_id, None)


accumulator = TokenAccumulator()

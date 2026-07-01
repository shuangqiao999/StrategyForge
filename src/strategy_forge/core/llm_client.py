"""Thin LLM adapter for the deduction engine — OpenAI-compatible API only.

Replaces the ~2800-line openakita.llm.client.LLMClient with a ~70-line wrapper.
Only implements what the deduction engine actually uses: chat().
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from .config import config
from .token_counter import TokenStats

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Minimal Message dataclass (replaces openakita.llm.types.Message)."""
    role: str
    content: str


@dataclass
class TextBlock:
    """Minimal TextBlock (replaces openakita.llm.types.TextBlock)."""
    text: str


class DeductionLLMResponse:
    """LLM response wrapper, compatible with the three parsing paths in _utils.extract_text()."""

    def __init__(self, content: str, token_stats: TokenStats | None = None):
        self.text = content
        self.content = content  # string path (simulator.py custom extract path)
        self.choices: list = []  # dict path
        self.token_stats = token_stats or TokenStats()

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class DeductionLLMClient:
    """Lightweight LLM client for StrategyForge deduction engine."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        from strategy_forge.core.providers import registry

        resolved = registry.resolve_for_llm_client()
        self.api_base = (api_base or resolved.get("api_base", "")).rstrip("/")
        self.api_key = api_key or resolved.get("api_key", "")
        self.model = model or resolved.get("model", "")
        self._http: httpx.AsyncClient | None = None

    async def _ensure_client(self):
        if self._http is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                headers=headers,
            )

    async def chat(
        self,
        messages: list[dict] | list[Message],
        system: str = "",
        tools=None,
        max_tokens: int = 0,
        temperature: float = 1.0,
        **kwargs,
    ) -> DeductionLLMResponse:
        await self._ensure_client()

        full_messages: list[dict] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        for m in messages:
            if isinstance(m, Message):
                full_messages.append({"role": m.role, "content": m.content})
            elif isinstance(m, dict):
                full_messages.append(m)
            else:
                full_messages.append({"role": "user", "content": str(m)})

        payload: dict = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        t0 = time.monotonic()
        try:
            resp = await self._http.post(
                f"{self.api_base}/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            stats = TokenStats(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                model=self.model,
                duration_ms=elapsed_ms,
            )
            # Auto-accumulate if context is set
            from .token_counter import _current_session, _current_phase, _current_round, accumulator
            sid = _current_session.get()
            if sid:
                accumulator.record(sid, _current_phase.get(), _current_round.get(), stats)
            else:
                logger.debug("[Token] session context not set, skipping accumulation (phase=%s tokens=%d)",
                             _current_phase.get(), stats.total_tokens)
            return DeductionLLMResponse(content, token_stats=stats)
        except Exception as e:
            logger.error("[LLM] Chat request failed: %s", e)
            raise

    async def close(self):
        if self._http:
            await self._http.aclose()
            self._http = None

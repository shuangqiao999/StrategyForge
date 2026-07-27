"""Thin LLM adapter for the deduction engine — OpenAI-compatible API only.

Replaces the ~2800-line openakita.llm.client.LLMClient with a ~70-line wrapper.
Only implements what the deduction engine actually uses: chat().
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass

import httpx

from .config import config
from .providers import registry as _reg
from .token_counter import TokenStats

logger = logging.getLogger(__name__)

# 全局 LLM 并发信号量——所有 client.chat() 统一经过此锁，
# 确保全推演（图构建/分类器/智能体/模拟/优化器）的总并发 ≤ 界面配置值
_global_sem: asyncio.Semaphore | None = None


def _ensure_global_sem() -> asyncio.Semaphore:
    global _global_sem
    if _global_sem is None:
        mc = max(1, _reg.max_concurrent)
        _global_sem = asyncio.Semaphore(mc)
    return _global_sem


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


class LLMConnectionError(Exception):
    """LLM 连接/传输彻底失败（重试耗尽后抛出，含可读上下文）。"""

    def __init__(self, msg: str, endpoint: str = "",
                 retries: int = 0, cause: str = ""):
        super().__init__(msg)
        self.endpoint = endpoint
        self.retries = retries
        self.cause = cause


# ── 预置 JSON Schema 定义，供 DeductionLLMClient.chat_json() 使用 ──
_JSON_SCHEMAS: dict[str, dict] = {
    "l1_entities": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "type"]
                }
            }
        },
        "required": ["entities"]
    },
    "l2_results": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tier": {"type": "integer", "minimum": 1, "maximum": 3},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "tier"]
                }
            }
        },
        "required": ["results"]
    },
    "l3_decisions": {
        "type": "object",
        "properties": {
            "downgrades": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name"]
                }
            },
            "merges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keep": {"type": "string"},
                        "discard": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["keep"]
                }
            },
            "notes": {"type": "string"},
        },
        "required": ["downgrades", "merges"]
    },
    # ── 扩展 Schema ──
    "persona": {
        "type": "object",
        "properties": {
            "persona": {"type": "string"},
            "background": {"type": "string"},
            "goals": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["persona", "background", "goals"]
    },
    "graph_extract": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["entity", "type"]
                }
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "relation": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["source", "target", "relation"]
                }
            },
        },
        "required": ["entities", "relations"]
    },
    "intel_sort": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "parent": {"type": "string"},
                        "sub_entities": {"type": "array", "items": {"type": "string"}},
                        "role": {"type": "string"},
                    },
                    "required": ["name"]
                }
            }
        },
        "required": ["entities"]
    },
    "ontology_gen": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "properties": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name"]
                }
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "from_type": {"type": "string"},
                        "to_type": {"type": "string"},
                    },
                    "required": ["name"]
                }
            },
        },
        "required": ["entities", "relations"]
    },
    "narrative_sort": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "include": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "include"]
                }
            }
        },
        "required": ["entities"]
    },
    "optimizer_eval": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "win_score": {"type": "number"},
            "cost": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["success", "win_score", "cost"]
    },
    "convergence": {
        "type": "object",
        "properties": {
            "resolved": {"type": "boolean"},
            "verdict": {"type": "string"},
        },
        "required": ["resolved"]
    },
    "domain_detect": {
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["domain", "confidence"]
    },
    "seed_extract": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "metrics": {"type": "object"},
                    },
                    "required": ["name", "metrics"]
                }
            }
        },
        "required": ["entities"]
    },
    "strategy_single": {
        "type": "object",
        "properties": {
            "action_type": {"type": "string"},
            "target": {"type": "string"},
            "intensity": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["action_type"]
    },
    "strategy_candidates": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "selected": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "content": {"type": "string"},
                        "rationale": {"type": "string"},
                        "risk_level": {"type": "string"},
                    },
                    "required": ["action"]
                }
            }
        },
    },
    "report_quantified": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_events": {"type": "array", "items": {"type": "string"}},
            "risk_alerts": {"type": "array", "items": {"type": "string"}},
            "conclusion": {"type": "string"},
        },
        "required": ["summary"]
    },
    "env_impact": {
        "type": "object",
        "properties": {
            "舆论风向": {"type": "number"},
            "抗议规模": {"type": "number"},
            "媒体关注": {"type": "number"},
            "国际压力": {"type": "number"},
            "社会分裂": {"type": "number"},
        },
    },
    "action_fallback": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "target": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["action"]
    },
}


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
        # 超时种子值在构造时确定（不依赖 _ensure_client），便于测试与环境注入
        # 优先级: FORGE_LLM_*_TIMEOUT env > UI保存值 > config默认值
        _env_conn = os.getenv("FORGE_LLM_CONNECT_TIMEOUT")
        self._conn_timeout = float(_env_conn) if _env_conn else max(10.0, _reg.connect_timeout)
        _env_gen = os.getenv("FORGE_LLM_GENERATION_TIMEOUT")
        self._gen_timeout = float(_env_gen) if _env_gen else _reg.generation_timeout

    async def _ensure_client(self):
        if self._http is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # 连接池上限：0 表示按并发自动派生，保证 >= FORGE_MAX_CONCURRENT 且留余量
            mc = max(1, _reg.max_concurrent)
            max_conn = config.deduction_http_max_connections or max(100, mc * 2)
            max_keep = config.deduction_http_max_keepalive or max(20, mc)
            # [A] 双层超时：连接(短)/生成(长) 在 __init__ 已算好；gen=0 表示无上限
            read_t = None if self._gen_timeout <= 0 else self._gen_timeout
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=self._conn_timeout,
                                      read=read_t,
                                      write=read_t,
                                      pool=self._conn_timeout),
                headers=headers,
                limits=httpx.Limits(max_connections=max_conn,
                                    max_keepalive_connections=max_keep),
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
            async with _ensure_global_sem():
                resp = await self._request_with_retry(payload)
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
                logger.warning("[Token] session context not set, skipping accumulation (phase=%s tokens=%d)",
                             _current_phase.get(), stats.total_tokens)
            return DeductionLLMResponse(content, token_stats=stats)
        except LLMConnectionError:
            raise  # 连接故障直接传播，不套 except Exception 吞掉
        except Exception as e:
            logger.error("[LLM] Chat request failed: %s", e)
            raise

    # 可重试的状态码：429 限流 + 5xx 服务端错误
    _RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

    async def _request_with_retry(self, payload: dict) -> httpx.Response:
        """发送 chat 请求，对 429/5xx 与传输类错误做指数退避重试（面向云端高并发）。

        非可重试的 4xx（400/401/403/404 等）立即抛出；重试用 asyncio.sleep（不阻塞其他并发请求）。
        429 若带 Retry-After 头则优先遵循该延迟。
        """
        url = f"{self.api_base}/chat/completions"
        max_retries = max(0, int(config.deduction_llm_max_retries))
        base = max(0.0, float(config.deduction_llm_retry_base))
        cap = max(base, float(config.deduction_llm_retry_cap))
        attempt = 0
        try:
            while True:
                try:
                    resp = await self._http.post(url, json=payload)
                    if resp.status_code in self._RETRYABLE_STATUS and attempt < max_retries:
                        delay = self._retry_delay(attempt, base, cap, resp)
                        logger.warning("[LLM] %s，第 %d/%d 次重试，%.1fs 后…",
                                       resp.status_code, attempt + 1, max_retries, delay)
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    resp.raise_for_status()
                    return resp
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    is_conn = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout))
                    is_gen = isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout))
                    if attempt >= max_retries:
                        if is_conn:
                            raise LLMConnectionError(
                                f"LLM 无法连接：{url}（{type(e).__name__}: {e}，"
                                f"已重试 {max_retries} 次仍无法建立连接。请检查服务端是否正在运行、URL 是否正确）",
                                endpoint=url, retries=max_retries, cause=str(e)) from e
                        if is_gen:
                            raise LLMConnectionError(
                                f"LLM 响应超时：{url}（{type(e).__name__}: {e}，"
                                f"已等待 {self._gen_timeout:.0f}s 无数据，已重试 {max_retries} 次。"
                                f"如需设上限可配 FORGE_LLM_GENERATION_TIMEOUT）",
                                endpoint=url, retries=max_retries, cause=str(e)) from e
                        raise LLMConnectionError(
                            f"LLM 请求失败：{url}（{type(e).__name__}: {e}，"
                            f"已重试 {max_retries} 次仍失败）",
                            endpoint=url, retries=max_retries, cause=str(e)) from e
                    delay = self._retry_delay(attempt, base, cap, None)
                    att_name = type(e).__name__
                    logger.warning("[LLM] %s(%s)，第 %d/%d 次重试，%.1fs 后…",
                                   "网络错误" if is_conn else ("超时" if is_gen else "传输错误"),
                                   att_name, attempt + 1, max_retries, delay)
                    await asyncio.sleep(delay)
                    attempt += 1
        finally:
            pass

    @staticmethod
    def _retry_delay(attempt: int, base: float, cap: float,
                     resp: httpx.Response | None) -> float:
        # 429 优先遵循 Retry-After（秒）
        if resp is not None and resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    return min(float(ra), cap)
                except (TypeError, ValueError):
                    pass
        # 指数退避 + 抖动，封顶 cap
        return min(base * (2 ** attempt) + random.uniform(0, base), cap)

    async def chat_json(
        self,
        messages: list[dict] | list[Message],
        system: str = "",
        schema_name: str = "l1_entities",
        max_tokens: int = 0,
        temperature: float = 0,
        **kwargs,
    ) -> DeductionLLMResponse:
        """使用 JSON Schema 约束的 chat 调用，根治 JSON 格式错误。
        
        schema_name 可选:
          l1_entities:  Layer1 归一化实体输出
          l2_results:   Layer2 分级判定结果
          l3_decisions: Layer3 交叉裁决输出
        """
        schema = _JSON_SCHEMAS.get(schema_name)
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            }
        return await self.chat(messages, system=system,
                               max_tokens=max_tokens, temperature=temperature, **kwargs)

    async def close(self):
        if self._http:
            await self._http.aclose()
            self._http = None


_client_instance: DeductionLLMClient | None = None

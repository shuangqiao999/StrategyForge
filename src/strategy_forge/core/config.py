"""Minimal configuration for StrategyForge — non-endpoint settings only.

All LLM/embedding endpoint resolution is delegated to core.providers.registry.
Hardcoded addresses and model names are FORBIDDEN here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _get_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path.cwd()


def _get_data_dir() -> Path:
    env_data = os.getenv("FORGE_DATA_DIR", "")
    if env_data:
        p = Path(env_data)
        if p.is_absolute():
            return p
        _log_env_once("FORGE_DATA_DIR", env_data, "not-absolute")
    else:
        _log_env_once("FORGE_DATA_DIR", "", "unset")
    return _get_root() / "data"

_data_logged: set = set()

def _log_env_once(key: str, val: str, reason: str):
    if key in _data_logged:
        return
    _data_logged.add(key)
    import logging
    logging.getLogger("strategy_forge").info("ENV %s=%s (%s → fallback to %s)",
        key, val[:80] if val else "(empty)", reason, str(_get_root() / "data" if reason != "unset" else "root/data"))


class DeductionConfig:
    """Non-endpoint configuration (rounds, agents, concurrency, data paths)."""

    def __init__(self):
        self.project_root = _get_root()
        self.deduction_data_dir = _get_data_dir()
        self.deduction_max_agents = int(os.getenv("FORGE_MAX_AGENTS", "10000"))
        self.deduction_default_rounds = int(os.getenv("FORGE_DEFAULT_ROUNDS", "10"))
        self.deduction_candidate_count = int(os.getenv("FORGE_CANDIDATE_COUNT", "3"))
        self.deduction_llm_temperature = float(os.getenv("FORGE_LLM_TEMPERATURE", "0.3"))
        self.deduction_max_concurrent = int(os.getenv("FORGE_MAX_CONCURRENT", "2"))
        self.deduction_retrieve_top_k = int(os.getenv("FORGE_RETRIEVE_TOP_K", "5"))
        self.deduction_similarity_threshold = float(os.getenv("FORGE_SIMILARITY_THRESHOLD", "0.4"))
        # 动态事件表混合检索(向量+BM25)开关，默认开启；开启时靠 RRF 排序而非余弦阈值。
        self.deduction_event_hybrid = os.getenv("FORGE_EVENT_HYBRID", "1") == "1"
        # 云端 API 并发容错：429/5xx/传输错误的指数退避重试（面向 vLLM/云端高并发）
        self.deduction_llm_max_retries = int(os.getenv("FORGE_LLM_MAX_RETRIES", "3"))
        self.deduction_llm_retry_base = float(os.getenv("FORGE_LLM_RETRY_BASE", "1.0"))
        self.deduction_llm_retry_cap = float(os.getenv("FORGE_LLM_RETRY_CAP", "30.0"))
        # httpx 连接池上限（0=按并发自动派生，保证 >= FORGE_MAX_CONCURRENT）
        self.deduction_http_max_connections = int(os.getenv("FORGE_HTTP_MAX_CONNECTIONS", "0"))
        self.deduction_http_max_keepalive = int(os.getenv("FORGE_HTTP_MAX_KEEPALIVE", "0"))
        # 模拟阶段 token 优化（Plan B）：控制每 agent 决策 prompt 的上下文规模。
        # others_ctx 只渲染 Top-K 最相关他方(其余合并为全局摘要)，砍掉 O(N^2) 与逐轮膨胀。
        self.deduction_sim_others_topk = int(os.getenv("FORGE_SIM_OTHERS_TOPK", "10"))
        # 模拟召回(原著/事件)片段上限 + 单块字符预算。
        self.deduction_sim_recall_topk = int(os.getenv("FORGE_SIM_RECALL_TOPK", "4"))
        self.deduction_sim_recall_chars = int(os.getenv("FORGE_SIM_RECALL_CHARS", "1200"))
        # 注入决策 prompt 的近期事件条数。
        self.deduction_sim_recent_events = int(os.getenv("FORGE_SIM_RECENT_EVENTS", "4"))

    def __getattr__(self, name: str):
        return None


config = DeductionConfig()

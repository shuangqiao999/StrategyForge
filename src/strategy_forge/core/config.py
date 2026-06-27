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
    return _get_root() / "data"


class DeductionConfig:
    """Non-endpoint configuration (rounds, agents, concurrency, data paths)."""

    def __init__(self):
        self.project_root = _get_root()
        self.deduction_data_dir = _get_data_dir()
        self.deduction_max_agents = int(os.getenv("FORGE_MAX_AGENTS", "200"))
        self.deduction_default_rounds = int(os.getenv("FORGE_DEFAULT_ROUNDS", "10"))
        self.deduction_candidate_count = int(os.getenv("FORGE_CANDIDATE_COUNT", "3"))
        self.deduction_llm_temperature = float(os.getenv("FORGE_LLM_TEMPERATURE", "0.3"))
        self.deduction_max_concurrent = int(os.getenv("FORGE_MAX_CONCURRENT", "8"))

    def __getattr__(self, name: str):
        return None


config = DeductionConfig()

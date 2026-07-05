"""Seed Data Extractor — LLM reads source material, extracts per-entity initial metrics.

Design: zero-config, single-LLM-call, graceful fallback to rule pack defaults.
Non-quantified (narrative) mode never calls this module.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SEED_PROMPT = """你是一个数据提取器。从以下战略文本中识别所有具有战略决策能力的实体（国家、企业、组织、关键人物），为每个实体评估以下指标值（0-100 的整数）。

## 可用指标
{metrics_list}

## 提取规则
- 只提取文本中明确提到的实体，不要凭空创建
- 如果文本给出了具体数值，直接使用；如果只有定性描述，映射到 0-100 的合理值
- 未提及的指标不要赋值（省略该字段）
- 不要添加上述列表中不存在的指标
- ★ 财务类指标特别规则：如果文本提到"净亏损"、"运营现金流为负"、"现金流承压"、"现金流偏紧"，该实体的 cash_flow 应低于 40（亏损企业再大也不可能现金流满分）
- ★ 如果文本描述"营收增长但持续亏损"的模式，market_share 可高（<80），但 cash_flow 应低（<40），supply_chain 也不能满分（依赖单一工厂）

## 输出 JSON（仅 JSON，无 markdown）
{{"entities": [
  {{"name": "实体名", "metrics": {{"strength": 85, "cash_flow": 60}}}}
]}}

## 文本
{source}"""


async def extract_seed_metrics(
    source: str,
    metrics: list[str],
    client: Any,
    max_chars: int = 20000,
) -> dict[str, dict[str, float]]:
    """Extract per-entity initial metrics from seed material via LLM.

    Args:
        source: Raw seed text (truncated to max_chars).
        metrics: List of metric names from the selected rule pack.
        client: DeductionLLMClient instance.
        max_chars: Max chars of source to send to LLM.

    Returns:
        {entity_name: {metric: value}}, empty dict on any failure.
    """
    if not source or not metrics:
        return {}

    prompt = _SEED_PROMPT.format(
        metrics_list=", ".join(metrics),
        source=source[:max_chars],
    )

    from strategy_forge.core.llm_client import Message
    try:
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是战略数据提取器，只输出 JSON。",
            temperature=0.1,
        )
    except Exception as e:
        logger.warning("[SeedExtractor] LLM call failed: %s", e)
        return {}

    raw = _extract_text(resp)
    data = _parse_json(raw)
    if not isinstance(data, dict):
        logger.warning("[SeedExtractor] Failed to parse LLM output as JSON")
        return {}

    result: dict[str, dict[str, float]] = {}
    metrics_set = set(metrics)
    for entity in data.get("entities", []):
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name", "")).strip()
        entity_metrics = entity.get("metrics", {})
        if not name or not isinstance(entity_metrics, dict):
            continue
        filtered = {}
        for m, v in entity_metrics.items():
            if m not in metrics_set:
                continue
            try:
                val = float(v)
                if val < 0 or val > 100:
                    val = max(0.0, min(100.0, val))
                filtered[m] = val
            except (TypeError, ValueError):
                continue
        if filtered:
            result[name] = filtered

    logger.info("[SeedExtractor] Extracted %d entities with custom initial metrics", len(result))
    return result


def _extract_text(resp: Any) -> str:
    if hasattr(resp, "text"):
        return resp.text
    if hasattr(resp, "content"):
        c = resp.content
        if isinstance(c, list):
            from strategy_forge.core.llm_client import TextBlock
            return "".join(b.text for b in c if isinstance(b, TextBlock))
        return str(c)
    if isinstance(resp, dict):
        choices = resp.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
        return str(resp)
    return str(resp)


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", raw)
    cleaned = re.sub(r"\n?```", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    for pat in (r'\{[\s\S]*?\}', r'\[[\s\S]*?\]'):
        m = re.search(pat, cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                continue
    return None

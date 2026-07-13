"""Narrative Sorter — LLM-based story character classification for narrative mode.

A lightweight alternative to IntelSorter, designed specifically for creative writing
and storytelling. Unlike IntelSorter which identifies "strategic decision-makers" for
quantified simulation, the NarrativeSorter identifies "story characters" — entities
that can drive plot, make decisions, and have memorable presence.

One LLM call per session. Output structure compatible with agent_factory's existing
intel_list filter logic.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_NARRATIVE_SORT_PROMPT = """你是故事编辑。阅读以下种子材料，列出所有实体名称，并判断每个实体在故事中应该作为"角色"还是"背景"。

## 种子材料
{source}

## 所有已提取的实体名称
{entity_names}

## 任务
1. 合并同一角色的不同称呼：简称→全名（如"莫雷诺"→"埃琳娜·莫雷诺"的别名）、英文名→中文名、职务头衔→对应人物
2. 纯文字头衔（如"总统""秘书长"）如果有具体人物在上下文中，则作为该人物的别名，不单独创建角色
3. 标记纯背景元素为背景，不生成角色：纯地理名称、天气描述、抽象概念（如"弓弦""平衡""压力""秩序"）、法律文件（如"协议""批文""纪要"）、基础设施（如"码头""港口""基地"）
4. 标记二元关系词（如"A与B""X和Y"）为背景

## 输出 JSON（仅 JSON，不要 markdown）
{{"entities": [
  {{"name": "规范名", "aliases": ["简称", "英文名", "头衔"], "include": true/false, "reason": "简短理由（≤15字）"}}
]}}

- name: 最规范、最完整的名称（优先中文全名）
- aliases: 该实体的其他所有称呼，同一角色的所有别名必须在此数组中
- include: true=故事角色，需要生成智能体；false=背景元素，不生成智能体
- reason: 简短说明分类理由
- 所有已提取的实体名都必须在输出中出现（作为 name 或 aliases 的一部分）

只返回 JSON。"""


async def sort_narrative_entities(
    source: str,
    entity_names: list[str],
    client: Any,
    max_source_chars: int = 20000,
) -> list[dict[str, Any]]:
    """Narrative mode entity classification via LLM.

    Returns a list compatible with agent_factory's intel_list filter logic:
    [{name, aliases, include_in_simulation, role}, ...]

    Returns empty list on failure (fallback to entity_type filtering).
    """
    if not source or not entity_names:
        return []

    prompt = _NARRATIVE_SORT_PROMPT.format(
        source=source[:max_source_chars],
        entity_names=", ".join(entity_names),
    )

    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import Message
    from strategy_forge.engine._utils import extract_text
    from strategy_forge.engine.intel_sorter import _extract_text, _parse_json

    try:
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是故事编辑，输出结构化 JSON。只输出 JSON。",
            temperature=0.3,
            max_tokens=config.deduction_intel_max_tokens,
        )
    except Exception as e:
        logger.warning("[NarrativeSorter] LLM call failed: %s", e)
        return []

    raw = _extract_text(resp)
    data = _parse_json(raw)
    if isinstance(data, list):
        data = {"entities": data}
    if not isinstance(data, dict):
        return []

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []

    result: list[dict[str, Any]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        aliases = e.get("aliases", [])
        if isinstance(aliases, list):
            aliases = [str(a).strip() for a in aliases if str(a).strip()]
        else:
            aliases = []
        include = bool(e.get("include", True))
        reason = str(e.get("reason", ""))[:80]
        result.append({
            "name": name,
            "aliases": aliases,
            "include_in_simulation": include,
            "role": reason,
        })

    return result

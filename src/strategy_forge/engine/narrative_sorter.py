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

_NARRATIVE_SORT_PROMPT = """你是故事编辑。基于文本概览和实体统计数据，列出所有实体名称，并判断每个实体在故事中应该作为"角色"还是"背景"。

## 文本概览（文本开头 + 中段各取一截，供你感知文风与故事基调）
{text_overview}

## 实体统计（按重要性排序：频次 = 全文出现次数，覆盖粒度 = 散布在多少个章节/分块中）
{entity_stats}

## 所有已提取的实体名称（必须全部出现在输出中）
{entity_names}

## 任务
1. 合并同一角色的不同称呼：简称→全名（如"莫雷诺"→"埃琳娜·莫雷诺"的别名）、英文名→中文名、职务头衔→对应人物
2. 纯文字头衔（如"总统""秘书长"）如果有具体人物在上下文中，则作为该人物的别名，不单独创建角色
3. 标记纯背景元素为背景，不生成角色：纯地理名称、天气描述、抽象概念（如"弓弦""平衡""压力""秩序"）、法律文件（如"协议""批文""纪要"）、基础设施（如"码头""港口""基地"）
4. 标记二元关系词（如"A与B""X和Y"）为背景
5. 高频次 + 高覆盖粒度的实体通常为核心角色，请优先判定为 include=true

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
    entity_frequencies: dict[str, int] | None = None,
    entity_chunk_coverage: dict[str, int] | None = None,
    chunk_texts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Narrative mode entity classification via LLM.

    For ultra-long texts (> 20000 chars), builds a structured "entity statistics"
    overview from preprocessor frequency/chunk-coverage data, supplemented by
    text samples from multiple parts of the text — avoiding the "first-chapter-only"
    blindness of raw source[:20000] truncation.

    Returns a list compatible with agent_factory's intel_list filter logic:
    [{name, aliases, include_in_simulation, role}, ...]

    Returns empty list on failure (fallback to entity_type filtering).
    """
    if not source or not entity_names:
        return []

    freq = entity_frequencies or {}
    cov = entity_chunk_coverage or {}
    chunks = chunk_texts or []

    # ── Build entity statistics table ──
    if freq and len(source) > max_source_chars:
        # Composite importance score: frequency * (1 + log(chunk_coverage + 1))
        import math
        def _score(name: str) -> float:
            f = freq.get(name, 0)
            c = cov.get(name, 0)
            return f * (1.0 + math.log(c + 1.0))
        # Show top-120 entities by importance, ranked
        ranked = sorted(entity_names, key=_score, reverse=True)[:120]
        stat_lines: list[str] = []
        for i, name in enumerate(ranked, 1):
            f = freq.get(name, 0)
            c = cov.get(name, 0)
            if f > 0 or c > 0:
                stat_lines.append(f"{i}. {name} （频次={f}, 覆盖={c}块）")
            else:
                stat_lines.append(f"{i}. {name} （频次=?, 覆盖=?）")
        # Add remaining "unknown" entities at the end
        shown = set(ranked)
        remaining = [n for n in entity_names if n not in shown]
        for name in remaining:
            stat_lines.append(f"- {name} （频次=?, 覆盖=?）")
        stats_text = "\n".join(stat_lines)
    else:
        # Short text: just list names
        stats_text = "\n".join(f"- {n}" for n in entity_names)

    # ── Build text overview (multi-point samples for long texts) ──
    if len(source) > max_source_chars and chunks:
        # Take: first chunk, a chunk from ~1/4, ~1/2, ~3/4 way through
        n = len(chunks)
        indices = [0]
        if n >= 4:
            indices.extend([n // 4, n // 2, 3 * n // 4])
        if n - 1 not in indices:
            indices.append(n - 1)
        samples = []
        for idx in indices:
            if idx < n:
                chunk_text = chunks[idx][:1500]
                samples.append(f"[文本样本·第{idx+1}/{n}块]\n{chunk_text}")
        text_overview = "\n\n---\n\n".join(samples)
    elif len(source) > max_source_chars:
        # Fallback: take samples from different parts of the raw text
        L = len(source)
        samples = [
            source[:4000],
            source[L//4: L//4 + 4000],
            source[L//2: L//2 + 4000],
            source[3*L//4: 3*L//4 + 4000],
        ]
        text_overview = "\n\n---\n\n".join(f"[位置 {i}]\n{s}" for i, s in enumerate(samples))
    else:
        text_overview = source[:max_source_chars]

    prompt = _NARRATIVE_SORT_PROMPT.format(
        text_overview=text_overview,
        entity_stats=stats_text,
        entity_names=", ".join(entity_names),
    )

    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import Message
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

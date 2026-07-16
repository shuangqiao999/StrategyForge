"""Narrative Sorter — LLM-based story character classification for narrative mode.

For texts with many entities (>60), splits entity list into batches and makes
multiple LLM calls to avoid output truncation. Each batch covers ~60 entities.
Results are merged and returned as a single intel_list-compatible structure.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_NARRATIVE_SORT_PROMPT = """你是故事编辑。基于文本概览，判断以下实体列表中每个实体应作为"角色"还是"背景"。

## 文本概览（文本多处采样，供你感知文风与故事基调）
{text_overview}

## 本批实体统计（频次 = 全文出现次数，覆盖粒度 = 散布在多少分块中）
{entity_stats}

## 本批实体名称（必须全部出现在输出中，不得遗漏）
{entity_names}

## 任务
1. 合并同一角色的不同称呼：简称→全名、英文名→中文名、职务头衔→对应人物
2. 纯文字头衔如有具体人物在上下文中，作为该人物的别名，不单独创建角色
3. 标记纯背景元素为背景：纯地理名称、天气、抽象概念、法律文件、基础设施
4. 标记二元关系词（如"A与B"）为背景
5. **组织/政党/国家/公司的判定**：
   - 若文本中存在该组织的**具体成员/领导人**作为独立角色 → 组织标记为背景（人物代表组织博弈）
   - 若该组织**没有具体成员**出现在角色列表中（如纯企业竞争、国家间博弈）→ 可标记为角色
   - 敌方/外部势力组织，若无具体人物代表，应标记为角色
6. 高频次 + 高覆盖粒度的实体通常为核心角色，优先判定为 include=true

## 输出 JSON（仅 JSON，不要 markdown）
{{"entities": [
  {{"name": "规范名", "aliases": ["简称", ...], "include": true/false, "reason": "≤15字理由"}}
]}}

只返回 JSON。"""

_ENTITY_BATCH_SIZE = 60


async def sort_narrative_entities(
    source: str,
    entity_names: list[str],
    client: Any,
    max_source_chars: int = 20000,
    entity_frequencies: dict[str, int] | None = None,
    entity_chunk_coverage: dict[str, int] | None = None,
    chunk_texts: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not source or not entity_names:
        return []

    freq = entity_frequencies or {}
    cov = entity_chunk_coverage or {}
    chunks = chunk_texts or []
    import math

    def _score(name: str) -> float:
        f = freq.get(name, 0)
        c = cov.get(name, 0)
        return f * (1.0 + math.log(c + 1.0))

    # ── Build text overview (once, shared by all batches) ──
    if len(source) > max_source_chars and chunks:
        n = len(chunks)
        indices = [0]
        if n >= 4:
            indices.extend([n // 4, n // 2, 3 * n // 4])
        if n - 1 not in indices:
            indices.append(n - 1)
        samples = []
        for idx in indices:
            if idx < n:
                samples.append(f"[文本样本·第{idx+1}/{n}块]\n{chunks[idx][:1200]}")
        text_overview = "\n\n---\n\n".join(samples)
    elif len(source) > max_source_chars:
        L = len(source)
        samples = [source[:3000], source[L//4: L//4+3000],
                   source[L//2: L//2+3000], source[3*L//4: 3*L//4+3000]]
        text_overview = "\n\n---\n\n".join(f"[位置{i}]\n{s}" for i, s in enumerate(samples))
    else:
        text_overview = source[:max_source_chars]

    # ── Split entities into batches ──
    if len(entity_names) <= _ENTITY_BATCH_SIZE:
        batches = [list(entity_names)]
    else:
        # Sort by importance so each batch gets a representative mix
        ranked = sorted(entity_names, key=_score, reverse=True)
        batches = [ranked[i:i + _ENTITY_BATCH_SIZE]
                   for i in range(0, len(ranked), _ENTITY_BATCH_SIZE)]

    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import Message
    from strategy_forge.engine.intel_sorter import _extract_text, _parse_json
    from strategy_forge.core.providers import registry as _reg

    async def _sort_one_batch(batch_names: list[str]) -> list[dict[str, Any]]:
        # Build stats for this batch only
        scored = sorted(batch_names, key=_score, reverse=True)
        stat_lines = []
        for i, name in enumerate(scored, 1):
            f = freq.get(name, 0)
            c = cov.get(name, 0)
            if f > 0 or c > 0:
                stat_lines.append(f"{i}. {name} （频次={f}, 覆盖={c}块）")
            else:
                stat_lines.append(f"{i}. {name} （频次=?, 覆盖=?）")

        prompt = _NARRATIVE_SORT_PROMPT.format(
            text_overview=text_overview,
            entity_stats="\n".join(stat_lines),
            entity_names=", ".join(batch_names),
        )
        batch_label = f"[NarrativeSorter batch {len(batch_names)} entities]"
        try:
            resp = await client.chat(
                [Message(role="user", content=prompt)],
                system="你是故事编辑，输出结构化 JSON。只输出 JSON。",
                temperature=0.3,
                max_tokens=config.deduction_intel_max_tokens,
            )
            raw = _extract_text(resp)
            data = _parse_json(raw)
            if isinstance(data, list):
                data = {"entities": data}
            if not isinstance(data, dict):
                logger.warning("%s LLM returned non-dict: %s", batch_label, type(data))
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
                result.append({
                    "name": name,
                    "aliases": aliases,
                    "include_in_simulation": bool(e.get("include", True)),
                    "role": str(e.get("reason", ""))[:80],
                })
            return result
        except Exception as e:
            logger.warning("%s failed: %s", batch_label, e)
            return []

    # ── Run all batches concurrently (shared overview, independent calls) ──
    sem = asyncio.Semaphore(max(1, _reg.max_concurrent))
    async def _guarded(batch_names):
        async with sem:
            return await _sort_one_batch(batch_names)

    all_results: list[dict[str, Any]] = []
    gathered = await asyncio.gather(*(_guarded(b) for b in batches))
    for r in gathered:
        if r:
            all_results.extend(r)

    if len(batches) > 1:
        logger.info("[NarrativeSorter] %d batches → %d entities classified",
                    len(batches), len(all_results))
    return all_results

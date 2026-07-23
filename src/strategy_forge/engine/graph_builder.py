"""Phase 2: GraphRAG — entity-driven extraction with hybrid retrieval.

Supports two modes:
  - With preprocessor: high-freq entities → LanceDB retrieval → targeted LLM extract
  - Without preprocessor (fallback): semantic chunk → per-chunk LLM extract
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from string import Template
from typing import Any

from strategy_forge.storage.graph_store import DeductionGraphStore

from .models import Ontology
from .preprocessor import DeductionPreprocessor
from ._utils import extract_text as _extract_text, extract_json
from strategy_forge.core.llm_client import LLMConnectionError

logger = logging.getLogger(__name__)


_EXTRACT_PROMPT = """从以下文本中抽取实体和关系的三元组，返回 JSON 数组。

## 全文概览（多处采样，供你感知文本主题与角色关系）
$text_overview

## 实体类型（仅使用以下类型）
$entity_types

## 关系类型（仅使用以下类型）
$relation_types

## 候选实体白名单（抽取的实体名必须是以下标准名之一）
$candidate_entities

## 重要约束
实体名必须严格来自上述白名单，禁止新增任何不在白名单中的实体名。如果文本提到了白名单外的概念，忽略它，不要将其作为实体输出。

## 别名映射表（发现别名时必须归一化为标准名）
$alias_map

## 输出格式 — 必须是纯 JSON 数组
[
  {"entity": "实体名(必须来自白名单)", "type": "类型", "description": "简短描述"},
  {"source": "实体A", "target": "实体B", "relation": "关系名", "evidence": "原文证据"}
]

## 规则
1. entity 字段的值必须来自候选实体白名单
2. 若发现别名，映射为标准名后再写入
3. 每个三元组需要 evidence（原文证据）
4. 仅提取本文本片段中实际出现的实体和关系——不要输出白名单中在本文本内未出现的实体名

【重要】只返回纯JSON数组。不要```json代码块。不要任何解释文字。

## 文本
$text"""


async def build_graph(
    source: str,
    graph: DeductionGraphStore,
    ontology: Ontology | None,
    log_fn: Callable[[str, str], None],
    preprocessor: DeductionPreprocessor | None = None,
) -> None:
    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    from strategy_forge.core.llm_client import Message
    from strategy_forge.core.config import config
    from strategy_forge.core.providers import registry as _reg

    client = LLMClient()

    entity_type_names = [e.name for e in ontology.entities] if ontology else [
        "Person", "Organization", "Event", "Concept", "Location"
    ]
    relation_type_names = [r.name for r in ontology.relations] if ontology else [
        "works_for", "involved_in", "located_in", "opposes", "supports"
    ]

    total_entities = 0
    total_relations = 0

    if preprocessor and preprocessor.result:
        # ── 智能模式: 实体驱动抽取 ──
        result = preprocessor.result
        high_freq = result.high_freq_entities
        low_freq = result.low_freq_entities
        all_aliases = {**high_freq, **low_freq}
        _reverse_alias = _build_reverse_alias(all_aliases)
        alias_map_str = json.dumps(
            {k: list(v) for k, v in all_aliases.items()}, ensure_ascii=False,
        )
        candidate_names = list(all_aliases.keys())

        # ── 高频实体 → 定向深度抽取 ──
        if high_freq:
            log_fn("graph", f"实体驱动模式: {len(high_freq)} 个高频实体定向抽取")
            system = "你是知识图谱构建专家。严格从候选白名单中抽取实体和关系三元组——禁止新增任何不在白名单中的实体名。只输出 JSON。"

            # ── Phase 1（顺序·廉价）：实体排名 + 动态上限 ──
            from strategy_forge.core.tokenizer import compress_to_keywords
            freq_map = getattr(result, "entity_frequencies", {}) or {}
            cov_map = getattr(result, "entity_chunk_coverage", {}) or {}
            def _entity_rank(item):
                name, aliases = item
                return (freq_map.get(name, 0), cov_map.get(name, 0), len(aliases))
            hf_sorted = sorted(high_freq.items(), key=_entity_rank, reverse=True)
            dyn_cap = max(50, len(hf_sorted) // 4)
            hf_items = hf_sorted[:dyn_cap]
            log_fn("graph", f"实体驱动模式: {len(hf_items)} 个高频实体定向抽取 (动态上限={dyn_cap})")

            # ── Phase 1.5: 构建抽取模板（白名单随 dyn_cap 自适应）──
            # 构建全文概览：从 chunks 多处采样，帮助 LLM 区分主角 vs 背景
            text_overview = ""
            if result.chunks:
                chunks_list = [c.content if hasattr(c, "content") else str(c)
                               for c in result.chunks]
                n = len(chunks_list)
                samples = [chunks_list[0]]
                if n >= 4:
                    samples.extend([chunks_list[n // 4],
                                    chunks_list[n // 2],
                                    chunks_list[3 * n // 4]])
                if n > 1 and n - 1 not in (0, n // 4, n // 2, 3 * n // 4):
                    samples.append(chunks_list[-1])
                text_overview = "\n\n---\n\n".join(s[:800] for s in samples if s.strip())
            elif len(source) > 10000:
                L = len(source)
                samples = [source[:1000], source[L // 4: L // 4 + 1000],
                           source[L // 2: L // 2 + 1000],
                           source[3 * L // 4: 3 * L // 4 + 1000]]
                text_overview = "\n\n---\n\n".join(samples)
            else:
                text_overview = source[:1500]

            white_count = min(len(candidate_names), max(200, dyn_cap * 2))
            _extract_base = Template(_EXTRACT_PROMPT).substitute(
                text="__TEXT__",
                text_overview=text_overview[:2000],
                entity_types=", ".join(entity_type_names),
                relation_types=", ".join(relation_type_names),
                candidate_entities=", ".join(candidate_names[:white_count]),
                alias_map=alias_map_str,
            )
            prompts: list[str | None] = []
            for std_name, aliases in hf_items:
                fragments = preprocessor.retrieve_for_entity(
                    std_name, max(_reg.retrieve_top_k, 3) * 2, must_contain=aliases)
                if not fragments:
                    prompts.append(None)
                    continue
                # 片段去重 + 按长度降序排列（长片段优先，信息密度高）
                seen: set[str] = set()
                deduped: list[str] = []
                for frag in fragments:
                    norm = frag.strip()
                    if not norm or norm in seen:
                        continue
                    seen.add(norm)
                    deduped.append(norm)
                deduped.sort(key=len, reverse=True)
                fused = "\n---\n".join(deduped)
                keywords = compress_to_keywords(fused, top_k=10)
                keyword_tag = f"\n\n## 关键词标签\n{', '.join(keywords)}" if keywords else ""
                # 累计拼接而非硬截断：避免关键信息在 3000 字符处被截掉
                prompt_text = deduped[0][:3000]
                for extra in deduped[1:]:
                    if len(prompt_text) + len(extra) + 5 <= 3000:
                        prompt_text += "\n---\n" + extra
                    else:
                        break
                prompts.append(_extract_base.replace("__TEXT__", prompt_text + keyword_tag))

            # ── Phase 2（并发·LLM 抽取，上限由全局 Semaphore 控制）──

            async def _extract_call(prompt: str) -> str | None:
                try:
                    resp = await client.chat(
                        [Message(role="user", content=prompt)], system=system, temperature=0)
                    return _extract_text(resp)
                except LLMConnectionError:
                    raise
                except Exception as e:
                    logger.warning("[Graph] Entity-driven extract failed: %s", e)
                    return None

            idxs = [k for k, p in enumerate(prompts) if p is not None]
            # 用 as_completed 替代 gather——每完成一个 LLM 调用立即 parse 并输出进度，
            # 避免 109 次调用全等完才开始显示进度（界面空等 27-55 分钟）
            _ent_pool: list[tuple[str, str, str, str]] = []
            _rel_pool: list[tuple[str, str, str, str]] = []
            all_aliases_map: dict[str, list[str]] = dict(high_freq)
            _seen_names: list[str] = []  # 已提取实体名——后续调用追加排除提示

            async def _extract_with_idx(idx: int, prompt: str) -> tuple[int, str | None]:
                # 在 prompt 末尾追加已发现实体列表——无需重复输出实体条目，
                # 但涉及这些实体的新关系仍可提取
                _max_exclude = min(200, len(_seen_names))
                if _max_exclude > 0:
                    prompt = f"{prompt}\n\n## 已在之前提取中发现的实体（无需输出条目，但涉及它们的新关系仍可提取）\n{', '.join(_seen_names[:_max_exclude])}"
                return (idx, await _extract_call(prompt))

            pending = [asyncio.ensure_future(_extract_with_idx(i, prompts[i])) for i in idxs]
            completed = 0
            for coro in asyncio.as_completed(pending):
                i, content = await coro
                completed += 1
                if content:
                    try:
                        entities, relations = _parse_extraction(content)
                    except Exception as e:
                        logger.warning("[Graph] parse '%s' failed: %s", hf_items[i][0], e)
                        continue
                    for ent in entities:
                        name = _reverse_alias.get(ent.get("entity", ""), ent.get("entity", ""))
                        _ent_pool.append((_make_id(name, ""), name,
                                          ent.get("type", ""), ent.get("description", "")))
                        if name and name not in _seen_names:
                            _seen_names.append(name)
                    for rel in relations:
                        sid = _make_id(
                            _reverse_alias.get(rel.get("source", ""), rel.get("source", "")), "")
                        tid = _make_id(
                            _reverse_alias.get(rel.get("target", ""), rel.get("target", "")), "")
                        _rel_pool.append((sid, tid, rel.get("relation", ""), rel.get("evidence", "")))
                if completed % 5 == 0 or completed == len(idxs):
                    log_fn("graph", f"  实体 {completed}/{len(idxs)}: pool={len(_ent_pool)} 实体, {len(_rel_pool)} 关系")
                if len(_ent_pool) >= 50_000:
                    log_fn("graph", f"  内存阈值触发: pool={len(_ent_pool)} 实体, 执行中间批量写入")
                    seen_names: set[str] = set()
                    deduped = [(eid, nm, et, ds) for eid, nm, et, ds in _ent_pool
                               if nm not in seen_names and not seen_names.add(nm)]
                    graph.upsert_entities_batch(deduped)
                    total_entities += len(deduped)
                    _ent_pool.clear()
            # 剩余实体最终写入（在后面 Phase 3 处理）

            # ── Phase 3（内存去重 + 一次批量写 + 别名合并）──
            seen_names: set[str] = set()
            deduped_ents: list[tuple[str, str, str, str]] = []
            for item in _ent_pool:
                nm = item[1]
                if nm not in seen_names:
                    seen_names.add(nm)
                    deduped_ents.append(item)
            total_entities = graph.upsert_entities_batch(deduped_ents)
            del _ent_pool, seen_names, deduped_ents  # 释放内存

            seen_rels: set[tuple[str, str, str]] = set()
            deduped_rels: list[tuple[str, str, str, str]] = []
            for item in _rel_pool:
                key = (item[0], item[1], item[2])
                if key not in seen_rels:
                    seen_rels.add(key)
                    deduped_rels.append(item)
            total_relations = graph.upsert_relations_batch(deduped_rels)
            del _rel_pool, seen_rels, deduped_rels  # 释放内存

            # 一次性全部别名合并
            for std_name, _aliases in all_aliases_map.items():
                try:
                    graph.merge_alias_nodes(std_name, _aliases)
                except Exception as e:
                    logger.warning("[Graph] alias merge failed for '%s': %s", std_name, e)
            log_fn("graph", f"图谱批量写入完成: {total_entities} 实体, {total_relations} 关系")

        # ── 低频实体 → 语义分块顺带抽取 ──
        if result.chunks and low_freq:
            log_fn("graph", f"分块顺带模式: {len(low_freq)} 个低频实体 + {len(result.chunks)} 个语义块")
            await _extract_from_chunks(
                client=client, chunks=result.chunks, graph=graph, log_fn=log_fn,
                entity_types=entity_type_names, relation_types=relation_type_names,
            )
    else:
        # ── 回退模式: 全量语义分块 (无预处理器时) ──
        from strategy_forge.core.chunker import TextChunker
        chunker = TextChunker(strategy="paragraph", max_chunk_size=1536)
        chunks = [c.content for c in chunker.chunk(source)]
        log_fn("graph", f"回退模式: {len(chunks)} 个语义块")
        await _extract_from_chunks(
            client=client, chunks=chunks, graph=graph, log_fn=log_fn,
            entity_types=entity_type_names, relation_types=relation_type_names,
        )


async def _extract_from_chunks(
    client, chunks, graph, log_fn,
    entity_types, relation_types,
) -> None:
    from strategy_forge.core.config import config
    from strategy_forge.core.providers import registry as _reg
    from strategy_forge.core.llm_client import Message
    system = "你是知识图谱构建专家。严格从候选白名单中抽取实体和关系三元组——禁止新增任何不在白名单中的实体名。只输出 JSON。"

    total_entities = 0
    total_relations = 0
    # 构建全文概览：从 chunks 多处采样
    texts = [(c if isinstance(c, str) else c.content) for c in chunks]
    _n = len(texts)
    _raw = "\n\n---\n\n".join(texts[i][:600] for i in [0, _n//4, _n//2, 3*_n//4, _n-1] if 0 <= i < _n)
    _overview = _raw[:1500] if _raw.strip() else "(无概览)"

    _chunk_base = Template(_EXTRACT_PROMPT).substitute(
        text="__TEXT__",
        text_overview=_overview,
        entity_types=", ".join(entity_types),
        relation_types=", ".join(relation_types),
        candidate_entities="(无限制)",
        alias_map="{}",
    )

    # 并发抽取（上限由全局 Semaphore 控制），随后按原顺序写库

    async def _chunk_call(text: str) -> str | None:
        try:
            resp = await client.chat(
                [Message(role="user", content=_chunk_base.replace("__TEXT__", text[:5000]))],
                system=system, temperature=0)
            return _extract_text(resp)
        except LLMConnectionError:
            raise
        except Exception as e:
            logger.warning("[Graph] Chunk extract failed: %s", e)
            return None

    texts = [(c if isinstance(c, str) else c.content) for c in chunks]
    contents = await asyncio.gather(*(_chunk_call(t) for t in texts))

    for i, content in enumerate(contents):
        if not content:
            continue
        try:
            entities, relations = _parse_extraction(content)
        except Exception as e:
            logger.warning("[Graph] Chunk %d parse failed: %s", i, e)
            continue
        for ent in entities:
            ent_id = _make_id(ent.get("entity", ""), "")
            graph.upsert_entity(ent_id, ent.get("entity", ""), ent.get("type", ""),
                               ent.get("description", ""))
            total_entities += 1
        for rel in relations:
            sid = _make_id(rel.get("source", ""), "")
            tid = _make_id(rel.get("target", ""), "")
            graph.upsert_relation(sid, tid, rel.get("relation", ""),
                                 evidence=rel.get("evidence", ""))
            total_relations += 1
        log_fn("graph", f"  块 {i+1}/{len(chunks)}: {len(entities)} 实体, {len(relations)} 关系")


def _build_reverse_alias(alias_map: dict[str, set[str]]) -> dict[str, str]:
    """Build O(1) reverse lookup: alias → standardized name."""
    rev: dict[str, str] = {}
    for std_name, aliases in alias_map.items():
        rev[std_name] = std_name
        for a in aliases:
            rev[a] = std_name
    return rev


def _parse_extraction(raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = extract_json(raw)
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    if isinstance(data, dict):
        entities = data.get("entities", [])
        relations = data.get("relations", [])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "entity" in item:
                    entities.append(item)
                elif "source" in item:
                    relations.append(item)
    return entities, relations


def _make_id(name: str, etype: str) -> str:
    import hashlib
    raw = f"{name}:{etype}".encode()
    return hashlib.md5(raw).hexdigest()[:12]

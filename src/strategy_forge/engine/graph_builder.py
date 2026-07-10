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
from strategy_forge.core.llm_client import LLMConnectionError

logger = logging.getLogger(__name__)


def _extract_text(response) -> str:
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        c = response.content
        if isinstance(c, list):
            from strategy_forge.core.llm_client import TextBlock
            return "".join(b.text for b in c if isinstance(b, TextBlock))
        return str(c)
    if isinstance(response, dict):
        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        return str(response)
    return str(response)


_EXTRACT_PROMPT = """从以下文本中抽取实体和关系的三元组，返回 JSON 数组。

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
            system = "You are a JSON-only knowledge graph builder. Extract entity-relation triples strictly from the allowed list — never invent new entity names. NO markdown."

            # Pre-format constant parts using Template (safe from { } in alias_map JSON)
            _extract_base = Template(_EXTRACT_PROMPT).substitute(
                text="__TEXT__",
                entity_types=", ".join(entity_type_names),
                relation_types=", ".join(relation_type_names),
                candidate_entities=", ".join(candidate_names[:200]),
                alias_map=alias_map_str,
            )

            # ── Phase 1（顺序·廉价）：混合检索 + 构建每实体抽取 prompt ──
            from strategy_forge.core.tokenizer import compress_to_keywords
            hf_items = sorted(high_freq.items(), key=lambda x: -len(x[1]))[:25]  # top-25 高频
            log_fn("graph", f"实体驱动模式: {len(hf_items)} 个高频实体定向抽取 (top-25)")
            prompts: list[str | None] = []
            for std_name, aliases in hf_items:
                fragments = preprocessor.retrieve_for_entity(
                    std_name, config.deduction_retrieve_top_k, must_contain=aliases)
                if not fragments:
                    prompts.append(None)
                    continue
                fused = "\n---\n".join(fragments)
                keywords = compress_to_keywords(fused, top_k=10)
                keyword_tag = f"\n\n## 关键词标签\n{', '.join(keywords)}" if keywords else ""
                prompts.append(_extract_base.replace("__TEXT__", fused[:3000] + keyword_tag))

            # ── Phase 2（并发·LLM 抽取，上限 = FORGE_MAX_CONCURRENT）──
            sem = asyncio.Semaphore(max(1, config.deduction_max_concurrent))

            async def _extract_call(prompt: str) -> str | None:
                async with sem:
                    try:
                        resp = await client.chat(
                            [Message(role="user", content=prompt)], system=system, temperature=0.1)
                        return _extract_text(resp)
                    except LLMConnectionError:
                        raise
                    except Exception as e:
                        logger.warning("[Graph] Entity-driven extract failed: %s", e)
                        return None

            idxs = [k for k, p in enumerate(prompts) if p is not None]
            gathered = await asyncio.gather(*(_extract_call(prompts[k]) for k in idxs))
            content_by_idx = dict(zip(idxs, gathered, strict=False))

            # ── Phase 2.5（内存缓冲·解析 + 积累，不写 Kuzu）──
            _ent_pool: list[tuple[str, str, str, str]] = []   # (id, name, type, desc)
            _rel_pool: list[tuple[str, str, str, str]] = []   # (sid, tid, relation, ev)
            all_aliases_map: dict[str, list[str]] = dict(high_freq)

            for i, (std_name, _aliases) in enumerate(hf_items):
                content = content_by_idx.get(i)
                if not content:
                    continue
                try:
                    entities, relations = _parse_extraction(content)
                except Exception as e:
                    logger.warning("[Graph] parse '%s' failed: %s", std_name, e)
                    continue
                for ent in entities:
                    name = _reverse_alias.get(ent.get("entity", ""), ent.get("entity", ""))
                    _ent_pool.append((_make_id(name, ""), name,
                                      ent.get("type", ""), ent.get("description", "")))
                for rel in relations:
                    sid = _make_id(
                        _reverse_alias.get(rel.get("source", ""), rel.get("source", "")), "")
                    tid = _make_id(
                        _reverse_alias.get(rel.get("target", ""), rel.get("target", "")), "")
                    _rel_pool.append((sid, tid, rel.get("relation", ""), rel.get("evidence", "")))
                if (i + 1) % 5 == 0 or i == len(hf_items) - 1:
                    log_fn("graph", f"  实体 {i+1}/{len(hf_items)}: pool={len(_ent_pool)} 实体, {len(_rel_pool)} 关系")

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
                except Exception:
                    pass
            log_fn("graph", f"图谱批量写入完成: {total_entities} 实体, {total_relations} 关系")

        # ── 低频实体 → 语义分块顺带抽取 ──
        if result.chunks and low_freq:
            log_fn("graph", f"分块顺带模式: {len(low_freq)} 个低频实体 + {len(result.chunks)} 个语义块")
            await _extract_from_chunks(
                client=client, chunks=result.chunks, graph=graph, log_fn=log_fn,
                entity_types=entity_type_names, relation_types=relation_type_names,
                total_entities=total_entities, total_relations=total_relations,
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
            total_entities=total_entities, total_relations=total_relations,
        )


async def _extract_from_chunks(
    client, chunks, graph, log_fn,
    entity_types, relation_types,
    total_entities: int = 0, total_relations: int = 0,
) -> None:
    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import Message
    system = "你是知识图谱构建专家。严格从候选白名单中抽取实体和关系三元组——禁止新增任何不在白名单中的实体名。只输出 JSON。"

    _chunk_base = Template(_EXTRACT_PROMPT).substitute(
        text="__TEXT__",
        entity_types=", ".join(entity_types),
        relation_types=", ".join(relation_types),
        candidate_entities="(无限制)",
        alias_map="{}",
    )

    # 并发抽取（上限 = FORGE_MAX_CONCURRENT），随后按原顺序写库
    sem = asyncio.Semaphore(max(1, config.deduction_max_concurrent))

    async def _chunk_call(text: str) -> str | None:
        async with sem:
            try:
                resp = await client.chat(
                    [Message(role="user", content=_chunk_base.replace("__TEXT__", text[:5000]))],
                    system=system, temperature=0.1)
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
    data = try_extract_json(raw)
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


def try_extract_json(raw: str):
    """容错 JSON 解析: 直接解析 → 去markdown → 提取首个JSON块 → 裸key包裹。

    Handles Qwen's unstable JSON output formats.
    """
    raw = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', raw)
    cleaned = re.sub(r'\n?```', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Extract first complete JSON block (object or array)
    for pat in (r'\{[\s\S]*\}', r'\[[\s\S]*\]'):
        m = re.search(pat, cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                continue

    # 4. Fallback: wrap bare key-value in braces (Qwen sometimes outputs raw)
    if raw.strip().startswith('"'):
        try:
            return json.loads("{" + raw.strip() + "}")
        except (json.JSONDecodeError, ValueError):
            pass

    # 5. Total failure
    return {} if raw.startswith("{") else []


def _make_id(name: str, etype: str) -> str:
    import hashlib
    raw = f"{name}:{etype}".encode()
    return hashlib.md5(raw).hexdigest()[:12]

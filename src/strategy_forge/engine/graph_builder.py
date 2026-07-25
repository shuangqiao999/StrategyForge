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

## 正确示例
文本片段："美国商务部将DeepSeek等多家中国科技企业列入实体清单"
→ [{"entity": "美国", "type": "国家", "description": "实施制裁的主权国家"}, {"source": "美国", "target": "DeepSeek", "relation": "制裁", "evidence": "将...列入实体清单"}]

## 错误示例（禁止）
文本片段："全球经济增长率从3.4%下降至2.8%"
→ [{"entity": "全球经济增长率", "type": "经济指标", "description": "..."}]  ← 经济指标非实体

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
        result = preprocessor.result
        log_fn("graph", f"全量语义分块抽取: {len(result.chunks)} 个语义块")
        await _extract_from_chunks(
            client=client, chunks=result.chunks, graph=graph, log_fn=log_fn,
            entity_types=entity_type_names, relation_types=relation_type_names,
        )
    else:
        # ── 回退模式: 全量语义分块 (无预处理器时) ──
        from strategy_forge.core.chunker import TextChunker
        from strategy_forge.core.config import config as _cfg
        _cs = max(256, _cfg.deduction_chunk_size)
        chunker = TextChunker(strategy="paragraph", max_chunk_size=_cs)
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

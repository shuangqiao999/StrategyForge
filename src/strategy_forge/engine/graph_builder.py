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

## 关系类型定义（含利益极性 foe/ally/neutral，优先选用体现利益冲突或合作的关系）
$relation_definitions

## 候选实体白名单（抽取的实体名必须是以下标准名之一）
$candidate_entities

## 重要约束
$entity_constraint

## 别名映射表（发现别名时必须归一化为标准名）
$alias_map

## 输出格式 — 必须是纯 JSON 对象
{
  "entities": [{"entity": "实体名", "type": "类型", "description": "简短描述"}],
  "relations": [{"source": "实体A", "target": "实体B", "relation": "关系名", "evidence": "原文证据"}]
}

## 规则
1. entity 字段的值必须来自候选实体白名单
2. 若发现别名，映射为标准名后再写入
3. 每个三元组需要 evidence（原文证据）
4. 仅提取本文本片段中实际出现的实体和关系——不要输出白名单中在本文本内未出现的实体名
5. 如果以上实体类型无一匹配该实体的本质特征，type 字段请填 "_UNKNOWN"
6. 若文本体现竞争/对抗/此消彼长，优先使用 polarity=foe 的关系类型；体现协同/共赢用 polarity=ally 的关系类型；不要用中性关系弱化实际存在的利益冲突
7. 实体归类优先使用语义桶而非字面：主权国家/政府归"国家/政权"，公司/集团/品牌/平台/车企/银行归"企业"，人物归"人物"，纯地理位置/区域归"地理区域"。不要因名称含"区域"就把国家归为地理区域

## 正确示例
文本片段："A国商务部将X科技等多家新兴市场科技企业列入贸易限制清单"
→ {"entities": [{"entity": "A国", "type": "国家", "description": "实施贸易限制的主权国家"}], "relations": [{"source": "A国", "target": "X科技", "relation": "制裁", "evidence": "将...列入清单"}]}

## 错误示例（禁止）
文本片段："全球经济增长率从3.4%下降至2.8%"
→ 不应输出任何实体——经济指标不是博弈主体

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

    # ── 领域无关的通用类型桶（B：治理"中国→地理区域"归类漂移）──
    # 无论 ontology 是否生成，以下核心类型恒注入，保证高频实体能归入正确桶。
    _UNIVERSAL_TYPE_BUCKETS = [
        "国家", "政权", "企业", "组织", "机构", "人物",
        "地理区域", "事件", "概念", "政策",
    ]

    entity_type_names = [e.name for e in ontology.entities] if ontology else [
        "Person", "Organization", "Event", "Concept", "Location"
    ]
    # 合并通用类型桶（去重保序）
    for _ut in _UNIVERSAL_TYPE_BUCKETS:
        if _ut not in entity_type_names:
            entity_type_names.append(_ut)
    # 确保 _UNKNOWN 在所有类型列表中（graph_builder prompt 中的安全阀）
    if "_UNKNOWN" not in entity_type_names:
        entity_type_names.append("_UNKNOWN")
    relation_type_names = [r.name for r in ontology.relations] if ontology else [
        "联盟", "对抗", "贸易", "制裁", "隶属", "合作", "竞争", "冲突",
        "投资", "供应", "外交",
    ]
    # 关系类型定义（含利益极性），供抽取 prompt 注入，缓解关系类型单一化
    relation_definitions = _build_relation_definitions(ontology)

    if preprocessor and preprocessor.result:
        result = preprocessor.result
        log_fn("graph", f"全量语义分块抽取: {len(result.chunks)} 个语义块")
        # 注入预处理器高频实体作为候选白名单，引导 LLM 图谱构建
        candidate_entities = ""
        alias_map_str = "{}"
        try:
            hi_freq = getattr(result, "high_freq_entities", {}) or {}
            lo_freq = getattr(result, "low_freq_entities", {}) or {}
            cand_names = sorted(set(list(hi_freq.keys()) + list(lo_freq.keys())))
            if cand_names:
                candidate_entities = "\n".join(cand_names[:200])
            # 别名映射: {标准名: 别名集合}
            alias_map: dict[str, list[str]] = {}
            for std, aliases in {**hi_freq, **lo_freq}.items():
                a = [x for x in aliases if x != std]
                if a:
                    alias_map[std] = a[:20]
            if alias_map:
                alias_map_str = json.dumps(alias_map, ensure_ascii=False)
        except Exception as e:
            logger.warning("[Graph] 候选实体构建失败，使用无限制: %s", e)
        await _extract_from_chunks(
            client=client, chunks=result.chunks, graph=graph, log_fn=log_fn,
            entity_types=entity_type_names, relation_types=relation_type_names,
            relation_definitions=relation_definitions,
            candidate_entities=candidate_entities, alias_map=alias_map_str,
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
            relation_definitions=relation_definitions,
        )


def _build_relation_definitions(ontology) -> str:
    """从 ontology 构建关系类型定义字符串（含利益极性），注入抽取 prompt。"""
    if not ontology or not ontology.relations:
        return "(未定义)"
    lines = []
    for r in ontology.relations:
        pol = getattr(r, "polarity", "neutral") or "neutral"
        desc = (getattr(r, "description", "") or "").strip()
        lines.append(f"- {r.name}（{pol}）：{desc}" if desc else f"- {r.name}（{pol}）")
    return "\n".join(lines)


async def _extract_from_chunks(
    client, chunks, graph, log_fn,
    entity_types, relation_types,
    relation_definitions: str = "",
    candidate_entities: str = "", alias_map: str = "",
) -> None:
    from strategy_forge.core.config import config
    from strategy_forge.core.providers import registry as _reg
    from strategy_forge.core.llm_client import Message
    system = "你是知识图谱构建专家。从原文中提取实体和关系三元组，只输出 JSON。"

    # 构建全文概览：从 chunks 多处采样
    texts = [(c if isinstance(c, str) else c.content) for c in chunks]
    _n = len(texts)
    _raw = "\n\n---\n\n".join(texts[i][:600] for i in [0, _n//4, _n//2, 3*_n//4, _n-1] if 0 <= i < _n)
    _overview = _raw[:1500] if _raw.strip() else "(无概览)"

    if not candidate_entities:
        candidate_entities = "(无限制)"
        entity_constraint = "提取实体时优先使用语义合理名称；可根据文本语义自行命名实体。"
    else:
        entity_constraint = "实体名必须来自上述候选白名单，禁止新增白名单外实体。"
    if not alias_map:
        alias_map = "{}"

    _chunk_base = Template(_EXTRACT_PROMPT).substitute(
        text="__TEXT__",
        text_overview=_overview,
        entity_types=", ".join(entity_types),
        relation_types=", ".join(relation_types),
        relation_definitions=relation_definitions or "(未定义)",
        candidate_entities=candidate_entities,
        entity_constraint=entity_constraint,
        alias_map=alias_map,
    )

    # 并发抽取（上限由全局 Semaphore 控制），随后按原顺序写库

    async def _chunk_call(text: str) -> str | None:
        try:
            chunk_limit = max(5000, config.deduction_chunk_size * 2)
            resp = await client.chat_json(
                [Message(role="user", content=_chunk_base.replace("__TEXT__", text[:chunk_limit]))],
                system=system, schema_name="graph_extract", temperature=0)
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
        # 质量预检: 过滤空名/纯标点/垃圾类型
        valid_entities = []
        for ent in entities:
            nm = (ent.get("entity") or "").strip()
            ett = (ent.get("type") or "").strip()
            if not nm or len(nm) < 1 or re.fullmatch(r'[\s\W_]+', nm):
                continue
            if ett == "_UNKNOWN" or not ett:
                ett = "_UNKNOWN"
            ent["type"] = ett
            valid_entities.append(ent)
        entities = valid_entities
        for ent in entities:
            ett = ent.get("type", "")
            ent_id = _make_id(ent.get("entity", ""), ett)
            graph.upsert_entity(ent_id, ent.get("entity", ""), ett,
                               ent.get("description", ""))
        name_to_type = {e.get("entity", ""): e.get("type", "") for e in entities}
        for rel in relations:
            st = name_to_type.get(rel.get("source", ""), "")
            tt = name_to_type.get(rel.get("target", ""), "")
            if not st or not tt:
                continue
            sid = _make_id(rel.get("source", ""), st)
            tid = _make_id(rel.get("target", ""), tt)
            graph.upsert_relation(sid, tid, rel.get("relation", ""),
                                 evidence=rel.get("evidence", ""))
        _report = f"  块 {i+1}/{len(chunks)}: {len(entities)} 实体, {len(relations)} 关系"
        if relations:
            _sample = [r.get("relation","?") for r in relations[:5]]
            _report += f" | 关系采样: {_sample}"
        log_fn("graph", _report)


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
    if data is None:
        logger.warning("[Graph] extract_json returned None for raw (len=%d): %.200s", len(raw), raw[:200])
        return entities, relations
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
    else:
        logger.warning("[Graph] Unexpected parse result type: %s", type(data).__name__)
    return entities, relations


def _make_id(name: str, etype: str) -> str:
    import hashlib
    raw = f"{name}:{etype}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]

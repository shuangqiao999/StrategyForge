"""Phase 1: Ontology Generation — LLM defines entity/relation types from source material."""
from __future__ import annotations

import json
import hashlib
import logging
import re
from string import Template

from ._utils import extract_text
from .models import EntityTypeDef, Ontology, RelationTypeDef

logger = logging.getLogger(__name__)

_ontology_cache: dict[str, Ontology] = {}

_PROMPT = """你是一个知识本体专家。请分析以下文本，定义其中涉及的实体类型和关系类型。

## 输出 JSON 格式
```json
{
  "entities": [
    {"name": "实体类型名", "description": "描述该类型实体的特征", "properties": ["属性1", "属性2"]}
  ],
  "relations": [
    {"name": "关系名", "description": "关系含义", "from_type": "源实体类型", "to_type": "目标实体类型"}
  ]
}
```

## 规则
1. 实体类型不超过 10 种，关系类型不超过 15 种
2. 每种关系必须指定 from_type 和 to_type
3. 实体类型名和关系名使用中文命名，描述应当简洁精确
4. 只返回 JSON，不要解释

## 示例（理解抽象粒度）
以下示例仅说明\"抽象到什么程度\"——具体类型必须根据你的文本内容来定义。

文本："A国与B国发生贸易争端，A国对B国商品加征25%关税，B国随后宣布限制稀土出口作为反制。"

正确粒度：
{"entities": [{"name": "国家", "description": "主权经济行为体", "properties": ["经济规模", "贸易政策"]}, {"name": "商品类别", "description": "受贸易政策影响的产品类型", "properties": ["关税税率", "所属行业"]}, {"name": "资源", "description": "战略物资", "properties": ["储量", "出口管制级别"]}], "relations": [{"name": "加征关税", "description": "对另一方的商品提高进口税率", "from_type": "国家", "to_type": "商品类别"}, {"name": "限制出口", "description": "以行政手段管控战略物资出境", "from_type": "国家", "to_type": "资源"}]}

过细（不推荐）：实体类型为\"A国\"、\"B国\"、\"稀土\"——每个具体名称都变成一个类型。
过粗（不推荐）：实体类型为\"参与者\"、\"物品\"——丢失了领域语义。

## 文本
$text"""


async def generate_ontology(text: str) -> Ontology:
    text_hash = hashlib.sha256(text[:8000].encode()).hexdigest()
    if text_hash in _ontology_cache:
        logger.info("[Ontology] 缓存命中 (hash=%s...)", text_hash[:8])
        return _ontology_cache[text_hash]

    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    from strategy_forge.core.llm_client import Message

    client = LLMClient()
    messages = [Message(role="user", content=Template(_PROMPT).substitute(text=text[:8000]))]
    system = "你是知识本体分析专家，只输出 JSON。"

    try:
        response = await client.chat(messages, system=system, temperature=0.1)
        content = extract_text(response)
        result = _parse_ontology(content)
    except Exception as e:
        logger.warning("[Deduction] Ontology LLM failed, using defaults: %s", e)
        result = _default_ontology()

    _ontology_cache[text_hash] = result
    return result


def _parse_ontology(raw: str) -> Ontology:
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return _default_ontology()
    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return _default_ontology()

    entities = [
        EntityTypeDef(name=e["name"], description=e.get("description", ""),
                       properties=e.get("properties", []))
        for e in data.get("entities", [])[:10]
    ]
    relations = [
        RelationTypeDef(name=r["name"], description=r.get("description", ""),
                         from_type=r.get("from_type", ""), to_type=r.get("to_type", ""))
        for r in data.get("relations", [])[:15]
    ]
    return Ontology(entities=entities, relations=relations) if entities else _default_ontology()


def _default_ontology() -> Ontology:
    return Ontology(
        entities=[
            EntityTypeDef("Person", "参与事件的人物", ["role"]),
            EntityTypeDef("Organization", "组织/机构", ["type"]),
            EntityTypeDef("Event", "事件", ["date", "location"]),
            EntityTypeDef("Concept", "抽象概念/主题", []),
            EntityTypeDef("Location", "地点", []),
        ],
        relations=[
            RelationTypeDef("works_for", "任职于", "Person", "Organization"),
            RelationTypeDef("involved_in", "参与事件", "Person", "Event"),
            RelationTypeDef("located_in", "位于", "Event", "Location"),
            RelationTypeDef("opposes", "反对/对抗", "Person", "Person"),
            RelationTypeDef("supports", "支持", "Person", "Person"),
        ],
    )

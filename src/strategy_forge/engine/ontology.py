"""Phase 1: Ontology Generation — LLM defines entity/relation types from source material."""
from __future__ import annotations

import json
import logging
import re
from string import Template

from ._utils import extract_text
from .models import EntityTypeDef, Ontology, RelationTypeDef

logger = logging.getLogger(__name__)

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

## 文本
$text"""


async def generate_ontology(text: str) -> Ontology:
    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    from strategy_forge.core.llm_client import Message

    client = LLMClient()
    messages = [Message(role="user", content=Template(_PROMPT).substitute(text=text[:8000]))]
    system = "你是知识本体分析专家，只输出 JSON。"

    try:
        response = await client.chat(messages, system=system, temperature=0.1)
        content = extract_text(response)
        return _parse_ontology(content)
    except Exception as e:
        logger.warning("[Deduction] Ontology LLM failed, using defaults: %s", e)
        return _default_ontology()


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

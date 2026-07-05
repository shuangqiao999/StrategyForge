"""Intelligence Sorter — LLM reads source material, classifies all entities.

Filters non-strategic entities (regulators, acquired companies, subordinate units)
before the agent factory creates decision-making profiles.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_INTEL_PROMPT = """你是情报分析师。请根据以下种子材料，整理实体关系清单。

## 所有已提取的实体名称
{entity_names}

## 种子材料（完整上下文）
{source}

## 任务

1. 判断每个实体是独立的战略决策者，还是某个实体的子部分/下属
2. 判断生命周期 —— 如果被收购，在推演时间窗口内是否仍独立存在
3. 过滤非战略实体 —— 监管机构(SEC/证监会)、评级机构(标普/穆迪)、指数、纯媒体 — 不参与推演
4. 建立层级关系 —— 工厂属于企业、部门属于组织

## 输出 JSON（仅 JSON，无 markdown）
{{"entities": [
  {{"name": "特斯拉", "type": "企业", "parent": null, "sub_entities": ["弗里蒙特工厂", "上海超级工厂"],
    "include_in_simulation": true, "role": "核心博弈者"}},
  {{"name": "SolarCity", "type": "企业", "parent": null, "sub_entities": [],
    "include_in_simulation": false, "role": "2016年被特斯拉收购，不再独立存在"}},
  {{"name": "SEC", "type": "监管机构", "parent": null, "sub_entities": [],
    "include_in_simulation": false, "role": "金融监管者，非商业博弈者"}},
  {{"name": "标普500", "type": "指数", "parent": null, "sub_entities": [],
    "include_in_simulation": false, "role": "市场指数，非决策实体"}}
]}}

- include_in_simulation: true = 独立决策者，需要生成智能体
- include_in_simulation: false = 子实体/监管/指数/已退出 —— 不生成智能体
- parent: null = 独立实体; 填写父实体名 = 从属关系
- 重要：如果某人是某组织的CEO/领导人/代表人物（如马斯克→特斯拉，特朗普→美国，普京→俄罗斯），将其 parent 设为该组织名称，include_in_simulation 设为 false。同时将该人添加到该组织的 sub_entities 列表中。组织本身保留为独立决策者。
- sub_entities: 该实体包含的子部分（工厂、部门、领导人等）
- sub_entities: 该实体包含的子部分（工厂、部门等）
- 不要遗漏任何已提取的实体名"""


async def sort_entities(
    source: str,
    entity_names: list[str],
    client: Any,
    max_source_chars: int = 25000,
) -> list[dict[str, Any]]:
    """LLM reads source material, outputs structured entity relationship list.

    Args:
        source: Full seed text.
        entity_names: All entity names extracted by graph builder.
        client: DeductionLLMClient instance.
        max_source_chars: Max chars of source to send (kept high for context).

    Returns:
        List of entity entries with classification. Empty on failure.
    """
    if not source or not entity_names:
        return []

    prompt = _INTEL_PROMPT.format(
        entity_names=", ".join(entity_names),
        source=source[:max_source_chars],
    )

    from strategy_forge.core.llm_client import Message
    try:
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是情报分析师，输出结构化 JSON。只输出 JSON。",
            temperature=0.1,
        )
    except Exception as e:
        logger.warning("[IntelSorter] LLM call failed: %s", e)
        return []

    raw = _extract_text(resp)
    data = _parse_json(raw)
    if not isinstance(data, dict):
        logger.warning("[IntelSorter] Failed to parse LLM output as JSON")
        return []

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []

    result = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        result.append({
            "name": str(e.get("name", "")).strip(),
            "type": str(e.get("type", "")).strip(),
            "parent": e.get("parent") or None,
            "sub_entities": list(e.get("sub_entities", [])) if isinstance(e.get("sub_entities"), list) else [],
            "include_in_simulation": bool(e.get("include_in_simulation", True)),
            "role": str(e.get("role", "")).strip(),
        })

    active = sum(1 for e in result if e["include_in_simulation"])
    logger.info("[IntelSorter] %d entities total, %d active for simulation", len(result), active)
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

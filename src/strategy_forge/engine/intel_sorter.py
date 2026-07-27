"""Intelligence Sorter — LLM reads source material, classifies all entities.

Filters non-strategic entities (regulators, acquired companies, subordinate units)
before the agent factory creates decision-making profiles.
"""
from __future__ import annotations

import logging
from typing import Any

from strategy_forge.core.llm_client import LLMConnectionError
from ._utils import extract_text as _extract_text, parse_json as _parse_json

logger = logging.getLogger(__name__)

_INTEL_PROMPT = """你是情报分析师。请根据以下种子材料，整理实体别名与层级关系。

## 所有已提取的实体名称
{entity_names}

## 种子材料（完整上下文）
{source}

## 任务（简化版——EntityRegistry 接管分类判定）

1. 别名合并 —— 识别同一实体的不同名称（中英文名、简称、全称），合并为一条，选最完整的中文全称作为 name
2. 建立层级关系 —— 判断哪些实体是其他实体的子部分/下属，填写 parent 字段

## 输出 JSON（仅 JSON，无 markdown）
{{"entities": [
  {{"name": "规范全称", "type": "企业/国家/组织等", "aliases": ["简称", "英文名"], "parent": null}},
  {{"name": "某部门", "type": "组织编制", "aliases": [], "parent": "上级组织名"}}
]}}

## 规则
1. 别名合并：同一实体的多个名称只输出一条，选最规范/最完整的名称作为 name，其余全部放入 aliases
2. 层级关系：部门→上级组织，编制→上级军队，子公司→母公司，人物→所属政党/组织
3. 不要遗漏任何已提取的实体名（作为别名合并进某条的除外）

## 领域背景
{domain_extra_rules}"""


# ── 领域示例：帮助 LLM 理解各领域的层级关系 ──


# ── 已废弃——EntityRegistry 接管分类判定，保留为兼容旧引用 ──
def _as_name(x: Any) -> str:
    """将实体名元素统一转为干净字符串。

    LLM 常把 sub_entities/aliases 返回成对象数组（如 {"name": "华为"}），
    若直接下游 join/strip 会抛 'expected str instance, dict found'。此处集中归一化。
    """
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("name", "entity", "title", "id"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if x is None:
        return ""
    return str(x).strip()


def _as_name_list(raw: Any) -> list[str]:
    """把 aliases/sub_entities 归一化为去空、去重（保序）的字符串列表。"""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        n = _as_name(item)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


async def sort_entities(
    source: str,
    entity_names: list[str],
    client: Any,
    max_source_chars: int = 25000,
    domain: str = "",
) -> list[dict[str, Any]]:
    """LLM reads source material, outputs structured entity relationship list.

    Args:
        source: Full seed text.
        entity_names: All entity names extracted by graph builder.
        client: DeductionLLMClient instance.
        max_source_chars: Max chars of source to send (kept high for context).
        domain: Domain key for injecting domain-specific examples.

    Returns:
        List of entity entries with classification. Empty on failure.
    """
    if not source or not entity_names:
        return []

    from strategy_forge.core.rule_templates import get_domain_prompt
    extra_rules = get_domain_prompt(domain, "intel_extra_rules")
    prompt = _INTEL_PROMPT.format(
        entity_names=", ".join(entity_names),
        source=source[:max_source_chars],
        domain_extra_rules=extra_rules or "（无）",
    )

    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import Message
    try:
        resp = await client.chat_json(
            [Message(role="user", content=prompt)],
            system="你是情报分析师，输出结构化 JSON。只输出 JSON。",
            schema_name="intel_sort", temperature=0,
            max_tokens=config.deduction_intel_max_tokens,
        )
    except LLMConnectionError:
        raise
    except Exception as e:
        logger.warning("[IntelSorter] LLM call failed: %s", e)
        return []

    raw = _extract_text(resp)
    data = _parse_json(raw)
    # 兼容顶层数组：模型偶尔省略 {"entities": ...} 外壳
    if isinstance(data, list):
        data = {"entities": data}
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
            "name": _as_name(e.get("name", "")),
            "type": str(e.get("type", "")).strip(),
            "aliases": _as_name_list(e.get("aliases")),
            "parent": e.get("parent") or None,
            "sub_entities": _as_name_list(e.get("sub_entities")),
            "role": str(e.get("role", "")).strip(),
        })

    logger.info("[IntelSorter] 总计 %d 实体，别名和层级关系已提取", len(result))
    return result


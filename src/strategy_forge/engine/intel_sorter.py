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
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是情报分析师，输出结构化 JSON。只输出 JSON。",
            temperature=0,
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
            "include_in_simulation": bool(e.get("include_in_simulation", True)),
            "role": str(e.get("role", "")).strip(),
        })

    demoted = _apply_safety_net(result)
    active = sum(1 for e in result if e["include_in_simulation"])
    excluded = [e["name"] for e in result if not e["include_in_simulation"]]
    logger.info("[IntelSorter] 总计 %d 实体 | 活跃 %d | 排除 %d | 安全网降级 %d",
                len(result), active, len(excluded), demoted)
    if excluded:
        logger.info("[IntelSorter] 排除实体: %s", excluded[:15])
    return result


# 二元关系/对抗词（精确集合）——非单一决策者
_DYAD_WORDS = frozenset({
    "俄乌", "美伊", "中美", "美中", "巴以", "以巴", "印巴", "美俄", "俄美",
    "俄乌冲突", "中美关系", "美中关系", "俄乌战争", "巴以冲突",
})
# 军队编制后缀——归上级
_UNIT_SUFFIX = ("舰队", "战区", "司令部", "集团军", "航母战斗群", "特遣队")
# 一国政府职能部门/机关——归上级国家
_DEPT_WORDS = ("国防部", "财政部", "外交部", "商务部", "内政部", "司法部", "央行",
               "中央银行", "最高法院", "最高法", "国务院", "白宫", "国会", "参议院",
               "众议院", "国台办", "发改委", "证监会")
# 纯职务/头衔后缀——对应人物/组织已单列
_TITLE_SUFFIX = ("总统", "总理", "首相", "部长", "秘书长", "司令", "主席", "领导人",
                 "议长", "行长", "总裁", "元首")

# ── 英文关键词（与中文关键词并行）──
_DEPT_WORDS_EN = (
    "defense department", "department of defense", "dod", "treasury",
    "state department", "department of state", "supreme court", "federal reserve",
    "pentagon", "congress", "senate", "white house", "central bank",
    "foreign ministry", "ministry of defence", "ministry of defense",
    "ministry of finance", "ministry of foreign affairs",
    "commerce department", "justice department", "interior department",
    "national security council", "joint chiefs",
)
_UNIT_SUFFIX_EN = ("fleet", "command", "division", "task force", "battalion",
                   "regiment", "brigade", "squadron", "carrier strike group")
_TITLE_SUFFIX_EN = ("president", "prime minister", "secretary", "secretary-general",
                    "chairman", "chief", "minister", "general", "admiral",
                    "commander", "governor", "mayor", "director", "premier")
_DYAD_WORDS_EN = frozenset({
    "russia-ukraine", "us-iran", "us-china", "china-us",
    "israel-palestine", "israel-hamas", "us-russia", "russia-us",
    "india-pakistan", "north korea-south korea",
})

_ALL_DYAD = _DYAD_WORDS | _DYAD_WORDS_EN

# ── 二元关系词模式检测：匹配 "A与B" "A和B" "A-B" "A/B" 等对抗/关系组合 ──
def _is_dyad_pattern(name: str) -> bool:
    imports = ("与", "和", "及", "对", "vs", "vs.", "/", "-")
    for sep in imports:
        parts = name.split(sep)
        if len(parts) == 2 and all(len(p.strip()) >= 1 for p in parts):
            return True
    return False

# ── 军队名称精确匹配降级（不含后缀的独立军队名）──
_MILITARY_NAMES = frozenset({
    "乌军", "俄军", "美军", "伊朗伊斯兰革命卫队", "以色列国防军",
    "朝鲜人民军", "韩国军队", "日本自卫队",
})


def _any_name_matches(e: dict, keywords: tuple[str, ...], mode: str) -> bool:
    """检查实体 name 及其 aliases 是否命中关键词（大小写不敏感）。"""
    candidates = [e.get("name", "")]
    candidates.extend(e.get("aliases") or [])
    for name in candidates:
        if not isinstance(name, str) or not name:
            continue
        low = name.lower()
        if mode == "exact":
            if low in keywords:
                return True
        elif mode == "suffix":
            if low.endswith(keywords):
                return True
        elif mode == "substring":
            if any(w in low for w in keywords):
                return True
    return False


def _apply_safety_net(result: list[dict[str, Any]]) -> int:
    """保守安全网：对高置信的"非独立决策者"强制 include_in_simulation=false。

    仅在 FORGE_INTEL_SAFETY_NET 开启时生效；只降级、不新增/删除实体，避免误伤唯一代表。
    返回被安全网降级的实体数量。
    """
    import os
    from strategy_forge.core.providers import registry as _reg
    safety_enabled = _reg.intel_safety_net
    if not safety_enabled:
        logger.warning("[IntelSorter] FORGE_INTEL_SAFETY_NET=0，安全网已关闭，完全依赖LLM判断")
        return 0
    logger.info("[IntelSorter] 安全网状态: 启用")

    def _demote(e: dict, note: str) -> bool:
        if not e["include_in_simulation"]:
            return False
        e["include_in_simulation"] = False
        base = e.get("role", "") or ""
        e["role"] = (base + f"｜安全网降级：{note}") if base else f"安全网降级：{note}"
        return True

    demoted = 0
    for e in result:
        if not e.get("name"):
            continue
        if _any_name_matches(e, _ALL_DYAD, mode="exact") or _is_dyad_pattern(e.get("name", "")):
            demoted += _demote(e, "二元关系词")
        elif _any_name_matches(e, _UNIT_SUFFIX, mode="suffix") or _any_name_matches(e, _UNIT_SUFFIX_EN, mode="suffix"):
            demoted += _demote(e, "军队编制归上级")
        elif _any_name_matches(e, _DEPT_WORDS, mode="substring") or _any_name_matches(e, _DEPT_WORDS_EN, mode="substring"):
            demoted += _demote(e, "政府部门归上级国家")
        elif _any_name_matches(e, _TITLE_SUFFIX, mode="suffix") or _any_name_matches(e, _TITLE_SUFFIX_EN, mode="suffix"):
            demoted += _demote(e, "职务头衔非实体")
        elif e.get("name") in _MILITARY_NAMES:
            demoted += _demote(e, "军队编制归上级")

    return demoted

"""SemanticMediator — 语义中介层：跨域统一基础类型映射 + 通用 tier 判定。

所有领域实体先映射到 7 大类基础类型（Agent/Subordinate/Resource/Geography/Contract/Event/Concept），
再根据基础类型 + 领域适配器规则做 tier 分级。

核心原则：
  - tier1: 具备独立完整决策 + 独立行动 + 独立利益诉求，不依附其他实体
  - tier2: 存在决策行为但依附上级 tier1，无独立博弈空间
  - tier3: 纯资源/工具/地点/概念/事件，无自主决策与行动
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategy_forge.engine.domain_adapter import DomainAdapter

logger = logging.getLogger(__name__)

# 7 大基础类型（全领域统一，不可修改）
BASE_TYPES = ["Agent", "Subordinate", "Resource", "Geography", "Contract", "Event", "Concept"]

# 基础类型 → 中文名
BASE_TYPE_CN: dict[str, str] = {
    "Agent": "独立决策主体",
    "Subordinate": "附属参与者",
    "Resource": "资源/工具/物资",
    "Geography": "地理空间",
    "Contract": "合约/条约/协议",
    "Event": "事件/项目/冲突",
    "Concept": "抽象概念",
}

# 基础类型 → tier 判定说明
BASE_TYPE_TIER_DESC: dict[str, str] = {
    "Agent": "具备独立完整决策+独立行动+独立利益诉求；最低tier1",
    "Subordinate": "有决策行为但依附上级Agent；最低tier2",
    "Resource": "纯资源/工具/物资，无自主决策；tier3",
    "Geography": "地理空间，无自主决策；tier3",
    "Contract": "合约/条约/协议文本，无自主决策；tier3",
    "Event": "事件/项目/冲突过程，无自主决策；tier3",
    "Concept": "抽象概念/政策/主义，无自主决策；tier3",
}


def map_to_base_type(entity_type: str, adapter: "DomainAdapter") -> str:
    """将领域实体类型映射到 7 大类基础类型。

    映射规则：
      1. 查找 adapter.base_type_mapping 中该类型属于哪个基础类型
      2. 未匹配 → 返回 "Unknown"
    """
    if not entity_type or not adapter.base_type_mapping:
        return "Unknown"

    for base_type, domain_types in adapter.base_type_mapping.items():
        if base_type not in BASE_TYPES:
            continue
        if entity_type in domain_types:
            return base_type

    return "Unknown"


def get_min_tier(base_type: str, adapter: "DomainAdapter") -> int:
    """基础类型的最低保证 tier。

    返回 0 表示无强制（由 LLM 自行判定）。
    返回 1/2/3 表示代码强制最低 tier（LLM 只能升级不可降级）。
    """
    if base_type in adapter.tier_rule.force_tier1_base_types:
        return 1
    if base_type in adapter.tier_rule.force_tier2_base_types:
        return 2
    if base_type in adapter.tier_rule.force_tier3_base_types:
        return 3
    return 0


def ensure_min_tier(llm_tier: int, base_type: str, adapter: "DomainAdapter") -> int:
    """用基础类型的最低保证 tier 修正 LLM 输出。

    规则：LLM 的 tier 显式 ≥ 最低保证 → 保留 LLM 结果；否则升级到最低保证。
    例如：LLM 判 tier3，但 base_type=Agent 强制 tier1 → 升级到 tier1。
    """
    min_t = get_min_tier(base_type, adapter)
    if min_t == 0:
        return llm_tier
    # tier 数字越小等级越高 → 取两者最小值（即更高优先级）
    return min(llm_tier, min_t)


def build_base_type_prompt_snippet(adapter: "DomainAdapter") -> str:
    """构建注入 LLM Prompt 的基础类型描述片段。

    用于 Layer2 分类 Prompt，让 LLM 理解基础类型含义。
    """
    lines = ["## 通用基础类型定义（全领域统一标准）"]
    for bt in BASE_TYPES:
        cn = BASE_TYPE_CN.get(bt, bt)
        desc = BASE_TYPE_TIER_DESC.get(bt, "")
        lines.append(f"- **{bt}** ({cn}): {desc}")
    return "\n".join(lines)


def build_tier_standard_prompt() -> str:
    """构建通用 tier 分级标准 Prompt 片段。"""
    return """## 通用 tier 分级标准（全领域通用，不绑定特定领域）

### tier1 — 核心博弈主体
实体具备 **独立完整决策权 + 独立行动能力 + 独立利益诉求**，不依附任何其他实体存在。
- 无论实体类型（人/企业/政权/组织），只要满足以上三条即为 tier1
- 例：主权国家、独立上市公司、独立军阀、有独立叙事弧的主角

### tier2 — 次级参与者
实体存在决策行为，但决策空间 **完全依附于上级 tier1 主体**，无独立博弈空间。
- 决策被上级覆盖、执行上级意志、无独立战略方向
- 例：CEO（依附公司）、元首（依附国家）、子公司（依附母公司）

### tier3 — 背景层（无需博弈）
纯资源、工具、地点、概念、事件——无任何自主决策与行动。
- 例：武器、原材料、地理区域、条约文本、数据指标、背景角色"""

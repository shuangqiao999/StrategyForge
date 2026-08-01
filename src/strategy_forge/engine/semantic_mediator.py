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


# 领域无关的 Unknown 启发式兜底（C：解决 base_type_mapping 静态表覆盖不到的动态类型）
# 仅当适配器映射未命中时启用；基于实体类型名的语义词，跨领域通用，无领域偏向。
# 注意：用多字关键词避免单字误命中（如"政"会误吞"政策/政治行动"），规则按具体→宽泛排序。
_UNKNOWN_HEURISTIC_RULES: list[tuple[str, frozenset[str]]] = [
    # 独立决策主体（先于 Resource，使"芯片厂商/汽车企业"这类含主体词的归 Agent）
    ("Agent", frozenset({
        "政权", "政府", "国家", "政党", "军队", "武装", "企业", "公司",
        "集团", "品牌", "平台", "车企", "车厂", "银行", "基金", "机构", "组织",
        "联盟", "同盟", "协会", "商会", "国际组织", "军阀", "财团", "投行",
        "券商", "评级机构", "监管部门", "厂商", "制造厂", "巨头", "独角兽",
    })),
    # 附属参与者（人物/下属）
    ("Subordinate", frozenset({
        "人物", "官员", "将领", "发言人", "员工", "部长", "总统", "主席",
        "创始人", "总裁", "总监", "议员", "代表", "经理", "负责人",
    })),
    # 资源/工具
    ("Resource", frozenset({
        "产品", "型号", "武器", "装备", "工具", "物资", "原材料",
        "芯片", "技术", "专利", "产能", "机型", "软件", "硬件", "车型",
        "设施", "设备", "资源",
    })),
    # 抽象概念
    ("Concept", frozenset({
        "概念", "政策", "主义", "指标", "意识形态", "口号", "思潮",
        "舆论", "规则", "制度", "战略", "模式", "理念", "份额", "数据",
        "税收", "法规", "理论", "观念", "标准", "体系",
    })),
    # 事件
    ("Event", frozenset({
        "事件", "冲突", "战争", "战役", "项目", "会议", "选举", "活动",
        "峰会", "谈判", "并购", "收购", "变动", "取消", "爆发", "协议",
    })),
    # 合约
    ("Contract", frozenset({
        "条约", "合同", "法案", "协定", "法令", "协议",
    })),
    # 地理空间（最后，避免"国家"被"区域"误判）
    ("Geography", frozenset({
        "地理", "地点", "区域", "城市", "海峡", "走廊", "海域", "领土",
        "港口", "河流", "山脉", "大洲",
    })),
]


def _heuristic_base_type(entity_type: str) -> str:
    """基于类型名的领域无关启发式映射。未命中返回空串。"""
    t = (entity_type or "").strip()
    if not t:
        return ""
    for base_type, kws in _UNKNOWN_HEURISTIC_RULES:
        for kw in kws:
            if kw in t:
                return base_type
    return ""


def map_to_base_type(entity_type: str, adapter: "DomainAdapter") -> str:
    """将领域实体类型映射到 7 大类基础类型。

    映射规则：
      1. 查找 adapter.base_type_mapping 中该类型属于哪个基础类型
      2. 未匹配 → 领域无关启发式兜底（C）
      3. 启发式也未命中 → 返回 "Unknown"
    """
    if not entity_type or not adapter.base_type_mapping:
        return "Unknown"

    for base_type, domain_types in adapter.base_type_mapping.items():
        if base_type not in BASE_TYPES:
            continue
        if entity_type in domain_types:
            return base_type

    heuristic = _heuristic_base_type(entity_type)
    if heuristic:
        return heuristic

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

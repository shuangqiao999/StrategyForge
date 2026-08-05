"""关系→敌友极性推导（共享模块）。

三层判定体系：
  Layer A: 结构化映射（ontology 动态标注 + 领域适配器覆盖）— 主路径，确定性
  Layer B: 关键字兜底 — 仅处理 A 未覆盖的关系，低成本+确定性
  Layer C: LLM 精判 — 对 A/B 均为 neutral 的高频交互边，开局批量让 LLM 判一次

本模块承载 Layer B 的共享关键字表与推导函数，供 ontology 解析兜底、
simulator 关系反哺等消费点统一复用，避免各模块维护各自的关键字表。
"""
from __future__ import annotations

# 关键字兜底（Layer B）：语义变体是无限的，此表只负责"确定性强"的常见情形，
# 覆盖面由 Layer A 的 ontology/适配器补足。禁止依赖它作为唯一判定手段。
REL_FOE_KW = (
    "敌", "对立", "对抗", "对手", "竞争", "冲突", "背叛", "仇", "攻击", "威胁",
    "抢占", "争夺", "蚕食", "围堵", "打压", "制裁", "封锁", "围剿", "遏制",
    "挤压", "吞并", "淘汰", "压制", "排斥", "瓦解", "颠覆", "报复", "反制",
    "利用", "盘剥",
    "rival", "enemy", "hostil", "oppos", "compet", "conflict", "betray",
    "threat", "sanction", "blockade",
)
REL_ALLY_KW = (
    "盟", "同盟", "结盟", "联盟", "支持", "合作", "友", "供应", "投资",
    "资助", "援助", "协作", "共建", "联合", "协同", "磋商", "伙伴", "入股",
    "ally", "allied", "support", "friend", "cooperat", "partner", "invest",
    "alliance", "supply",
)
# 中立关系关键字：单向从属/怀柔/朝贡等不应自动归为敌对或盟友
# 优先级高于 foe/ally 关键字匹配，防止"招抚"误判为 foe
REL_NEUTRAL_KW = (
    "招抚", "招安", "绥靖", "依赖", "隶属", "依附", "朝贡",
    "互害",  # 互相伤害 = 无明确利益倾向
)

VALID_POLARITIES = ("foe", "ally", "neutral")


def normalize_polarity(value: str | None) -> str:
    """归一化极性值，非法输入回退 neutral。"""
    if not value:
        return "neutral"
    v = str(value).strip().lower()
    return v if v in VALID_POLARITIES else "neutral"


def infer_polarity(relation_name: str) -> str:
    """关键字兜底推导。仅在 ontology/适配器未显式标注时使用。

    三层匹配（命中即返回）：
      1. REL_NEUTRAL_KW 优先 — 明确不应判定为 foe/ally 的关系
      2. REL_FOE_KW — 零和/冲突关系
      3. REL_ALLY_KW — 共赢/协同关系
      未命中 → neutral
    """
    r = (relation_name or "").lower()
    if any(k in r for k in REL_NEUTRAL_KW):
        return "neutral"
    if any(k in r for k in REL_FOE_KW):
        return "foe"
    if any(k in r for k in REL_ALLY_KW):
        return "ally"
    return "neutral"


def merge_polarity_map(*maps: dict | None) -> dict[str, str]:
    """按序合并多个极性映射，后者覆盖前者；仅保留有效极性。

    顺序约定：ontology 标注 < 适配器覆盖。
    """
    out: dict[str, str] = {}
    for m in maps:
        if not m or not isinstance(m, dict):
            continue
        for k, v in m.items():
            p = normalize_polarity(v)
            if p != "neutral":
                out[str(k)] = p
    return out

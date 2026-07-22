"""Seed Data Extractor — LLM reads source material, extracts per-entity initial metrics.

Design: zero-config, single-LLM-call, graceful fallback to rule pack defaults.
Non-quantified (narrative) mode never calls this module.
"""
from __future__ import annotations

import logging
from typing import Any

from ._utils import extract_text as _extract_text, parse_json as _parse_json

logger = logging.getLogger(__name__)

# 指标刻度锚点：帮助 LLM 将定性描述映射到 0-100 的具体数值
_METRIC_ANCHORS: dict[str, str] = {
    "strength": "军力/竞争力（100=压倒性优势, 50=势均力敌, 20=残破不堪）",
    "morale": "士气/满意度（80=昂扬, 50=正常, 20=低迷）",
    "supply": "补给（100=充足, 50=勉强维持, 20=告急, 5=断供）",
    "fatigue": "疲劳度（10=精力充沛, 50=明显疲态, 90=精疲力竭）",
    "leadership": "领导力（90=卓越, 50=合格, 20=指挥混乱）",
    "market_share": "市场份额（100=垄断, 30=前三, 5=边缘玩家）",
    "cash_flow": "现金流（100=充裕, 50=平衡, 20=紧张, 5=濒临断裂）",
    "brand": "品牌（90=顶级, 50=有知名度, 20=无辨识度）",
    "rnd": "研发（90=行业领先, 50=跟随, 20=落后）",
    "supply_chain": "供应链韧性（100=完全自主, 50=有备选, 20=单一节点风险, 5=高度脆弱）",
    "support_rate": "支持率（80=稳固, 50=浮动, 20=危险, 5=崩溃边缘）",
    "economy": "经济（100=繁荣, 50=平稳, 20=衰退）",
    "unity": "团结度（90=铁板一块, 50=有裂隙, 20=分裂边缘）",
    "intl_relations": "国际关系（80=广泛友好, 50=中性, 20=孤立）",
    "legislative_power": "立法权（100=绝对控制, 50=需谈判, 20=边缘化）",
    "population": "人口/用户量（100=超级大国/亿级用户, 50=中等, 20=少数）",
    "resources": "资源（100=富足, 50=自给, 20=匮乏）",
    "pollution": "污染（10=清洁, 50=中度, 90=严重危机）",
    "biodiversity": "生物多样性（90=完好, 50=退化中, 20=崩溃）",
    "stability": "稳定性（90=稳固, 50=有波动, 20=动荡）",
    "employment": "就业（90=充分, 50=正常, 20=高失业）",
    "infrastructure": "基础设施（90=先进, 50=够用, 20=陈旧）",
    "finance": "财政（100=盈余, 50=平衡, 20=赤字, 5=破产）",
    "satisfaction": "满意度（80=满意, 50=中性, 20=不满）",
    "tech_lead": "技术领先度（90=行业标杆, 50=紧跟, 20=落后一代）",
    "chip_stock": "芯片储备（100=充足, 50=勉强, 20=紧缺, 5=断供）",
    "talent_pool": "人才池（90=人才富集, 50=正常流动, 20=流失严重）",
    "patent_barrier": "专利壁垒（90=垄断级, 50=有护城河, 20=无保护）",
    "commercialization": "商业化能力（90=变现强劲, 50=有收入, 20=烧钱）",
    "narrative_dominance": "舆情主导力（80=引领舆论, 50=有存在感, 20=失语）",
    "public_trust": "公信力（80=高度信任, 50=中性, 20=质疑, 5=失信）",
    "polarization": "极化度（90=极度分裂, 50=有分化, 10=高度共识）",
    "media_reach": "媒体触达（90=全覆盖, 50=有传播, 20=边缘）",
}

_SEED_PROMPT = """你是一个数据提取器。从以下战略文本中识别所有具有战略决策能力的实体（国家、企业、组织、关键人物），为每个实体评估以下指标值（0-100 的整数）。

## 可用指标及参考刻度
{metrics_descriptions}

## 提取规则
- 只提取文本中明确提到的实体，不要凭空创建
- 如果文本给出了具体数值，直接使用；如果只有定性描述，依据上方的参考刻度映射到 0-100 的合理值
- 未提及的指标不要赋值（省略该字段）
- 不要添加上述列表中不存在的指标
$finance_rules
## 实体选择规则
- 提取具有独立战略决策权的实体（公司、主权国家、核心人物），不提取下属/部门/项目/产品
- 如果文本只提了行业泛指而未提具体公司 → 不提取
- 行业协会、联盟、论坛本身不提取，其成员公司才提取

## 提取示例
文本："比亚迪以26%份额领跑市场，特斯拉Model Y排名第二。碳酸锂价格反弹至20万元/吨，多家车企成本承压。欧盟对华加征反补贴税。"
正确提取：比亚迪(cash_flow=70, market_share=85)、特斯拉(cash_flow=55, market_share=60)
错误提取：← 不应提取"碳酸锂"（原材料）"欧盟"（仅作为政策背景，无战略决策描述）"多家车企"（泛指）

## 输出 JSON（仅 JSON，无 markdown）
{{"entities": [
  {{"name": "实体名", "metrics": {{"strength": 85, "cash_flow": 60}}}}
]}}

## 文本
{source}"""


async def extract_seed_metrics(
    source: str,
    metrics: list[str],
    client: Any,
    max_chars: int = 20000,
) -> dict[str, dict[str, float]]:
    """Extract per-entity initial metrics from seed material via LLM.

    Args:
        source: Raw seed text (truncated to max_chars).
        metrics: List of metric names from the selected rule pack.
        client: DeductionLLMClient instance.
        max_chars: Max chars of source to send to LLM.

    Returns:
        {entity_name: {metric: value}}, empty dict on any failure.
    """
    if not source or not metrics:
        return {}

    # Conditionally inject finance-specific rules only when relevant metrics exist
    finance_keys = {"cash_flow", "market_share", "supply_chain"}
    if any(m in metrics for m in finance_keys):
        finance_rules = (
            "- ★ 财务类指标特别规则：如果文本提到\"净亏损\"、\"运营现金流为负\"、\"现金流承压\"、\"现金流偏紧\"，"
            "该实体的 cash_flow 应低于 40（亏损企业再大也不可能现金流满分）\n"
            "- ★ 如果文本描述\"营收增长但持续亏损\"的模式，market_share 可高（<80），"
            "但 cash_flow 应低（<40），supply_chain 也不能满分（依赖单一工厂）"
        )
    else:
        finance_rules = ""

    prompt = _SEED_PROMPT.format(
        metrics_descriptions="\n".join(
            f"- {m}: {_METRIC_ANCHORS.get(m, '(无描述)')}" for m in metrics
        ),
        source=source[:max_chars],
        finance_rules=finance_rules,
    )

    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import Message
    try:
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是战略数据提取器，只输出 JSON。",
            temperature=0.1,
            max_tokens=config.deduction_seed_max_tokens,
        )
    except Exception as e:
        logger.warning("[SeedExtractor] LLM call failed: %s", e)
        return {}

    raw = _extract_text(resp)
    data = _parse_json(raw)
    # 兼容顶层数组：模型偶尔省略 {"entities": ...} 外壳，直接给出对象数组
    if isinstance(data, list):
        data = {"entities": data}
    if not isinstance(data, dict):
        logger.warning("[SeedExtractor] Failed to parse LLM output as JSON")
        return {}

    result: dict[str, dict[str, float]] = {}
    metrics_set = set(metrics)
    for entity in data.get("entities", []):
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name", "")).strip()
        entity_metrics = entity.get("metrics", {})
        if not name or not isinstance(entity_metrics, dict):
            continue
        filtered = {}
        for m, v in entity_metrics.items():
            if m not in metrics_set:
                continue
            try:
                val = float(v)
                if val < 0 or val > 100:
                    val = max(0.0, min(100.0, val))
                filtered[m] = val
            except (TypeError, ValueError):
                continue
        if filtered:
            result[name] = filtered

    logger.info("[SeedExtractor] Extracted %d entities with custom initial metrics", len(result))
    return result

"""DomainDetector — 自动领域探测：基于文本关键词 + 实体类型统计，识别输入文本所属领域。

手动传入 domain 参数优先；domain="" 时触发自动探测。
探测失败 → 返回 "universal_neutral" 通用中立适配器。
"""

from __future__ import annotations

import logging
from typing import Any

from strategy_forge.engine.domain_adapter import list_adapters, load_adapter

logger = logging.getLogger(__name__)

# 实体类型 → 域方向 映射（用于实体类型统计推断）
_PERSON_TYPES = {"Person", "人物", "角色"}
_ORG_TYPES = {"Organization", "组织", "国际组织", "InternationalOrganization",
              "PoliticalParty", "政党", "政治组织"}
_COUNTRY_TYPES = {"Country", "国家", "政权", "割据势力"}
_COMPANY_TYPES = {"Company", "Enterprise", "企业", "公司", "Organization", "组织", "机构", "车企", "品牌"}
_MILITARY_TYPES = {"MilitaryUnit", "军队", "军事组织", "武装部队", "武装组织"}
_GEO_TYPES = {"Location", "地点", "地理区域", "地理空间"}


def detect_domain(
    source: str,
    entity_list: list[dict],
    log_fn: Any = None,
) -> str:
    """自动探测输入文本所属领域。

    探测策略：
      1. 收集所有 DomainAdapter 的 detect_keywords，在原文中加权计数
      2. 统计实体类型分布（Person比例、Organization比例、Company比例等）
      3. 综合打分，返回最高置信度 domain_id
      4. 置信度不足 → "universal_neutral"

    返回: domain_id 字符串
    """
    if not source and not entity_list:
        if log_fn:
            log_fn("agents", "域探测: 无输入，回退 universal_neutral")
        return "universal_neutral"

    available = list_adapters()
    if not available:
        logger.warning("[DomainDetector] 无可用的适配器，回退 universal_neutral")
        return "universal_neutral"

    # ── 1. 关键词特征匹配 ──
    scores: dict[str, float] = {}
    for domain_id in available:
        try:
            adapter = load_adapter(domain_id)
        except Exception:
            continue
        keywords = adapter.meta.detect_keywords or []
        if not keywords:
            continue
        score = 0.0
        for kw_entry in keywords:
            if isinstance(kw_entry, dict):
                kw = str(kw_entry.get("keyword", ""))
                weight = float(kw_entry.get("weight", 1))
            elif isinstance(kw_entry, str):
                kw = kw_entry
                weight = 1.0
            else:
                continue
            if kw and kw in source:
                # 每命中一次加权（高频域词汇更有区分度）
                count = source.count(kw)
                score += weight * min(count, 10)  # 上限防通胀
        if score > 0:
            scores[domain_id] = score

    # ── 2. 实体类型统计推断 ──
    if entity_list:
        total = len(entity_list)
        if total > 0:
            n_person = sum(1 for e in entity_list if str(e.get("type", "")) in _PERSON_TYPES)
            n_org = sum(1 for e in entity_list if str(e.get("type", "")) in _ORG_TYPES)
            n_country = sum(1 for e in entity_list if str(e.get("type", "")) in _COUNTRY_TYPES)
            n_company = sum(1 for e in entity_list if str(e.get("type", "")) in _COMPANY_TYPES)
            n_military = sum(1 for e in entity_list if str(e.get("type", "")) in _MILITARY_TYPES)
            n_geo = sum(1 for e in entity_list if str(e.get("type", "")) in _GEO_TYPES)

            person_ratio = n_person / total
            org_ratio = (n_org + n_country) / total
            company_ratio = n_company / total
            military_ratio = n_military / total
            geo_ratio = n_geo / total

            # 实体类型加权推断
            def _boost(domain: str, val: float) -> None:
                scores[domain] = scores.get(domain, 0) + val * 5.0  # 实体类型权重大于关键词

            if person_ratio > 0.5:
                _boost("novel", person_ratio)
                _boost("narrative", person_ratio * 0.8)
                if org_ratio < 0.2:
                    _boost("history", person_ratio * 0.6)
                if company_ratio > 0.2:
                    _boost("business_narrative", person_ratio * 0.7)
            if company_ratio > 0.3:
                _boost("business", company_ratio)
                if person_ratio > 0.2:
                    _boost("business_narrative", company_ratio * 0.9)
            if company_ratio > 0.15 and person_ratio < 0.5 and org_ratio < 0.3:
                _boost("business_narrative", company_ratio * 0.6)
            if military_ratio > 0.25:
                _boost("military", military_ratio)
                _boost("geo_strategy", military_ratio * 0.6)
            if org_ratio > 0.4:
                _boost("geo_strategy", org_ratio)
                _boost("politics", org_ratio * 0.7)
                _boost("business", org_ratio * 0.3)
            if geo_ratio > 0.3:
                _boost("geo_strategy", geo_ratio * 0.5)

    # ── 3. 选最高分 ──
    if not scores:
        if log_fn:
            log_fn("agents", "域探测: 无显著特征，回退 universal_neutral")
        return "universal_neutral"

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    # 置信度阈值：最高分 < 3.0 且实体类型信息很少时回退
    if best_score < 3.0 and (not entity_list or len(entity_list) < 5):
        if log_fn:
            log_fn("agents", f"域探测: 最佳'{best_domain}'得分{best_score:.1f}<3.0，回退 universal_neutral")
        return "universal_neutral"

    if log_fn:
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        detail = ", ".join(f"{d}={s:.1f}" for d, s in top3)
        log_fn("agents", f"域探测: {detail} → 选择 '{best_domain}'")

    logger.info("[DomainDetector] 探测: %s (得分 %.1f), 候选: %s",
                best_domain, best_score,
                {d: round(s, 1) for d, s in sorted(scores.items(), key=lambda x: -x[1])[:5]})

    return best_domain


def resolve_domain(
    domain: str,
    source: str,
    entity_list: list[dict],
    log_fn: Any = None,
) -> str:
    """综合解析领域：手动优先 + 自动兜底。

    返回: domain_id 字符串（保证有效适配器存在）。
    """
    # 手动传入 → 验证适配器是否存在
    if domain and domain.strip():
        available = list_adapters()
        if domain in available:
            return domain
        if log_fn:
            log_fn("agents", f"手动域 '{domain}' 未找到适配器，自动探测...")
        logger.warning("[DomainDetector] 手动域 '%s' 无适配器，自动探测", domain)

    # 自动探测
    detected = detect_domain(source, entity_list, log_fn)
    return detected

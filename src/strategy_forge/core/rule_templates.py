"""内置领域规则包（量化推演 L2 模板）。

规则包数据驱动 EntityState：metrics / initial_metrics / metric_ranges / thresholds /
actions / self_effects(作用于自身) / target_effects(作用于目标) / weather/terrain 修正。
新增领域只需在此添加一个条目（或由用户上传自定义 JSON 覆盖），无需改引擎代码。

约定：
- 数值为 0-100 归一化（除非 metric_ranges 另行指定）。
- thresholds: 指标 <= 阈值即判该实体出局(not alive)；只对"越高越好且代表存亡"的指标设置。
- self_effects / target_effects 的数值为"满强度(intensity=1)"基准，实际按 intensity 线性缩放。
"""
from __future__ import annotations

from typing import Any

DEFAULT_RANGE = [0.0, 100.0]


RULE_TEMPLATES: dict[str, dict[str, Any]] = {
    "military": {
        "domain": "military",
        "display_name": "军事战争",
        "metrics": ["strength", "morale", "supply", "fatigue", "leadership"],
        "initial_metrics": {"strength": 100, "morale": 80, "supply": 90, "fatigue": 10, "leadership": 70},
        "thresholds": {"strength": 10, "morale": 10},
        "actions": ["attack", "defend", "invest", "maneuver", "diplomacy", "observe"],
        "self_effects": {
            "attack": {"strength": -12, "supply": -10, "fatigue": 8, "morale": 5},
            "defend": {"strength": -5, "morale": -3, "fatigue": 2, "supply": -2},
            "invest": {"supply": -15, "strength": 8, "morale": 5, "fatigue": -3},
            "maneuver": {"supply": -6, "fatigue": 5},
            "diplomacy": {"morale": 3, "supply": -2},
            "observe": {"fatigue": -4},
        },
        "target_effects": {
            "attack": {"strength": -18, "morale": -8, "supply": -5},
            "maneuver": {"morale": -3},
            "diplomacy": {"morale": 2},
        },
        "weather_modifiers": {
            "rain": {"morale": -5, "supply": -8},
            "snow": {"fatigue": 10, "supply": -12},
            "clear": {"morale": 3, "fatigue": -3},
        },
        "terrain_modifiers": {
            "mountain": {"fatigue": 8, "strength": -5},
            "plain": {"morale": 3},
            "forest": {"supply": 3, "strength": -3},
        },
    },

    "business": {
        "domain": "business",
        "display_name": "商业竞争",
        "metrics": ["market_share", "cash_flow", "brand", "rnd", "morale"],
        "initial_metrics": {"market_share": 20, "cash_flow": 60, "brand": 60, "rnd": 50, "morale": 65},
        "thresholds": {"cash_flow": 10, "market_share": 5},
        "actions": ["price_war", "invest_rnd", "marketing", "expand", "partner", "observe"],
        "self_effects": {
            "price_war": {"cash_flow": -15, "market_share": 8, "brand": -5},
            "invest_rnd": {"cash_flow": -12, "rnd": 12, "morale": 3},
            "marketing": {"cash_flow": -8, "brand": 10, "market_share": 4},
            "expand": {"cash_flow": -18, "market_share": 6, "morale": -3},
            "partner": {"cash_flow": 5, "brand": 4},
            "observe": {"cash_flow": 2},
        },
        "target_effects": {
            "price_war": {"market_share": -10, "cash_flow": -6},
            "marketing": {"market_share": -4},
            "expand": {"market_share": -5},
        },
    },

    "politics": {
        "domain": "politics",
        "display_name": "政治博弈",
        "metrics": ["support_rate", "legislative_power", "intl_relations", "economy", "unity"],
        "initial_metrics": {"support_rate": 50, "legislative_power": 40, "intl_relations": 60, "economy": 55, "unity": 60},
        "thresholds": {"support_rate": 10},
        "actions": ["campaign", "legislate", "diplomacy", "reform", "attack_opponent", "observe"],
        "self_effects": {
            "campaign": {"support_rate": 10, "economy": -5, "unity": -3},
            "legislate": {"legislative_power": 8, "support_rate": -4},
            "diplomacy": {"intl_relations": 10, "support_rate": 2},
            "reform": {"economy": 10, "support_rate": -6, "unity": -4},
            "attack_opponent": {"support_rate": 3, "unity": -5},
            "observe": {"unity": 2},
        },
        "target_effects": {
            "attack_opponent": {"support_rate": -8, "unity": -5},
            "campaign": {"support_rate": -3},
            "legislate": {"legislative_power": -4},
        },
    },

    "ecology": {
        "domain": "ecology",
        "display_name": "生态环境",
        "metrics": ["population", "resources", "pollution", "biodiversity", "stability"],
        "initial_metrics": {"population": 60, "resources": 70, "pollution": 20, "biodiversity": 65, "stability": 60},
        "thresholds": {"population": 10, "resources": 5},
        "actions": ["exploit", "conserve", "pollute_control", "expand_habitat", "compete", "observe"],
        "self_effects": {
            "exploit": {"resources": -15, "population": 8, "pollution": 10},
            "conserve": {"resources": 10, "biodiversity": 8, "population": -2},
            "pollute_control": {"pollution": -15, "resources": -8, "stability": 5},
            "expand_habitat": {"population": 10, "resources": -10},
            "compete": {"population": 5, "biodiversity": -5},
            "observe": {"stability": 2},
        },
        "target_effects": {
            "compete": {"population": -8, "resources": -5},
            "exploit": {"resources": -6},
        },
    },

    "urban": {
        "domain": "urban",
        "display_name": "城市规划",
        "metrics": ["population", "employment", "infrastructure", "finance", "satisfaction"],
        "initial_metrics": {"population": 50, "employment": 60, "infrastructure": 50, "finance": 55, "satisfaction": 60},
        "thresholds": {"finance": 10, "satisfaction": 10},
        "actions": ["build_infra", "develop_industry", "welfare", "attract_talent", "regulate", "observe"],
        "self_effects": {
            "build_infra": {"finance": -15, "infrastructure": 12, "satisfaction": 5},
            "develop_industry": {"finance": -10, "employment": 10, "population": 5, "satisfaction": -2},
            "welfare": {"finance": -12, "satisfaction": 12},
            "attract_talent": {"finance": -8, "population": 8, "employment": 4},
            "regulate": {"satisfaction": -4, "finance": 5, "infrastructure": 2},
            "observe": {"finance": 2},
        },
        "target_effects": {
            "attract_talent": {"population": -6, "employment": -4},
            "develop_industry": {"population": -3},
        },
    },
}


def list_domains() -> list[dict[str, str]]:
    """供前端下拉使用的领域清单。"""
    return [{"domain": k, "display_name": v["display_name"]} for k, v in RULE_TEMPLATES.items()]


def get_template(domain: str) -> dict[str, Any] | None:
    return RULE_TEMPLATES.get(domain)

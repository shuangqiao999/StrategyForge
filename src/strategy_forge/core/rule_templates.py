"""领域规则包 — JSON 外部化 + 动态加载 + 缓存。

规则包以 JSON 文件存放在 data/rule/ 目录下：
  data/rule/rules.json           — 内置默认规则包（所有领域在一份文件）
  data/rule/custom/*.json        — 用户自定义规则包（上传 / 手动放置）

启动时自动加载；无文件时回退到 FALLBACK_RULES（与旧版 RULE_TEMPLATES 一致，保证向后兼容）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import _get_data_dir

logger = logging.getLogger(__name__)

DEFAULT_RANGE = [0.0, 100.0]

# 最小硬编码兜底（与旧版 RULE_TEMPLATES 逐字一致，确保降级安全）
_FALLBACK_RULES: dict[str, dict[str, Any]] = {
    "military": {
        "name": "\u2694\ufe0f \u519b\u4e8b\u6218\u4e89",
        "domain": "military",
        "display_name": "\u519b\u4e8b\u6218\u4e89",
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
        "name": "\ud83d\udcca \u5546\u4e1a\u7ade\u4e89",
        "domain": "business",
        "display_name": "\u5546\u4e1a\u7ade\u4e89",
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
        "name": "\ud83c\udfdb\ufe0f \u653f\u6cbb\u535a\u5f08",
        "domain": "politics",
        "display_name": "\u653f\u6cbb\u535a\u5f08",
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
        "name": "\ud83c\udf3f \u751f\u6001\u73af\u5883",
        "domain": "ecology",
        "display_name": "\u751f\u6001\u73af\u5883",
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
        "name": "\ud83c\udfd9\ufe0f \u57ce\u5e02\u89c4\u5212",
        "domain": "urban",
        "display_name": "\u57ce\u5e02\u89c4\u5212",
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

# ── 动态加载缓存 ──
_RULE_CACHE: dict[str, dict[str, Any]] = {}
_rules_loaded_from_file: bool = False


def _load_json_file(path: Path) -> dict[str, dict[str, Any]] | None:
    """加载单个 JSON 文件，返回 domain→rule 映射。"""
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception:
        logger.debug("[rule_templates] 无法加载 %s，跳过", path)
        return None
    # 支持两份格式：{"domain1":{...}, "domain2":{...}} 或 {"domain":"...", ...}
    if all(isinstance(v, dict) and ("domain" in v or "metrics" in v) for v in data.values()):
        rules: dict[str, dict[str, Any]] = {}
        for k, v in data.items():
            dom = v.get("domain", k)
            if "name" not in v:
                v["name"] = v.get("display_name", dom)
            rules[dom] = v
        return rules
    if isinstance(data, dict) and "domain" in data:
        dom = data.get("domain", "")
        if dom:
            if "name" not in data:
                data["name"] = data.get("display_name", dom)
            return {dom: data}
    return None


def reload_rules() -> None:
    """扫描内置与自定义规则包。

    打包环境：
      FORGE_RULE_DIR  → 内置规则（安装目录，只读，随安装包更新）
      FORGE_DATA_DIR  → 用户自定义规则 (data/rule/custom/)，持久化、卸载不丢
    开发环境（FORGE_RULE_DIR 未设置）：回退到单一 data/rule/ 目录。
    """
    global _RULE_CACHE, _rules_loaded_from_file
    try:
        data_dir = _get_data_dir()
    except Exception:
        logger.warning("[rule_templates] 无法确定数据目录，使用兜底规则")
        _RULE_CACHE = dict(_FALLBACK_RULES)
        _rules_loaded_from_file = False
        return

    import os
    rule_root = os.getenv("FORGE_RULE_DIR", "")
    loaded: dict[str, dict[str, Any]] = {}
    if rule_root:
        bundle_dir = Path(rule_root)
        if bundle_dir.is_dir():
            # 1) 内置规则包（安装包提供）
            default_file = bundle_dir / "rules.json"
            if default_file.exists():
                rules = _load_json_file(default_file)
                if rules:
                    loaded.update(rules)
                    logger.info("[rule_templates] 加载内置规则(FORGE_RULE_DIR): %d 个领域", len(rules))
    else:
        # 开发模式：回退到 data/rule/
        bundle_dir = data_dir / "rule"
        if bundle_dir.is_dir():
            default_file = bundle_dir / "rules.json"
            if default_file.exists():
                rules = _load_json_file(default_file)
                if rules:
                    loaded.update(rules)
                    logger.info("[rule_templates] 加载内置规则(data/rule): %d 个领域", len(rules))
    # 2) 用户自定义规则（持久化目录，卸载不丢）
    custom_dir = data_dir / "rule" / "custom"
    if custom_dir.is_dir():
        for f in sorted(custom_dir.glob("*.json")):
            rules = _load_json_file(f)
            if rules:
                conflicts = [k for k in rules if k in loaded]
                if conflicts:
                    logger.warning("[rule_templates] 自定义规则覆盖内置: %s → %s",
                                   f.name, ", ".join(conflicts))
                loaded.update(rules)
                logger.info("[rule_templates] 加载自定义规则: %s", f.name)

    if loaded:
        _RULE_CACHE = loaded
        _rules_loaded_from_file = True
    else:
        _RULE_CACHE = dict(_FALLBACK_RULES)
        _rules_loaded_from_file = False


def get_template(domain: str) -> dict[str, Any] | None:
    return _RULE_CACHE.get(domain)


def list_domains() -> list[dict[str, str]]:
    """供前端下拉使用的领域清单（仅返回从 JSON 文件加载成功的规则包）。

    _rules_loaded_from_file 为 False 时返回空列表——前端如实显示"无规则包"。
    """
    if not _rules_loaded_from_file:
        return []
    def _safe(s: str) -> str:
        return "".join(ch if ord(ch) < 0xD800 or ord(ch) > 0xDFFF else "?" for ch in s)
    return [{"domain": k, "name": _safe(v.get("name", v.get("display_name", k)))}
            for k, v in _RULE_CACHE.items()]


# ── 模块加载即初始化 ──
reload_rules()

"""migrate_adapters.py — 一次性脚本：从旧的 layer3_config.yaml + domain_prompts.json + entity_alias.json
自动生成统一的 data/domain_adapters/*.yaml 领域适配器文件。

用法：
  python scripts/migrate_adapters.py

输出：
  data/domain_adapters/
  ├── geo_strategy.yaml
  ├── novel.yaml
  ├── history.yaml
  ├── business.yaml
  ├── military.yaml
  ├── politics.yaml
  ├── ecology.yaml
  ├── urban.yaml
  ├── tech.yaml
  ├── info_war.yaml
  ├── narrative.yaml
  └── universal_neutral.yaml
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
DATA_RULE = ROOT / "data" / "rule"
OUTPUT_DIR = ROOT / "data" / "domain_adapters"

# ── 域列表（旧配置中存在的域）──
ALL_DOMAINS = [
    "geo_strategy", "novel", "history", "business", "military",
    "politics", "ecology", "urban", "tech", "info_war", "narrative",
]

# ── 域元信息（手动补充，旧配置中没有）──
DOMAIN_META = {
    "geo_strategy": {
        "display_name": "地缘战略",
        "methodology_mode": "geo",
        "detect_keywords": [
            {"keyword": "主权国家", "weight": 5},
            {"keyword": "制裁", "weight": 3},
            {"keyword": "地缘", "weight": 5},
            {"keyword": "军事联盟", "weight": 4},
            {"keyword": "国际关系", "weight": 3},
            {"keyword": "领土", "weight": 3},
            {"keyword": "外交", "weight": 3},
        ],
    },
    "novel": {
        "display_name": "小说",
        "methodology_mode": "narrative",
        "detect_keywords": [
            {"keyword": "主角", "weight": 3},
            {"keyword": "章节", "weight": 3},
            {"keyword": "小说", "weight": 5},
            {"keyword": "情节", "weight": 2},
            {"keyword": "内心独白", "weight": 2},
        ],
    },
    "history": {
        "display_name": "历史",
        "methodology_mode": "narrative",
        "detect_keywords": [
            {"keyword": "年", "weight": 1},
            {"keyword": "朝", "weight": 3},
            {"keyword": "皇帝", "weight": 3},
            {"keyword": "起兵", "weight": 3},
            {"keyword": "割据", "weight": 3},
            {"keyword": "崇祯", "weight": 5},
            {"keyword": "大明", "weight": 5},
        ],
    },
    "business": {
        "display_name": "商业经济",
        "methodology_mode": "geo",
        "detect_keywords": [
            {"keyword": "企业", "weight": 3},
            {"keyword": "公司", "weight": 3},
            {"keyword": "供应链", "weight": 3},
            {"keyword": "市场份额", "weight": 3},
            {"keyword": "融资", "weight": 3},
            {"keyword": "上市", "weight": 3},
            {"keyword": "制裁清单", "weight": 2},
        ],
    },
    "military": {
        "display_name": "军事战争",
        "methodology_mode": "geo",
        "detect_keywords": [
            {"keyword": "军队", "weight": 4},
            {"keyword": "武装", "weight": 3},
            {"keyword": "战区", "weight": 3},
            {"keyword": "舰队", "weight": 3},
            {"keyword": "导弹", "weight": 2},
            {"keyword": "占领", "weight": 3},
        ],
    },
    "politics": {
        "display_name": "政治博弈",
        "methodology_mode": "geo",
        "detect_keywords": [
            {"keyword": "政党", "weight": 4},
            {"keyword": "选举", "weight": 3},
            {"keyword": "议会", "weight": 3},
            {"keyword": "选民", "weight": 2},
            {"keyword": "法案", "weight": 2},
        ],
    },
    "ecology": {
        "display_name": "生态环境",
        "methodology_mode": "geo",
        "detect_keywords": [
            {"keyword": "碳排放", "weight": 3},
            {"keyword": "环保", "weight": 3},
            {"keyword": "污染", "weight": 3},
            {"keyword": "气候", "weight": 3},
            {"keyword": "NGO", "weight": 2},
        ],
    },
    "urban": {
        "display_name": "城市管理",
        "methodology_mode": "geo",
        "detect_keywords": [
            {"keyword": "城市", "weight": 3},
            {"keyword": "规划", "weight": 3},
            {"keyword": "交通", "weight": 2},
            {"keyword": "房地产", "weight": 3},
            {"keyword": "市政", "weight": 3},
        ],
    },
    "tech": {
        "display_name": "科技竞争",
        "methodology_mode": "geo",
        "detect_keywords": [
            {"keyword": "技术", "weight": 2},
            {"keyword": "专利", "weight": 3},
            {"keyword": "开源", "weight": 2},
            {"keyword": "芯片", "weight": 3},
            {"keyword": "AI", "weight": 3},
            {"keyword": "制裁", "weight": 2},
        ],
    },
    "info_war": {
        "display_name": "信息战/舆论",
        "methodology_mode": "geo",
        "detect_keywords": [
            {"keyword": "媒体", "weight": 3},
            {"keyword": "舆论", "weight": 3},
            {"keyword": "假新闻", "weight": 3},
            {"keyword": "社交平台", "weight": 3},
            {"keyword": "宣传", "weight": 3},
        ],
    },
    "narrative": {
        "display_name": "叙事/故事",
        "methodology_mode": "narrative",
        "detect_keywords": [
            {"keyword": "角色", "weight": 2},
            {"keyword": "故事", "weight": 3},
            {"keyword": "情节", "weight": 2},
            {"keyword": "叙事", "weight": 3},
        ],
    },
}

# ── 基础类型映射（每域的 7 大类映射，手动定义）──
def _build_base_type_mapping(domain: str) -> dict[str, list[str]]:
    """根据域类型构建基础类型映射表。"""
    common = {
        "Agent": ["Organization", "组织", "Country", "国家", "政权", "Company", "Enterprise",
                   "企业", "公司", "InternationalOrganization", "国际组织", "PoliticalParty",
                   "政党", "政治组织", "割据势力", "叛军", "军阀", "武装组织", "NGO",
                   "非政府组织", "军事组织", "MilitaryUnit"],
        "Subordinate": ["Person", "人物", "角色", "官员", "将领", "发言人", "员工",
                         "子公司", "部门", "Division", "Department",
                         "议员", "代表", "政治党派"],
        "Resource": ["武器", "资源", "物资", "兵器", "原材料", "产品", "弹药",
                      "技术标准", "TechnologyStandard", "基础设施"],
        "Geography": ["Location", "地点", "地理区域", "地理空间", "城市", "海峡",
                       "走廊", "区域", "国家", "Country"],
        "Contract": ["条约", "协议", "法案", "合同", "贸易协定"],
        "Event": ["事件", "冲突", "战争", "项目", "活动", "会议", "选举"],
        "Concept": ["概念", "政策", "主义", "数据指标", "经济指标", "意识形态", "规则",
                     "政治口号", "思潮", "舆论"],
    }
    if domain in ("novel", "history", "narrative"):
        result = dict(common)
        result["Geography"] = ["Location", "地点", "地理区域", "场景", "地名"]
        return result
    return common


# ── Jaccard 阈值 ──
JACCARD_THRESHOLDS = {
    "geo_strategy": 0.40, "military": 0.40,
    "business": 0.45, "politics": 0.40,
    "novel": 0.50, "history": 0.50,
    "ecology": 0.45, "urban": 0.45,
    "tech": 0.45, "info_war": 0.40,
    "narrative": 0.50,
}

# ── 域所属组（Layer3 跳过 / 采样模式）──
NARRATIVE_DOMAINS = {"novel", "history", "narrative"}


def load_layer3_config() -> dict:
    path = DATA_RULE / "layer3_config.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_domain_prompts() -> dict:
    path = DATA_RULE / "domain_prompts.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_entity_alias() -> dict:
    path = DATA_RULE / "entity_alias.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_adapter_yaml(domain: str, l3cfg: dict, prompts: dict, alias: dict) -> dict:
    meta = DOMAIN_META.get(domain, {
        "display_name": domain,
        "methodology_mode": "neutral",
        "detect_keywords": [],
    })

    is_narrative = domain in NARRATIVE_DOMAINS

    # ── layer3_config 字段提取 ──
    domain_l3 = l3cfg.get(domain, {}) or {}
    # 排除 _defaults 等非域 key
    if not isinstance(domain_l3, dict) or not domain_l3:
        # 回退
        fallback = "novel" if is_narrative else "geo_strategy"
        domain_l3 = l3cfg.get(fallback, {}) or {}

    defaults = l3cfg.get("_defaults", {}) or {}

    token_raw = domain_l3.get("token", defaults.get("token", {}))

    # ── domain_prompts 字段提取 ──
    dp = prompts.get(domain, prompts.get("_default", {})) or {}

    # ── entity_alias 字段提取 ──
    domain_aliases = alias.get(domain, {}) or {}
    force_keep_all = (alias.get("_force_keep", {}) or {}).get("all", [])
    force_keep_domain = (alias.get("_force_keep", {}) or {}).get(domain, [])
    force_keep_list = list(force_keep_all) + list(force_keep_domain)
    builtin_org = alias.get("_builtin_org_members", {}) or {}
    builtin_pc = alias.get("_builtin_person_country", {}) or {}
    strategic_regions = alias.get("_strategic_regions", [])

    # ── 采样配置 ──
    if is_narrative:
        sampling = {"head_size": 1500, "tail_size": 1500, "lit_mode": True}
    else:
        sampling = {"head_size": 2000, "tail_size": 2000, "lit_mode": False}

    jaccard = JACCARD_THRESHOLDS.get(domain, 0.45)

    # ── 组装 YAML ──
    adapter = {
        "domain_meta": {
            "domain_id": domain,
            "display_name": meta["display_name"],
            "detect_keywords": meta.get("detect_keywords", []),
            "methodology_mode": meta["methodology_mode"],
        },
        "param_config": {
            "shard_threshold": int(domain_l3.get("shard_threshold", defaults.get("shard_threshold", 12000))),
            "shard_size": int(domain_l3.get("shard_size", defaults.get("shard_size", 7500))),
            "shard_overlap": int(domain_l3.get("shard_overlap", defaults.get("shard_overlap", 800))),
            "jaccard_threshold": float(jaccard),
            "max_sample_chars": int(domain_l3.get("sample_chars", defaults.get("sample_chars", 8000))),
            "sampling": sampling,
            "token": {
                "l1_normalize": token_raw.get("l1_normalize", {"base": 200, "per": 150, "cap": 12000}),
                "l1_shard": token_raw.get("l1_shard", {"base": 200, "per": 200, "cap": 6000}),
                "l1_refine": token_raw.get("l1_refine", {"base": 200, "per": 120, "cap": 8000}),
                "l2_classify": token_raw.get("l2_classify", {"base": 150, "per": 100, "cap": 4000}),
                "l3_cross": token_raw.get("l3_cross", {"base": 300, "per": 60, "cap": 3000}),
            },
            "cache": {
                "l1_enabled": bool(domain_l3.get("l1_cache_enabled", defaults.get("l1_cache_enabled", True))),
                "l1_max_entries": int(domain_l3.get("l1_cache_max_entries", defaults.get("l1_cache_max_entries", 16))),
                "l3_enabled": bool(domain_l3.get("cache_enabled", defaults.get("cache_enabled", True))),
                "l3_max_entries": int(domain_l3.get("cache_max_entries", defaults.get("cache_max_entries", 64))),
                "ttl_sec": 1800,
            },
        },
        "base_type_mapping": _build_base_type_mapping(domain),
        "tier_rule": {
            "force_tier1_base_types": ["Agent"],
            "force_tier2_base_types": ["Subordinate"],
            "force_tier3_base_types": ["Resource", "Geography", "Contract", "Event", "Concept"],
            "extra_rules": (dp.get("l2_entity_rules") or dp.get("l2_tier_table") or "").strip(),
        },
        "layer_config": {
            "skip_layer3": is_narrative,
            "min_kept_for_check": int(domain_l3.get("min_kept_for_check", defaults.get("min_kept_for_check", 3))),
            "warn_threshold": int(domain_l3.get("warn_threshold", defaults.get("warn_threshold", 8))),
            "desc_truncate": int(domain_l3.get("desc_truncate", defaults.get("desc_truncate", 80))),
            "sample_chars_l3": int(domain_l3.get("sample_chars", defaults.get("sample_chars", 5000))),
            "monarchs_force_tier1": bool(domain_l3.get("monarchs_force_tier1", defaults.get("monarchs_force_tier1", False))),
            "warlord_force_tier1": bool(domain_l3.get("warlord_force_tier1", defaults.get("warlord_force_tier1", False))),
            "strategic_geo_tier2": bool(domain_l3.get("strategic_geo_tier2", defaults.get("strategic_geo_tier2", False))),
            "log_file": str(domain_l3.get("log_file", defaults.get("log_file", ""))),
            "fallback_rules": {
                "org_overlap_threshold": int((domain_l3.get("fallback_rules", {}) or {}).get("org_overlap_threshold", 3)),
                "org_members_map": dict(builtin_org),
                "person_country_map": dict(builtin_pc),
            },
        },
        "prompts": {
            "l2_entity_rules": dp.get("l2_entity_rules", ""),
            "l2_tier_table": dp.get("l2_tier_table", ""),
            "l3_system_prompt": domain_l3.get("system_prompt", ""),
            "l3_redundancy_rules": dp.get("l3_redundancy_rules", ""),
            "l3_downgrade_rules": dp.get("l3_downgrade_rules", ""),
        },
        "aliases": {
            "_org_members": dict(builtin_org),
            "_person_country": dict(builtin_pc),
            "_force_keep": force_keep_list,
            "entity_aliases": domain_aliases,
            "_strategic_regions": strategic_regions if not is_narrative else [],
        },
    }
    return adapter


def build_universal_neutral(l3cfg: dict, prompts: dict) -> dict:
    """构建通用中立兜底适配器。"""
    defaults = l3cfg.get("_defaults", {}) or {}
    dp = prompts.get("_default", {}) or {}

    return {
        "domain_meta": {
            "domain_id": "universal_neutral",
            "display_name": "通用中立领域",
            "detect_keywords": [],
            "methodology_mode": "neutral",
        },
        "param_config": {
            "shard_threshold": 12000,
            "shard_size": 7500,
            "shard_overlap": 800,
            "jaccard_threshold": 0.45,
            "max_sample_chars": 8000,
            "sampling": {"head_size": 2000, "tail_size": 2000, "lit_mode": False},
            "token": {
                "l1_normalize": {"base": 200, "per": 150, "cap": 12000},
                "l1_shard": {"base": 200, "per": 200, "cap": 6000},
                "l1_refine": {"base": 200, "per": 120, "cap": 8000},
                "l2_classify": {"base": 150, "per": 100, "cap": 4000},
                "l3_cross": {"base": 300, "per": 60, "cap": 3000},
            },
            "cache": {
                "l1_enabled": True,
                "l1_max_entries": 16,
                "l3_enabled": True,
                "l3_max_entries": 64,
                "ttl_sec": 1800,
            },
        },
        "base_type_mapping": _build_base_type_mapping("universal_neutral"),
        "tier_rule": {
            "force_tier1_base_types": ["Agent"],
            "force_tier2_base_types": ["Subordinate"],
            "force_tier3_base_types": ["Resource", "Geography", "Contract", "Event", "Concept"],
            "extra_rules": "仅依据实体独立决策能力分级。拥有完整独立行动与利益诉求为tier1；依附其他主体决策为tier2；纯资源/工具/概念/地理/事件为tier3。\n\n## 活性检验参考\n- 实体执行独立行为（宣布/决定/发起/签署/制裁/拒绝/要求/投资）→ 独立决策\n- 实体仅被报道/引用/分析 → 被动背景，tier3\n- 政党/政治派别 → tier2，依附更高层级政权\n- 频次陷阱：1次关键决策 > 10次背景提及",
        },
        "layer_config": {
            "skip_layer3": False,
            "min_kept_for_check": 3,
            "warn_threshold": 8,
            "desc_truncate": 80,
            "sample_chars_l3": 5000,
            "monarchs_force_tier1": False,
            "warlord_force_tier1": False,
            "strategic_geo_tier2": False,
            "log_file": "",
            "fallback_rules": {"org_overlap_threshold": 3, "org_members_map": {}, "person_country_map": {}},
        },
        "prompts": {
            "l2_entity_rules": "你是通用博弈分析专家。\n## 本域基本博弈单位\n- 有独立行为+独立决策权的实体 = tier1——不论类型（企业/国家/组织/独立政权/军阀/有独立叙事弧的角色）\n- 该实体的领导/负责人 = tier2（决策⊆上级决策）\n- 该实体的下属/部门/子品牌/成员 = tier3（决策被覆盖）\n- 政党/政治派别/意识形态运动 = tier2（国内政治子系统，依附更高级政权）\n- 工具/资源/概念/数据/地点/报告/指标 = tier3\n\n## 活性检验（独立决策 vs 被动背景）\n- 实体执行了独立行为（宣布/决定/发起/签署/制裁/拒绝/要求/投资/出兵）→ 独立决策\n- 实体仅被报道/提及/引用/分析，无主动行为 → 被动背景，tier3\n- 实体名称含\"报告/数据/指标/趋势/调查\" → 信息产物，tier3\n- 频次陷阱：高频≠重要，1次关键决策 > 10次背景提及",
            "l2_tier_table": "| 独立决策实体（不论类型） | tier1 | 始终——该域最高博弈单位 |\n| 该实体的领导/负责人 | tier2 | 决策⊆上级决策 |\n| 政党/政治派别 | tier2 | 国内政治子系统 |\n| 该实体的下属/部门/子品牌 | tier3 | 决策空间被覆盖 |\n| 工具/资源/概念/数据/地点/报告 | tier3 | 永远",
            "l3_system_prompt": "你是通用博弈实体冗余检测专家。检测实体间是否存在决策权重叠。\n## 核心原则\n- 一个实体决策空间 ⊆ 另一实体时 → 前者冗余\n- 仅当重叠 ≥ 60% 且一方完全覆盖另一方时才判定冗余\n- 不确定 → 保留不降级",
            "l3_redundancy_rules": "## 通用冗余检测（中立，无领域偏见）\n- 领导-实体重叠 → 保留实体，降级领导\n- 组织-成员重叠 → 核心成员≥3已独立列席时降级组织；否则保留\n- 子品牌/产品线/下级机构-主体重叠 → 合并到主体\n- 政党/政治派别-政权重叠 → 降级政党（国内子系统）\n- 政策口号/主义-国家重叠 → 降级口号（非独立博弈者）\n- 平行对等主体 → 互不冗余，分别保留\n- 不确定者 → 保守保留，不降级",
            "l3_downgrade_rules": "tier2",
        },
        "aliases": {
            "_org_members": {},
            "_person_country": {},
            "_force_keep": [],
            "entity_aliases": {},
            "_strategic_regions": [],
        },
    }


def _yaml_str_presenter(dumper, data):
    """多行字符串 Literal Block Scalar 样式。"""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def main():
    print("Loading old configs...")
    l3cfg = load_layer3_config()
    prompts = load_domain_prompts()
    alias = load_entity_alias()

    print(f"  layer3_config.yaml: {len(l3cfg)} sections")
    print(f"  domain_prompts.json: {len(prompts)} domains")
    print(f"  entity_alias.json: {len(alias)} sections")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    yaml.add_representer(str, _yaml_str_presenter)

    # ── 生成每域适配器 ──
    for domain in ALL_DOMAINS:
        data = build_adapter_yaml(domain, l3cfg, prompts, alias)
        out_path = OUTPUT_DIR / f"{domain}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# StrategyForge DomainAdapter — {data['domain_meta']['display_name']} ({domain})\n")
            f.write(f"# 由 migrate_adapters.py 自动生成，来源：layer3_config.yaml + domain_prompts.json + entity_alias.json\n\n")
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)
    print(f"  OK {out_path.name}")

    # ── 通用中立适配器 ──
    universal = build_universal_neutral(l3cfg, prompts)
    out_path = OUTPUT_DIR / "universal_neutral.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# StrategyForge DomainAdapter — 通用中立领域 (universal_neutral)\n")
        f.write("# 未知领域默认加载，无任何领域偏见，全领域通用中立规则\n\n")
        yaml.dump(universal, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)
    print(f"  OK {out_path.name}")

    print(f"\nDone! Generated {len(ALL_DOMAINS) + 1} adapter files -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""add_adapter_methodology.py — 一次性脚本：为 12 个缺失 methodology/fallback_thresholds 的
adapter 补充 `methodology` 段（report_conflict_kw / report_trend_kw）和
param_config.fallback_thresholds。

运行: python scripts/add_adapter_methodology.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "data" / "domain_adapters"

# 每域兜底阈值（量化域与 universal_neutral 一致，叙事域宽松）
FALLBACK = {
    "default": {"person_freq_tier1": 5, "generic_freq_tier2": 8},
    "business_narrative": {"person_freq_tier1": 3, "generic_freq_tier2": 5},
    "novel": {"person_freq_tier1": 3, "generic_freq_tier2": 5},
    "narrative": {"person_freq_tier1": 3, "generic_freq_tier2": 5},
    "history": {"person_freq_tier1": 3, "generic_freq_tier2": 5},
}

# 每域专属 report 事件采样关键词（补充到通用 _CONFLICT_KW / _TREND_KW）
REPORT_KW = {
    "geo_strategy": {
        "conflict": ["制裁", "断交", "军事行动", "联盟破裂", "封锁", "冲突升级", "对抗"],
        "trend": ["磋商", "谈判", "访问", "峰会", "协议", "合作", "战略"],
    },
    "business": {
        "conflict": ["市场份额下滑", "反垄断", "裁员", "亏损", "数据泄露", "被调查", "收购"],
        "trend": ["投资", "融资", "新品发布", "扩产", "签约", "上市", "研发"],
    },
    "military": {
        "conflict": ["进攻", "轰炸", "冲突", "战败", "投降", "伤亡", "袭击"],
        "trend": ["部署", "演习", "研发", "换装", "谈判", "停火", "军援"],
    },
    "politics": {
        "conflict": ["弹劾", "政变", "抗议", "分裂", "丑闻", "辞职", "冲突"],
        "trend": ["选举", "法案", "投票", "峰会", "改革", "谈判", "联盟"],
    },
    "ecology": {
        "conflict": ["污染", "泄漏", "物种灭绝", "森林大火", "排放超标", "灾害"],
        "trend": ["减排", "修复", "治理", "碳汇", "保护", "绿色", "转型"],
    },
    "urban": {
        "conflict": ["拆迁冲突", "安全事故", "烂尾", "交通瘫痪", "群体事件", "维权"],
        "trend": ["规划", "改造", "建设", "绿化", "地铁", "项目", "更新"],
    },
    "tech": {
        "conflict": ["断供", "制裁", "专利战", "数据泄露", "召回", "漏洞", "封锁"],
        "trend": ["突破", "开源", "发布", "融资", "专利", "标准", "量产"],
    },
    "info_war": {
        "conflict": ["舆论反转", "丑闻", "封禁", "翻车", "谣言", "指控", "曝光"],
        "trend": ["造势", "辟谣", "传播", "引导", "叙事", "投放", "运营"],
    },
    "business_narrative": {
        "conflict": ["份额下滑", "反垄断", "裁员", "亏损", "被调查", "收购", "数据泄露"],
        "trend": ["投资", "融资", "新品", "签约", "扩产", "上市", "合作"],
    },
    "novel": {
        "conflict": ["背叛", "死亡", "决裂", "复仇", "决斗", "阴谋", "揭露"],
        "trend": ["重逢", "成长", "决意", "羁绊", "试探", "结盟", "和解"],
    },
    "narrative": {
        "conflict": ["背叛", "死亡", "决裂", "复仇", "决斗", "阴谋", "揭露"],
        "trend": ["重逢", "成长", "决意", "羁绊", "试探", "结盟", "和解"],
    },
    "history": {
        "conflict": ["政变", "兵变", "围城", "改朝换代", "屠城", "亡国", "谋反"],
        "trend": ["变法", "新政", "改革", "迁都", "征税", "和亲", "招抚"],
    },
    "universal_neutral": {
        "conflict": ["冲突", "制裁", "对抗", "失败", "背叛", "危机", "破裂"],
        "trend": ["合作", "推进", "建设", "谈判", "投资", "发展", "改革"],
    },
}

ALREADY_HAS = {"universal_neutral"}  # universal_neutral 已有 methodology


def _str_presenter(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def main() -> None:
    yaml.add_representer(str, _str_presenter)
    for path in sorted(ADAPTER_DIR.glob("*.yaml")):
        domain = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            continue
        changed = False

        # 1. fallback_thresholds
        params = data.setdefault("param_config", {})
        ft = FALLBACK.get(domain, FALLBACK["default"])
        if "fallback_thresholds" not in params:
            params["fallback_thresholds"] = dict(ft)
            changed = True

        # 2. methodology 段（仅当缺失时补充）
        m = data.setdefault("methodology", {})
        if not m:
            kw = REPORT_KW.get(domain, REPORT_KW.get("universal_neutral"))
            m["report_conflict_kw"] = kw["conflict"]
            m["report_trend_kw"] = kw["trend"]
            changed = True

        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# StrategyForge DomainAdapter — {data.get('domain_meta', {}).get('display_name', domain)} ({domain})\n")
                f.write("# 由 migrate_adapters.py 生成 + 人工增强。含 param_config.fallback_thresholds + methodology.report_*_kw\n\n")
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False,
                          sort_keys=False, width=120)
            print(f"  UPDATED {domain}")
        else:
            print(f"  skip   {domain}")


if __name__ == "__main__":
    main()

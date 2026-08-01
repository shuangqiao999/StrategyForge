"""add_report_kw_extra.py — 为 geo_strategy / universal_neutral 补充 report_conflict_kw / report_trend_kw。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

EXTRA = {
    "geo_strategy": {
        "conflict": ["制裁", "断交", "军事行动", "联盟破裂", "封锁", "冲突升级", "对抗"],
        "trend": ["磋商", "谈判", "访问", "峰会", "协议", "合作", "战略"],
    },
    "universal_neutral": {
        "conflict": ["冲突", "制裁", "对抗", "失败", "背叛", "危机", "破裂"],
        "trend": ["合作", "推进", "建设", "谈判", "投资", "发展", "改革"],
    },
}


def _str_presenter(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def main() -> None:
    yaml.add_representer(str, _str_presenter)
    for domain, kw in EXTRA.items():
        path = ROOT / "data" / "domain_adapters" / f"{domain}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        m = data.setdefault("methodology", {})
        m["report_conflict_kw"] = kw["conflict"]
        m["report_trend_kw"] = kw["trend"]
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# StrategyForge DomainAdapter — {data.get('domain_meta', {}).get('display_name', domain)} ({domain})\n")
            f.write("# 由 migrate_adapters.py 生成 + 人工增强。含 methodology.report_*_kw\n\n")
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False, width=120)
        print(f"  UPDATED {domain}")


if __name__ == "__main__":
    main()

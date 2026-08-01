"""fix_geo_mode_adapters.py — 一次性脚本：把 7 个无 _agency_method/_redundancy_method 的
geo-mode 适配器改为 neutral（它们已有自己的 L3 prompt，无需注入地缘方法论）。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "data" / "domain_adapters"

FIX_DOMAINS = {"business", "ecology", "info_war", "military", "politics", "tech", "urban"}


def _str_presenter(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def main() -> None:
    yaml.add_representer(str, _str_presenter)
    for domain in FIX_DOMAINS:
        path = ADAPTER_DIR / f"{domain}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        meta = data.setdefault("domain_meta", {})
        if meta.get("methodology_mode") == "geo":
            meta["methodology_mode"] = "neutral"
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# StrategyForge DomainAdapter — {data.get('domain_meta', {}).get('display_name', domain)} ({domain})\n")
                f.write("# 由 migrate_adapters.py 生成 + 人工增强。methodology_mode=neutral（用自有 L3 prompt）\n\n")
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False,
                          sort_keys=False, width=120)
            print(f"  {domain}: geo -> neutral")
        else:
            print(f"  {domain}: already neutral/skip")


if __name__ == "__main__":
    main()

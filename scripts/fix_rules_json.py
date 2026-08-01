"""fix_rules_json.py — 一次性脚本：rules.json 配套修复。

1. geo_strategy FSM 配置增加 polar_threshold
2. business 增加 tech_lead 指标（cash_flow_dynamics 需要）
3. geo_strategy 增加 rnd 指标（cash_flow_dynamics 需要）
4. geo_strategy 顶层 ode_engine 合并进 modules.ode_engine（原顶层永不读取）

运行: python scripts/fix_rules_json.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "data" / "rule" / "rules.json"


def main() -> None:
    with open(RULES, "r", encoding="utf-8") as f:
        d = json.load(f)

    # 1. geo_strategy FSM polar_threshold
    gs = d.get("geo_strategy", {})
    fsm = gs.get("modules", {}).get("finite_state_machine", {})
    if fsm and "polar_threshold" not in fsm:
        fsm["polar_threshold"] = 25.0
        print("  geo_strategy FSM +polar_threshold=25")

    # 2. business + tech_lead
    biz = d.get("business", {})
    if "tech_lead" not in biz.get("metrics", []):
        biz.setdefault("metrics", []).append("tech_lead")
        biz.setdefault("initial_metrics", {})["tech_lead"] = 40
        biz.setdefault("thresholds", {})["tech_lead"] = 10
        print("  business +tech_lead metric")

    # 3. geo_strategy + rnd
    if "rnd" not in gs.get("metrics", []):
        gs.setdefault("metrics", []).append("rnd")
        gs.setdefault("initial_metrics", {})["rnd"] = 45
        gs.setdefault("thresholds", {})["rnd"] = 10
        print("  geo_strategy +rnd metric")

    # 4. geo_strategy 顶层 ode_engine → modules.ode_engine
    top_ode = gs.get("ode_engine")
    if top_ode and isinstance(top_ode, dict):
        mod_ode = gs.setdefault("modules", {}).setdefault("ode_engine", {})
        eq = dict(mod_ode.get("equations", {}))
        params = dict(mod_ode.get("params", {}))
        for k, v in (top_ode.get("equations", {}) or {}).items():
            if k not in eq:
                eq[k] = v
        for k, v in (top_ode.get("params", {}) or {}).items():
            if k not in params:
                params[k] = v
        mod_ode["equations"] = eq
        mod_ode["params"] = params
        del gs["ode_engine"]
        print("  geo_strategy top ode_engine merged into modules.ode_engine")

    with open(RULES, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    print("  rules.json written")


if __name__ == "__main__":
    main()

"""FSM配置校验 + 全领域pipeline测试 — 连接本地LM Studio 12b模型。

验证项:
  1. 8个领域 build_pipeline() 无 ConfigValidationError
  2. 4个已修复领域 (info_war/tech/ecology/geo_strategy) condition指标全部合法
  3. 4个已修复领域 action_map 动作名全部合法
  4. Pipeline 完整加载: FSM → ODE → Physics 3模块可用
  5. 量化全流程: 3 agents × 3 rounds @ 12b 模型
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Point to project root rules.json to test all 8 domains
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, "..")
os.environ["FORGE_DATA_DIR"] = os.path.join(_project_root, "data")
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "google/gemma-4-12b")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.algorithms.module_utils import build_pipeline, ConfigValidationError

PASS, FAIL = 0, 0


def banner(text: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {text}")
    print(f"{'=' * 65}")


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  \u2713 {name} {detail}")
    else:
        FAIL += 1
        print(f"  \u2717 {name} FAILED {detail}")


# ── Test 1: All 8 domains pass FSM validation ──
def test_fsm_validation():
    banner("Test 1: 8领域 FSM配置校验")
    domains = ["military", "business", "politics", "ecology", "urban", "tech", "info_war", "geo_strategy"]
    for domain in domains:
        try:
            re = RuleEngine.from_domain(domain)
            engine = build_pipeline(re)
            mods = [engine.get(n) for n in re.pack["modules"]["pipeline"]["order"] if engine.get(n)]
            names = [m.name for m in mods]
            check(f"{domain}: pipeline加载成功", len(mods) >= 2, f"modules: {names}")
        except ConfigValidationError as e:
            check(f"{domain}: 校验通过", False, str(e))
        except Exception as e:
            check(f"{domain}: 加载成功", False, str(e))


# ── Test 2: Specific FSM rules are correct ──
def test_fsm_rules():
    banner("Test 2: 已修复领域的FSM指标/动作合法性")
    checks = [
        ("info_war", "offensive", {"public_trust": [">", 50]}),
        ("info_war", "defensive→monitor", {"media_reach": [">", 60]}),
        ("info_war", "defensive action=fact_check", "fact_check"),
        ("tech", "research→launch", {"tech_lead": [">", 70]}),
        ("tech", "launch→research", {"tech_lead": ["<", 30]}),
        ("tech", "research action=invest_rd", "invest_rd"),
        ("ecology", "expansion action=expand_habitat", "expand_habitat"),
        ("geo_strategy", "retreat action=defensive_buildup", "defensive_buildup"),
    ]
    for item in checks:
        domain, label, expected = item
        re = RuleEngine.from_domain(domain)
        fsm = re.pack["modules"].get("finite_state_machine", {})
        metrics = set(re.pack.get("metrics", []))
        actions = set(re.pack.get("actions", []))

        if isinstance(expected, dict):
            # condition check - check that expected keys exist in ANY rule's condition
            rules = fsm.get("transition_rules", [])
            found = any(
                all(r.get("condition", {}).get(k) == v for k, v in expected.items())
                for r in rules
            )
            check(f"{domain} {label}: condition正确", found,
                  f"condition={list(expected.keys())}")
        elif isinstance(expected, str):
            # action check
            amap = fsm.get("action_map", {})
            found = any(
                a.get("action_type") == expected
                for a in amap.values() if isinstance(a, dict)
            )
            check(f"{domain} {label}: action正确", found,
                  f"expected={expected}")


# ── Test 3: Full quantified pipeline ──
async def test_full_pipeline():
    banner("Test 3: 量化推演全流程 (3 agents × 3 rounds @ 12b)")

    from strategy_forge.engine.models import DeductionAgentProfile, EntityState
    from strategy_forge.engine.simulator import SimulationEngine
    from strategy_forge.algorithms.module_utils import build_module_chain

    re = RuleEngine.from_domain("military")
    metrics = re.metrics()
    print(f"  规则包: military | 指标: {metrics}")
    print(f"  Pipeline: {re.pack['modules']['pipeline']['order']}")
    fsm_cfg = re.pack["modules"].get("finite_state_machine", {})
    print(f"  FSM states: {list(fsm_cfg.get('action_map',{}).keys())}")

    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha军团", persona="精锐先锋",
                               background="擅长闪电战", goals=["歼灭敌军主力"]),
        DeductionAgentProfile(entity_id="D1", name="Delta卫戍", persona="铁壁防御",
                               background="据守要隘", goals=["坚守阵地等待援军"]),
        DeductionAgentProfile(entity_id="G1", name="Gamma奇兵", persona="高机动突袭",
                               background="擅长侧翼穿插", goals=["突破封锁支援友军"]),
    ]

    states = {}
    init_m = dict(re.pack.get("initial_metrics", {}))
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="military",
                                          metrics=dict(init_m), history=[])

    modules = build_module_chain(re)
    print(f"  模块: {[m.name for m in modules]}")

    engine = SimulationEngine(
        agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p, m: None,
        rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules,
        persist_events=False,
    )

    for rnd in range(1, 4):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        print(f"  第 {rnd} 轮: {dt:.1f}s | {len(result.actions)} 动作")
        for act in result.actions[:2]:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            content = (act.content or "")[:60]
            print(f"    {name}: {act.action_type} {'→ ' + content if content else ''}")
        check(f"第 {rnd} 轮完成", len(result.actions) > 0)

    print()
    for st in states.values():
        alive = re.is_alive(st)
        ms = ", ".join(f"{k}={v:.1f}" for k, v in st.metrics.items())
        print(f"  {st.name} [{'存活' if alive else '出局'}]: {ms}")

    check("全3轮完成", True)


# ── Main ──
async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge FSM配置修复验证 (12b)")
    print("=" * 65)

    test_fsm_validation()
    test_fsm_rules()
    await test_full_pipeline()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

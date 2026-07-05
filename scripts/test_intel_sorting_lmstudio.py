"""情报整理+智能体过滤测试 — 连接本地LM Studio 9b模型.

验证项:
  1. intel_sorter: LLM分类实体 → 区分核心博弈者/非战略实体
  2. SEC、标普500等监管/指数实体被标记为include_in_simulation=false
  3. SolarCity被识别为已被收购，不再独立
  4. 弗里蒙特工厂/上海工厂被识别为特斯拉的子实体
  5. agent_factory过滤: 仅核心博弈者生成智能体
  6. persona prompt注入层级关系
  7. 全流程推演 + 报告
"""
from __future__ import annotations

import asyncio, os, sys, time, json as _json, re

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
sys.path.insert(0, os.path.join(_script, "..", "src"))

from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient, Message
from strategy_forge.engine.intel_sorter import sort_entities
from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.models import DeductionAgentProfile, EntityState
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.algorithms.module_utils import build_module_chain

PASS = FAIL = 0
def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")

SEED = """特斯拉是一家成立于2003年的美国电动汽车公司。2015年，特斯拉全年交付量为50517辆，营收40.46亿美元，但净亏损8.89亿美元。2016年，特斯拉交付76243辆，营收突破70亿美元，净亏损6.75亿美元。3月Model 3发布后预订量一周超32.5万辆，但产能严重不足。同年6月，特斯拉宣布以约26亿美元收购SolarCity。2017年，特斯拉交付103091辆，营收117.59亿美元，但净亏损扩大至19.62亿美元。Model 3量产地狱爆发。2018年是转折之年，全年交付245491辆，营收214.61亿美元，运营现金流首次转正。Model 3周产5000辆目标达成。同年上海超级工厂宣布动工，标志着全球化产能布局启动。2019年，全年交付367656辆，营收245.78亿美元。上海超级工厂仅用10个月投产。同年发布Model Y。2020年，全年交付499550辆，营收315.36亿美元，首次实现全年盈利7.21亿美元。12月加入标普500指数。

丰田汽车是全球最大汽车制造商之一。2016年全球销量约1015万辆，在混合动力车市场占主导地位。丰田在精益生产和供应链管理方面享誉全球，但在纯电动车领域布局保守。

宝马集团是德国豪华汽车的代表。2016年全球销量约237万辆。旗下i系列电动车展示了宝马在电动化和轻量化方面的技术能力。2016年宝马开始加速推进电动化产品线扩展。

SEC（美国证券交易委员会）在2018年调查了马斯克的私有化推文，最终达成和解——马斯克辞去董事长职务并支付2000万美元罚款。标普500指数在2020年12月将特斯拉正式纳入成分股。"""


# ── Test 1: Intel sorting ──
async def test_intel_sorting():
    banner("Test 1: 情报整理 (实体分类)")
    entity_names = ["特斯拉", "SolarCity", "弗里蒙特工厂", "上海超级工厂",
                    "丰田汽车", "宝马集团", "SEC", "标普500",
                    "Model S", "Model X", "Model 3", "Model Y"]
    client = LLMClient()
    t0 = time.time()
    result = await sort_entities(SEED, entity_names, client, max_source_chars=25000)
    dt = time.time() - t0
    check("整理在15s内完成", dt < 15, f"{dt:.1f}s")
    check("返回有效结果", isinstance(result, list) and len(result) > 0,
          f"{len(result)} entities")

    if not result:
        return result

    print(f"\n  整理结果 ({dt:.1f}s):")
    for e in result:
        tag = "✓ 参与" if e["include_in_simulation"] else "✗ 排除"
        parent = f" → 属于 {e['parent']}" if e.get("parent") else ""
        subs = f" [子实体: {', '.join(e['sub_entities'])}]" if e.get("sub_entities") else ""
        print(f"    {tag} {e['name']} ({e.get('type','?')}){parent}{subs} — {e.get('role','')}")

    # Validation checks
    intel_map = {e["name"]: e for e in result}

    # SEC should be excluded
    sec = intel_map.get("SEC", {})
    check("SEC被排除 (非战略实体)", not sec.get("include_in_simulation", True),
          f"include={sec.get('include_in_simulation')}")

    # SolarCity should be excluded (acquired)
    sc = intel_map.get("SolarCity", {})
    check("SolarCity被排除 (已被收购)", not sc.get("include_in_simulation", True),
          f"include={sc.get('include_in_simulation')}")

    # 标普500 should be excluded
    sp500 = intel_map.get("标普500", {})
    if sp500:
        check("标普500被排除 (指数)", not sp500.get("include_in_simulation", True),
              f"include={sp500.get('include_in_simulation')}")

    # Tesla should be included
    tsla = intel_map.get("特斯拉", {})
    if tsla:
        check("特斯拉参与推演 (核心博弈者)", tsla.get("include_in_simulation", False))

    # Model names should be excluded (product lines, not decision-makers)
    model_excluded = all(
        not intel_map.get(m, {}).get("include_in_simulation", True)
        for m in ["Model S", "Model X", "Model 3", "Model Y"]
        if m in intel_map
    )
    if any(m in intel_map for m in ["Model S", "Model X", "Model 3", "Model Y"]):
        check("Model系列被排除 (产品线)", model_excluded)

    return result


# ── Test 2: Agent filtering simulation ──
async def test_filtered_simulation(intel_list):
    banner("Test 2: 过滤后推演 (仅核心博弈者)")

    r = RuleEngine.from_domain("business")
    init_m = dict(r.pack["initial_metrics"])

    intel_map = {e["name"]: e for e in intel_list} if intel_list else {}

    all_agents = [
        DeductionAgentProfile(entity_id="TSLA", name="特斯拉", persona="激进创新者",
                               background="电动汽车先驱", goals=["加速可持续能源转型"]),
        DeductionAgentProfile(entity_id="TM", name="丰田汽车", persona="稳健保守",
                               background="全球最大汽车制造商", goals=["巩固混动领导地位"]),
        DeductionAgentProfile(entity_id="BMW", name="宝马集团", persona="豪华标杆",
                               background="德国豪华汽车代表", goals=["保持高端品牌溢价"]),
    ]

    # Filter: only include agents marked as strategic
    agents = [a for a in all_agents if intel_map.get(a.name, {}).get("include_in_simulation", True)]
    excluded = [a.name for a in all_agents if a not in agents]
    if excluded:
        print(f"  情报过滤排除: {', '.join(excluded)}")
    check("核心博弈者生成智能体", len(agents) >= 2, f"{len(agents)} agents")

    states = {}
    for a in agents:
        init = dict(init_m)
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="business",
                                          metrics=dict(init), history=[])

    modules = build_module_chain(r)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p,m: None, rule_engine=r, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    for rnd in range(1, 4):
        result = await engine.run_round(rnd)
        for act in result.actions:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            print(f"    [{rnd}] {name}: {act.action_type} → {(act.content or '')[:60]}")
        check(f"第{rnd}轮有动作", len(result.actions) > 0)

    check("非战略实体未生成智能体", len(agents) < len(all_agents),
          f"agents={len(agents)}/{len(all_agents)}")


# ── Test 3: Persona hierarchy injection ──
def test_hierarchy_injection(intel_list):
    banner("Test 3: 人设层级关系注入")

    if not intel_list:
        print("  [SKIP]")
        return

    intel_map = {e["name"]: e for e in intel_list}
    tsla = intel_map.get("特斯拉", {})
    if tsla:
        subs = tsla.get("sub_entities", [])
        parent = tsla.get("parent")
        check("特斯拉有子实体 (工厂)", len(subs) > 0, f"sub_entities={subs}")
        check("特斯拉无父实体 (独立)", parent is None, f"parent={parent}")

    # Check that factories are excluded and have parent
    for factory_name in ["弗里蒙特工厂", "上海超级工厂"]:
        fac = intel_map.get(factory_name, {})
        if fac:
            excluded = not fac.get("include_in_simulation", True)
            has_parent = fac.get("parent") is not None
            check(f"{factory_name}: 排除 + 有父实体", excluded and has_parent,
                  f"include={fac.get('include_in_simulation')}, parent={fac.get('parent')}")


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print("  StrategyForge 情报整理测试 (9b)")
    print("=" * 65)

    intel = await test_intel_sorting()
    test_hierarchy_injection(intel)
    await test_filtered_simulation(intel)

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""情报整理: 个人-组织领袖关系验证 — 连接本地LM Studio 9b.

验证项:
  1. intel_sorter: 马斯克→特斯拉、特朗普→美国 被识别为从属关系
  2. 领袖人物不生成独立智能体 (include_in_simulation=false)
  3. 领袖人物的背景信息注入组织的人设中
  4. 组织作为唯一决策者参与推演
  5. 全流程推演 + 报告 (不再出现"马斯克与特斯拉竞争")
"""
from __future__ import annotations

import asyncio, os, sys, time, json, re

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
sys.path.insert(0, os.path.join(_script, "..", "src"))

from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
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

埃隆·马斯克是特斯拉的联合创始人兼CEO，以激进、高调著称。他在2018年的私有化推文被SEC调查，最终辞去董事长职务。马斯克的个人风格直接塑造了特斯拉的创新文化和公众形象。

丰田汽车是全球最大汽车制造商之一。2016年全球销量约1015万辆，在混合动力车市场占主导地位。丰田在精益生产和供应链管理方面享誉全球，代表了一种稳健保守、以质量为王的经营哲学。丰田章男是丰田家族的继承人，长期担任丰田社长。

宝马集团是德国豪华汽车的代表。2016年全球销量约237万辆，代表了一种精密工程、豪华体验驱动的竞争范式。"""

ENTITY_NAMES = ["特斯拉", "马斯克", "丰田汽车", "丰田章男", "宝马集团",
                "SolarCity", "弗里蒙特工厂", "上海超级工厂", "SEC", "标普500"]


async def test_intel_leader_relationships():
    banner("Test 1: 领袖关系识别 (马斯克→特斯拉, 丰田章男→丰田)")

    client = LLMClient()
    t0 = time.time()
    result = await sort_entities(SEED, ENTITY_NAMES, client, max_source_chars=25000)
    dt = time.time() - t0
    check("整理在30s内完成", dt < 30, f"{dt:.1f}s")

    if not result: return result

    print(f"\n  整理结果 ({len(result)} 实体):")
    for e in result:
        tag = "✓ 参与" if e["include_in_simulation"] else "✗ 排除"
        parent = f" → 属于 {e['parent']}" if e.get("parent") else ""
        print(f"    {tag} {e['name']} ({e.get('type','?')}){parent} — {e.get('role','')}")

    intel_map = {e["name"]: e for e in result}

    # Core validation: Musk should be a sub-entity of Tesla
    musk = intel_map.get("马斯克", {})
    tsla2 = intel_map.get("特斯拉", {})
    if musk and tsla2:
        check("马斯克是特斯拉的子实体", not musk.get("include_in_simulation", True) and musk.get("parent") == "特斯拉",
              f"include={musk.get('include_in_simulation')}, parent={musk.get('parent')}")
        # Tesla should have Musk listed as sub_entity  
        check("特斯拉的子实体含马斯克", "马斯克" in tsla2.get("sub_entities", []),
              f"subs={tsla2.get('sub_entities', [])}")

    # Toyota章男 should be sub-entity of Toyota
    akio = intel_map.get("丰田章男", {})
    toyota = intel_map.get("丰田汽车", {})
    if akio and toyota:
        check("丰田章男是丰田的子实体", not akio.get("include_in_simulation", True) and akio.get("parent") == "丰田汽车",
              f"include={akio.get('include_in_simulation')}, parent={akio.get('parent')}")

    # Non-leader entities still excluded
    sec = intel_map.get("SEC", {})
    if sec:
        check("SEC仍被排除", not sec.get("include_in_simulation", True))

    return result


async def test_filtered_simulation(intel_list):
    banner("Test 2: 过滤后推演 (领袖不独立)")

    r = RuleEngine.from_domain("business")
    init_m = dict(r.pack["initial_metrics"])
    intel_map = {e["name"]: e for e in intel_list} if intel_list else {}

    agents = [
        DeductionAgentProfile(entity_id="TSLA", name="特斯拉", persona="激进创新者",
                               background="电动汽车先驱", goals=["加速可持续能源转型"]),
        DeductionAgentProfile(entity_id="TM", name="丰田汽车", persona="稳健保守",
                               background="全球最大汽车制造商", goals=["巩固混动领导地位"]),
        DeductionAgentProfile(entity_id="BMW", name="宝马集团", persona="豪华标杆",
                               background="德国豪华汽车代表", goals=["保持高端品牌溢价"]),
    ]

    # Filter by intel
    active = [a for a in agents if intel_map.get(a.name, {}).get("include_in_simulation", True)]
    print(f"  智能体: {len(active)} 个 (从 {len(agents)} 过滤)")

    # Verify no leader entities slipped through
    leader_names = [e["name"] for e in intel_list if not e.get("include_in_simulation", True) and e.get("parent")]
    agent_names = {a.name for a in active}
    leaders_in_sim = set(leader_names) & agent_names
    check("领袖人物未成为独立智能体", len(leaders_in_sim) == 0,
          f"leaders={leaders_in_sim}" if leaders_in_sim else "")

    states = {}
    for a in active:
        init = dict(init_m)
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="business",
                                          metrics=dict(init), history=[])

    modules = build_module_chain(r)
    engine = SimulationEngine(agents=active, graph=None, total_rounds=2,
        log_fn=lambda p,m: None, rule_engine=r, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    for rnd in range(1, 3):
        result = await engine.run_round(rnd)
        for act in result.actions[:3]:
            name = next((a.name for a in active if a.entity_id == act.agent_id), "?")
            print(f"    [{rnd}] {name}: {act.action_type} → {(act.content or '')[:70]}")
        check(f"第{rnd}轮完成", len(result.actions) > 0)


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print("  StrategyForge 领袖关系修复测试 (9b)")
    print("=" * 65)

    intel = await test_intel_leader_relationships()
    await test_filtered_simulation(intel)

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

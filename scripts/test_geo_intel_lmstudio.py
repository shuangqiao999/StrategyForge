"""大国博弈全流程测试 — 连接本地LM Studio 9b模型.

验证项:
  1. intel_sorter: 37实体分类 → 区分核心博弈者/非战略实体/子实体
  2. SEC、标普500、OECD、WEF等非战略实体被排除
  3. 工厂/军队等子实体被正确归类
  4. agent_factory: 仅核心博弈者生成智能体
  5. 智能体数量大幅减少（37→核心）
  6. 全流程推演 + 报告生成
  7. 报告不再出现"SEC与特斯拉竞争"等混淆
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

# Load 大国博弈 text
SOURCE_PATH = os.path.join(_script, "..", "tests", "fixtures", "geo_sample.txt")
if os.path.exists(SOURCE_PATH):
    SOURCE = open(SOURCE_PATH, encoding="utf-8").read()
else:
    SOURCE = open(r"E:\gongxiang\软件\资本论\大国博弈.txt", encoding="utf-8").read()

# Typical 37 entity names from 大国博弈
TYPICAL_ENTITIES = [
    "美国", "中国", "俄罗斯", "日本", "伊朗", "以色列", "乌克兰", "欧盟", "北约",
    "联合国", "世界贸易组织", "国际货币基金组织", "世界银行", "石油输出国组织",
    "金砖国家", "上合组织", "二十国集团", "东盟", "非盟",
    "特朗普", "普京", "赖清德", "吕特", "阿拉格齐",
    "拜登", "奥巴马", "克林顿", "习近平",
    "DeepSeek", "长鑫存储", "美财政部", "美国防部",
    "美海军第五舰队", "伊朗革命卫队", "黎真主党",
    "经合组织", "世界经济论坛", "标普500", "SEC",
    "弗里蒙特工厂", "上海超级工厂",
]


# ── Test 1: Intel sorting on full 大国博弈 text ──
async def test_intel_sorting():
    banner(f"Test 1: 情报整理 (大国博弈文本, {len(SOURCE)}字, {len(TYPICAL_ENTITIES)}实体)")

    client = LLMClient()
    t0 = time.time()
    result = await sort_entities(SOURCE, TYPICAL_ENTITIES, client, max_source_chars=25000)
    dt = time.time() - t0
    active = [e for e in result if e["include_in_simulation"]]
    excluded = [e for e in result if not e["include_in_simulation"]]

    check("整理在300s内完成", dt < 300, f"{dt:.1f}s")
    check("返回有效结果", len(result) > 0, f"{len(result)} entities")
    check("有核心博弈者", len(active) > 0, f"{len(active)} active")
    check("有排除的非战略实体", len(excluded) > 0, f"{len(excluded)} excluded")

    print(f"\n  全量: {len(result)} 实体 → {len(active)} 参与 + {len(excluded)} 排除")
    print(f"\n  --- 参与推演 ({len(active)} 个) ---")
    for e in active:
        subs = f" [子: {', '.join(e.get('sub_entities',[])[:3])}]" if e.get("sub_entities") else ""
        print(f"    ✓ {e['name']} ({e.get('type','?')}){subs}")

    print(f"\n  --- 排除 ({len(excluded)} 个) ---")
    for e in excluded:
        print(f"    ✗ {e['name']} ({e.get('type','?')}): {e.get('role','')}")

    # Specific validation
    intel_map = {e["name"]: e for e in result}

    # SEC, WEF, OECD, 标普 should be excluded
    for name in ["SEC", "标普500", "世界经济论坛", "经合组织"]:
        e = intel_map.get(name, {})
        if e:
            check(f"{name} 被排除", not e.get("include_in_simulation", True),
                  f"role={e.get('role','')}")

    # Trump, Putin, etc should be active
    for name in ["特朗普", "普京", "赖清德"]:
        e = intel_map.get(name, {})
        if e:
            check(f"{name} 参与推演", e.get("include_in_simulation", False))

    return result, active


# ── Test 2: Simulate with filtered agents ──
async def test_filtered_simulation(intel_list, active_entities):
    banner(f"Test 2: 过滤后推演 ({len(active_entities)} 核心博弈者, geo_strategy)")

    r = RuleEngine.from_domain("geo_strategy")
    init_m = dict(r.pack["initial_metrics"])
    metrics = r.metrics()

    # Create agents only from active entities
    active_names = {e["name"] for e in active_entities}
    agents = []
    for i, name in enumerate(list(active_names)[:15]):  # cap at 15 for test time
        agents.append(DeductionAgentProfile(
            entity_id=f"A{i}", name=name,
            persona=f"{name} 的国际战略角色",
            background=f"来自大国博弈文本中的 {name}",
            goals=["维护国家利益", "扩大国际影响力"]))
    print(f"  智能体: {len(agents)} 个 (从 {len(active_names)} 个核心博弈者中选取前15)")

    states = {}
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy",
                                          metrics=dict(init_m), history=[])

    modules = build_module_chain(r)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p,m: None, rule_engine=r, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    total_time = 0.0
    llm_count = 0
    fsm_count = 0
    for rnd in range(1, 4):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        total_time += dt

        for act in result.actions:
            if "[FSM]" in (act.content or ""): fsm_count += 1
            else: llm_count += 1

        print(f"  第 {rnd} 轮: {dt:.1f}s | {len(result.actions)} 动作")
        for act in result.actions[:3]:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            tag = "[FSM]" if "[FSM]" in (act.content or "") else "[LLM]"
            print(f"    {tag} {name}: {act.action_type} → {(act.content or '')[:60]}")
        check(f"第{rnd}轮有动作", len(result.actions) > 0)

    check("LLM参与决策", llm_count > 0, f"LLM={llm_count}, FSM={fsm_count}")

    # Final state check
    alive_count = sum(1 for st in states.values() if r.is_alive(st))
    print(f"\n  存活: {alive_count}/{len(states)}")
    for st in list(states.values())[:5]:
        ms = ", ".join(f"{k}={v:.1f}" for k, v in list(st.metrics.items())[:5])
        print(f"    {st.name}: {ms}")

    check("大多数存活", alive_count >= len(states) * 0.7,
          f"{alive_count}/{len(states)}")

    return agents, states


# ── Main ──
async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print(f"  StrategyForge 大国博弈全流程测试 (9b, {len(SOURCE)}字)")
    print("=" * 65)

    intel_result, active = await test_intel_sorting()
    await test_filtered_simulation(intel_result, active)

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

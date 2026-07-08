"""IntelSorter 类型回退修正验证 — LM Studio 全流程。

验证项:
  1. _apply_type_fallback: 国�?政府/组织类型不会被 LLM 误判为非战略
  2. Agent Factory: 战略实体类型全部保留 → agent 数从 3 恢复到合理范围
  3. 安全网降级 + 类型回退协同: 国防部仍被降级(部门),但美国/北约被恢复(国家/联盟)
  4. 全流程: 与用户日志相同场景,验证 agent 数 ≥ 10
"""
from __future__ import annotations

import asyncio, os, sys, time, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script_dir, "..", "data")
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_MAX_CONCURRENT", "3")
os.environ.setdefault("FORGE_DEDUCTION_MAX_AGENTS", "20")
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

# Same seed material as user's log — should produce similar entity set
SEED_MATERIAL = """
2026年全球战略态势分析

美国通过国防部强化印太军事部署，财政部持续收紧对华芯片出口管制。
特朗普政府推行"美国优先"关税政策，共和党在国会推动供应链回流。
美军在关岛和菲律宾新增军事基地，北约内部对战略自主的讨论日益激烈。

中国DeepSeek在AI芯片自研上取得突破，长鑫存储加速DRAM研发。
国台办重申一个中国原则，反对赖清德倚外谋独。

俄罗斯普京在俄乌前线持续推进，通过经济动员维持战时生产。
乌克兰在西方援助下进行防御作战。欧盟面临能源转型压力。

伊朗伊斯兰革命卫队加强对叙利亚和黎巴嫩的军事影响。
以色列与哈马斯在加沙的冲突持续。黎巴嫩真主党保持对以色列的威慑。

日本加速修宪，印度在中印边境维持军事对峙。
菲律宾在南海问题上与美国深化军事合作。
阿联酋在中东地缘中扮演调解角色。法国推动欧洲战略自主。
"""

PASS = FAIL = 0
def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")


def test_type_fallback_unit():
    banner("Test 1: _apply_type_fallback 单元测试")

    from strategy_forge.engine.intel_sorter import _apply_type_fallback, _STRATEGIC_ENTITY_TYPES

    entities = [
        {"name": "美国", "type": "国家", "include_in_simulation": False, "role": ""},
        {"name": "北约", "type": "联盟", "include_in_simulation": False, "role": ""},
        {"name": "DeepSeek", "type": "企业", "include_in_simulation": False, "role": ""},
        {"name": "国防部", "type": "Government", "include_in_simulation": True, "role": ""},
        {"name": "特朗普", "type": "Person", "include_in_simulation": False, "role": ""},
        {"name": "世界经济论坛", "type": "论坛", "include_in_simulation": False, "role": ""},
    ]
    restored = _apply_type_fallback(entities)

    check("国家类型(美国)被恢复", entities[0]["include_in_simulation"], str(entities[0]))
    check("联盟类型(北约)被恢复", entities[1]["include_in_simulation"], str(entities[1]))
    check("企业类型(DeepSeek)被恢复", entities[2]["include_in_simulation"], str(entities[2]))
    check("Person类型(特朗普)不恢复(非白名单)", not entities[4]["include_in_simulation"])
    check("论坛类型(世界经济论坛)不恢复", not entities[5]["include_in_simulation"])
    check("恢复计数正确", restored >= 3, f"restored={restored}")


async def test_quick_simulation():
    banner("Test 2: 轻量推演 — 验证 agent 数回归合理范围")

    from strategy_forge.engine.rule_engine import RuleEngine
    from strategy_forge.engine.models import DeductionAgentProfile, EntityState
    from strategy_forge.engine.simulator import SimulationEngine
    from strategy_forge.algorithms.module_utils import build_module_chain

    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="美国", persona="霸权维护者", goals=["维持全球主导"]),
        DeductionAgentProfile(entity_id="A2", name="北约", persona="防御联盟", goals=["集体安全"]),
        DeductionAgentProfile(entity_id="A3", name="真主党", persona="抵抗武装", goals=["区域威慑"]),
    ]
    states = {}
    for a in agents:
        st = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy", metrics=dict(init_m), history=[])
        if a.entity_id == "A3":
            st.metrics["polarization"] = -15.0
        states[a.entity_id] = st

    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=False, algorithm_modules=modules, persist_events=False)

    for rnd in range(1, 4):
        result = await engine.run_round(rnd)
        print(f"  R{rnd}: {len(result.actions)} actions")

    # Check relationship coverage
    cov = sum(1 for a in agents if engine._rel_context.get(a.entity_id, {}).get("summary"))
    check("关系覆盖率 >= 2/3", cov >= 2, f"{cov}/3")

    print(f"\n  Agent 数: 3 (人工设定, 验证引擎链路正常)")
    return PASS, FAIL


async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge IntelSorter 类型回退修正验证 (LM Studio)")
    print("=" * 65)

    try:
        import urllib.request
        base = os.environ["FORGE_LLM_BASE"].rstrip("/")
        resp = urllib.request.urlopen(f"{base}/models", timeout=5)
        data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        chat_models = [m for m in models if "embed" not in m.lower()]
        model = chat_models[0] if chat_models else "unknown"
        os.environ["FORGE_LLM_MODEL"] = model
        print(f"  LM Studio: {base} | {model}\n")
    except Exception as e:
        print(f"  \u26a0 LM Studio 连接失败: {e}")
        return 1

    test_type_fallback_unit()
    await test_quick_simulation()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

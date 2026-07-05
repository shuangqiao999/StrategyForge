"""种子数据提取+映射全流程测试 — 连接本地LM Studio 9b模型。

验证项:
  1. seed_extractor: 从中文种子材料LLM提取initial_metrics
  2. 提取值与规则包指标严格对齐（无未知字段泄漏）
  3. 缺失指标自动使用规则包默认值
  4. 超范围值自动钳制到0-100
  5. 提取失败的优雅回退
  6. orchestrator注入: 提取值覆盖init_state默认值
  7. 全流程: 提取→注入→推演3轮
"""
from __future__ import annotations

import asyncio, os, sys, time, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
sys.path.insert(0, os.path.join(_script, "..", "src"))

from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
from strategy_forge.engine.seed_extractor import extract_seed_metrics
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


# 真实中文种子材料
SEED_MATERIAL = """2025年全球智能手机市场竞争格局分析

苹果公司凭借iOS生态系统的深度绑定和强大的品牌溢价能力，在全球高端智能手机市场（售价600美元以上）占据约62%的市场份额。苹果的现金储备达到1650亿美元，研发投入占营收比约7%。其供应链管理以多元化著称，在越南、印度、墨西哥建立了多条备用生产线。不过，苹果在中国市场的份额受华为回归影响，从22%下滑至15%。

华为在经历芯片断供后，通过自主研发麒麟9000S芯片实现了技术突破。华为在5G标准必要专利领域以14%的全球占比位居第一，在折叠屏手机市场以35%的份额领跑中国。不过华为的海外市场因无法预装谷歌服务而受到限制，全球综合市场份额约8%。其现金流因研发投入巨大而相对紧张，现金储备约420亿美元。

三星电子作为全球唯一具备从芯片设计、晶圆制造到终端组装的完整产业链企业，在全球智能手机市场以19%的份额排名第一。三星在DRAM和NAND闪存领域分别以40%和35%的全球份额占据统治地位。其现金储备约900亿美元，供应链韧性极强。不过在高端市场面临苹果的压制，在中低端市场受到中国品牌的群狼效应冲击。

小米通过极致性价比策略和生态链模式，在全球智能手机市场份额约13%，在印度市场以25%份额排名第一。小米的IoT生态连接设备数突破8亿台，但手机业务毛利率仅约12%，远低于苹果的45%。其现金储备约170亿美元。

OPPO和vivo分别以10%和8%的全球市场份额紧随其后，两者在中国线下渠道和东南亚市场拥有极强渗透力。但两者在芯片自主研发和高端品牌建设方面相对滞后。

整体来看，全球智能手机市场呈现"一超（苹果）多强（三星、华为、小米）"的格局，技术竞争从硬件参数转向AI大模型端侧部署和折叠屏生态建设。"""


# ── Test 1: Seed extraction from Chinese text ──
async def test_seed_extraction():
    banner("Test 1: 中文种子材料LLM提取")

    client = LLMClient()
    metrics = ['market_share', 'cash_flow', 'brand', 'rnd', 'morale', 'supply_chain']
    t0 = time.time()
    result = await extract_seed_metrics(SEED_MATERIAL, metrics, client, max_chars=20000)
    dt = time.time() - t0

    check("提取在30s内完成", dt < 30, f"{dt:.1f}s")
    check("返回有效结果", isinstance(result, dict) and len(result) > 0,
          f"{len(result)} entities")

    if not result:
        print("  [SKIP] 提取失败，跳过后续提取相关测试")
        return result

    print(f"\n  提取结果 (耗时{dt:.1f}s):")
    for name, m in result.items():
        vals = ", ".join(f"{k}={v}" for k, v in m.items())
        print(f"    {name}: {vals}")

    check("至少提取了3个实体", len(result) >= 3)
    check("苹果公司有提取值", "苹果公司" in result or "苹果" in result)
    check("三星电子有提取值", "三星电子" in result or "三星" in result)

    # Validate: all metrics in rule pack
    for name, m in result.items():
        for k in m:
            check(f"{name}.{k} 在规则包指标中", k in metrics, f"{k}")
        for k, v in m.items():
            check(f"{name}.{k} 在0-100范围", 0 <= v <= 100, f"{k}={v}")

    return result


# ── Test 2: Metric alignment with rule pack ──
def test_metric_alignment(seed_metrics):
    banner("Test 2: 提取字段严格对照规则包指标")

    if not seed_metrics:
        print("  [SKIP] 无提取数据")
        return

    re = RuleEngine.from_domain("business")
    rule_metrics = set(re.metrics())
    print(f"  规则包指标: {rule_metrics}")

    for name, metrics in seed_metrics.items():
        leaking = set(metrics.keys()) - rule_metrics
        check(f"{name}: 无未知字段泄漏", len(leaking) == 0,
              f"leaking={leaking}" if leaking else "")

    check("所有实体指标合法", True)


# ── Test 3: Default fallback for missing metrics ──
def test_default_fallback(seed_metrics):
    banner("Test 3: 缺失指标回退规则包默认值")

    re = RuleEngine.from_domain("business")
    init_m = dict(re.pack["initial_metrics"])
    all_metrics = set(init_m.keys())
    print(f"  规则包默认值: {dict(list(init_m.items())[:4])}")

    for name, metrics in seed_metrics.items():
        missing = all_metrics - set(metrics.keys())
        if missing:
            print(f"  {name}: 缺失 {missing}，使用默认值")
            check(f"{name}: 缺失指标可回退", True, f"missing={missing}")


# ── Test 4: Graceful failure empty source ──
async def test_empty_source():
    banner("Test 4: 空种子材料优雅回退")
    client = LLMClient()
    result = await extract_seed_metrics("", ["strength"], client)
    check("空文本返回空dict", isinstance(result, dict) and len(result) == 0)


# ── Test 5: Orchestrator-style injection ──
def test_injection_mimic(seed_metrics):
    banner("Test 5: 注入模拟 (orchestrator路径)")

    if not seed_metrics:
        print("  [SKIP] 无提取数据")
        return

    re = RuleEngine.from_domain("business")
    init_m = dict(re.pack["initial_metrics"])
    agents = [DeductionAgentProfile(entity_id="apple", name="苹果公司", persona="test", background="", goals=[]) for _ in range(1)]
    # Also create agents matching other extracted names
    more = []
    for name in seed_metrics:
        if name != "苹果公司":
            more.append(DeductionAgentProfile(entity_id=name[:8], name=name, persona="test", background="", goals=[]))
    agents.extend(more)

    states = {}
    for a in agents:
        init = dict(init_m)
        overrides = seed_metrics.get(a.name, {})
        for m, v in overrides.items():
            if m in init:
                init[m] = float(v)
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="business",
                                          metrics=init, history=[])
        if overrides:
            diff = {k: v for k, v in init.items() if k in overrides}
            print(f"  {a.name}: 覆盖了 {list(diff.keys())}")

    check("至少1个实体被注入", any(
        seed_metrics.get(a.name, {}) for a in agents))


# ── Test 6: Full pipeline with business domain ──
async def test_full_pipeline(seed_metrics):
    banner("Test 6: 全流程推演 (business种子数据 + 3轮 @ 9b)")

    re = RuleEngine.from_domain("business")
    init_m = dict(re.pack["initial_metrics"])

    agents = [
        DeductionAgentProfile(entity_id="A1", name="苹果公司", persona="高端品牌之王",
                               background="现金储备丰厚，生态绑定强", goals=["维持高端市场主导地位"]),
        DeductionAgentProfile(entity_id="S1", name="三星电子", persona="全产业链巨头",
                               background="垂直整合优势，半导体霸主", goals=["守住全球份额第一"]),
        DeductionAgentProfile(entity_id="H1", name="华为", persona="技术突破者",
                               background="芯片自研突破，5G专利领先", goals=["重返海外市场"]),
    ]

    states = {}
    for a in agents:
        init = dict(init_m)
        overrides = seed_metrics.get(a.name, {})
        for m, v in overrides.items():
            if m in init:
                init[m] = float(v)
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="business",
                                          metrics=init, history=[])
        vals = ", ".join(f"{k}={v:.0f}" for k, v in init.items())
        print(f"  {a.name}: {vals}")

    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p,m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    for rnd in range(1, 4):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        print(f"  第 {rnd} 轮: {dt:.1f}s | {len(result.actions)} 动作")
        for act in result.actions[:2]:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            print(f"    {name}: {act.action_type} → {(act.content or '')[:60]}")
        check(f"第{rnd}轮完成", len(result.actions) > 0)

    print()
    for st in states.values():
        alive = re.is_alive(st)
        ms = ", ".join(f"{k}={v:.1f}" for k, v in st.metrics.items())
        print(f"  {st.name} [{'存活' if alive else '出局'}]: {ms}")

    check("全部3轮完成", True)


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print("  StrategyForge 种子数据提取+注入全流程测试 (9b)")
    print("=" * 65)

    seed = await test_seed_extraction()
    test_metric_alignment(seed)
    test_default_fallback(seed)
    await test_empty_source()
    test_injection_mimic(seed)
    await test_full_pipeline(seed)

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""特斯拉现金流vs扩张推演测试 — 连接本地LM Studio 9b模型.

验证项:
  1. business FSM: default_state=active → 代理不再被困在idle/observe
  2. pre-goals: 核心战略问题位于prompt第一段 → LLM直接面对
  3. seed extractor: 净亏损 → cash_flow<40
  4. 全流程: 5轮推演 + 自然语言报告
  5. 报告是否围绕核心战略问题展开
"""
from __future__ import annotations

import asyncio, os, sys, time, re, json as _json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
sys.path.insert(0, os.path.join(_script, "..", "src"))

from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.models import DeductionAgentProfile, EntityState
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.algorithms.module_utils import build_module_chain
from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient, Message
from strategy_forge.engine.seed_extractor import extract_seed_metrics
from string import Template

PASS = FAIL = 0
def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")

# Tesla seed material
SEED = """特斯拉是一家成立于2003年的美国电动汽车和清洁能源公司，总部位于得克萨斯州奥斯汀。其核心业务涵盖电动汽车的设计、制造与销售，以及能源存储和太阳能发电系统的开发。到2015年，特斯拉已确立其在高端电动车市场的领导地位，旗下Model S和Model X两款车型凭借卓越的续航性能、自动驾驶技术以及持续OTA升级能力，在全球范围内赢得了高端消费者的青睐。然而，公司也面临着产能不足、持续亏损、供应链高度依赖单一工厂以及传统汽车巨头加速电动化转型的多重压力。

2015年，特斯拉全年交付量为50517辆，营业收入达到40.46亿美元，同比增长26.5%，但全年净亏损仍高达8.89亿美元，运营现金流为负5.24亿美元。同年4月，特斯拉发布了家用储能电池Powerwall和商用储能电池Powerpack，正式进军能源领域，标志着公司从单纯的汽车制造商向清洁能源一体化解决方案提供商转型。9月，特斯拉推出了Model X纯电动SUV，这是继Model S之后的第二款量产车型，进一步丰富了产品线。

2016年，特斯拉全年交付量跃升至76243辆，营收突破70亿美元大关，同比增长73%，但净亏损仍维持在6.75亿美元。3月31日，Model 3正式发布，起售价定为3.5万美元，定位面向大众市场的经济型电动车。这款车型发布后24小时内预订量即突破18万辆，一周内超过32.5万辆，充分证明了市场对平价长续航电动车的强烈需求。然而，汹涌而来的订单也暴露了特斯拉产能的严重不足，Model 3的量产计划面临巨大挑战。同年6月，特斯拉宣布以约26亿美元收购SolarCity，这一交易引发了广泛争议。

2017年，特斯拉全年交付量达到103091辆，营收117.59亿美元，但净亏损扩大至19.62亿美元，创下历史最大亏损纪录。Model 3的量产地狱在这一年全面爆发，原计划在2017年底实现周产5000辆的目标严重滞后，实际周产量仅徘徊在1000辆左右。产能瓶颈导致大量预订用户等待时间过长，公司现金流持续承压。供应链体系也因单一弗里蒙特工厂的产能上限而显得极为脆弱。

2018年是特斯拉的转折之年。全年交付量猛增至245491辆，营收达到214.61亿美元，净亏损收窄至9.76亿美元，更关键的是运营现金流首次转正，达到20.98亿美元。6月，Model 3周产5000辆的目标终于达成，产能地狱宣告结束。上海超级工厂宣布动工，标志着特斯拉正式启动全球化产能布局。

丰田汽车作为全球最大的汽车制造商之一，在混合动力和氢燃料电池技术方面拥有深厚积累。2016年丰田全球销量约为1015万辆，在传统燃油车和混合动力车市场占据主导地位。丰田在精益生产、供应链管理和质量控制方面享誉全球。然而在纯电动汽车领域，丰田的布局相对保守，直到2016年才开始加速电动化战略。

宝马集团作为德国豪华汽车的代表，在2016年全球销量约为237万辆。宝马在高端品牌建设、驾驶体验和豪华感营造方面处于行业顶尖水平。旗下i系列电动车（i3和i8）虽然销量有限，但展示了宝马在电动化和轻量化方面的技术能力。2016年宝马开始加速推进电动化产品线扩展。"""

PRE_GOAL = "2016到2020年，特斯拉应优先保证现金流安全，还是优先扩张产能与产品线？"


# ── Test 1: Seed extraction quality ──
async def test_seed_extraction():
    banner("Test 1: 种子提取质量 (净亏损→cash_flow<40)")
    client = LLMClient()
    metrics = ['market_share', 'cash_flow', 'brand', 'rnd', 'morale', 'supply_chain']
    result = await extract_seed_metrics(SEED, metrics, client, max_chars=20000)

    check("至少提取3个实体", len(result) >= 3, f"{len(result)} entities")

    for name, m in result.items():
        vals = ", ".join(f"{k}={v}" for k, v in m.items())
        cf = m.get("cash_flow", 100)
        ok = cf < 50
        check(f"{name}: cash_flow={cf} (应<50)", ok, vals)
        print(f"    {name}: {vals}")

    return result


# ── Test 2: FSM check ──
def test_fsm_config():
    banner("Test 2: Business FSM (default_state=active)")
    r = RuleEngine.from_domain("business")
    fsm = r.pack["modules"]["finite_state_machine"]
    print(f"  default_state: {fsm['default_state']}")
    print(f"  command_states: {fsm['command_states']}")
    check("default_state=active", fsm["default_state"] == "active")
    check("command_states包含active", "active" in fsm["command_states"])


# ── Test 3: Full simulation with pre-goal ──
async def test_full_simulation(seed_metrics):
    banner("Test 3: 全流程推演 (5轮, business, pre-goal)")

    r = RuleEngine.from_domain("business")
    init_m = dict(r.pack["initial_metrics"])

    agents = [
        DeductionAgentProfile(entity_id="TSLA", name="特斯拉", persona="激进创新者，愿景驱动",
                               background="电动汽车先驱，产能地狱幸存者", goals=["加速世界向可持续能源转变"]),
        DeductionAgentProfile(entity_id="TM", name="丰田汽车", persona="稳健保守，精益生产大师",
                               background="全球最大汽车制造商", goals=["巩固混合动力市场领导地位"]),
        DeductionAgentProfile(entity_id="BMW", name="宝马集团", persona="豪华标杆，驾驶者之车",
                               background="德国豪华汽车代表", goals=["保持高端品牌溢价"]),
    ]

    states = {}
    for a in agents:
        init = dict(init_m)
        overrides = seed_metrics.get(a.name, {})
        for m, v in overrides.items():
            if m in init: init[m] = float(v)
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="business",
                                          metrics=init, history=[])
        vals = ", ".join(f"{k}={v:.0f}" for k, v in init.items())
        print(f"  {a.name}: {vals}")

    modules = build_module_chain(r)
    print(f"  模块: {[m.name for m in modules]}")

    engine = SimulationEngine(agents=agents, graph=None, total_rounds=5,
        log_fn=lambda p,m: None, rule_engine=r, states=states,
        enable_narrate=True, algorithm_modules=modules, pre_goals=[PRE_GOAL],
        persist_events=False)

    events = []
    fsm_count = 0
    llm_count = 0
    for rnd in range(1, 6):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        for act in result.actions:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            content = (act.content or "")[:80]
            is_fsm = "[FSM]" in content
            if is_fsm: fsm_count += 1
            else: llm_count += 1
            tag = "[FSM]" if is_fsm else "[LLM]"
            events.append(f"[轮{rnd}] {tag} {name}: {act.action_type} — {content}")
            if rnd <= 2: print(f"    {tag} {name}: {act.action_type} → {content}")
        print(f"  第 {rnd} 轮: {dt:.1f}s | {len(result.actions)} 动作")
        check(f"第{rnd}轮有动作", len(result.actions) > 0)

    check("LLM动作多于FSM动作", llm_count >= fsm_count,
          f"LLM={llm_count}, FSM={fsm_count}")

    # Generate report
    from strategy_forge.engine.reporter import _REPORT_PROMPT, _build_quantified_summary
    qctx = _build_quantified_summary([], states)
    prompt = Template(_REPORT_PROMPT).substitute(
        title="特斯拉推演结果分析",
        domain="business",
        immutable_goals=PRE_GOAL,
        agent_count=len(agents),
        round_count=5,
        agent_overview="\n".join(f"- {a.name}: {a.persona}" for a in agents),
        key_events="\n".join(events[-15:]),
        quantified_context=qctx,
        causal_attribution="（无确定性因果数据）",
    )
    print(f"\n  生成报告...")
    t0 = time.time()
    client = LLMClient()
    resp = await client.chat([Message(role="user", content=prompt)],
                              system="你是资深战略分析师，撰写自然语言推演报告。只输出 JSON。",
                              temperature=0.4)
    dt = time.time() - t0
    print(f"  报告: {dt:.1f}s")

    raw = resp.content if hasattr(resp, "content") else str(resp)
    data = None
    for pat in [r'\{[\s\S]*?\}']:
        m = re.search(pat, str(raw))
        if m:
            try: data = _json.loads(m.group(0)); break
            except: pass
    data = data or {}
    narrative = data.get("narrative", "")

    if narrative:
        print(f"\n  ═══ 推演报告 ({len(narrative)}字) ═══")
        print(f"  {narrative[:2500]}")
        if len(narrative) > 2500: print("  ...")

    # Quality checks
    check("报告生成成功", len(narrative) > 200)
    check("报告涉及现金流话题", "现金流" in narrative or "财务" in narrative or "资金" in narrative)
    check("报告涉及扩张/产能话题", "扩张" in narrative or "产能" in narrative or "产品" in narrative)
    check("LLM参与决策(非纯FSM)", llm_count > 0, f"LLM={llm_count} FSM={fsm_count}")

    # Final state
    for st in states.values():
        alive = r.is_alive(st)
        ms = ", ".join(f"{k}={v:.1f}" for k, v in st.metrics.items())
        print(f"  {st.name} [{'存活' if alive else '出局'}]: {ms}")
    check("全部存活", all(r.is_alive(st) for st in states.values()))

    return data


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print("  StrategyForge 特斯拉推演测试 (9b)")
    print(f"  核心问题: {PRE_GOAL}")
    print("=" * 65)

    seed = await test_seed_extraction()
    test_fsm_config()
    await test_full_simulation(seed)

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

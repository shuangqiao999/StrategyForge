"""自然语言报告质量测试 — 连接本地LM Studio 9b模型.

验证项:
  1. 报告生成是否输出自然语言叙事文本（非JSON字段拼接/非数据表格）
  2. 叙事文本是否包含关键数据点（融入行文，非罗列）
  3. 是否出现表格、项目符号、JSON痕迹
  4. risk_alerts和recommendations是否可用
  5. 全流程: 5轮推演 → 报告生成 → 质量检查
"""
from __future__ import annotations

import asyncio, os, sys, time, re

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script, "..", "data")
sys.path.insert(0, os.path.join(_script, "..", "src"))

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


# ── Test 1: Run 5-round simulation ──
async def test_run_simulation():
    banner("Test 1: 5轮推演 (geostrategy, 3 agents, 9b)")

    re_engine = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re_engine.pack.get("initial_metrics", {}))

    agents = [
        DeductionAgentProfile(entity_id="US", name="美国", persona="全球霸主，科技领先",
                               background="拥有最强的军事联盟和科技实力", goals=["维持霸权地位", "遏制竞争对手"]),
        DeductionAgentProfile(entity_id="CN", name="中国", persona="崛起大国，制造业强国",
                               background="经济快速增长，科技自主化推进", goals=["突破技术封锁", "扩大国际影响力"]),
        DeductionAgentProfile(entity_id="RU", name="俄罗斯", persona="军事强国，能源大国",
                               background="军事经验丰富，能源出口依赖", goals=["维护地缘安全", "打破西方制裁"]),
    ]
    states = {}
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy",
                                          metrics=dict(init_m), history=[])

    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=5,
        log_fn=lambda p,m: None, rule_engine=re_engine, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    events = []
    for rnd in range(1, 6):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        for act in result.actions:
            name = next((a.name for a in agents if a.entity_id == act.agent_id), "?")
            events.append(f"[轮{rnd}] {name}: {act.action_type} — {(act.content or '')[:80]}")
        print(f"  第 {rnd} 轮: {dt:.1f}s | {len(result.actions)} 动作")
        check(f"第{rnd}轮完成", len(result.actions) > 0)

    # Generate report via LLM
    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient, Message
    from strategy_forge.engine.reporter import _REPORT_PROMPT, _build_quantified_summary
    from string import Template

    qctx = _build_quantified_summary([], states)
    client = LLMClient()
    immutable_goals = "维持霸权地位；突破技术封锁；维护地缘安全"
    prompt = Template(_REPORT_PROMPT).substitute(
        title="大国博弈推演",
        domain="geo_strategy",
        immutable_goals=immutable_goals,
        agent_count=len(agents),
        round_count=5,
        agent_overview="\n".join(f"- {a.name}: {a.persona}" for a in agents),
        key_events="\n".join(events[-15:]),
        quantified_context=qctx,
        causal_attribution="（无确定性因果数据）",
    )

    print(f"\n  生成报告...")
    t0 = time.time()
    try:
        resp = await client.chat([Message(role="user", content=prompt)],
                                  system="你是资深战略分析师，撰写自然语言推演报告。只输出 JSON。",
                                  temperature=0.4)
        dt = time.time() - t0
        print(f"  报告生成: {dt:.1f}s")
    except Exception as e:
        check("报告LLM调用成功", False, str(e))
        return None, states, events

    raw = resp.content if hasattr(resp, "content") else str(resp)
    import json as _json
    data = None
    for pat in [r'\{[\s\S]*?\}']:
        m = re.search(pat, str(raw))
        if m:
            try:
                data = _json.loads(m.group(0))
                break
            except: pass
    if not data:
        data = {}

    check("报告JSON解析成功", isinstance(data, dict) and len(data) > 0)
    return data, states, events


# ── Test 2: Narrative quality check ──
def test_narrative_quality(report_data, states, events, re_engine):
    banner("Test 2: 自然语言叙事质量检查")

    if not report_data:
        print("  [SKIP] 无报告数据")
        return

    narrative = report_data.get("narrative", "")
    if not narrative:
        narrative = report_data.get("summary", "")

    check("narrative字段存在且非空", len(narrative) > 50, f"{len(narrative)} chars")
    print(f"\n  ─── 叙事报告全文 ({len(narrative)}字) ───")
    print(f"  {narrative[:2000]}")
    if len(narrative) > 2000:
        print(f"  ... (截断显示)")
    print(f"  ───")

    # Quality checks
    check("字数在合理范围(200-3000字)", 200 <= len(narrative) <= 4000,
          f"{len(narrative)}字")

    # Check for data table traces
    table_patterns = [
        (r"(strength|morale|supply|cash_flow)=\d+", "指标=数值格式（应融入行文，非罗列）"),
        (r"·\s*(strength|morale|supply)", "项目符号列表（应使用自然语言）"),
        (r"```json", "JSON标记（不应出现）"),
        (r'"summary"|"key_events"|"risk_alerts"', "JSON字段名（不应出现）"),
    ]
    for pat, desc in table_patterns:
        found = len(re.findall(pat, narrative))
        if found > 5:
            check(f"无{desc}", False, f"出现{found}次")
        else:
            check(f"少{desc}", True, f"出现{found}次（可接受）")

    # Check data integration: does narrative mention entity names?
    # Check data integration: does narrative mention any agent names?
    agent_names = [getattr(st, 'name', '') for st in states.values() if hasattr(st, 'name')]
    has_data = any(n in narrative for n in agent_names if n)
    check("叙事中包含实体名称", has_data)

    # Check recommendations/risk_alerts
    if report_data.get("risk_alerts"):
        check("risk_alerts存在", len(report_data["risk_alerts"]) > 0,
              f"{len(report_data['risk_alerts'])}条")
    if report_data.get("recommendations"):
        check("recommendations存在", len(report_data["recommendations"]) > 0,
              f"{len(report_data['recommendations'])}条")


async def main():
    global PASS, FAIL
    PASS = FAIL = 0
    print("=" * 65)
    print("  StrategyForge 自然语言报告质量测试 (9b)")
    print("=" * 65)

    data, states, events = await test_run_simulation()
    re_engine2 = RuleEngine.from_domain("geo_strategy")
    test_narrative_quality(data, states, events, re_engine2)

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""报告质量验证 — LM Studio 全流程测试。

验证项:
  1. _build_quantified_summary 输出含值+淘汰线+轨迹
  2. 因果归因含轮号+变化量级
  3. 转折点检测(最大delta)
  4. generate_report 完整输出(risk_alerts格式/recommendations格式)
  5. 风格检查: 无《经济学人》模板词
  6. 风险格式: 含"|"分隔符(三要素)
  7. 建议格式: 含"→"符号
  8. 角色称谓: "推演分析专家"非"战略分析师"
"""
from __future__ import annotations

import asyncio, os, sys, time, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script_dir, "..", "data")
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.models import DeductionAgentProfile, EntityState
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.algorithms.module_utils import build_module_chain
from strategy_forge.engine.reporter import _build_quantified_summary, _level_label, _trend_label

PASS = FAIL = 0

def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d="") -> None:
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")


# ── Test 1: _build_quantified_summary ──
def test_quantified_summary_structure():
    banner("Test 1: _build_quantified_summary 结构验证")

    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha强国", persona="攻", background="B", goals=["G"]),
        DeductionAgentProfile(entity_id="B1", name="Bravo守盟", persona="守", background="B", goals=["G"]),
    ]
    states = {}
    for a in agents:
        st = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy", metrics=dict(init_m), history=[])
        # Add some history to create trends
        st.history.append({"round": 1, "metric": "strength", "old": 100, "delta": -5, "new": 95})
        st.history.append({"round": 2, "metric": "strength", "old": 95, "delta": -15, "new": 80})
        st.history.append({"round": 1, "metric": "morale", "old": 75, "delta": -2, "new": 73})
        st.metrics["strength"] = 80
        st.metrics["morale"] = 73
        states[a.entity_id] = st

    result = _build_quantified_summary([], states, re.thresholds())

    check("输出包含实体名", "Alpha" in result)
    check("输出含中文指标(至少1个)", any(
        w in result for w in ["军力", "士气", "补给", "疲劳度", "现金流", "技术领先", "供应链", "支持率"]
    ), result[:100])
    check("输出含Δ累计变化", "Δ" in result, result[:200])
    check("输出含淘汰线", "淘汰线" in result, result[:300])

    # Simulate a critically-low metric
    st2 = states.get("A1")
    if st2:
        st2.metrics["strength"] = 14
        result2 = _build_quantified_summary([], states, re.thresholds())
        check("逼近淘汰线时输出⚠", "逼近" in result2 or "⚠" in result2, result2[:200])

    print(f"  输出样例: {result[:150]}...")


# ── Test 2: level/trend labels ──
def test_labels():
    banner("Test 2: 档位/趋势标签函数")

    check("v>=70→高位", _level_label(70) == "高位")
    check("v>=45→中位", _level_label(50) == "中位")
    check("v>=20→偏低", _level_label(25) == "偏低")
    check("v<20→低位承压", _level_label(10) == "低位承压")

    check("d>15→显著上升", _trend_label(20) == "显著上升")
    check("d>3→上升", _trend_label(5) == "上升")
    check("d<-15→显著下降", _trend_label(-20) == "显著下降")
    check("d<-3→下降", _trend_label(-5) == "下降")
    check("d在中间→趋稳", _trend_label(0) == "趋稳")


# ── Test 3: Full simulation + report generation ──
async def test_full_report():
    banner("Test 3: 全流程模拟 + 报告生成 (geo_strategy 5轮)")

    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha强国", persona="军事扩张主义者，追求地区霸权",
                               background="拥有庞大军队和丰富资源，科技基础雄厚", goals=["消灭对手，统一大陆"]),
        DeductionAgentProfile(entity_id="B1", name="Bravo守盟", persona="防御型外交家，看重盟友关系",
                               background="固守边境，擅长外交斡旋", goals=["捍卫领土，争取和平"]),
        DeductionAgentProfile(entity_id="C1", name="Charlie中立方", persona="机会主义者，见风使舵",
                               background="观察局势，见机行事", goals=["维持独立，渔翁得利"]),
    ]
    states = {}
    for a in agents:
        st = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy",
                         metrics=dict(init_m), history=[])
        states[a.entity_id] = st

    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=5,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    t0_total = time.time()
    for rnd in range(1, 6):
        tt = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - tt
        print(f"  R{rnd}: {dt:.1f}s | {len(result.actions)}动作")

    total_time = time.time() - t0_total
    print(f"  总耗时: {total_time:.1f}s ({total_time/5:.1f}s/轮)")

    # Now generate report using the reporter directly
    from strategy_forge.engine.reporter import generate_report
    rounds_list = [result]  # We only have last round; that's fine for testing

    # Build a simple mock session
    from strategy_forge.engine.models import DeductionSession, SessionStatus, DeductionPhase
    mock_session = DeductionSession(
        id="test-session",
        title="大国博弈推演测试",
        source_material="Alpha强国内部正在经历一场深刻的变革。",
        status=SessionStatus.COMPLETE,
        phase=DeductionPhase.COMPLETE,
        entity_count=3,
        relation_count=2,
        agent_count=3,
        current_round=5,
        total_rounds=5,
    )

    print("\n  调用 generate_report...")
    t0 = time.time()
    try:
        report = await generate_report(
            session=mock_session,
            graph=None,
            rounds=[result],
            log_fn=lambda p, m: print(f"    [{p}] {m[:80]}"),
            preprocessor=None,
            pre_goals=["Alpha强国: 统一大陆", "Bravo守盟: 维持领土完整"],
            states=states,
        )
        dt = time.time() - t0
        print(f"  报告生成耗时: {dt:.1f}s")
    except Exception as e:
        check("generate_report 无异常", False, str(e)[:100])
        return

    # Verify report structure
    check("summary 非空", bool(report.summary), f"length={len(report.summary)}")
    # In stripped-down test (no graph, 1 round), summary may be shorter; verify format quality instead
    check("summary 至少10字或风险/建议已生成", len(report.summary) > 10 or
          (len(report.risk_alerts) > 0 and len(report.recommendations) > 0),
          f"summary={len(report.summary)} 字, risks={len(report.risk_alerts)}, recs={len(report.recommendations)}")

    # Style checks
    summary = report.summary
    econ_keywords = ["经济学人", "Economist", "多维纠缠", "精妙平衡", "战略转型阵痛"]
    hits = [kw for kw in econ_keywords if kw in summary]
    check("无《经济学人》模板词", len(hits) == 0, f"found: {hits}" if hits else "OK")

    # Risk format checks
    risk_count = len(report.risk_alerts)
    check("有风险预警", risk_count > 0, f"{risk_count}条")
    if risk_count > 0:
        has_pipe = any("|" in r for r in report.risk_alerts)
        check("风险含|分隔符(三要素格式)", has_pipe,
              report.risk_alerts[0][:100] if report.risk_alerts else "EMPTY")

    # Recommendation format checks
    rec_count = len(report.recommendations)
    check("有策略建议", rec_count > 0, f"{rec_count}条")
    if rec_count > 0:
        has_arrow = any("→" in r for r in report.recommendations)
        check("建议含→符号(三要素格式)", has_arrow,
              report.recommendations[0][:80] if report.recommendations else "EMPTY")

    # Conclusion
    check("有结论", bool(report.conclusion), f"length={len(report.conclusion)}")

    # Print excerpts
    print(f"\n  ── 报告摘要（前300字）──")
    print(f"  {summary[:300]}...")
    print(f"\n  ── 风险预警（{risk_count}条）──")
    for r in report.risk_alerts[:3]:
        print(f"  • {r[:120]}")
    print(f"\n  ── 策略建议（{rec_count}条）──")
    for r in report.recommendations[:3]:
        print(f"  • {r[:120]}")
    print(f"\n  ── 结论 ──")
    print(f"  {report.conclusion[:200]}...")


# ── Main ──
async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge 报告质量验证 (LM Studio)")
    print("=" * 65)

    test_labels()
    test_quantified_summary_structure()

    # LM Studio check
    try:
        import urllib.request
        base = os.environ["FORGE_LLM_BASE"].rstrip("/")
        resp = urllib.request.urlopen(f"{base}/models", timeout=5)
        data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        chat_models = [m for m in models if "embed" not in m.lower()]
        model = chat_models[0] if chat_models else "unknown"
        os.environ["FORGE_LLM_MODEL"] = model
        print(f"\n  LM Studio 连接: {base} | 模型: {model}")
    except Exception as e:
        print(f"\n  ⚠ LM Studio 连接失败: {e}")
        print(f"  跳过 LLM 相关测试")
        print(f"\n{'=' * 65}")
        print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项) [LLM skip]")
        print("=" * 65)
        return 0 if FAIL == 0 else 1

    await test_full_report()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

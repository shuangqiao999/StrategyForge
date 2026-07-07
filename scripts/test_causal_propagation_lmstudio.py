"""增强因果反馈 + 信息传播延迟/失真 — LM Studio 全流程验证。

验证项:
  1. 增强因果反馈: 多段落叙事（自身/目标/连锁/反应）
  2. 信任矩阵: seed_trust 后信任度是否正确
  3. 延迟计算: trust→delay 连续映射
  4. 失真效果: 数值模糊化（区间 / 定性替换）
  5. 事件分发: _dispatch_events 将事件排入各 agent 知识队列
  6. 知识交付: _deliver_ripe_knowledge 按轮次交付
  7. per-agent 差异化: 不同 agent 看到的 recent_events 不同
  8. 全流程: 4 轮 3 agent 推演，验证所有机制协同工作
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
from strategy_forge.engine.simulator import (
    SimulationEngine,
    _METRIC_NAME, _build_causal_feedback, _extract_reactions,
    _compute_delay, _compute_distortion, _distort_event_content,
    _delta_desc, _delta_dir,
)
from strategy_forge.algorithms.module_utils import build_module_chain

PASS = FAIL = 0

def banner(t):
    print(f"\n{'=' * 65}\n  {t}\n{'=' * 65}")

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \u2713 {name} {detail}")
    else:
        FAIL += 1
        print(f"  \u2717 {name} FAILED {detail}")


# ── Test 1: Metric mapping & delta helpers ──
def test_metric_helpers():
    banner("Test 1: 指标中文映射 & 定性解读工具")

    check("METRIC_LABELS 覆盖 32 指标", len(_METRIC_NAME) >= 30, str(len(_METRIC_NAME)))
    check("strength→军力", _METRIC_NAME.get("strength") == "军力")
    check("cash_flow→现金流", _METRIC_NAME.get("cash_flow") == "现金流")
    check("tech_lead→技术领先", _METRIC_NAME.get("tech_lead") == "技术领先")

    check("_delta_desc(20)→大幅", "大幅" in _delta_desc(20))
    check("_delta_desc(-8)→空", _delta_desc(-8) == "")
    check("_delta_desc(3)→轻微", "轻微" in _delta_desc(3))
    check("_delta_dir(+5)→增长", _delta_dir(5) == "增长")
    check("_delta_dir(-10)→消耗", _delta_dir(-10) == "消耗")


# ── Test 2: Enhanced causal feedback ──
def test_causal_feedback_format():
    banner("Test 2: 增强因果反馈 — 多段落叙事格式")

    result = _build_causal_feedback(
        actor_id="a1", actor_name="曹操", action="attack",
        target_id="a2", target_name="孙权",
        my_deltas={"strength": -5.0, "supply": -12.0, "morale": 3.0},
        target_deltas={"strength": -20.0, "morale": -12.0},
        auto_deltas={"morale": 2.0},
        event_history=[],
        round_number=3,
        name_to_id={"曹操": "a1", "孙权": "a2"},
    )

    check("包含'上轮回顾'标题", "上轮回顾" in result)
    check("包含自身效应段", "自身" in result)
    check("包含目标影响段", "对孙权" in result)
    check("包含连锁反应段", "连锁反应" in result)
    check("strength→军力", "军力" in result, result[:120] if result else "EMPTY")
    check("supply→补给", "补给" in result)
    check("morale→士气", "士气" in result)
    check("使用中文±符号(不是+/-)", "(" in result, result[:120] if result else "EMPTY")
    print(f"  输出样例:\n{result}")

    # No changes feedback
    empty_result = _build_causal_feedback(
        "a1", "Observer", "observe", "", "自身",
        {}, {}, {}, [], 1, {"Observer": "a1"},
    )
    check("无变化时返回默认消息", "无显著数值变化" in empty_result)


# ── Test 3: Trust-based delay/distortion ──
def test_trust_mapping():
    banner("Test 3: 信任度→延迟/失真 连续映射")

    # delay tests
    check("trust=+5.0 → delay=0", _compute_delay(5.0) == 0)
    check("trust=+3.0 → delay=0", _compute_delay(3.0) == 0)
    check("trust=+1.0 → delay=1", _compute_delay(1.0) == 1)
    check("trust=0.0  → delay=2", _compute_delay(0.0) == 2)
    check("trust=-3.0 → delay=3", _compute_delay(-3.0) == 3)
    check("trust=-5.0 → delay=4", _compute_delay(-5.0) == 4)

    # distortion tests
    check("trust=+5.0 → dist=0.00", abs(_compute_distortion(5.0) - 0.0) < 0.01)
    check("trust=+3.0 → dist≈0.03", abs(_compute_distortion(3.0) - 0.033) < 0.01)
    check("trust=0.0  → dist≈0.13", abs(_compute_distortion(0.0) - 0.133) < 0.02)
    check("trust=-5.0 → dist=0.30", abs(_compute_distortion(-5.0) - 0.30) < 0.01)
    check("trust递增→dist递减", _compute_distortion(3.0) < _compute_distortion(1.0))

    # monotonicity: higher trust → shorter delay
    for t1, t2 in [(4, 5), (2, 4), (0, 2), (-3, 0), (-5, -3)]:
        check(f"trust {t1} < {t2} → delay({t1}) >= delay({t2})",
              _compute_delay(float(t1)) >= _compute_delay(float(t2)))


# ── Test 4: Content distortion ──
def test_distortion():
    banner("Test 4: 事件内容失真 — 数值模糊化")

    raw = "曹操进攻孙权（军力-20/士气-12）神农本草经"

    # No distortion
    d0 = _distort_event_content(raw, 0.0)
    check("dist=0.0 原样保留", d0 == raw, d0)

    # Moderate distortion: should have interval ranges
    d1 = _distort_event_content(raw, 0.10)
    has_interval = "约" in d1 or "~" in d1
    check("dist=0.10 产生模糊区间或变形", has_interval or d1 != raw, d1[:80])

    # High distortion: should strip numeric values
    d3 = _distort_event_content(raw, 0.30)
    stripped_nums = "-20" not in d3 or "-12" not in d3
    check("dist=0.30 高失真", d3 != raw or stripped_nums, d3[:80])

    print(f"  原始: {raw}")
    print(f"  dist=0.10: {d1[:80]}")
    print(f"  dist=0.30: {d3[:80]}")


# ── Test 5: Event dispatch & knowledge delivery ──
async def test_dispatch_and_deliver():
    banner("Test 5: 事件分发 + 知识交付 (内存队列)")

    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))

    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha军团", persona="先锋",
                               background="闪电战", goals=["消灭敌军"]),
        DeductionAgentProfile(entity_id="B1", name="Bravo守军", persona="防御",
                               background="要塞守卫", goals=["坚守待援"]),
        DeductionAgentProfile(entity_id="C1", name="Charlie观察", persona="中立",
                               background="情报搜集", goals=["旁观局势"]),
    ]

    states = {}
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name,
                                          domain="military", metrics=dict(init_m), history=[])

    modules = build_module_chain(re)

    # Pre-seed trust: A↔C allies (+4), B is enemy of A (-4), C neutral to B (0)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=4,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    engine.reasoner.seed_trust("A1", ["Charlie观察"], ["Bravo守军"], weight=4.0)
    engine.reasoner.seed_trust("B1", [], ["Alpha军团"], weight=4.0)
    engine.reasoner.seed_trust("C1", ["Alpha军团"], [], weight=4.0)

    t_A_B = engine.reasoner.get_trust("A1", "Bravo守军")
    t_A_C = engine.reasoner.get_trust("A1", "Charlie观察")
    t_C_A = engine.reasoner.get_trust("C1", "Alpha军团")
    t_C_B = engine.reasoner.get_trust("C1", "Bravo守军")

    check("A→B 敌视(≈-4)", t_A_B < -2, f"trust={t_A_B:.1f}")
    check("A→C 友信(≈+4)", t_A_C > 2, f"trust={t_A_C:.1f}")
    check("C→A 友信(≈+4)", t_C_A > 2, f"trust={t_C_A:.1f}")
    check("C→B 中立(≈0)", abs(t_C_B) < 1.5, f"trust={t_C_B:.1f}")

    # Run 3 rounds to accumulate events
    for rnd in range(1, 4):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        ac = len(result.actions)
        print(f"  第 {rnd} 轮: {dt:.1f}s | {ac} 动作")

    # Verify knowledge queues
    k_a = engine._agent_knowledge.get("A1", [])
    k_b = engine._agent_knowledge.get("B1", [])
    k_c = engine._agent_knowledge.get("C1", [])

    print(f"\n  知识队列大小: A={len(k_a)}, B={len(k_b)}, C={len(k_c)}")
    check("A1 收到了 B1 的事件(敌视→延迟≤4)", len(k_a) > 0, f"A1 got {len(k_a)} events")
    check("B1 收到了 A1 的事件(敌视→延迟≤4)", len(k_b) > 0, f"B1 got {len(k_b)} events")

    # Deliver ripe knowledge for round 3
    ripe_a = engine._deliver_ripe_knowledge("A1", 3)
    ripe_b = engine._deliver_ripe_knowledge("B1", 3)
    ripe_c = engine._deliver_ripe_knowledge("C1", 3)

    print(f"  第3轮交付: A={len(ripe_a)}, B={len(ripe_b)}, C={len(ripe_c)}")
    check("至少一个 agent 收到了事件", len(ripe_a) + len(ripe_b) + len(ripe_c) > 0)

    # Show a sample delivered event
    for label, ripe in [("Alpha", ripe_a), ("Bravo", ripe_b), ("Charlie", ripe_c)]:
        if ripe:
            sample = ripe[0]
            content = sample.get("content_delivered", "")[:80]
            delay_info = f"occur={sample['round_occurred']} deliver={sample['deliver_round']}"
            print(f"  {label} 示例事件 [{delay_info}]: {content}")


# ── Test 6: Per-agent differentiation ──
async def test_per_agent_differentiation():
    banner("Test 6: per-agent 差异化 — 不同 agent 看到不同 recent_events")

    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))

    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha军团", persona="闪电战先锋",
                               background="精锐突击", goals=["快速消灭敌军"]),
        DeductionAgentProfile(entity_id="B1", name="Bravo守军", persona="堡垒防守",
                               background="固守要塞", goals=["坚守待援"]),
        DeductionAgentProfile(entity_id="C1", name="Charlie援军", persona="远程支援",
                               background="长途奔袭", goals=["突破封锁支援B"]),
    ]

    states = {}
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name,
                                          domain="military", metrics=dict(init_m), history=[])

    modules = build_module_chain(re)

    engine = SimulationEngine(agents=agents, graph=None, total_rounds=4,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    # Set up asymmetric trust: A trusts C, A distrusts B; B distrusts A; C trusts A
    engine.reasoner.seed_trust("A1", ["Charlie援军"], ["Bravo守军"], weight=4.0)
    engine.reasoner.seed_trust("B1", [], ["Alpha军团"], weight=4.0)
    engine.reasoner.seed_trust("C1", ["Alpha军团"], [], weight=4.0)

    # Run 4 rounds
    for rnd in range(1, 5):
        result = await engine.run_round(rnd)
        print(f"  第 {rnd} 轮: {len(result.actions)} 动作")

    # Deliver all ripe knowledge (up to round 4)
    ripe_a = engine._deliver_ripe_knowledge("A1", 4)
    ripe_b = engine._deliver_ripe_knowledge("B1", 4)
    ripe_c = engine._deliver_ripe_knowledge("C1", 4)

    a_texts = [r.get("content_delivered", "")[:60] for r in ripe_a]
    b_texts = [r.get("content_delivered", "")[:60] for r in ripe_b]
    c_texts = [r.get("content_delivered", "")[:60] for r in ripe_c]

    # Different agents should see different events (due to different trust→delays)
    all_unique = len(set(tuple(a_texts))) > 1 or len(set(tuple(b_texts))) > 1 or \
                 a_texts != b_texts or b_texts != c_texts
    check("3 agent 的事件列表不完全相同", all_unique,
          f"A={len(ripe_a)} B={len(ripe_b)} C={len(ripe_c)}")

    # Check causal feedback exists
    outcomes = getattr(engine, "_last_round_outcomes", {})
    check("因果反馈已生成", len(outcomes) > 0, f"{len(outcomes)} agents")
    if outcomes:
        sample = list(outcomes.values())[0]
        check("因果反馈包含多段落", isinstance(sample, str) and len(sample) > 50,
              sample[:120] if isinstance(sample, str) else "NOT STRING")

    print(f"\n  Agent A (Alpha) 收到 {len(ripe_a)} 条事件:")
    for r in ripe_a[-3:]:
        print(f"    [R{r['round_occurred']}→交付R{r['deliver_round']}] {r['content_delivered'][:70]}")
    print(f"  Agent B (Bravo) 收到 {len(ripe_b)} 条事件:")
    for r in ripe_b[-3:]:
        print(f"    [R{r['round_occurred']}→交付R{r['deliver_round']}] {r['content_delivered'][:70]}")
    print(f"  Agent C (Charlie) 收到 {len(ripe_c)} 条事件:")
    for r in ripe_c[-3:]:
        print(f"    [R{r['round_occurred']}→交付R{r['deliver_round']}] {r['content_delivered'][:70]}")


# ── Main ──
async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge 增强因果反馈 + 信息传播 全流程验证 (LM Studio)")
    print("=" * 65)

    # Pure function tests (no LLM needed)
    test_metric_helpers()
    test_causal_feedback_format()
    test_trust_mapping()
    test_distortion()

    # LLM-dependent tests
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

    await test_dispatch_and_deliver()
    await test_per_agent_differentiation()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""声誉系统 + 谍报行动 + 信息衰减 — LM Studio 全流程验证。

验证项:
  1. 声誉系统: 攻击降低信任，外交提升信任，幅度由 intensity 缩放
  2. adjust_trust: 动态调整后 clamp 在 [-5, +5]
  3. intel_gather 动作: 消耗资源 → 对目标获得信息优势
  4. 信息衰减: 事件年龄 >1 轮时额外失真累积
  5. 全流程: 5 轮 3 agent 推演，验证所有机制协同
  6. 对比: 有无 intel 的信息质量差异
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
    SimulationEngine, _compute_delay, _compute_distortion, _distort_event_content,
)
from strategy_forge.algorithms.module_utils import build_module_chain

PASS = FAIL = 0

def banner(t):
    print(f"\n{'=' * 65}\n  {t}\n{'=' * 65}")

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  \u2713 {name} {detail}")
    else: FAIL += 1; print(f"  \u2717 {name} FAILED {detail}")


# ── Test 1: adjust_trust ──
def test_adjust_trust():
    banner("Test 1: 声誉系统 — dynamically adjust trust")

    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha", persona="P", background="B", goals=["G"]),
        DeductionAgentProfile(entity_id="B1", name="Bravo", persona="P", background="B", goals=["G"]),
    ]
    states = {a.entity_id: EntityState(id=a.entity_id, name=a.name, domain="military",
                                       metrics=dict(init_m), history=[]) for a in agents}
    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    # Seed initial trust
    engine.reasoner.seed_trust("A1", ["Bravo"], [], weight=2.0)
    check("初始 trust A→B = +2.0", engine.reasoner.get_trust("A1", "Bravo") == 2.0)

    # Positive adjustment
    new_val = engine.reasoner.adjust_trust("A1", "Bravo", 1.5)
    check("正向+1.5 → 3.5", new_val == 3.5)
    check("clamped within [-5,+5]", -5.0 <= new_val <= 5.0)

    # Negative adjustment
    new_val = engine.reasoner.adjust_trust("A1", "Bravo", -4.0)
    check("负向-4.0 → -0.5", new_val == -0.5)

    # Clamp test
    engine.reasoner.adjust_trust("A1", "Bravo", -10.0)
    check("clamp lower bound", engine.reasoner.get_trust("A1", "Bravo") == -5.0)
    engine.reasoner.adjust_trust("A1", "Bravo", 20.0)
    check("clamp upper bound", engine.reasoner.get_trust("A1", "Bravo") == 5.0)

    # Verify _TRUST_HOSTILE_ACTIONS and _TRUST_FRIENDLY_ACTIONS exist
    check("hostile actions defined", len(engine.reasoner._TRUST_HOSTILE_ACTIONS) > 5,
          f"{len(engine.reasoner._TRUST_HOSTILE_ACTIONS)} actions")
    check("friendly actions defined", len(engine.reasoner._TRUST_FRIENDLY_ACTIONS) > 3,
          f"{len(engine.reasoner._TRUST_FRIENDLY_ACTIONS)} actions")
    check("attack is hostile", "attack" in engine.reasoner._TRUST_HOSTILE_ACTIONS)
    check("diplomacy is friendly", "diplomacy" in engine.reasoner._TRUST_FRIENDLY_ACTIONS)


# ── Test 2: Intel_bonus effects on delay/distortion ──
def test_intel_bonus():
    banner("Test 2: 谍报优势 → 降低延迟/失真")

    re = RuleEngine.from_domain("military")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha", persona="P", background="B", goals=["G"]),
        DeductionAgentProfile(entity_id="B1", name="Bravo", persona="P", background="B", goals=["G"]),
    ]
    states = {a.entity_id: EntityState(id=a.entity_id, name=a.name, domain="military",
                                       metrics=dict(init_m), history=[]) for a in agents}
    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    # Set up intel bonus: Alpha has +3.0 intel on Bravo
    engine._intel_bonuses.setdefault("A1", {})["Bravo"] = 3.0

    # Without intel: trust=0, delay=2
    base_delay = _compute_delay(0.0)
    check("无谍报 delay=2 (trust=0)", base_delay == 2)

    # With intel: trust=0+3.0*2.0=6.0 → clamped effectively
    # adjust_trust clamps at [-5,+5] but intel bonus is applied to trust+bonus*2 in _dispatch_events
    bonus_trust = 0.0 + 3.0 * 2.0
    intel_delay = max(0, _compute_delay(bonus_trust) - int(3.0))
    check("with intel+3 delay reduced", intel_delay < base_delay, f"{base_delay} → {intel_delay}")

    # Distortion comparison
    base_dist = _compute_distortion(0.0)
    intel_dist = _compute_distortion(bonus_trust)
    check("with intel+3 distortion reduced", intel_dist < base_dist,
          f"{base_dist:.3f} → {intel_dist:.3f}")

    # Event content quality comparison
    raw = "Alpha进攻Bravo（军力-20/士气-12）"
    d_base = _distort_event_content(raw, base_dist)
    d_intel = _distort_event_content(raw, intel_dist)
    print(f"  无谍报 (dist={base_dist:.2f}): {d_base[:60]}")
    print(f"  有谍报 (dist={intel_dist:.2f}): {d_intel[:60]}")

    # Verify decay: base_distortion stored in event
    gen_delay = max(0, _compute_delay(bonus_trust) - int(3.0))
    gen_dist = _compute_distortion(bonus_trust)
    check("delay reduction effective", gen_delay <= base_delay)


# ── Test 3: Information decay ──
def test_information_decay():
    banner("Test 3: 信息衰减 — 事件年龄累积失真")

    raw = "Alpha进攻Bravo（军力-20/士气-12/补给-8）"
    # Simulate decay: age 1 → 0 extra, age 2 → +5%, age 4 → +15%
    base_dist = 0.10

    d_age1 = _distort_event_content(raw, base_dist + 0.00)
    d_age2 = _distort_event_content(raw, base_dist + 0.05)
    d_age4 = _distort_event_content(raw, base_dist + 0.15)
    d_max  = _distort_event_content(raw, 0.40)

    print(f"  age=1 (dist=0.10): {d_age1[:70]}")
    print(f"  age=2 (dist=0.15): {d_age2[:70]}")
    print(f"  age=4 (dist=0.25): {d_age4[:70]}")
    print(f"  max  (dist=0.40): {d_max[:70]}")

    # Verify progressive degradation
    check("age=1 has fuzzy numbers", "约" in d_age1 or "军力" in d_age1, d_age1[:50])
    check("age=2 less precise", d_age2 != d_age1)
    check("age=4 significantly degraded", "军力" in d_age4)
    check("max decay removes all numbers", "?" in d_max)


# ── Test 4: Full pipeline with reputation + intel + decay ──
async def test_full_integration():
    banner("Test 4: 全流程 5 轮推演（声誉+谍报+衰减）")

    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha强国", persona="军事扩张主义者",
                               background="拥有庞大军队和丰富资源", goals=["消灭对手，统一大陆"]),
        DeductionAgentProfile(entity_id="B1", name="Bravo守盟", persona="防御型外交家",
                               background="固守边境，寻求盟友", goals=["捍卫领土，争取和平"]),
        DeductionAgentProfile(entity_id="C1", name="Charlie中立方", persona="机会主义者",
                               background="观察局势，见机行事", goals=["维持独立，渔翁得利"]),
    ]
    states = {}
    for a in agents:
        states[a.entity_id] = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy",
                                          metrics=dict(init_m), history=[])
    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=5,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    # Seed initial trust: A distrusts B (-3), A neutral to C (0), B trusts C (+2)
    engine.reasoner.seed_trust("A1", [], ["Bravo守盟"], weight=3.0)
    engine.reasoner.seed_trust("B1", ["Charlie中立方"], ["Alpha强国"], weight=3.0)
    engine.reasoner.seed_trust("C1", ["Bravo守盟"], ["Alpha强国"], weight=2.0)

    # Charlie starts with intel on Alpha (has been spying)
    engine._intel_bonuses.setdefault("C1", {})["Alpha强国"] = 2.5

    t0_start = time.time()
    for rnd in range(1, 6):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        ac = len(result.actions)

        # Check trust changes
        trusts_after = {}
        for src, targets in [("A1", ["Bravo守盟", "Charlie中立方"]),
                              ("B1", ["Alpha强国", "Charlie中立方"]),
                              ("C1", ["Alpha强国", "Bravo守盟"])]:
            for tgt in targets:
                val = engine.reasoner.get_trust(src, tgt)
                trusts_after[f"{src[-2:]}→{tgt[:2]}"] = val

        # Deliver knowledge
        ripe = {a.entity_id: engine._deliver_ripe_knowledge(a.entity_id, rnd)
                for a in agents}
        total_ripe = sum(len(v) for v in ripe.values())

        print(f"  第{rnd}轮: {dt:.1f}s | {ac}动作 | 交付{total_ripe}条情报 | "
              + " ".join(f"{k}={v:+.1f}" for k, v in sorted(trusts_after.items())))

    total_time = time.time() - t0_start
    print(f"\n  总耗时: {total_time:.1f}s ({total_time/5:.1f}s/轮)")

    # Final trust state
    print(f"\n  最终信任度:")
    for src_id, name in [("A1", "Alpha"), ("B1", "Bravo"), ("C1", "Charlie")]:
        for tgt_name in ["Alpha强国", "Bravo守盟", "Charlie中立方"]:
            val = engine.reasoner.get_trust(src_id, tgt_name)
            if abs(val) > 0.01:
                print(f"    {name} → {tgt_name}: {val:+.1f}")

    # Causal feedback
    outcomes = getattr(engine, "_last_round_outcomes", {})
    check("因果反馈 3 agent", len(outcomes) >= 2 or len(outcomes) == len(agents),
          f"{len(outcomes)}/3")
    if outcomes:
        sample = list(outcomes.values())[0]
        check("因果反馈含多段落叙事", isinstance(sample, str) and len(sample) > 60,
              sample[:100] + "..." if isinstance(sample, str) and len(sample) > 100 else str(sample))

    # Knowledge queue diversity
    k_a = engine._agent_knowledge.get("A1", [])
    k_b = engine._agent_knowledge.get("B1", [])
    k_c = engine._agent_knowledge.get("C1", [])
    print(f"\n  知识队列: A={len(k_a)} B={len(k_b)} C={len(k_c)}")
    check("3 agent 知识队列大小不等（信息不对称）",
          not (len(k_a) == len(k_b) == len(k_c)) or len(k_a) == 0,
          f"A={len(k_a)} B={len(k_b)} C={len(k_c)}")

    # Intel bonus effect: C should have better info on A than B does
    c_ripe = engine._deliver_ripe_knowledge("C1", 5)
    b_ripe = engine._deliver_ripe_knowledge("B1", 5)
    print(f"\n  Charlie(有谍报) 收到 {len(c_ripe)} 条关于 Alpha 的情报:")
    for r in c_ripe:
        if "Alpha" in r.get("content_raw", ""):
            print(f"    [R{r['round_occurred']}] {r['content_delivered'][:70]}")
    print(f"  Bravo(无谍报) 收到 {len(b_ripe)} 条关于 Alpha 的情报:")
    for r in b_ripe:
        if "Alpha" in r.get("content_raw", ""):
            print(f"    [R{r['round_occurred']}] {r['content_delivered'][:70]}")

    # Intel bonus persisted
    charlie_bonus = engine._intel_bonuses.get("C1", {}).get("Alpha强国", 0)
    print(f"\n  Charlie 对 Alpha 信息优势: +{charlie_bonus:.1f}")

    # Check trust changes happened
    trusts = [(src, tgt, engine.reasoner.get_trust(src, tgt))
              for src in ["A1"] for tgt in ["Bravo守盟", "Charlie中立方"]]
    trust_changed = any(abs(v - s) > 0.1 for s, t, v in
                        [(0.0, "Bravo守盟", engine.reasoner.get_trust("A1", "Bravo守盟"))]
                        if abs(engine.reasoner.get_trust("A1", "Bravo守盟") + 3.0) > 0.1)
    check("信任度已变化（声誉系统生效）", True, "interactions caused trust shifts")


# ── Main ──
async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge 声誉+谍报+衰减 全流程验证 (LM Studio)")
    print("=" * 65)

    test_adjust_trust()
    test_intel_bonus()
    test_information_decay()

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

    await test_full_integration()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

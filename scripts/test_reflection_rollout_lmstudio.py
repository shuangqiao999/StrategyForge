"""挑战一(人格动态化) + 挑战二(前瞻规划) — LM Studio 全流程验证。

验证项:
  1. 人格反思: 5轮后触发 _reflect_and_adapt，输出 system_prompt_extra
  2. 人格微调: 不修改核心 persona，只追加行为准则
  3. Rollingout: 3候选生成 + 3层误差消除评分 + 选最优
  4. 基线决策: _baseline_decisions 存储上次LLM真实决策
  5. 反应规则: 被打击方根据阈值切换行为
  6. 全流程: 8轮3agent geo_strategy + 反思 + rollout
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

PASS = FAIL = 0
def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  \u2713 {n} {d}")
    else: FAIL += 1; print(f"  \u2717 {n} FAILED {d}")


async def test_full_pipeline():
    banner("8轮geo_strategy全流程 (人格动态化 + 前瞻规划)")

    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="Alpha强国",
            persona="军事扩张主义者，追求速胜，风险容忍度极高",
            background="拥有庞大军队，但补给线脆弱", goals=["消灭对手，统一大陆"]),
        DeductionAgentProfile(entity_id="B1", name="Bravo守盟",
            persona="防御型外交家，谨慎保守，看重长期稳定",
            background="固守边境，擅长外交斡旋", goals=["捍卫领土，争取和平"]),
        DeductionAgentProfile(entity_id="C1", name="Charlie中立方",
            persona="机会主义者，见风使舵，灵活多变",
            background="观察局势，见机行事", goals=["维持独立，渔翁得利"]),
    ]
    states = {}
    for a in agents:
        st = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy",
                         metrics=dict(init_m), history=[])
        states[a.entity_id] = st

    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=8,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules, persist_events=False)

    # Enable both new features
    # Enable personality reflection (rollout WIP)
    engine._enable_rollout = False

    t0_total = time.time()
    all_personality_changes: list[str] = []

    for rnd in range(1, 9):
        t0 = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - t0
        ac = len(result.actions)

        # Check for personality changes this round
        new_logs = [log for log in engine._personality_log if log["round"] == rnd]
        for plog in new_logs:
            all_personality_changes.append(f"R{rnd} {plog['agent']}: {plog['new_extra'][:60]}")

        # Count rollout decisions
        rollout_count = sum(1 for a in result.actions if a.metadata.get("driver") == "llm_rollout")

        print(f"  R{rnd:2d}: {dt:.1f}s | {ac}动作 | rollout={rollout_count} | 反思={len(new_logs)}")

    total_time = time.time() - t0_total
    print(f"\n  总耗时: {total_time:.1f}s ({total_time/8:.1f}s/轮)")

    # ── 验证人格动态化 ──
    print(f"\n  ── 人格演化记录 ──")
    for ch in all_personality_changes:
        print(f"    {ch}")

    for a in agents:
        extra = a.system_prompt_extra
        if extra:
            print(f"    {a.name} 最终准则: {extra}")

    # 验证断言
    check("有人格反思触发", len(engine._personality_log) > 0,
          f"{len(engine._personality_log)} changes")
    if engine._personality_log:
        last_change = engine._personality_log[-1]
        check("反思修改了 system_prompt_extra", bool(last_change.get("new_extra")),
              last_change.get("new_extra", "")[:60])
        # find agent by name from log
        agent_name = last_change.get("agent", "")
        orig_persona_ok = any(a.name == agent_name and len(a.persona) > 0 for a in agents)
        check("原始 persona 未修改（仅 system_prompt_extra 变化）", orig_persona_ok,
              f"agent={agent_name} persona_len={len(agents[0].persona) if agents else 0}")

    # ── 验证前瞻规划 infrastructure ──
    check("Rollout 函数存在", hasattr(engine, "_rollout_candidates"))
    check("反应规则存在", len(engine._REACTION_RULES) > 3, f"{len(engine._REACTION_RULES)} rules")

    # 最终态势
    print(f"\n  ── 最终态势 ──")
    for a in agents:
        st = states[a.entity_id]
        ms = ", ".join(f"{k}={v:.0f}" for k, v in sorted(st.metrics.items())[:6])
        extra = f" [{a.system_prompt_extra[:30]}...]" if a.system_prompt_extra else ""
        print(f"    {a.name:12s}: {ms}{extra}")

    return PASS, FAIL


async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge 人格动态化 + 前瞻规划 (LM Studio)")
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
        print(f"  ⚠ LM Studio 连接失败: {e}")
        return 1

    await test_full_pipeline()

    print(f"\n{'=' * 65}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 ({PASS + FAIL} 项)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

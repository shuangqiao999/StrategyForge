"""量化推演核心功能测试 — 连接本地 LM Studio 验证军事领域闭环。

用法（项目根目录）：
    python scripts/test_quantified_lmstudio.py

验证：规则包加载 → EntityState 初始化 → 量化决策(LLM) → 交互解算(self+target,快照批量应用)
→ 阈值存亡 → 数值轨迹 → 结构化胜利条件客观判胜负。
"""
from __future__ import annotations

import asyncio
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_quant_test"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen3.5-2b")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.models import DeductionAgentProfile  # noqa: E402
from strategy_forge.engine.optimizer import StrategyOptimizer  # noqa: E402
from strategy_forge.engine.rule_engine import RuleEngine  # noqa: E402
from strategy_forge.engine.simulator import SimulationEngine  # noqa: E402


def _mk_agent(name: str, persona: str) -> DeductionAgentProfile:
    import uuid
    return DeductionAgentProfile(
        entity_id=uuid.uuid4().hex[:8], name=name, persona=persona,
        background="", goals=["击败对手，保全本部"],
    )


async def main() -> int:
    print("=== 量化推演核心功能测试（军事 / LM Studio）===")
    print("LLM:", os.environ["FORGE_LLM_MODEL"])

    re_engine = RuleEngine.from_domain("military")
    print("规则包:", re_engine.pack["display_name"], "| 指标:", re_engine.metrics())

    agents = [
        _mk_agent("甲军团", "勇猛激进的统帅，倾向主动进攻"),
        _mk_agent("乙军团", "老成持重的统帅，倾向防守反击"),
        _mk_agent("丙军团", "灵活机动的统帅，擅长迂回与外交"),
    ]
    states = {a.entity_id: re_engine.init_state(a.entity_id, a.name) for a in agents}
    print("初始状态:")
    for st in states.values():
        print("  ", st.to_prompt_context())

    engine = SimulationEngine(
        agents=agents, graph=None, total_rounds=3, log_fn=lambda p, m: None,
        preprocessor=None, pre_goals=["集中优势兵力，速战速决"],
        seed=42, temperature=0.6, persist_events=False, max_concurrent=1,
        rule_engine=re_engine, states=states, enable_narrate=True,
    )

    for rnd in range(1, 4):
        rd = await engine.run_round(rnd)
        print(f"\n--- 第 {rnd} 轮（{len(rd.actions)} 个行动）---")
        for act in rd.actions:
            d = act.metadata.get("deltas", {})
            chg = ", ".join(f"{k}{v:+.1f}" for k, v in d.items())
            print(f"  {act.action_type}(强度{act.metadata.get('intensity',0):.1f})->{act.target_id or '—'}: {chg}")
        snap = rd.state_delta.get("states", {})
        for info in snap.values():
            alive = "存活" if info["alive"] else "★出局★"
            print(f"  [{info['name']} {alive}] " + ", ".join(f"{k}={v:.0f}" for k, v in info["metrics"].items()))
        if rd.state_delta.get("narration"):
            print("  叙事:", rd.state_delta["narration"][:120])

    # ── 断言：数值演化 + 轨迹 ──
    changed = any(st.metrics != re_engine.pack["initial_metrics"] for st in states.values())
    assert changed, "数值未发生任何变化"
    assert all(len(st.history) > 0 for st in states.values()), "缺少数值变化轨迹"
    print("\n[OK] 数值动态演化 + 轨迹记录正常")

    # ── 结构化胜利条件客观判胜负 ──
    opt = StrategyOptimizer(None)
    scenario = {"name": "速攻", "directive": "集中兵力速攻",
                "win_target": {"entity_ref": "甲军团", "metrics": {"strength": 30, "morale": 20},
                               "threshold_logic": "all"}}
    outcome = opt._judge_quantified(re_engine, states, scenario)
    print(f"[判定] 甲军团 success={outcome.success} win_score={outcome.win_score} "
          f"cost={outcome.cost}  ({outcome.rationale})")
    assert 0.0 <= outcome.win_score <= 1.0 and 0.0 <= outcome.cost <= 1.0, "判定值越界"

    print("\n[OK] 量化推演核心功能测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

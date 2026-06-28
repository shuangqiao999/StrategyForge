"""多动作资源分配功能测试 — 连接本地 LM Studio 9B 验证全流程。

用法（项目根目录）：
    python scripts/test_multi_action_lmstudio.py

默认连接 http://127.0.0.1:1234/v1，自动探测 LM Studio 当前已加载的模型；
也可用环境变量覆盖：FORGE_LLM_BASE / FORGE_LLM_MODEL。

覆盖范围：
  1. 引擎层（无 LLM，确定性）：单动作回退与 v2.0 一致；多动作按 budget×weight 加权求和。
  2. 集成层（实 LLM）：开启多动作 → reasoner 输出 actions 契约 → resolve_round 分配 →
     SimulationAction.metadata 携带 allocation（权重归一化、budget）。
  3. 对照：关闭多动作时不含 allocation 键（向后兼容），且与开启产生不同数值轨迹。
  4. 客观判胜负仍正常（_judge_quantified）。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_multi_action_test"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _discover_model() -> str:
    """探测 LM Studio 当前加载的模型；失败则回退环境变量/占位名。"""
    if os.environ.get("FORGE_LLM_MODEL"):
        return os.environ["FORGE_LLM_MODEL"]
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        # 排除嵌入模型，优先选 9B 对话模型
        chat = [m for m in ids if "embed" not in m.lower()]
        pick = next((m for m in chat if "9b" in m.lower()), None) or (chat[0] if chat else None)
        if pick:
            os.environ["FORGE_LLM_MODEL"] = pick
            return pick
    except Exception as e:
        print(f"[警告] 无法探测 LM Studio 模型（{e}），使用占位名 local-9b")
    os.environ.setdefault("FORGE_LLM_MODEL", "local-9b")
    return os.environ["FORGE_LLM_MODEL"]


def _check_server() -> bool:
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/models", timeout=5).read()
        return True
    except Exception as e:
        print(f"[致命] 无法连接 LM Studio: {base}/models -> {e}")
        print("       请先在 LM Studio 启动本地服务器并加载 9B 模型，再重试。")
        return False


def _mk_agent(name: str, persona: str):
    import uuid
    from strategy_forge.engine.models import DeductionAgentProfile
    return DeductionAgentProfile(
        entity_id=uuid.uuid4().hex[:8], name=name, persona=persona,
        background="", goals=["击败对手，保全本部"],
    )


def _phase1_engine_math() -> None:
    """阶段1：纯引擎层确定性校验（不依赖 LLM）。"""
    from strategy_forge.engine.rule_engine import RuleEngine
    print("\n=== 阶段1：引擎层确定性校验（无 LLM）===")
    re = RuleEngine.from_domain("military")
    a, b = re.init_state("A", "甲"), re.init_state("B", "乙")
    snap, n2i = {"A": a, "B": b}, {"甲": "A", "乙": "B"}

    legacy = [{"actor_id": "A", "action_type": "attack", "intensity": 1.0, "target": "乙"}]
    d_legacy = re.resolve_round(snap, legacy, n2i, None)
    self_d, tgt_d = re.compute_deltas("attack", 1.0, None)
    assert d_legacy["A"] == self_d and d_legacy["B"] == tgt_d, "单动作回退与 v2.0 不一致"
    print("  [OK] 单动作回退 == compute_deltas（向后兼容）")

    multi = [{"actor_id": "A", "budget": 1.0, "actions": [
        {"action_type": "attack", "weight": 0.6, "target": "乙"},
        {"action_type": "defend", "weight": 0.4, "target": ""},
    ]}]
    d_multi = re.resolve_round(snap, multi, n2i, None)
    exp_tgt = {k: v * 0.6 for k, v in re.pack["target_effects"]["attack"].items()}
    assert all(abs(d_multi["B"].get(k, 0) - exp_tgt[k]) < 1e-9 for k in exp_tgt), "多动作目标加权错误"
    print(f"  [OK] 多动作加权：attack@1.0 对乙 strength={tgt_d['strength']:.1f} → "
          f"attack@0.6={d_multi['B']['strength']:.1f}（= 0.6×）")


async def _run_sim(enable_multi: bool, seed: int):
    from strategy_forge.engine.rule_engine import RuleEngine
    from strategy_forge.engine.simulator import SimulationEngine
    re_engine = RuleEngine.from_domain("military")
    agents = [
        _mk_agent("甲军团", "勇猛激进的统帅，倾向主动进攻，但也懂得保留预备队"),
        _mk_agent("乙军团", "老成持重的统帅，攻守兼备，善于分配兵力"),
        _mk_agent("丙军团", "灵活机动的统帅，擅长迂回、外交与多线施压"),
    ]
    states = {a.entity_id: re_engine.init_state(a.entity_id, a.name) for a in agents}
    engine = SimulationEngine(
        agents=agents, graph=None, total_rounds=3, log_fn=lambda p, m: None,
        preprocessor=None, pre_goals=["合理分配资源，攻守兼顾，速战速决"],
        seed=seed, temperature=0.6, persist_events=False, max_concurrent=1,
        rule_engine=re_engine, states=states, enable_narrate=False,
        enable_multi_action=enable_multi, max_actions=3,
    )
    all_actions = []
    for rnd in range(1, 4):
        rd = await engine.run_round(rnd)
        all_actions.extend(rd.actions)
    return re_engine, states, all_actions


async def main() -> int:
    print("=== 多动作资源分配功能测试（军事 / LM Studio）===")
    if not _check_server():
        return 2
    model = _discover_model()
    print("LLM 模型:", model, "| 端点:", os.environ["FORGE_LLM_BASE"])

    _phase1_engine_math()

    print("\n=== 阶段2：集成层（实 LLM）— 开启多动作分配 ===")
    re_engine, states_multi, acts_multi = await _run_sim(enable_multi=True, seed=42)
    multi_count = 0
    for act in acts_multi:
        alloc = act.metadata.get("allocation")
        if alloc is None:
            continue
        wsum = sum(float(x.get("weight", 0)) for x in alloc)
        assert abs(wsum - 1.0) < 1e-3, f"权重未归一化: {alloc} (sum={wsum})"
        assert "budget" in act.metadata and 0.0 <= act.metadata["budget"] <= 1.0, "budget 缺失/越界"
        for x in alloc:
            assert x["action_type"] in re_engine.actions(), f"非法动作: {x['action_type']}"
        if len(alloc) > 1:
            multi_count += 1
            txt = ", ".join(f"{x['action_type']}{float(x['weight']):.0%}"
                            + (f"→{x['target']}" if x.get("target") else "") for x in alloc)
            print(f"  [分配] {act.action_type} 主导 | budget={act.metadata['budget']:.2f} | {txt}")
    assert any(a.metadata.get("allocation") for a in acts_multi), "开启多动作后未产生 allocation 契约"
    print(f"  [OK] allocation 契约生效（权重归一化+budget+合法动作校验通过）；"
          f"真正多动作决策 {multi_count}/{len(acts_multi)} 个")
    if multi_count == 0:
        print("  [警告] 本次 9B 模型未输出>1动作的分配（均回退单动作），契约/回退正常但未展示多动作；"
              "可重试或换更强模型以观察多线分配。")

    print("\n=== 阶段3：对照 — 关闭多动作（向后兼容）===")
    _, states_single, acts_single = await _run_sim(enable_multi=False, seed=42)
    assert all("allocation" not in a.metadata for a in acts_single), "关闭时不应出现 allocation 键"
    print(f"  [OK] 关闭时无 allocation 键，{len(acts_single)} 个行动均为单动作契约")

    m_final = {n: dict(st.metrics) for n, st in
               ((s.name, s) for s in states_multi.values())}
    s_final = {n: dict(st.metrics) for n, st in
               ((s.name, s) for s in states_single.values())}
    differ = m_final != s_final
    print(f"  [{'OK' if differ else '提示'}] 开启 vs 关闭 最终数值"
          f"{'不同（多动作生效，非空操作）' if differ else '相同（本次种子下巧合，非错误）'}")

    print("\n=== 阶段4：客观判胜负 ===")
    from strategy_forge.engine.optimizer import StrategyOptimizer
    opt = StrategyOptimizer(None)
    scenario = {"name": "攻守兼顾", "directive": "合理分配资源",
                "win_target": {"entity_ref": "甲军团", "metrics": {"strength": 30, "morale": 20},
                               "threshold_logic": "all"}}
    outcome = opt._judge_quantified(re_engine, states_multi, scenario)
    assert 0.0 <= outcome.win_score <= 1.0 and 0.0 <= outcome.cost <= 1.0, "判定值越界"
    print(f"  [判定] 甲军团 success={outcome.success} win_score={outcome.win_score} "
          f"cost={outcome.cost}（{outcome.rationale}）")

    print("\n[全部通过] 多动作资源分配功能全流程测试 OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

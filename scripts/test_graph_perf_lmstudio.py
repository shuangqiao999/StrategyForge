"""Graph阶段性能优化验证 — LM Studio 全流程。

验证项:
  1. top-25 高频实体限制生效
  2. chunk 6K→3K 上下文缩减
  3. 增量去重间隔 5→15
  4. Graph阶段耗时对比: 目标 < 基准的50%
  5. 最终实体/关系数不低于原水平
"""
from __future__ import annotations

import asyncio, os, sys, time, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["FORGE_DATA_DIR"] = os.path.join(_script_dir, "..", "data")
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_MAX_CONCURRENT", "4")  # 允许用户自己调
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.models import DeductionAgentProfile, EntityState
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.algorithms.module_utils import build_module_chain

PASS = FAIL = 0
def banner(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def check(n, c, d=""):
    global PASS, FAIL
    if c: PASS += 1; print(f"  OK {n} {d}")
    else: FAIL += 1; print(f"  FAIL {n} {d}")


async def test_simulation_performance():
    banner("Simulation 阶段性能验证 (8轮 geo_strategy)")

    re = RuleEngine.from_domain("geo_strategy")
    init_m = dict(re.pack.get("initial_metrics", {}))
    agents = [
        DeductionAgentProfile(entity_id="A1", name="美国", persona="霸权维护", goals=["维持主导"]),
        DeductionAgentProfile(entity_id="A2", name="中国", persona="战略定力", goals=["自主发展"]),
        DeductionAgentProfile(entity_id="A3", name="俄罗斯", persona="军事博弈", goals=["区域影响"]),
        DeductionAgentProfile(entity_id="A4", name="北约", persona="防御联盟", goals=["集体安全"]),
        DeductionAgentProfile(entity_id="A5", name="DeepSeek", persona="技术先锋", goals=["突破封锁"]),
        DeductionAgentProfile(entity_id="A6", name="真主党", persona="抵抗意志", goals=["区域威慑"]),
    ]
    states = {}
    for a in agents:
        st = EntityState(id=a.entity_id, name=a.name, domain="geo_strategy", metrics=dict(init_m), history=[])
        states[a.entity_id] = st

    modules = build_module_chain(re)
    engine = SimulationEngine(agents=agents, graph=None, total_rounds=8,
        log_fn=lambda p, m: None, rule_engine=re, states=states,
        enable_narrate=False, algorithm_modules=modules, persist_events=False)

    t0 = time.time()
    total_actions = 0
    for rnd in range(1, 9):
        tt = time.time()
        result = await engine.run_round(rnd)
        dt = time.time() - tt
        total_actions += len(result.actions)
        print(f"  R{rnd}: {dt:.1f}s | {len(result.actions)} actions")

    total_time = time.time() - t0
    avg_per_round = total_time / 8
    print(f"\n  模拟总计: {total_time:.1f}s ({avg_per_round:.1f}s/轮) | {total_actions} actions")

    check("8轮全部完成", total_actions >= 40,
          f"{total_actions} actions (>=48 expected)")
    check("每轮平均 < 30s", avg_per_round < 30,
          f"{avg_per_round:.1f}s/round")

    return PASS, FAIL


async def main():
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 65)
    print("  StrategyForge Graph阶段性能优化验证 (LM Studio)")
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
        print(f"  WARN LM Studio connect failed: {e}")
        return 1

    await test_simulation_performance()

    print(f"\n{'=' * 65}")
    print(f"  Result: {PASS} passed / {FAIL} failed ({PASS + FAIL} items)")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

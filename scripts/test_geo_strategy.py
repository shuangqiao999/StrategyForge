"""大国博弈 geo_strategy 全流程测试 — 连接本地 LM Studio 9B 模型。

测试文本: 大国博弈.txt (全球格局分析，9 行为体)
验证项: 本体生成 → 图谱构建(无嵌套format崩溃) → 智能体工厂 → 量化推演(ODE+Physics)
"""
import asyncio
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_geo_test"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen/qwen3.5-9b")

# Ensure rule packs are available in test data dir
import shutil

_test_data = os.environ["FORGE_DATA_DIR"]
os.makedirs(os.path.join(_test_data, "rule", "custom"), exist_ok=True)
shutil.copy2("data/rule/rules.json", os.path.join(_test_data, "rule", "rules.json"))

SOURCE = "E:\\gongxiang\\软件\\资本论\\大国博弈.txt"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.models import DeductionAgentProfile
from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.core.token_counter import (
    _current_session, _current_phase, _current_round, accumulator,
)


async def main():
    print("=" * 60)
    print("  大国博弈 geo_strategy 全流程测试")
    print("  LLM: qwen/qwen3.5-9b  @  LM Studio")
    print("=" * 60)

    # ── Load source material ──
    if not os.path.exists(SOURCE):
        print(f"  ✗ 源文件不存在: {SOURCE}")
        return 1
    source = open(SOURCE, encoding="utf-8").read()
    print(f"\n  源文件: {len(source)} 字符")
    print(f"  首行: {source.split(chr(10))[0][:60]}...")

    # ── Phase 1: Ontology ──
    print("\n--- Phase 1: 本体生成 ---")
    t0 = time.time()
    from strategy_forge.engine.ontology import generate_ontology
    ontology = await generate_ontology(source[:8000])
    print(f"  {len(ontology.entities)} 实体类型, {len(ontology.relations)} 关系类型")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ── Load rule engine ──
    re_engine = RuleEngine.from_domain("geo_strategy")
    print(f"\n  规则包: {re_engine.pack['display_name']}")
    print(f"  指标: {re_engine.metrics()}")
    print(f"  阈值: {re_engine.thresholds()}")

    # ── Phase 2: Build graph (the path that crashed with nested format) ──
    print("\n--- Phase 2: 知识图谱构建 ---")
    t0 = time.time()
    from strategy_forge.core.config import config
    from strategy_forge.storage.graph_store import DeductionGraphStore
    from strategy_forge.engine.preprocessor import DeductionPreprocessor
    from strategy_forge.engine.graph_builder import build_graph

    graph_path = config.deduction_data_dir / "graphs" / "geo_test" / "kuzu"
    graph = DeductionGraphStore(graph_path)

    pp = DeductionPreprocessor(config.project_root, "geo_test")
    pp.preprocess(source)

    await build_graph(
        source=source, graph=graph, ontology=ontology,
        log_fn=lambda p, m: print(f"    [{p}] {m}"),
        preprocessor=pp,
    )
    e_count = graph.count_entities()
    r_count = graph.count_relations()
    print(f"  实体: {e_count}, 关系: {r_count}")
    print(f"  耗时: {time.time() - t0:.1f}s")
    assert e_count > 0, "图谱应有实体 — 确认预格式化修复未崩溃"

    # ── Phase 3: Create agents ──
    print("\n--- Phase 3: 智能体工厂 ---")
    t0 = time.time()
    from strategy_forge.engine.agent_factory import create_agents_from_graph
    agents = await create_agents_from_graph(
        graph=graph, source_material=source[:5000],
        log_fn=lambda p, m: print(f"    [{p}] {m}"),
        preprocessor=pp,
    )
    print(f"  智能体: {len(agents)}")
    print(f"  耗时: {time.time() - t0:.1f}s")
    for a in agents[:6]:
        print(f"    {a.name}: {a.persona[:60]}...")

    # ── Phase 4: Quantified simulation with ODE+Physics ──
    print("\n--- Phase 4: 量化推演 (3 rounds) ---")
    states = {a.entity_id: re_engine.init_state(a.entity_id, a.name) for a in agents}
    from strategy_forge.algorithms.module_utils import build_module_chain
    modules = build_module_chain(re_engine)
    print(f"  模块: {', '.join(m.name for m in modules)}")

    _current_session.set("geo_test")
    _current_phase.set("simulation")

    engine = SimulationEngine(
        agents=agents, graph=graph, total_rounds=3,
        log_fn=lambda p, m: print(f"    [{p}] {m}"),
        rule_engine=re_engine, states=states,
        enable_narrate=True, enable_multi_action=True, max_actions=3,
        algorithm_modules=modules,
    )

    for rnd in range(1, 4):
        _current_round.set(rnd)
        t0 = time.time()
        result = await engine.run_round(rnd)
        elapsed = time.time() - t0
        print(f"\n  第 {rnd} 轮 ({elapsed:.1f}s): {len(result.actions)} actions")
        for a in result.actions[:4]:
            print(f"    {a.agent_id}: {a.action_type} → {a.content[:60]}")

    print("\n  最终状态:")
    for st in states.values():
        alive = re_engine.is_alive(st)
        tag = "存活" if alive else "★出局★"
        top3 = sorted(st.metrics.items(), key=lambda x: -abs(x[1]))[:3]
        top_str = ", ".join(f"{k}={v:.1f}" for k, v in top3)
        print(f"    {st.name:12s} [{tag}]  {top_str}")

    # ── Token stats ──
    st = accumulator.get_session_stats("geo_test")
    if st and st.get("total_tokens", 0) > 0:
        print(f"\n  Token: {st['total_tokens']} total ({st['total_prompt_tokens']} in / {st['total_completion_tokens']} out)")

    graph.close()
    print("\n" + "=" * 60)
    print("  全部通过 ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

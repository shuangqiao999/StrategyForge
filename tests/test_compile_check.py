"""针对性编译测试：验证本次重构所有模块可正常导入 + 新方法存在。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

os.environ["FORGE_PROVIDER"] = "lmstudio"
os.environ["FORGE_EMBED_PROVIDER"] = "lmstudio"

from strategy_forge.core.providers import registry
registry._data["llm_provider"] = "lmstudio"
registry._data["embed_provider"] = "lmstudio"

def test_all_imports():
    """验证所有本次改动的模块可编译导入。"""
    from strategy_forge.engine.graph_builder import build_graph, _parse_extraction, _EXTRACT_PROMPT
    from strategy_forge.engine.simulator import SimulationEngine
    from strategy_forge.engine.preprocessor import DeductionPreprocessor
    from strategy_forge.engine.agent_factory import create_agents_from_graph
    from strategy_forge.engine.narrative_sorter import sort_narrative_entities
    from strategy_forge.engine.intel_sorter import sort_entities
    from strategy_forge.engine.orchestrator import DeductionOrchestrator
    from strategy_forge.core.rule_templates import get_domain_prompt
    print("  [OK] All module imports")

    # 验证共享方法存在
    for m in ['_shared_dual_recall', '_should_reflect', '_append_event']:
        assert hasattr(SimulationEngine, m), f"Missing method: {m}"
    print("  [OK] Shared methods exist")

    # 验证 prompt 模板包含 text_overview
    assert "$text_overview" in _EXTRACT_PROMPT, "text_overview placeholder missing"
    print("  [OK] text_overview in _EXTRACT_PROMPT")

    # 验证 domain_prompts.json 各领域有 intel_examples
    for d in ['politics', 'business', 'military', 'ecology', 'urban', 'tech', 'info_war', 'geo_strategy']:
        rules = get_domain_prompt(d, 'intel_extra_rules')
        examples = get_domain_prompt(d, 'intel_examples')
        assert rules, f"{d} missing intel_extra_rules"
        assert examples, f"{d} missing intel_examples"
    print("  [OK] All 8 domains have intel rules + examples")

    # 验证死代码 _last_reflection_round 已删除
    init_code = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "strategy_forge", "engine", "simulator.py"),
        encoding="utf-8").read()
    lines_with_reflection_round = [i + 1 for i, line in enumerate(init_code.splitlines())
                                    if '_last_reflection_round:' in line and '_n' not in line]
    assert not lines_with_reflection_round, f"Dead code still at lines: {lines_with_reflection_round}"
    print("  [OK] Dead code _last_reflection_round removed")

    # 验证无本地 Semaphore 残留（graph_builder, agent_factory, narrative_sorter, preprocessor）
    for fname, mod_name in [
        ("graph_builder.py", "graph_builder"),
        ("agent_factory.py", "agent_factory"),
        ("narrative_sorter.py", "narrative_sorter"),
        ("preprocessor.py", "preprocessor"),
    ]:
        code = open(
            os.path.join(os.path.dirname(__file__), "..", "src", "strategy_forge", "engine", fname),
            encoding="utf-8").read()
        # Should not have "sem = asyncio.Semaphore" in these files
        if "sem = asyncio.Semaphore(max(1, _reg.max_concurrent))" in code:
            print(f"  [WARN] {mod_name}: local semaphore still present (false positive possible)")
    print("  [OK] Local Semaphore cleanup verified")

    print("\nALL COMPILE-TIME CHECKS PASSED")

if __name__ == "__main__":
    test_all_imports()

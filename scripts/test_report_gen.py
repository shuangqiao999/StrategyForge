"""Quick test: 1-2 round quantified deduction → report generation via LM Studio 9B."""
import asyncio, os, sys, time, tempfile, shutil, uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_report_test"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen/qwen3.5-9b")

_test_data = os.environ["FORGE_DATA_DIR"]
os.makedirs(os.path.join(_test_data, "rule", "custom"), exist_ok=True)
if not os.path.exists(os.path.join(_test_data, "rule", "rules.json")):
    shutil.copy2("data/rule/rules.json", os.path.join(_test_data, "rule", "rules.json"))

sys.path.insert(0, "src")

from strategy_forge.engine.models import DeductionAgentProfile
from strategy_forge.engine.rule_engine import RuleEngine
from strategy_forge.engine.simulator import SimulationEngine
from strategy_forge.engine.ontology import generate_ontology
from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.engine.preprocessor import DeductionPreprocessor
from strategy_forge.engine.graph_builder import build_graph
from strategy_forge.engine.agent_factory import create_agents_from_graph
from strategy_forge.engine.reporter import generate_report
from strategy_forge.algorithms.module_utils import build_module_chain
from strategy_forge.core.config import config
from strategy_forge.core.token_counter import _current_session, _current_phase, _current_round, accumulator

SOURCE = """三国演义第一回：宴桃园豪杰三结义 斩黄巾英雄首立功

话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。及秦灭之后，楚、汉分争，又并入于汉。汉朝自高祖斩白蛇而起义，一统天下，后来光武中兴，传至献帝，遂分为三国。

推其致乱之由，殆始于桓、灵二帝。桓帝禁锢善类，崇信宦官。及桓帝崩，灵帝即位，大将军窦武、太傅陈蕃共相辅佐。时有宦官曹节等弄权，窦武、陈蕃谋诛之，机事不密，反为所害，中涓自此愈横。

榜文行到涿县，引出涿县中一个英雄。那人不甚好读书；性宽和，寡言语，喜怒不形于色；素有大志，专好结交天下豪杰；生得身长七尺五寸，两耳垂肩，双手过膝，目能自顾其耳，面如冠玉，唇若涂脂；中山靖王刘胜之后，汉景帝阁下玄孙，姓刘名备，字玄德。

玄德见他形貌异常，问其姓名。其人曰："某姓张名飞，字翼德。世居涿郡，颇有庄田，卖酒屠猪，专好结交天下豪杰。恰才见公看榜而叹，故此相问。"玄德曰："我本汉室宗亲，姓刘，名备。今闻黄巾倡乱，有志欲破贼安民，恨力不能，故长叹耳。" 

正饮间，见一大汉，推着一辆车子，到店门首歇了，入店坐下，便唤酒保："快斟酒来吃，我待赶入城去投军。"玄德看其人：身长九尺，髯长二尺；面如重枣，唇若涂脂；丹凤眼，卧蚕眉，相貌堂堂，威风凛凛。玄德就邀他同坐，叩其姓名。其人曰："吾姓关名羽，字云长，河东解良人也。因本处势豪倚势凌人，被吾杀了，逃难江湖，五六年矣。今闻此处招军破贼，特来应募。"玄德遂以己志告之，云长大喜。同到张飞庄上，共议大事。"""


async def main():
    print("=" * 50)
    print("  Report Generation Quick Test (2 rounds)")
    print("=" * 50)

    sid = uuid.uuid4().hex[:8]
    _current_session.set(sid)
    _current_phase.set("ontology")

    # Phase 1: Ontology
    print("\n--- Phase 1: Ontology ---")
    ontology = await generate_ontology(SOURCE[:8000])
    print(f"  {len(ontology.entities)} types, {len(ontology.relations)} relations")

    # Load rule engine
    re = RuleEngine.from_domain("military")
    print(f"  Rules: {re.pack['display_name']}")

    # Phase 2: Graph
    print("\n--- Phase 2: Graph ---")
    gpath = config.deduction_data_dir / "graphs" / sid / "kuzu"
    graph = DeductionGraphStore(gpath)
    pp = DeductionPreprocessor(config.project_root, sid)
    pp.preprocess(SOURCE)
    await build_graph(source=SOURCE, graph=graph, ontology=ontology,
                      log_fn=lambda p, m: None, preprocessor=pp)
    print(f"  {graph.count_entities()} entities, {graph.count_relations()} relations")

    # Phase 3: Agents
    print("\n--- Phase 3: Agents ---")
    agents = await create_agents_from_graph(graph=graph, source_material=SOURCE[:5000],
                                             log_fn=lambda p, m: None, preprocessor=pp)
    print(f"  {len(agents)} agents")

    # Phase 4: Simulation (2 rounds)
    print("\n--- Phase 4: Simulation (2 rounds) ---")
    states = {a.entity_id: re.init_state(a.entity_id, a.name) for a in agents}
    modules = build_module_chain(re)
    print(f"  Modules: {[m.name for m in modules]}")
    _round_cache = getattr(pp, "clear_round_cache", None)

    engine = SimulationEngine(
        agents=agents, graph=graph, total_rounds=2,
        log_fn=lambda p, m: print(f"    [{p}] {m}"),
        rule_engine=re, states=states,
        enable_narrate=True, algorithm_modules=modules,
    )

    rounds = []
    for rnd in range(1, 3):
        _current_phase.set("simulation")
        _current_round.set(rnd)
        t0 = time.time()
        result = await engine.run_round(rnd)
        rounds.append(result)
        print(f"  Round {rnd}: {len(result.actions)} actions ({time.time()-t0:.1f}s)")

    # Phase 5: Report
    print("\n--- Phase 5: Report ---")
    _current_phase.set("report")

    from strategy_forge.engine.models import DeductionSession, SessionStatus, DeductionPhase
    mock_session = DeductionSession(
        id=sid, title="三英结义", source_material=SOURCE,
    )
    mock_session.entity_count = graph.count_entities()
    mock_session.relation_count = graph.count_relations()
    mock_session.agent_count = len(agents)
    mock_session.current_round = 2

    t0 = time.time()
    report = await generate_report(
        session=mock_session,
        graph=graph, rounds=rounds, log_fn=lambda p, m: print(f"    [{p}] {m}"),
        preprocessor=pp, states=states,
    )
    print(f"  Report generated in {time.time()-t0:.1f}s")
    print(f"  Summary: {report.summary[:100]}...")
    print(f"  Causal chains: {len(report.causal_summary)}")
    print(f"  Stage narratives: {len(report.stage_narratives)}")
    print(f"  Deviation analysis: {len(report.deviation_analysis)}")
    print(f"  Risk alerts: {len(report.risk_alerts)}")
    print(f"  Recommendations: {len(report.recommendations)}")
    print(f"  Conclusion: {report.conclusion[:80]}...")

    # Verify orchestrator json issue is fixed by simulating the _phase5_report path
    import json as test_json
    payload = {
        "summary": report.summary, "key_events": report.key_events,
        "risk_alerts": report.risk_alerts, "recommendations": report.recommendations,
        "causal_summary": report.causal_summary, "stage_narratives": report.stage_narratives,
        "deviation_analysis": report.deviation_analysis, "conclusion": report.conclusion,
    }
    serialized = test_json.dumps(payload, ensure_ascii=False)
    assert isinstance(serialized, str) and len(serialized) > 100, "json serialization failed!"
    print("  ✓ Report JSON serialization OK")

    graph.close()

    st = accumulator.get_session_stats(sid)
    if st:
        print(f"\n  Token: {st['total_tokens']} total")

    print("\n" + "=" * 50)
    print("  ALL PASSED ✓")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

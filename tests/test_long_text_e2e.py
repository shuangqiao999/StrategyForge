"""E2E test: ultra-long text entity extraction pipeline with LM Studio gemma-4-12b.
Tests only Phase 1-3 (ontology→graph→agents), skips simulation.
Validates: chunk batching, entity frequency ranking, sorter batching, agent count.
"""
import sys, os, asyncio, json, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["FORGE_PROVIDER"] = "lmstudio"
os.environ["FORGE_EMBED_PROVIDER"] = "lmstudio"

from strategy_forge.core.config import config
from strategy_forge.core.providers import registry

# Override forge_config.json saved values with our test settings
registry._data["llm_provider"] = "lmstudio"
registry._data["llm_model"] = "qwen/qwen3.5-9b"
registry._data["embed_provider"] = "lmstudio"
registry._data["embedding_model_name"] = "text-embedding-embeddinggemma-300m-qat"
registry._data["llm_base_url"] = ""
registry._data["embed_base_url"] = ""
registry._data["llm_api_key"] = ""
from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.storage.session_store import SessionStore
from strategy_forge.engine.engine import DeductionEngine
from strategy_forge.engine.ontology import generate_ontology
from strategy_forge.engine.preprocessor import DeductionPreprocessor
from strategy_forge.engine.graph_builder import build_graph
from strategy_forge.engine.narrative_sorter import sort_narrative_entities
from strategy_forge.engine.agent_factory import create_agents_from_graph
from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient

SOURCE_PATH = r"E:\gongxiang\软件\资本论\水浒传.txt"
TEST_CHARS = 80_000  # ~8万 chars, enough entity diversity for validation

def load_text(path: str, max_chars: int) -> str:
    raw = open(path, encoding="utf-8").read()
    pos = raw.find("第一回")
    if pos > 0:
        raw = raw[pos:]
    return raw[:max_chars]

def log(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

async def main():
    t0 = time.time()
    print("=" * 70)
    print("超长文本 E2E 测试 — 水浒传 · 叙事模式")
    print(f"LM Studio: {registry.llm_model}  |  嵌入: {registry.embedding_model_name}")
    print("=" * 70)

    # ── load text ──
    source = load_text(SOURCE_PATH, TEST_CHARS)
    print(f"\n文本: {len(source):,} 字符 (水浒传片段)")
    print(f"LLM 并发: {registry.max_concurrent}")

    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="forge_test_")
    # Clean up any previous LanceDB tables from same session name
    from strategy_forge.core.config import config as _cfg
    lance_dir = str(_cfg.deduction_data_dir / "lancedb")
    import lancedb
    db_lance = lancedb.connect(lance_dir)
    for t in list(db_lance.table_names()):
        if "test_session" in t:
            try:
                db_lance.drop_table(t)
            except Exception:
                pass
    db_path = os.path.join(tmp, "test.db")

    # ── Phase 1: Ontology ──
    log("阶段1: 本体生成...")
    ontology = await generate_ontology(source)
    log(f"本体: {len(ontology.entities)} 实体类型, {len(ontology.relations)} 关系类型")

    # ── Preprocessor ──
    log("预处理: 分块 + jieba + LanceDB...")
    pp = DeductionPreprocessor(config.project_root, "test_session")
    await pp.preprocess(source)
    presult = pp.result
    log(f"分块: {presult.total_chunks} 块, 实体: {presult.total_entities} (高频 {len(presult.high_freq_entities)}, 低频 {len(presult.low_freq_entities)})")

    if presult.entity_frequencies:
        top5 = sorted(presult.entity_frequencies.items(), key=lambda x: -x[1])[:10]
        log(f"Top-10 高频实体: {', '.join(f'{n}({c}次)' for n,c in top5)}")

    # ── Phase 2: Graph ──
    log("阶段2: 图谱构建...")
    graph_path = os.path.join(tmp, "graphs", "test_session", "kuzu")
    graph = DeductionGraphStore(graph_path)
    await build_graph(
        source=source, graph=graph, ontology=ontology,
        log_fn=lambda p, m: log(f"[{p}] {m}"),
        preprocessor=pp,
    )
    e_count = graph.count_entities()
    r_count = graph.count_relations()
    log(f"图谱: {e_count} 实体, {r_count} 关系")

    # ── Narrative sorter ──
    log("叙事实体分类...")
    entity_names = list(graph.get_entity_names())
    log(f"待分类实体: {len(entity_names)} 个")
    client = LLMClient()
    chunk_texts = [c.content for c in presult.chunks] if presult.chunks else None
    intel_list = await sort_narrative_entities(
        source, entity_names, client,
        entity_frequencies=getattr(presult, "entity_frequencies", None),
        entity_chunk_coverage=getattr(presult, "entity_chunk_coverage", None),
        chunk_texts=chunk_texts,
    )
    active = sum(1 for e in intel_list if e.get("include_in_simulation"))
    passive = len(intel_list) - active
    log(f"分类结果: {len(intel_list)} 实体 → {active} 角色 + {passive} 背景")

    if active > 0:
        top_roles = sorted(
            [e for e in intel_list if e.get("include_in_simulation")],
            key=lambda e: presult.entity_frequencies.get(e["name"], 0) if presult.entity_frequencies else 0,
            reverse=True,
        )[:20]
        log(f"Top-20 角色: {', '.join(e['name'] for e in top_roles)}")

    # ── Phase 3: Agent Factory ──
    log("阶段3: 智能体生成...")
    agents = await create_agents_from_graph(
        graph=graph, source_material=source,
        log_fn=lambda p, m: log(f"[{p}] {m}"),
        preprocessor=pp, intel_list=intel_list,
    )
    log(f"智能体: {len(agents)} 个")
    for a in agents[:10]:
        log(f"  [{a.entity_type}] {a.name}: {a.persona[:60]}...")

    # ── Analysis ──
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("测试分析")
    print("=" * 70)
    print(f"耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"实体发现: {presult.total_entities} 实体 (jieba+LLM)")
    print(f"图谱实体: {e_count}, 关系: {r_count}")
    print(f"分类覆盖: {len(intel_list)}/{e_count} ({len(intel_list)*100//max(1,e_count)}%)" if e_count > 0 else "N/A")
    print(f"角色/背景: {active}/{passive}")
    print(f"智能体: {len(agents)}")

    # Key character check
    key_chars = ["宋江", "武松", "林冲", "鲁智深", "李逵", "吴用", "卢俊义",
                 "高俅", "洪太尉", "史进", "王进"]
    agent_names = {a.name for a in agents}
    intel_names = {e["name"] for e in intel_list if e.get("include_in_simulation")}
    graph_names = set(entity_names)
    print("\n关键角色检查:")
    for c in key_chars:
        in_graph = c in graph_names or any(c in gn for gn in graph_names)
        in_intel = c in intel_names
        in_agent = c in agent_names
        status = "✅" if in_agent else ("⚠️ intel" if in_intel else ("❌" if not in_graph else "🔍 graph only"))
        print(f"  {status} {c}: graph={in_graph}, intel={in_intel}, agent={in_agent}")

    pp.close()
    graph.close()
    print(f"\n临时文件: {tmp}")

if __name__ == "__main__":
    asyncio.run(main())

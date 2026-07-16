"""E2E test: full pipeline Phase 1-3 with 20K chars, LM Studio qwen3.5-9b.
Should complete in ~20-30 min with ~20 LLM calls total.
"""
import sys, os, asyncio, json, time, shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

os.environ["FORGE_PROVIDER"] = "lmstudio"
os.environ["FORGE_EMBED_PROVIDER"] = "lmstudio"

from strategy_forge.core.config import config as _cfg
from strategy_forge.core.providers import registry

registry._data["llm_provider"] = "lmstudio"
registry._data["llm_model"] = "qwen/qwen3.5-9b"
registry._data["embed_provider"] = "lmstudio"
registry._data["embedding_model_name"] = "text-embedding-embeddinggemma-300m-qat"

from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.engine.engine import DeductionEngine
from strategy_forge.engine.ontology import generate_ontology
from strategy_forge.engine.preprocessor import DeductionPreprocessor
from strategy_forge.engine.graph_builder import build_graph
from strategy_forge.engine.narrative_sorter import sort_narrative_entities
from strategy_forge.engine.agent_factory import create_agents_from_graph
from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient

SOURCE = r"E:\gongxiang\软件\资本论\水浒传.txt"

def load(source_path, max_chars):
    raw = open(source_path, encoding="utf-8").read()
    pos = raw.find("\u7b2c\u4e00\u56de")  # 第一回
    if pos > 0:
        raw = raw[pos:]
    return raw[:max_chars]  # 2w chars, 第1回~第3回中段

async def main():
    t0 = time.time()
    tmp = os.path.join(os.environ["TEMP"], "forge_e2e_test")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    print("=" * 60)
    print(f"E2E: full pipeline Phase 1-3")
    print(f"Model: {registry.llm_model} | Embed: {registry.embedding_model_name}")
    print(f"Concurrency: {registry.max_concurrent}")
    print("=" * 60)

    source = load(SOURCE, 20000)
    print(f"\nText: {len(source):,} chars")
    
    def progress(cur, tot):
        pct = cur * 100 // tot
        elapsed = time.time() - t0
        print(f"  [t={elapsed:.0f}s] LLM entity discovery: {cur}/{tot} ({pct}%)", flush=True)

    # Phase 1
    print("\n[1/4] Ontology...")
    ontology = await generate_ontology(source)
    print(f"  -> {len(ontology.entities)} types, {len(ontology.relations)} relations")

    # Preprocess
    print("\n[2/4] Preprocess + LLM entity discovery...")
    pp = DeductionPreprocessor(os.path.join(tmp, "data"), "e2e_test")
    pp.set_progress_callback(progress)
    await pp.preprocess(source)
    presult = pp.result
    print(f"  -> {presult.total_chunks} chunks, {presult.total_entities} entities ({len(presult.high_freq_entities)} hi, {len(presult.low_freq_entities)} lo)")
    if presult.entity_frequencies:
        top10 = sorted(presult.entity_frequencies.items(), key=lambda x: -x[1])[:10]
        print(f"  -> Top freq: {', '.join(f'{n}({f})' for n,f in top10)}")

    # Phase 2
    print("\n[3/4] Graph build...")
    graph_path = os.path.join(tmp, "graphs", "e2e_test", "kuzu")
    graph = DeductionGraphStore(graph_path)
    await build_graph(
        source=source, graph=graph, ontology=ontology,
        log_fn=lambda p, m: None,
        preprocessor=pp,
    )
    e_count = graph.count_entities()
    r_count = graph.count_relations()
    print(f"  -> {e_count} entities, {r_count} relations")

    # Sorter
    print("\n  Narrative sorter...")
    entity_names = list(graph.get_entity_names())
    print(f"  -> {len(entity_names)} entities to classify")
    client = LLMClient()
    chunks = [c.content for c in presult.chunks] if presult.chunks else None
    intel_list = await sort_narrative_entities(
        source, entity_names, client,
        entity_frequencies=getattr(presult, "entity_frequencies", None),
        entity_chunk_coverage=getattr(presult, "entity_chunk_coverage", None),
        chunk_texts=chunks,
    )
    active = sum(1 for e in intel_list if e.get("include_in_simulation"))
    print(f"  -> {len(intel_list)} classified: {active} roles + {len(intel_list)-active} bg")
    if active:
        roles = sorted(
            [e for e in intel_list if e.get("include_in_simulation")],
            key=lambda e: presult.entity_frequencies.get(e["name"], 0) if presult.entity_frequencies else 0,
            reverse=True,
        )[:15]
        print(f"  -> Top roles: {', '.join(e['name'] for e in roles)}")

    # Phase 3
    print("\n[4/4] Agent factory...")
    agents = await create_agents_from_graph(
        graph=graph, source_material=source,
        log_fn=lambda p, m: None,
        preprocessor=pp, intel_list=intel_list,
    )
    print(f"  -> {len(agents)} agents generated")
    for a in agents[:10]:
        print(f"    [{a.entity_type}] {a.name}: {a.persona[:50]}...")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"RESULT: {len(agents)} agents from {e_count} entities")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("=" * 60)

    pp.close()
    graph.close()
    print(f"\nPassed. Temp: {tmp}")

if __name__ == "__main__":
    asyncio.run(main())

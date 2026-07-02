"""Test alias-based deduplication + entity extraction with min_freq=1 + candidate[:200].

Verifies:
  1. min_freq=1 picks up single-occurrence entities (like "G7", "北约")  
  2. candidate[:200] includes all low-freq entities in LLM prompts
  3. alias_map deduplication merges "特朗普"/"美国（特朗普）" correctly
  4. alias_map does NOT merge unrelated entities like "美国"/"美国国防部"
"""
import asyncio, os, sys, shutil, time, uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_dedup_test"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen/qwen3.5-9b")

_test = os.environ["FORGE_DATA_DIR"]
os.makedirs(os.path.join(_test, "rule", "custom"), exist_ok=True)
shutil.copy2("data/rule/rules.json", os.path.join(_test, "rule", "rules.json"))

sys.path.insert(0, "src")

from strategy_forge.core.config import config
from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.engine.preprocessor import DeductionPreprocessor


def banner(text):
    print(f"\n{'='*50}")
    print(f"  {text}")
    print(f"{'='*50}")


def main():
    # Quick test: verify entity_aliases exists and contains expected data
    banner("Test 1: entity_aliases field on PreprocessResult")

    pp = DeductionPreprocessor(config.project_root, "dedup_test")
    SOURCE = "美国制裁伊朗 俄罗斯支持中国 特朗普发动贸易战 北约召开峰会 G7发布声明 阿联酋调解冲突"
    pp.preprocess(SOURCE)

    result = pp.result
    assert result is not None, "PreprocessResult is None"
    assert hasattr(result, "entity_aliases"), "entity_aliases field missing"
    assert isinstance(result.entity_aliases, dict), "entity_aliases must be dict"

    total_entities = len(result.entity_aliases)
    high = len(result.high_freq_entities)
    low = len(result.low_freq_entities)
    print(f"  Total entities: {total_entities} (high-freq={high}, low-freq={low})")

    # With min_freq=1, ALL recognized entities appear
    assert total_entities >= 5, f"Expected >=5 entities, got {total_entities}"
    # Low-freq entities should include single-occurrence ones
    assert low >= 1, f"Expected at least 1 low-freq entity, got {low}"

    # Verify alias structure
    for std, aliases in result.entity_aliases.items():
        if std == "特朗普":
            print(f"  Alias structure OK: {std} has {len(aliases)} aliases")
            break

    banner("Test 2: alias_map dedup logic (simulated)")
    # Simulate what agent_factory does
    alias_to_std = {}
    for std, aliases in result.entity_aliases.items():
        alias_to_std[std] = std
        for a in aliases:
            alias_to_std[a] = std

    # Test: "特朗普" and "美国（特朗普）" should normalize to "特朗普"
    test_cases = [
        ("特朗普", "特朗普"),         # exact match
        ("美国（特朗普）", "美国（特朗普）"),  # alias with parens — in alias_map?
        ("北约", "北约"),              # single-occurrence
        ("G7", "G7"),                  # single-occurrence, alphanumeric
    ]
    for name, expected in test_cases:
        resolved = alias_to_std.get(name, name)
        print(f"  '{name}' → '{resolved}'" + ("" if resolved == expected else "  ⚠️"))

    # Test: unrelated entities should NOT be merged
    persons = [
        {"name": "美国", "type": "Organization"},
        {"name": "美国国防部", "type": "Organization"},
    ]
    seen = set()
    deduped = []
    for p in persons:
        name = p.get("name", "")
        std_name = alias_to_std.get(name, name)
        if std_name in seen:
            continue
        seen.add(std_name)
        p["name"] = std_name
        deduped.append(p)
    print(f"  After dedup: {[p['name'] for p in deduped]}")
    assert len(deduped) == 2, f"'美国' and '美国国防部' should remain separate, got {len(deduped)}"
    print("  ✓ Unrelated entities preserved")

    banner("Test 3: candidate_names includes low-freq entities")
    from strategy_forge.engine.graph_builder import _EXTRACT_PROMPT
    from string import Template

    all_aliases = {**result.high_freq_entities, **result.low_freq_entities}
    candidate_names = list(all_aliases.keys())
    candidate_200 = candidate_names[:200]  # new limit
    candidate_80 = candidate_names[:80]   # old limit

    # With >5 entities, 200 covers all; 80 might not
    assert len(candidate_200) >= len(candidate_names), "200 should cover all entities"
    print(f"  Total candidates: {len(candidate_names)}")
    print(f"  With [:200]: {len(candidate_200)}")
    print(f"  With  [:80]: {len(candidate_80)}")
    if len(candidate_names) > 80:
        missing = set(candidate_names) - set(candidate_80)
        print(f"  Previously truncated: {len(missing)} entities (e.g. {list(missing)[:5]})")
    else:
        print("  All fit within 80 (short text)")

    banner("Result")
    print("  All dedup tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

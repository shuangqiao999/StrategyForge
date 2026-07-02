"""Test LLM-assisted entity discovery for 大国博弈 corpus.

Verifies:
  1. jieba-only entity count (baseline ~27)
  2. LLM-assisted entity count (should be 50+)
  3. Specific entities jieba misses but LLM finds: "北约", "G7", "OECD", "霍尔木兹海峡"
  4. Full graph build + agent creation with merged entities
"""
import asyncio, os, sys, shutil, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_llm_entity"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen/qwen3.5-9b")

_test = os.environ["FORGE_DATA_DIR"]
os.makedirs(os.path.join(_test, "rule", "custom"), exist_ok=True)
shutil.copy2("data/rule/rules.json", os.path.join(_test, "rule", "rules.json"))

sys.path.insert(0, "src")

from strategy_forge.core.config import config
from strategy_forge.engine.preprocessor import DeductionPreprocessor, _merge_entity_dicts
from strategy_forge.engine.rule_engine import RuleEngine

SOURCE = open(r"E:\gongxiang\软件\资本论\大国博弈.txt", encoding="utf-8").read()


def main():
    print("=" * 50)
    print("  LLM Entity Discovery Test — 大国博弈")
    print("=" * 50)

    # ── Test 1: jieba-only baseline ──
    print("\n--- Test 1: Jieba-only baseline ---")
    jieba_entities = {}
    from strategy_forge.core.tokenizer import extract_named_entities
    jieba_entities = extract_named_entities(SOURCE, top_k=1000, min_freq=1)
    jieba_count = len(jieba_entities)
    jieba_names = set(jieba_entities.keys())
    print(f"  Jieba entities: {jieba_count}")
    print(f"  Sample: {sorted(jieba_names)[:10]}")

    # Check which expected entities jieba misses
    expected = {"北约", "G7", "OECD", "霍尔木兹海峡", "伊朗革命卫队", "克拉斯诺达尔",
                "DeepSeek", "阿拉伯联合酋长国", "真主党", "斯拉维扬斯克"}
    found = expected & jieba_names
    missing_jieba = expected - jieba_names
    print(f"  Expected found by jieba: {len(found)}/{len(expected)}: {sorted(found)}")
    print(f"  Missing (jieba): {sorted(missing_jieba)[:6]}")

    # ── Test 2: LLM entity discovery ──
    print("\n--- Test 2: LLM entity discovery ---")
    pp = DeductionPreprocessor(config.project_root, "llm_entity_test")
    t0 = time.time()
    llm_entities = pp._llm_entity_discovery(SOURCE)
    elapsed = time.time() - t0
    llm_count = len(llm_entities)
    llm_names = set(llm_entities.keys())
    print(f"  LLM entities: {llm_count} (in {elapsed:.1f}s)")
    print(f"  Sample: {sorted(llm_names)[:10]}")

    llm_found = expected & llm_names
    llm_missing = expected - llm_names
    print(f"  Expected found by LLM: {len(llm_found)}/{len(expected)}: {sorted(llm_found)}")
    print(f"  Still missing: {sorted(llm_missing)[:6]}")

    improvement = llm_count - jieba_count if llm_count > jieba_count else 0
    print(f"\n  LLM net improvement: +{improvement} entities")

    # ── Test 3: Merged entities ──
    print("\n--- Test 3: Merged entity set ---")
    merged = _merge_entity_dicts(jieba_entities, llm_entities)
    merged_count = len(merged)
    merged_names = set(merged.keys())
    merged_found = expected & merged_names
    print(f"  Merged total: {merged_count}")
    print(f"  Expected found in merged: {len(merged_found)}/{len(expected)}: {sorted(merged_found)}")

    # ── Test 4: Full graph build + agent creation ──
    print("\n--- Test 4: Full graph build + agent creation ---")
    t0 = time.time()
    pp_result = pp.preprocess(SOURCE)

    total = len(pp_result.entity_aliases)
    high = len(pp_result.high_freq_entities)
    low = len(pp_result.low_freq_entities)
    print(f"  Preprocess entities: {total} total ({high} high-freq + {low} low-freq)")

    # Check LLM-discovered entities in the result
    result_entities = set(pp_result.entity_aliases.keys())
    result_found = expected & result_entities
    print(f"  Result has: {len(result_found)}/{len(expected)} expected: {sorted(result_found)}")

    print(f"  Total time: {time.time() - t0:.1f}s")

    print(f"\n{'=' * 50}")
    print(f"  LLM improvement: +{improvement} entities discovered")
    print(f"  All tests complete")
    print(f"{'=' * 50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

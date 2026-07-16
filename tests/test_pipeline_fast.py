"""Fast pipeline validation: preprocessor + frequency ranking + fallback logic.
No LLM calls — validates jieba extraction, chunk coverage, frequency ranking,
and the agent_factory fallback threshold computation. Completes in <30 seconds.
"""
import sys, os, re, math, statistics, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy_forge.core.tokenizer import extract_named_entities
from strategy_forge.core.chunker import TextChunker

SOURCE_PATH = r"E:\gongxiang\软件\资本论\水浒传.txt"
TEST_CHARS = 150_000

def load_text(path, max_chars):
    raw = open(path, encoding="utf-8").read()
    pos = raw.find("第一回")
    if pos > 0:
        raw = raw[pos:]
    return raw[:max_chars]

t0 = time.time()
print("=" * 70)
print("预处理 + 频次排名 + 兜底逻辑 验证")
print("=" * 70)

source = load_text(SOURCE_PATH, TEST_CHARS)
print(f"\n文本: {len(source):,} 字符")

# 1. Chunking
chunker = TextChunker(strategy="paragraph", max_chunk_size=1536)
chunks = chunker.chunk(source, file_type=".txt")
chunk_texts = [c.content for c in chunks]
print(f"分块: {len(chunks)} 块")

# 2. Jieba entity extraction
all_entities = extract_named_entities(source, top_k=1000, min_freq=1)
entity_freq = {}
entity_cov = {}
high_freq = {}
low_freq = {}
for name, aliases in all_entities.items():
    count = len(re.findall(re.escape(name), source))
    entity_freq[name] = count
    entity_cov[name] = sum(1 for ct in chunk_texts if name in ct)
    if count >= 2:
        high_freq[name] = aliases
    else:
        low_freq[name] = aliases

print(f"实体: {len(all_entities)} (高频 {len(high_freq)}, 低频 {len(low_freq)})")

# 3. Frequency ranking (Fix B logic)
def _entity_rank(item):
    name, aliases = item
    return (entity_freq.get(name, 0), entity_cov.get(name, 0), len(aliases))

hf_sorted = sorted(high_freq.items(), key=_entity_rank, reverse=True)
dyn_cap = max(50, len(hf_sorted) // 4)
print(f"\n动态 Top-N: {dyn_cap} (high_freq={len(hf_sorted)}, formula=max(50, {len(hf_sorted)}//4))")

# 4. Top entities by frequency
top30 = hf_sorted[:30]
print("\nTop-30 高频实体（频次, 覆盖分块数, 别名数）:")
for i, (name, aliases) in enumerate(top30, 1):
    f = entity_freq.get(name, 0)
    c = entity_cov.get(name, 0)
    print(f"  {i:2d}. {name:8s}  freq={f:4d}  chunks={c:3d}  aliases={len(aliases)}")

# 5. Simulate sorter batching (Fix C logic)
entity_names = list(all_entities.keys())
_ENTITY_BATCH_SIZE = 60
if len(entity_names) <= _ENTITY_BATCH_SIZE:
    n_batches = 1
else:
    n_batches = math.ceil(len(entity_names) / _ENTITY_BATCH_SIZE)
print(f"\n叙事分类 batch 数: {n_batches} ({len(entity_names)} 实体 / {_ENTITY_BATCH_SIZE})")

# 6. Simulate frequency fallback (Fix D logic)
# Assume sorter classified top-15 highest-frequency entities as active
mock_intel_list = [
    {"name": name, "include_in_simulation": True}
    for name, _ in hf_sorted[:15]
]
included_freqs = [entity_freq.get(e["name"], 0) for e in mock_intel_list]
freq_threshold = int(statistics.median(included_freqs) // 3) if included_freqs else 5
freq_threshold = max(3, freq_threshold)
fallback_added = []
for name in entity_freq:
    if name in {e["name"] for e in mock_intel_list}:
        continue
    if entity_freq.get(name, 0) >= freq_threshold:
        fallback_added.append((name, entity_freq[name]))

fallback_added.sort(key=lambda x: -x[1])
print(f"\n频率兜底阈值: >= {freq_threshold} 次")
print(f"兜底恢复实体: {len(fallback_added)} 个")
for name, f in fallback_added[:20]:
    print(f"  {name} (freq={f})")

# 7. Key character check
key_chars = ["宋江", "武松", "林冲", "鲁智深", "李逵", "吴用", "卢俊义",
             "高俅", "洪太尉", "史进", "王进", "杨志", "花荣", "燕青"]
print("\n关键角色检查:")
all_included = {e["name"] for e in mock_intel_list} | {n for n, _ in fallback_added}
for c in key_chars:
    freq = entity_freq.get(c, 0)
    cov = entity_cov.get(c, 0)
    in_mock = c in {e["name"] for e in mock_intel_list}
    in_fallback = c in {n for n, _ in fallback_added}
    in_any = c in all_included
    status = "OK" if in_any else ("!! not extracted" if freq == 0 else "-- low freq")
    print(f"  {status} {c}: freq={freq}, chunks={cov}, sorter={in_mock}, fallback={in_fallback}")

elapsed = time.time() - t0
print(f"\n耗时: {elapsed:.1f}s")
print("=" * 70)

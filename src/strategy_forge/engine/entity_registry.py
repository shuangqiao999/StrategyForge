"""Entity Registry — 实体注册中心：多层 LLM 流水线式实体识别与归类。

架构：
  Layer 1: 实体归一化
    - 快速路径 (≤12K字): 单次 LLM 全量归一化
    - 分片路径 (>12K字):  滑动分片 → 保守归一化 → 内存合并 → LLM 精修
  Layer 2: 逐批角色判定 — 10 个一组并行判定 KEEP/DISCARD + 证据
  Layer 3: 交叉裁决 — LLM 全局冗余检测 (地缘/商业域)；文学/历史域跳过

用法：
  registry = await build_registry(graph, preprocessor, intel_list, source_material=source)
  kept = registry.get_kept()
"""
from __future__ import annotations

import asyncio
import hashlib
import json as _json
import logging
import re as _re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── LRU 缓存 (P0-1: 防止内存溢出 + TTL过期) ──
class _LRUCache:
    """简易 LRU 缓存，带 TTL 过期（默认30分钟）。"""
    def __init__(self, maxsize: int = 64, ttl_sec: int = 1800):
        self._maxsize = maxsize
        self._ttl = ttl_sec
        self._data: dict[str, object] = {}
        self._timestamps: dict[str, float] = {}
        self._order: list[str] = []

    def get(self, key: str) -> object | None:
        import time
        if key in self._data:
            if self._ttl > 0 and time.time() - self._timestamps.get(key, 0) > self._ttl:
                self._order.remove(key)
                del self._data[key]
                del self._timestamps[key]
                return None
            self._order.remove(key)
            self._order.append(key)
            return self._data[key]
        return None

    def set(self, key: str, value: object) -> None:
        import time
        if key in self._data:
            self._order.remove(key)
        elif len(self._data) >= self._maxsize:
            oldest = self._order.pop(0)
            del self._data[oldest]
            self._timestamps.pop(oldest, None)
            logger.debug("[LRU] 淘汰缓存: %s", oldest[:16])
        self._data[key] = value
        self._timestamps[key] = time.time()
        self._order.append(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


# ── 模块级缓存 (P0-1 + P1-3) ──
_layer3_decision_cache = _LRUCache(64)     # Layer3 哈希缓存
_layer1_normalize_cache = _LRUCache(16)    # Layer1 归一化结果缓存
_layer3_variance_log: list[dict] = []      # P0-2: 方差日志
_token_defaults: dict | None = None        # 统一 token 配置缓存


def _token_cfg(key: str, n: int, domain: str = "geo_strategy") -> int:
    """读取统一 token 配置。模块级缓存避免重复 YAML 解析。"""
    global _token_defaults
    if _token_defaults is None:
        _token_defaults = _load_layer3_config(domain).get("token", {})
    return _get_token_limit({"token": _token_defaults}, key, n)


# ── 字典加载 (P0-3: 统一数据源) ──
def _load_alias_json() -> dict:
    """P0-3: 从 entity_alias.json 加载全部内置映射 + 别名词典。"""
    try:
        rule_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "rule"
        path = rule_dir / "entity_alias.json"
        if not path.exists():
            import os
            env_dir = os.environ.get("FORGE_RULE_DIR", "")
            if env_dir:
                path = Path(env_dir) / "entity_alias.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return _json.load(f)
    except Exception as e:
        logger.warning("[EntityRegistry] 加载 entity_alias.json 失败: %s", e)
    return {}

# 模块加载时读取内置字典（仅一次文件IO）
_alias_data = _load_alias_json()
_ORG_MEMBERS: dict[str, frozenset[str]] = {
    k: frozenset(v) for k, v in _alias_data.get("_builtin_org_members", {}).items()
}
_PERSON_COUNTRY: dict[str, str] = dict(_alias_data.get("_builtin_person_country", {}))

# 方法论块：注入 LLM Prompt，不参与代码逻辑
_A_METHODOLOGY = _alias_data.get("_methodology", {})
_A_AGENCY_METHOD = _A_METHODOLOGY.get("entity_agency", {}).get("framework", "")
_A_ALIAS_METHOD = _A_METHODOLOGY.get("alias_detection", {}).get("framework", "")
_A_REDUNDANCY_METHOD = _A_METHODOLOGY.get("redundancy_detection", {}).get("framework", "")
_A_TYPE_METHOD = _A_METHODOLOGY.get("type_normalization", {}).get("framework", "")

_SHARD_SIZE = 7500
_SHARD_OVERLAP = 800
_SHARD_THRESHOLD = 12000  # 超过此字数启用分片路径 (P1-2: 默认值，运行时可用 config 覆盖)

# Jaccard 阈值：地缘政治实体名较短(2-4字)，文学叙事实体名较长(3-6字)
_JACCARD_THRESHOLDS = {
    "geo_strategy": 0.40, "military": 0.40,
    "business": 0.45, "politics": 0.40,
    "novel": 0.50,
    "ecology": 0.45, "urban": 0.45,
    "tech": 0.45, "info_war": 0.40,
    "_default": 0.45,
}


# ── Data Classes ──

@dataclass
class RegisteredEntity:
    id: str = ""
    name: str = ""
    type: str = ""
    freq: int = 0
    chunk_coverage: int = 0
    decision: str = ""
    reason: str = ""
    parent: str = ""
    aliases: list[str] = field(default_factory=list)
    rich_description: str = ""
    tier: int = 0           # 1=核心博弈者 2=次级参与者 3=纯背景 0=未判定
    tier_evidence: str = "" # 分级证据（原文引用）
    group: str = ""         # L2 归属分组（所属国家/组织）


@dataclass
class EntityRegistry:
    entities: dict[str, RegisteredEntity] = field(default_factory=dict)
    total: int = 0
    kept: int = 0
    discarded: int = 0
    tier1_count: int = 0
    tier2_count: int = 0
    discard_reasons: dict[str, int] = field(default_factory=dict)

    def get_kept(self) -> list[RegisteredEntity]:
        return sorted(
            [e for e in self.entities.values() if e.decision == "KEEP"],
            key=lambda e: -e.freq)

    def get_tier1(self) -> list[RegisteredEntity]:
        """一级核心博弈者 → 生成独立智能体。"""
        return sorted(
            [e for e in self.entities.values() if e.tier == 1],
            key=lambda e: -e.freq)

    def get_tier2(self) -> list[RegisteredEntity]:
        """二级次级参与者 → 保留不生成智能体。"""
        return sorted(
            [e for e in self.entities.values() if e.tier == 2],
            key=lambda e: -e.freq)

    def get_tier12(self) -> list[RegisteredEntity]:
        """全部保留实体（一级+二级），向后兼容旧 get_kept()。"""
        return sorted(
            [e for e in self.entities.values() if e.tier in (1, 2)],
            key=lambda e: -e.freq)

    def get_by_type(self, etype: str) -> list[RegisteredEntity]:
        return [e for e in self.entities.values() if e.type == etype]

    def summary(self) -> str:
        lines = [f"EntityRegistry: {self.total} total, {self.kept} KEEP, {self.discarded} DISCARD"]
        if self.discard_reasons:
            detail = " | ".join(f"{k}:{v}" for k, v in
                                sorted(self.discard_reasons.items(), key=lambda x: -x[1]))
            lines.append(f"  DISCARD: {detail}")
        for e in self.get_kept()[:50]:
            lines.append(f"    {e.name}  {e.type}  freq={e.freq}  → {e.reason}")
        return "\n".join(lines)

    def find(self, name: str) -> RegisteredEntity | None:
        e = self.entities.get(name)
        if e:
            return e
        for ent in self.entities.values():
            if name in ent.aliases:
                return ent
        return None


# ── JSON 解析工具 ──

def _extract_balanced(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == opener: depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_llm_json(raw: str) -> dict:
    s = str(raw).strip()
    for _name, parser in [
        ("direct",   lambda x: _json.loads(x)),
        ("no_md",    lambda x: _json.loads(
            _re.sub(r'```(?:json)?\s*\n?', '', x).replace('```', '').strip())),
        ("greedy",   lambda x: _json.loads(
            (_re.search(r'\{[\s\S]*\}', x) or _re.search(r'\[[\s\S]*\]', x)).group(0)
            if (_re.search(r'\{[\s\S]*\}', x) or _re.search(r'\[[\s\S]*\]', x)) else "")),
        ("balanced", lambda x: _json.loads(_extract_balanced(x, '{', '}') or x)),
        ("last_brace", lambda x: _json.loads(
            x[:x.rfind('}') + 1] if '}' in x else "")),
        ("repair",   lambda x: _json.loads(_repair_json(x))),
    ]:
        try:
            data = parser(s)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    raise ValueError(f"JSON parse failed. Raw(length={len(s)}):\n{s[:500]}")


def _repair_json(s: str) -> str:
    """修复常见的 LLM JSON 语法错误：双逗号、截断数组、孤立逗号、缺失值。"""
    # 1. 去除多余逗号：`, ,` → `,`
    s = _re.sub(r',\s*,', ',', s)
    # 2. 逗号后无值截断：`"aliases": ,` → `"aliases": []`
    s = _re.sub(r'":\s*,', '": [],', s)
    # 2b. 截断在 `"aliases": ` 处（换行或末尾）
    s = _re.sub(r'":\s*$', '": []', s, flags=_re.MULTILINE)
    s = _re.sub(r'":\s*\n', '": [],\n', s)
    # 2c. `"key":` 后跟换行 + 另一个 key（值完全丢失）
    s = _re.sub(r'":\s*\n\s*"', '": "",\n  "', s)
    # 3. 截断的未完成条目：删掉最后一个不完整的 key-value
    s = _re.sub(r',\s*"[a-z_]+\s*$', '', s)
    # 4. 孤立方括号修复：`, ]` → `]`
    s = _re.sub(r',\s*\]', ']', s)
    # 4b. `"],\n]"` → `"\n]`
    s = _re.sub(r'"\],\s*\n\s*\]', '"]\n]', s)
    # 5. 丢失的闭合：如果 `]` 缺失但 `}` 存在，补 `]}`
    if _re.search(r'"[a-z_]+\s*":\s*$', s):
        s = s.rstrip().rstrip(',') + ']}'
    # 6. 双冒号：`"": "` → `": "`
    s = _re.sub(r'""\s*:\s*"', '": "', s)
    return s


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────

def _smart_sample(source: str, max_chars: int = 8000, domain: str = "") -> str:
    """智能采样：头 + 均匀中段 + 尾，覆盖全文叙事弧。
    文学域增加中段权重（关键剧情多在中段），地缘域保持均匀。"""
    n = len(source)
    if n <= max_chars:
        return source
    # 文学域：头1500 + 中段(多取) + 尾1500
    # 其他域：头2000 + 中段(均匀) + 尾2000
    _lit = domain in ("novel", "history", "narrative")
    head_size = 1500 if _lit else 2000
    tail_size = 1500 if _lit else 2000
    mid_budget = max_chars - head_size - tail_size
    if mid_budget <= 0:
        mid_budget = max_chars // 2
        head_size = max_chars // 4
        tail_size = max_chars // 4
    head = source[:head_size]
    tail = source[-tail_size:]
    mid_start = head_size
    mid_end = n - tail_size
    if remaining <= 0:
        return head + "\n...(中段省略)...\n" + tail
    # 均匀采样中段
    mid_start = 2000
    mid_end = n - 2000
    mid_len = mid_end - mid_start
    if mid_len <= mid_budget:
        return source
    step = max(1, mid_len // 4)
    samples = []
    for i in range(4):
        pos = mid_start + i * step
        chunk = source[pos:pos + mid_budget // 4]
        samples.append(f"\n--- [片段 {i+1}] ---\n{chunk}")
    return head + "".join(samples) + "\n--- [结尾] ---\n" + tail


def _shard_source(text: str, shard_size: int = _SHARD_SIZE,
                  overlap: int = _SHARD_OVERLAP) -> list[str]:
    """滑动分片，重叠区确保边界实体不丢失。"""
    shards = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + shard_size, n)
        shards.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return shards


def _get_token_limit(cfg: dict, key: str, n: int) -> int:
    """从统一 token 配置读取 max_tokens: min(cap, base + n*per)。"""
    tk = cfg.get("token", {}).get(key, {})
    base = tk.get("base", 200)
    per = tk.get("per", 100)
    cap = tk.get("cap", 4000)
    return min(cap, base + n * per)


def _char_jaccard(a: str, b: str) -> float:
    """字符级 Jaccard 相似度。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _match_entities_to_shards(
    shards: list[str],
    entities: list[dict],
) -> list[list[dict]]:
    """纯文本匹配：实体名或别名出现在分片中 → 归入该分片。零 LLM 成本。"""
    result: list[list[dict]] = [[] for _ in shards]
    for e in entities:
        name = e.get("name", "")
        aliases: list[str] = e.get("aliases", []) or []
        match_names = [name] + [a for a in aliases if a]
        assigned = False
        for i, shard in enumerate(shards):
            if any(nm in shard for nm in match_names):
                result[i].append(e)
                assigned = True
        if not assigned:
            # 未匹配任何分片 → 归入第 0 分片（保守保留）
            result[0].append(e)
    return result


# ────────────────────────────────────────────────────────────
# 快速路径：单次 Layer 1 归一化 (≤12K 字)
# ────────────────────────────────────────────────────────────

_LAYER1_SYSTEM = """你是一个实体归一化专家。你的任务是：接收一批从原文各段落中分别提取的"实体碎片"，将它们合并为规范实体列表。

## 任务
1. 识别同义异名实体（例如"史密斯"与"史先生"是同一人，"X国"与"X方"是同一个国家）
2. 融合描述：将同一实体的所有描述片段合并为一段完整描述（100字以内）
3. 修正不一致的类型标签（如有的碎片标"国家"有的标"Organization"，统一为合理类型）
4. 输出规范实体列表

## 规则
- 同名实体在不同块出现，描述互补则融合，描述冲突则取多数
- 别名字符重叠≥2且语义相同→合并（如"史密斯"与"史先生"）
- 上下级实体不合并（"财政部"≠"X国政府"）
- 二元关系词不合并（"X-Y贸易关系"≠"X"也不合并，单独保留）
- 无法确定是否同义时，保守不合并
- 每种类型统一为中文标签（国家/人物/组织/国际组织/企业/地点/概念/事件/...）

## 输出 JSON
{
  "entities": [
    {
      "name": "规范实体名",
      "aliases": ["别名1", "别名2"],
      "type": "统一类型",
      "description": "融合后的完整描述，≤100字"
    }
  ]
}

只输出 JSON。"""

_LAYER1_USER = """## 原文全文（理解实体关系的依据）
{source}

## 从各段落提取的实体碎片
{fragments}

请合并同义实体，融合描述，统一类型。只输出 JSON。"""


async def _layer1_normalize(
    raw_fragments: list[dict],
    source: str,
    freq_map: dict[str, int],
    log_fn: Any = None,
) -> list[dict]:
    """Layer 1 快速路径：单次 LLM 全量归一化。>15 实体自动拆批并行。"""
    n_total = len(raw_fragments)
    batch_size = 15  # qwen3.5-9b 在此规模 JSON 输出稳定

    if n_total <= batch_size:
        return await _layer1_normalize_batch(raw_fragments, source, freq_map, 1, 1, log_fn)

    # 拆批 + 并行 + 代码合并
    batches = [raw_fragments[i:i + batch_size] for i in range(0, n_total, batch_size)]
    total = len(batches)
    tasks = [_layer1_normalize_batch(b, source, freq_map, i + 1, total, log_fn) for i, b in enumerate(batches)]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: dict[str, dict] = {}
    for i, result in enumerate(all_results):
        if isinstance(result, Exception):
            logger.warning("[Layer1] 批次 %d 失败: %s", i + 1, result)
            # 失败批次 → 保守保留原始碎片
            for p in batches[i]:
                nm = p.get("name", "")
                if nm and nm not in merged:
                    merged[nm] = {"name": nm, "aliases": [], "type": p.get("type", "Unknown"),
                                  "description": (p.get("description") or "")[:200],
                                  "id": p.get("id", "")}
            continue
        for e in result:
            nm = e["name"]
            if nm in merged:
                # 合并：描述拼接，别名归并，类型投票
                existing = merged[nm]
                if e.get("description") and e["description"] != existing.get("description"):
                    existing["description"] = existing["description"] + "；" + e["description"]
                existing.setdefault("aliases", []).extend(a for a in e.get("aliases", []) if a not in existing.get("aliases", []))
            else:
                merged[nm] = dict(e)

    final = list(merged.values())
    if log_fn:
        log_fn("agents", f"Layer1 分批归一化: {n_total} 碎片(×{total}批) → {len(final)} 规范实体")
    logger.info("[Layer1] 分批: %d 碎片(×%d批) → %d 规范实体", n_total, total, len(final))
    return final


async def _layer1_normalize_batch(
    raw_fragments: list[dict],
    source: str,
    freq_map: dict[str, int],
    batch_idx: int,
    total_batches: int,
    log_fn: Any = None,
) -> list[dict]:
    """Layer 1 单批归一化。"""

    lines = []
    for i, p in enumerate(raw_fragments, 1):
        nm = p.get("name", "?")
        tp = p.get("type", "") or "未知"
        desc = (p.get("description") or "")[:200]
        fm = freq_map.get(nm, "?")
        lines.append(f"  {i}. 名={nm}  类型={tp}  频次={fm}  描述={desc}")
    frags_text = "\n".join(lines)

    source_trim = _smart_sample(source, 8000)

    prompt = _LAYER1_USER.format(source=source_trim, fragments=frags_text)
    if total_batches > 1:
        prompt += f"\n\n（批次{batch_idx}/{total_batches}，{len(raw_fragments)} 个碎片。仅合并本批内同义实体——跨批合并由后续步骤处理。）"

    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    client = DeductionLLMClient()
    resp = await client.chat_json(
        [Message(role="user", content=prompt)],
        system=_LAYER1_SYSTEM,
        schema_name="l1_entities",
        temperature=0,
        max_tokens=_token_cfg("l1_normalize", len(raw_fragments)),
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(b.text for b in content if hasattr(b, "text"))

    raw = str(content)
    # 主解析 + 截断恢复（输出过长被 max_tokens 截断时）
    try:
        data = _parse_llm_json(raw)
    except Exception:
        last_complete = raw.rfind('"name"')
        if last_complete > 0:
            truncated = raw[:last_complete - 1] + "\n]}"
            try:
                data = _parse_llm_json(truncated)
                logger.warning("[Layer1] 输出被截断，恢复到最后一个完整条目")
            except Exception:
                raise
        else:
            raise
    entities = data.get("entities", [])
    if not isinstance(entities, list) or len(entities) == 0:
        raise ValueError(f"Layer 1 empty entities. Raw: {str(content)[:300]}")

    result = []
    for e in entities:
        if isinstance(e, dict) and e.get("name"):
            result.append({
                "name": str(e["name"]).strip(),
                "aliases": [str(a).strip() for a in e.get("aliases", []) if a],
                "type": str(e.get("type", "Unknown")).strip(),
                "description": str(e.get("description", "")).strip()[:200],
            })

    if log_fn:
        log_fn("agents", f"Layer1 归一化: {len(raw_fragments)} 碎片 → {len(result)} 规范实体")
    logger.info("[Layer1] 归一化: %d 碎片 → %d 规范实体", len(raw_fragments), len(result))
    return result


# ────────────────────────────────────────────────────────────
# 分片路径：超长文本 Layer 1
# ────────────────────────────────────────────────────────────

_SHARD_SYSTEM = """你是一个实体归一化专家。你当前看到的文本，只是全文的其中一个局部分片，信息是局部、残缺、不完整的。

你的本轮任务不是「一次性完成全文最终归一化」，而是：输出高质量、可被后续全局合并模块融合的「局部规范实体碎片」。

## 分片级保守原则（最重要）
- 局部分不清是否同一实体 → 绝对不合并
- 局部看不出别名关系 → 不强行关联
- 局部只看到单次出场 → 必须保留，不得丢弃
- 同名但局部语义不明 → 分开保留，留给全局合并决策
所有「全局级判断」全部禁止在分片层执行，交由后续内存合并阶段。

## 实体归一化规范（分片级）
1. 局部同义严格合并：仅合并本片段可 100% 确认的同义（同一人物不同称谓、同一国家不同表述、同一组织简称全称）
2. 禁止跨语境强行合并
3. 描述生成：基于当前分片可见行为写局部描述，不脑补、不预测全文。40–100 字，记录：出场行为、立场、动作、关联事件。
4. 类型标准化：国家/政权/人物/军事组织/国际联盟/企业/机构/地理区域/资源/经济概念/条约/事件

## 必须保留的实体（分片宁可多保留、不可漏）
- 只出现一次但有独立动作、表态、参与事件的实体
- 临时联盟、临时机构、临时派系
- 小众势力、边缘参与者、次级官员

## 分片层绝对禁止行为
- 禁止基于局部信息做全局去重
- 禁止主观判断 "不重要所以删除"
- 禁止合并局部疑似、不确定实体
- 禁止丢弃低频、小众、临时实体
- 禁止补全文剧情、全局总结

## 输出 JSON
{
  "entities": [
    {
      "name": "标准实体名",
      "aliases": ["本片段出现的别名、简称、别称"],
      "type": "标准化中文类型",
      "description": "基于当前分片局部事实的精准短描述，不全局脑补"
    }
  ]
}

只输出 JSON。"""


async def _layer1_shard_normalize(
    shard_text: str,
    shard_entities: list[dict],
    shard_idx: int,
    total_shards: int,
    log_fn: Any = None,
) -> list[dict]:
    """分片级保守归一化：宁杂勿缺。"""

    lines = []
    for i, p in enumerate(shard_entities, 1):
        nm = p.get("name", "?")
        tp = p.get("type", "") or "未知"
        desc = (p.get("description") or "")[:200]
        lines.append(f"  {i}. 名={nm}  类型={tp}  描述={desc}")
    frags_text = "\n".join(lines) if lines else "（本分片无匹配实体碎片）"

    prompt = f"## 当前分片原文 (第 {shard_idx}/{total_shards} 片)\n{shard_text}\n\n"
    prompt += f"## 本分片匹配到的 Kuzu 实体碎片\n{frags_text}\n\n"
    prompt += "请对以上实体做保守归一化。只输出 JSON。"

    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    client = DeductionLLMClient()
    resp = await client.chat_json(
        [Message(role="user", content=prompt)],
        system=_SHARD_SYSTEM,
        schema_name="l1_entities",
        temperature=0,
        max_tokens=_token_cfg("l1_shard", len(shard_entities)),
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(b.text for b in content if hasattr(b, "text"))

    try:
        data = _parse_llm_json(str(content))
    except Exception as e:
        logger.warning("[Shard L1] 片 %d JSON 解析失败: %s", shard_idx, e)
        # 解析失败 → 直接返回原始碎片（宁漏勿误删）
        return [{
            "name": p.get("name", ""),
            "aliases": p.get("aliases", []),
            "type": p.get("type", "Unknown"),
            "description": (p.get("description") or "")[:150],
        } for p in shard_entities]

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        entities = []

    result = []
    for e in entities:
        if isinstance(e, dict) and e.get("name"):
            result.append({
                "name": str(e["name"]).strip(),
                "aliases": [str(a).strip() for a in e.get("aliases", []) if a],
                "type": str(e.get("type", "Unknown")).strip(),
                "description": str(e.get("description", "")).strip()[:150],
            })

    # 补漏: 原始碎片中未出现在 LLM 输出中的 → 保守保留
    llm_names = {r["name"] for r in result}
    for p in shard_entities:
        nm = p.get("name", "")
        if nm and nm not in llm_names:
            result.append({
                "name": nm,
                "aliases": p.get("aliases", []),
                "type": p.get("type", "Unknown"),
                "description": (p.get("description") or "")[:150],
            })

    if log_fn:
        log_fn("agents", f"  Layer1 分片[{shard_idx}/{total_shards}]: "
               f"{len(shard_entities)} 碎片 → {len(result)} 局部实体")
    return result


# ────────────────────────────────────────────────────────────
# 内存合并层 (纯代码, 无 LLM)
# ────────────────────────────────────────────────────────────

def _memory_merge(all_shard_entities: list[dict], domain: str = "") -> tuple[list[dict], list[dict]]:
    """纯内存合并所有分片的局部实体。返回 (merged, conflicts)。

    merged:   代码已合并的规范实体列表
    conflicts: 需 LLM 裁定的歧义实体对 [{"a":..., "b":..., "reason":...}, ...]
    """
    jaccard_threshold = _JACCARD_THRESHOLDS.get(domain, _JACCARD_THRESHOLDS["_default"])
    # 1. HashIndex: name → 出现记录
    by_name: dict[str, list[dict]] = defaultdict(list)
    # alias → canonical_name
    alias_map: dict[str, str] = {}

    for e in all_shard_entities:
        name = e["name"]
        by_name[name].append(e)
        for a in e.get("aliases", []):
            # 多地注册别名：取首次
            if a not in alias_map:
                alias_map[a] = name

    # 2. 别名链归并：某分片把"闯王"当独立实体，但另一分片标注"闯王"为"李自成"的别名
    merge_sources: set[str] = set()
    name_infos: dict[str, dict] = {}

    for name, records in by_name.items():
        all_types = []
        all_descs = []
        all_aliases: set[str] = set()
        for r in records:
            all_types.append(r.get("type", ""))
            all_descs.append(r.get("description", ""))
            all_aliases.update(r.get("aliases", []))
        name_infos[name] = {
            "types": all_types,
            "descs": all_descs,
            "aliases": all_aliases,
        }

    # 归并：别名指向主名 → 主名吸收全部信息
    for alias, target in alias_map.items():
        if alias != target and alias in name_infos:
            if target not in name_infos:
                name_infos[target] = {"types": [], "descs": [], "aliases": set()}
            # 迁移
            name_infos[target]["types"].extend(name_infos[alias]["types"])
            name_infos[target]["descs"].extend(name_infos[alias]["descs"])
            name_infos[target]["aliases"].update(name_infos[alias]["aliases"])
            name_infos[target]["aliases"].add(alias)
            merge_sources.add(alias)

    # 3. 精确同名合并：投票类型 + 去重描述
    merged = []
    for name, info in name_infos.items():
        if name in merge_sources:
            continue
        # 类型投票
        types = [t for t in info["types"] if t and t != "Unknown"]
        majority_type = max(set(types), key=types.count) if types else "Unknown"
        # 描述拼接去重
        descs = list(dict.fromkeys(d for d in info["descs"] if d))
        merged_desc = "；".join(descs)[:300]
        merged.append({
            "name": name,
            "aliases": sorted(info["aliases"] - {name}),
            "type": majority_type,
            "description": merged_desc,
        })

    # 4. Jaccard 高相似名检测 → 输出为冲突对供 LLM 裁定
    # 仅同类型实体对比，避免"美军"(Military) vs "美企"(Company) 误匹配
    conflicts = []
    merged_names = [m["name"] for m in merged]
    merged_types = [m["type"] for m in merged]
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            ti, tj = merged_types[i], merged_types[j]
            # 类型过滤：不同类型直接跳过（Unknown 除外）
            if ti != tj and ti != "Unknown" and tj != "Unknown":
                continue
            sim = _char_jaccard(merged_names[i], merged_names[j])
            if sim > jaccard_threshold:
                conflicts.append({
                    "a": merged[i]["name"],
                    "b": merged[j]["name"],
                    "a_type": ti, "b_type": tj,
                    "similarity": round(sim, 2),
                    "same_type": ti == tj or ti == "Unknown" or tj == "Unknown",
                })

    return merged, conflicts


# ────────────────────────────────────────────────────────────
# LLM 全局精修 (仅处理代码无法裁定的歧义)
# ────────────────────────────────────────────────────────────

_REFINE_SYSTEM = """你是实体合并专家。以下是代码预合并后的实体列表，以及代码无法裁定的歧义实体对。

## 你的任务
1. 每条实体描述精炼到 ≤100 字（合并重复信息，去除冗余）
2. 裁定冲突对：
   - 高相似名且类型相同 → 基本可判定为同一实体，合并描述，任选一规范名作为主名
   - 高相似名但类型不同 → 需结合原文裁决（如"X集团"vs"X手机"可能是同一企业的主品牌和产品线）
   - 无法确定 → 保守不合并，分别保留
3. 类型标准化为中文标签：国家/政权/人物/军事组织/国际联盟/企业/机构/地理区域/经济概念/条约/事件

## 输出 JSON
{
  "entities": [
    {
      "name": "规范实体名",
      "aliases": ["别名1", "别名2"],
      "type": "统一中文类型",
      "description": "精炼描述 ≤100字"
    }
  ]
}

只输出 JSON。"""


async def _layer1_global_refine(
    merged: list[dict],
    conflicts: list[dict],
    source: str,
    log_fn: Any = None,
) -> list[dict]:
    """LLM 全局精修：裁定歧义 + 描述精炼。"""

    sample = _smart_sample(source, 6000)

    prompt_parts = [
        "## 原文采样",
        sample,
        "\n## 代码预合并后的实体列表 (描述可能冗余，需精炼)",
        _json.dumps(merged, ensure_ascii=False, indent=2),
    ]
    if conflicts:
        prompt_parts.append("\n## 待裁定歧义实体对 (需判断是否合并)")
        prompt_parts.append(_json.dumps(conflicts, ensure_ascii=False, indent=2))
    else:
        prompt_parts.append("\n## 无歧义实体对需要裁定")
    prompt_parts.append("\n请精炼描述并裁定歧义。只输出 JSON。")

    prompt = "\n".join(prompt_parts)

    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    client = DeductionLLMClient()
    resp = await client.chat_json(
        [Message(role="user", content=prompt)],
        system=_REFINE_SYSTEM,
        schema_name="l1_entities",
        temperature=0,
        max_tokens=_token_cfg("l1_refine", len(merged)),
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(b.text for b in content if hasattr(b, "text"))

    data = _parse_llm_json(str(content))
    entities = data.get("entities", [])
    if not isinstance(entities, list) or len(entities) == 0:
        logger.warning("[Global Refine] LLM 返回空，使用代码合并结果")
        return merged

    result = []
    for e in entities:
        if isinstance(e, dict) and e.get("name"):
            result.append({
                "name": str(e["name"]).strip(),
                "aliases": [str(a).strip() for a in e.get("aliases", []) if a],
                "type": str(e.get("type", "Unknown")).strip(),
                "description": str(e.get("description", "")).strip()[:200],
            })

    if log_fn:
        log_fn("agents", f"Layer1 全局精修: {len(merged)} 预合并 → {len(result)} 规范实体"
               f" (裁定 {len(conflicts)} 对歧义)")
    logger.info("[Layer1 Global] 预合并 %d → 精修 %d, 裁定 %d 歧义对",
                len(merged), len(result), len(conflicts))
    return result


# ────────────────────────────────────────────────────────────
# Layer 2: 逐批角色判定
# ────────────────────────────────────────────────────────────

_LAYER2_SYSTEM = """你是战略分析专家。对每个实体进行分层判定——先分组归属，再判定决策主权，最后分配 tier。

## 第一步：归属分组
将每个实体归入其所属的博弈单元（通常是其所属主权国家或最高组织）：

1. **政治人物**（总统/总理/主席/首相/部长/将领）→ 归入其所属**主权国家**
2. **军事组织/军队/武装** → 归入其所属**国家**
3. **政府机构/职能部门** → 归入其所属**上级主体**（国家或独立组织）
4. **主权国家/最高政权** → 自身即根节点
5. **跨国企业/国际组织/NGO** → 若原文明确其跨越多国管辖边界独立行动 → 独立成组；否则归入主要关联国
6. **无法确定归属的实体** → 标记为 unaffiliated，后续独立评估

## 第二步：组内决策主权判定

每个博弈单元（组）内最多产生 **1 个 tier1**——即该组在博弈中的最高决策代表：

| 实体角色 | tier | 条件 |
|---------|------|------|
| 主权国家/最高政权 | tier1 | 始终——国家是不可再分的最高博弈单位 |
| 该国元首/政府首脑 | tier2 | 其决策 = 该国决策，不应另建独立 agent |
| 该国其他官员/部长/将领 | tier3 | 决策空间已被元首覆盖 |
| 该国地方势力/中层将领 | tier3 | 除非原文提供 ≥3 项**独立于中央**的决策证据 |
| 独立军阀/叛军/割据势力 | tier1 | 需原文明确 ≥3 项独立决策证据（独立交火/占领/征税/结盟） |
| 跨国企业/独立机构 | tier1 | 若原文明确其决策跨越 3 个以上国家的管辖边界 |
| 国际组织/军事联盟 | tier2 | 若其核心成员国 ≥3 已独立列席 → tier3 |
| 战略地缘区域 | tier2 | 关键海峡/主要盆地/边境走廊作为争夺舞台 |
| 工具/资源/概念/关系词 | tier3 | 永远——非决策实体 |

## 第三步：tier 分配汇总

- tier1（生成独立 agent）：主权国家、独立武装、跨国巨头——每一组最多 1 个
- tier2（保留数据、不建 agent）：归入上级的核心人物、战略地缘区域、未满 3 国的国际组织
- tier3（DISCARD）：被上级覆盖的官员/军队/机构、纯工具/资源/概念/背景名词

## 特殊规则

- 白名单中的实体 → 无条件 tier1（但其属下的官员仍按上述规则降级）
- 原文确实出现独立军阀/叛乱实体的 → 独立建组，不影响原国家组保留 1 个 tier1
- 频次陷阱：高频≠重要，1 次关键决策 > 10 次背景提及
- 不确定归属时 → 保守保留为 tier2 并注明原因

## 输出 JSON
{"results": [{"name":"实体名","tier":1|2|3,"reason":"≤30字理由及原文行为证据","group":"所属组名或独立"}]}

tier 取 1/2/3。只输出 JSON。"""

_LAYER2_USER = """## 原文全文
{source}

## 待判定实体（已归一化，含融合描述）
{batch}

请逐实体判定 tier (1/2/3)。只输出 JSON。"""


async def _layer2_classify_batch(
    batch: list[dict],
    source: str,
    batch_idx: int,
    total_batches: int,
    log_fn: Any = None,
) -> list[dict]:
    """Layer 2: 判定一批实体。"""

    lines = []
    for e in batch:
        desc = e.get("description", "")[:300]
        tp = e.get("type", "?")
        fm = e.get("freq", "?")
        aliases = ",".join(e.get("aliases", [])) or "无"
        lines.append(
            f"  - 名={e['name']}  类型={tp}  频次={fm}  别名={aliases}\n"
            f"    描述: {desc}"
        )
    batch_text = "\n".join(lines)

    source_trim = _smart_sample(source, 8000)

    prompt = _LAYER2_USER.format(source=source_trim, batch=batch_text)

    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    client = DeductionLLMClient()
    resp = await client.chat_json(
        [Message(role="user", content=prompt)],
        system=_LAYER2_SYSTEM,
        schema_name="l2_results",
        temperature=0,
        max_tokens=_token_cfg("l2_classify", len(batch)),
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(b.text for b in content if hasattr(b, "text"))

    data = _parse_llm_json(str(content))
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Layer 2 batch {batch_idx} bad format: {str(content)[:200]}")

    _tag = f"L2[{batch_idx}/{total_batches}]"
    t1 = t2 = 0
    out = []
    for r in results:
        if isinstance(r, dict) and r.get("name"):
            # 优先读 tier，兼容旧 decision 格式
            tier_raw = r.get("tier")
            if tier_raw is not None:
                tier = int(tier_raw)
                if tier not in (1, 2, 3):
                    tier = 3
            else:
                d = str(r.get("decision", "")).upper().strip()
                tier = 1 if d == "KEEP" else 3
            if tier == 1:
                t1 += 1
            elif tier == 2:
                t2 += 1
            out.append({
                "name": str(r["name"]).strip(),
                "tier": tier,
                "reason": str(r.get("reason", ""))[:60],
                "decision": "KEEP" if tier in (1, 2) else "DISCARD",
                "group": str(r.get("group", "")).strip(),
            })
    if log_fn:
        log_fn("agents", f"  Layer2 批次{_tag}: tier1={t1} tier2={t2} tier3={len(out)-t1-t2}/{len(out)}")
    logger.info("[Layer2] 批次%s: t1=%d t2=%d / %d", _tag, t1, t2, len(out))
    return out


async def _layer2_classify_all(
    entities: list[dict],
    source: str,
    batch_size: int = 10,
    log_fn: Any = None,
) -> dict[str, dict]:
    """Layer 2: 分批并行判定。"""

    batches = []
    for i in range(0, len(entities), batch_size):
        batches.append(entities[i:i + batch_size])

    total = len(batches)
    tasks = [
        _layer2_classify_batch(batch, source, idx + 1, total, log_fn)
        for idx, batch in enumerate(batches)
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    decisions: dict[str, dict] = {}
    for i, result in enumerate(all_results):
        if isinstance(result, Exception):
            logger.warning("[Layer2] 批次 %d 失败: %s", i + 1, result)
            if log_fn:
                log_fn("agents", f"  Layer2 批次{i+1} 失败，拆为单实体重试...")
            # 拆分失败批次为单实体重试
            retry_tasks = [
                _layer2_classify_batch([e], source, 0, 1, log_fn)
                for e in batches[i]
            ]
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
            recovered = 0
            for j, rr in enumerate(retry_results):
                e = batches[i][j]
                if isinstance(rr, Exception):
                    decisions[e["name"]] = {"decision": "DISCARD", "tier": 3,
                                            "reason": f"批次{i+1}重试失败"}
                elif rr and len(rr) > 0:
                    decisions[e["name"]] = {
                        "decision": rr[0]["decision"],
                        "tier": rr[0].get("tier", 3),
                        "reason": rr[0]["reason"],
                        "group": rr[0].get("group", ""),
                    }
                    recovered += 1
                else:
                    decisions[e["name"]] = {"decision": "DISCARD", "tier": 3,
                                            "reason": f"批次{i+1}LLM失败"}
            if log_fn and recovered:
                log_fn("agents", f"  Layer2 重试恢复 {recovered}/{len(batches[i])} 个实体")
            continue
        for r in result:
            decisions[r["name"]] = {
                "decision": r["decision"],
                "tier": r.get("tier", 3),
                "reason": r["reason"],
                "group": r.get("group", ""),
            }

    for e in entities:
        if e["name"] not in decisions:
            decisions[e["name"]] = {"decision": "DISCARD", "tier": 3, "reason": "LLM未覆盖"}

    return decisions


# ────────────────────────────────────────────────────────────
# 兜底 + 层次修正
# ────────────────────────────────────────────────────────────

def _fallback_classify(
    registry: EntityRegistry,
    entities: list[RegisteredEntity],
    log_fn: Any = None,
) -> None:
    """LLM 不可用时的兜底规则。优先保留国家/组织为 tier1，人物降为 tier2。"""
    # 按 group 分组（兜底时大多数实体无 group，从 entity_alias.json 推断）
    country_group: dict[str, list[str]] = {}
    for e in entities:
        if e.type in ("Country", "国家"):
            country_group.setdefault(e.name, []).append(e.name)
        elif e.type in ("Person", "人物") and e.group:
            country_group.setdefault(e.group, []).append(e.name)

    for e in entities:
        if e.type in ("Country", "国家", "Organization", "组织") and e.freq >= 1:
            e.decision = "KEEP"
            e.tier = 1
            e.reason = "兜底(国家/组织)"
            registry.kept += 1
            registry.tier1_count += 1
        elif e.type in ("Person", "人物") and e.group:
            # 有归属组 → tier2（决策被所属国家覆盖）
            e.decision = "KEEP"
            e.tier = 2
            e.reason = f"兜底(人物归属{e.group})"
            registry.kept += 1
            registry.tier2_count += 1
        elif e.type in ("Person", "人物") and e.freq >= 5:
            e.decision = "KEEP"
            e.tier = 2
            e.reason = "兜底(高频人物)"
            registry.kept += 1
            registry.tier2_count += 1
        elif e.freq >= 8:
            e.decision = "KEEP"
            e.tier = 2
            e.reason = f"兜底(高频≥8)"
            registry.kept += 1
            registry.tier2_count += 1
        else:
            e.decision = "DISCARD"
            e.reason = "兜底排除"
            registry.discarded += 1
            registry.discard_reasons["兜底排除"] = registry.discard_reasons.get("兜底排除", 0) + 1
    if log_fn:
        log_fn("agents", f"兜底规则分类: {registry.kept} 保留 (LLM不可用)")


# ────────────────────────────────────────────────────────────
# Layer 3: LLM 交叉裁决 — 全局冗余检测 (完整改造)
#   新增: 哈希缓存 / 别名词典预合并 / 配置解耦 / merge / 权重联动 / 方差日志
# ────────────────────────────────────────────────────────────

def _layer3_cache_key(registry: EntityRegistry) -> str:
    """Defect #4: 基于所有 KEEP 实体生成 MD5 哈希。"""
    kept = sorted(registry.get_kept(), key=lambda e: e.name)
    parts = []
    for e in kept:
        parts.append(f"{e.name}|{e.type}|{e.freq}")
    raw = ";".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _load_alias_dict(domain: str) -> dict[str, set[str]]:
    """Defect #2: 加载别名词典。返回 {主名: {别名集合}}。"""
    try:
        rule_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "rule"
        path = rule_dir / "entity_alias.json"
        if not path.exists():
            # 尝试环境变量路径
            import os
            env_dir = os.environ.get("FORGE_RULE_DIR", "")
            if env_dir:
                path = Path(env_dir) / "entity_alias.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        return {}

    domain_data = data.get(domain, {})
    result: dict[str, set[str]] = {}
    for main_name, aliases in domain_data.items():
        if isinstance(aliases, list):
            result[main_name] = {str(a) for a in aliases if a}
    return result


def _pre_merge_aliases(
    registry: EntityRegistry,
    alias_dict: dict[str, set[str]],
) -> int:
    """Defect #2: 代码别名词典预合并。返回合并的实体数。"""
    # 构建反向索引: 别名 → 主名
    rev: dict[str, str] = {}
    for main_name, aliases in alias_dict.items():
        rev[main_name] = main_name
        for a in aliases:
            rev[a] = main_name

    kept = registry.get_kept()
    kept_names = {e.name for e in kept}
    # 找出所有可与字典匹配的实体对
    merge_map: dict[str, str] = {}  # source → target
    for main_name in alias_dict:
        if main_name not in kept_names:
            continue
        aliases = alias_dict[main_name]
        for e in kept:
            ename = e.name
            if ename == main_name:
                continue
            if ename in aliases:
                merge_map[ename] = main_name

    # 执行代码层合并
    merged_count = 0
    for source, target in merge_map.items():
        src_entity = registry.entities.get(source)
        tgt_entity = registry.entities.get(target)
        if not src_entity or not tgt_entity:
            continue
        if src_entity.decision != "KEEP":
            continue
        # 合并: 频次累加, 别名归入, 描述拼接
        tgt_entity.freq += src_entity.freq
        if src_entity.rich_description:
            tgt_entity.rich_description += "；" + src_entity.rich_description
        if source not in tgt_entity.aliases:
            tgt_entity.aliases.append(source)
        # 标记源为 DISCARD
        src_entity.decision = "DISCARD"
        src_entity.reason = f"代码别名合并→{target}"
        registry.kept -= 1
        registry.discarded += 1
        reason_key = "代码别名合并"
        registry.discard_reasons[reason_key] = registry.discard_reasons.get(reason_key, 0) + 1
        merged_count += 1
        logger.info("[Layer3 PreMerge] %s → %s (代码别名)", source, target)

    return merged_count


def _load_layer3_config(domain: str) -> dict:
    """Defect #3: 加载 Layer3 配置文件。"""
    config_paths = [
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "rule" / "layer3_config.yaml",
    ]
    import os
    env_dir = os.environ.get("FORGE_RULE_DIR", "")
    if env_dir:
        config_paths.insert(0, Path(env_dir) / "layer3_config.yaml")

    for path in config_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    domain_cfg = data.get(domain, {})
                    if isinstance(domain_cfg, dict):
                        return domain_cfg
                    # 文学/历史/叙事域回退到 novel 配置
                    fallback_domain = "novel" if domain in ("novel", "history", "narrative") else "geo_strategy"
                    defaults = data.get(fallback_domain, {})
                    if isinstance(defaults, dict):
                        return defaults
    # 内置默认
    return {
        "system_prompt": "",
        "min_kept_for_check": 3,
        "warn_threshold": 8,
        "sample_chars": 5000,
        "desc_truncate": 80,
        "token": {
            "l1_normalize": {"base": 200, "per": 150, "cap": 12000},
            "l1_shard":     {"base": 200, "per": 200, "cap": 6000},
            "l1_refine":    {"base": 200, "per": 120, "cap": 8000},
            "l2_classify":  {"base": 150, "per": 100, "cap": 4000},
            "l3_cross":     {"base": 300, "per": 60,  "cap": 3000},
        },
        "log_file": "",
        "cache_enabled": True,
        "fallback_rules": {"org_overlap_threshold": 3},
    }


def _reconcile_weights(registry: EntityRegistry) -> None:
    """Defect #6: 降级/合并后归一化所有保留实体的频次权重。"""
    kept = registry.get_kept()
    if not kept:
        return
    # 频次归一化: 每个保留实体 freq = max(1, freq)
    for e in kept:
        e.freq = max(1, e.freq)
    logger.info("[Layer3 W] 权重联动: %d 个实体频次已归一化", len(kept))


def _log_variance(
    total_kept: int,
    downgrades: list,
    merges: list,
    notes: str,
    domain: str,
) -> None:
    """Defect #8: 方差量化日志。"""
    entry = {
        "kept_before": total_kept,
        "downgrade_count": len(downgrades),
        "downgrade_names": [d.get("name", "?") for d in downgrades if isinstance(d, dict)],
        "merge_count": len(merges),
        "merge_pairs": [
            f"{m.get('keep','?')}←{','.join(m.get('discard',[]))}"
            for m in merges if isinstance(m, dict)
        ],
        "notes": notes,
        "domain": domain,
    }
    _layer3_variance_log.append(entry)

    # 简易跨轮统计
    if len(_layer3_variance_log) >= 2:
        prev = _layer3_variance_log[-2]
        curr = entry
        prev_names = set(prev.get("downgrade_names", []))
        curr_names = set(curr.get("downgrade_names", []))
        if prev_names != curr_names:
            added = curr_names - prev_names
            removed = prev_names - curr_names
            diff_rate = len(added | removed) / max(len(prev_names | curr_names), 1)
            logger.info(
                "[Layer3 Variance] 降级变化: +%s -%s (差异率=%.0f%%)",
                list(added) if added else "无",
                list(removed) if removed else "无",
                diff_rate * 100,
            )


def _persist_variance(entry: dict, log_file: str) -> None:
    """P0-2: 方差日志持久化 — 追加一行 JSON 到文件。"""
    if not log_file:
        return
    try:
        import datetime
        entry["_ts"] = datetime.datetime.now().isoformat()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[Layer3 Log] 日志持久化失败: %s", e)


# ── Layer 3 主函数 (签名不变，内部全部改造) ──

async def _layer3_cross_validate(
    registry: EntityRegistry,
    source: str,
    log_fn: Any = None,
    domain: str = "",
) -> None:
    """Layer 3: LLM 交叉裁决 (改造版)。

    文学/历史/叙事域跳过 Layer 3 —— 其角色冗余场景极少，
    9B 模型易将核心角色误判为背景/附属，反而过杀。
    地缘/商业/军事域正常执行 LLM 冗余检测。
    """

    # ── 0. 配置加载 (Defect #3) ──
    from strategy_forge.core.config import config as _cfg
    _domain_raw = domain or getattr(_cfg, "active_domain", "") or getattr(_cfg, "domain", "")
    domain = _domain_raw or "geo_strategy"

    # 内容特征检测：决定是否跳过 Layer 3
    kept_before = registry.get_kept() if hasattr(registry, "get_kept") else []
    _skip = False
    if _domain_raw in ("novel", "history", "narrative"):
        _skip = True
    elif _domain_raw == "geo_strategy":
        _skip = False
    elif not _domain_raw:
        # 域名未知 → 内容特征检测
        person_types = {"Person", "人物"}
        geo_types = {"Country", "国家", "Organization", "组织", "国际组织"}
        n_person = sum(1 for e in kept_before if e.type in person_types)
        n_geo = sum(1 for e in kept_before if e.type in geo_types)
        total = len(kept_before)
        if total > 0 and n_person / total > 0.5 and n_geo < 3:
            _skip = True  # 人物为主+少组织 → 文学叙事
        # 否则不跳过 → 按地缘处理

    if _skip:
        kept = registry.get_kept()
        if log_fn:
            log_fn("agents", f"Layer3 跳过 (域={_domain_raw or '未知/文学特征'}: {len(kept)} KEEP，不执行冗余检测)")
        # 仍执行权重联动
        _reconcile_weights(registry)
        return
    cfg = _load_layer3_config(domain)
    min_kept = cfg.get("min_kept_for_check", 3)
    warn_threshold = cfg.get("warn_threshold", 8)
    sample_chars = cfg.get("sample_chars", 5000)
    desc_trunc = cfg.get("desc_truncate", 80)
    cache_enabled = cfg.get("cache_enabled", True)

    kept = registry.get_kept()
    total_kept = len(kept)
    if total_kept < min_kept:
        if log_fn:
            log_fn("agents", f"Layer3 跳过 (KEEP={total_kept}<{min_kept})")
        return

    # ── 1. 哈希缓存 (Defect #4) ──
    if cache_enabled:
        ck = _layer3_cache_key(registry)
        cached = _layer3_decision_cache.get(ck)
        if cached:
            logger.info("[Layer3 Cache] 命中, 跳过LLM")
            _apply_layer3_result(registry, cached)
            _reconcile_weights(registry)
            if log_fn:
                log_fn("agents",
                       f"Layer3 缓存命中: {total_kept} KEEP → {registry.kept} KEEP"
                       f" (降级 {len(cached.get('downgrades',[]))}, "
                       f"合并 {len(cached.get('merges',[]))})")
            return

    # ── 2. 代码别名词典预合并 (Defect #2) ──
    alias_dict = _load_alias_dict(domain)
    merged_count = 0
    if alias_dict:
        merged_count = _pre_merge_aliases(registry, alias_dict)
        if merged_count and log_fn:
            log_fn("agents", f"Layer3 代码预合并: {merged_count} 个实体 (减少LLM负担)")
        # 重新读取 KEEP 列表 (可能已缩减)
        kept = registry.get_kept()
        total_kept = len(kept)

    # ── 3. LLM 交叉裁决 (Defect #5: 新增 merge 支持) ──
    # 构建实体表格
    entity_lines = []
    for i, e in enumerate(kept, 1):
        tp = e.type or "?"
        desc = (e.rich_description or e.reason or "")[:desc_trunc]
        entity_lines.append(
            f"{i}. {e.name} | {tp} | 频次={e.freq} | {desc}"
        )
    entity_table = "\n".join(entity_lines)

    sample = _smart_sample(source, sample_chars)
    system_prompt = cfg.get("system_prompt", "").strip()
    if not system_prompt:
        system_prompt = """你是博弈实体冗余检测专家。基于整体种子材料，检测实体间是否存在决策权重叠。

## 核心原则：决策覆盖链

一个实体的决策空间 ⊆ 另一个实体的决策空间时 → 前者冗余。

检测路由（沿归属链自上而下）：
```
主权国家/最高政权（根节点）
 └─ 该国的政府首脑/元首 → 决策 = 国家决策 → 降级人物（tier2，不另建 agent）
     └─ 该国的部长/将领/官员 → 决策被元首覆盖 → 降级（tier3）
         └─ 地方势力/中层将领 → 无原文独立决策证据 → 降级（tier3）
             └─ 独立军阀/叛军/割据势力 → 有独立证据 → 保留

国际组织
 └─ 核心成员国已独立列席 ≥3 个 → 组织的决策由其成员投票决定 → 降级组织
 └─ 核心成员国未完全列席 → 组织有独立决策空间 → 保留
     └─ 组织隶属的秘书长/主席 → 决策 = 组织决策 → 降级人物

跨国企业/独立机构
 └─ 原文明确其决策跨越 3 个以上国家的管辖边界 → 独立组 → 保留
 └─ 主要依附单一国家 → 归入该国组 → 降级
```

## 冗余检测类型

| 类型 | 检测对象 | 判定 |
|------|---------|------|
| 人物-国家重叠 | 元首/总理 vs 其所属国家 | 降级人物 → tier2 |
| 人物-组织重叠 | 官员/职员 vs 其所属组织 | 降级人物 → tier3 |
| 下级-上级重叠 | 部长/将领 vs 政府/国家 | 降级下级 → tier3 |
| 组织-成员重叠 | 国际联盟 vs 其核心成员 | 核心成员 ≥3 已列席 → 降级组织 |
| 政府机构重叠 | 职能部门 vs 国家 | 降级机构 → tier3 |
| 军队-国家重叠 | 军队/武装 vs 所属国家 | 默认降级军队（仅原文明确独立军政行为时例外） |
| 同义名重叠 | 两个名称指向同一实体 | 合并 |

## 铁律

- **每一国家组最多保留 1 个 tier1**——国家本身为其最高博弈代表
- 独立军阀/叛军除外——它们是独立博弈单元，不受国家组 tier1 上限约束
- 优先保留国家/最高组织，降级其下属人员——而非反过来
- 无法确定归属 → 保留，不降级

## 输出 JSON
{"downgrades": [{"name":"实体名","new_tier":2|3,"reason":"≤30字"}],
 "merges": [{"source":"被合并名","target":"保留名","reason":"≤30字"}],
 "notes": "≤80字总结合并/降级逻辑"}

只输出 JSON。"""

    # 注入方法论 (entity_alias.json) — 文学叙事域不注入代理力/冗余，小说角色规则不同
    extra_method = []
    if domain not in ("novel", "history", "narrative"):
        if _A_REDUNDANCY_METHOD:
            extra_method.append(f"## 冗余检测方法论\n{_A_REDUNDANCY_METHOD}")
        if _A_AGENCY_METHOD:
            extra_method.append(f"## 实体代理力判定方法论\n{_A_AGENCY_METHOD}")
    else:
        # 文学叙事使用独立方法论：角色≠博弈主体，每个有独立弧的角色都是独立实体
        extra_method.append("""## 文学叙事冗余判断规则
- 文学叙事中，每个有【独立行动 + 独立对话 + 独立心理描写】的角色都是独立实体
- 核心角色绝不可降级：统治者/领袖≠其所属国家/组织（其个人决策、内心挣扎、性格弧线构成叙事核心）
- 「国家意志化身」「组织代表」等理由不可用于降级——个体角色是独立叙事主体，非工具
- 角色-组织关系不适用「组织-成员」冗余检测：将领≠其所属军队，个体≠其所属团体
- 主要反派/外部威胁≠「背景」——只要原文有具体描写其行动、动机、决策，即为核心叙事主体
- 上下级不降级：除非角色在文中完全无独立行为（仅作为他人命令的执行工具）
- 保守原则强化：文学叙事宁可多留角色，不可合并关键叙事主体。5-10 个角色是合理的下限""")
    if extra_method:
        system_prompt = system_prompt.rstrip() + "\n\n" + "\n\n".join(extra_method)

    prompt = (
        f"## 原文采样\n{sample}\n\n"
        f"## 当前保留的博弈实体 ({total_kept} 个)\n{entity_table}\n\n"
        f"请检测冗余，输出降级列表和合并列表。只输出 JSON。"
    )

    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    try:
        client = DeductionLLMClient()
        resp = await client.chat_json(
            [Message(role="user", content=prompt)],
            system=system_prompt,
            schema_name="l3_decisions",
            temperature=0,
            max_tokens=_get_token_limit(cfg, "l3_cross", total_kept),
        )
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(b.text for b in content if hasattr(b, "text"))
        data = _parse_llm_json(str(content))
    except Exception as e:
        logger.warning("[Layer3] LLM 调用失败: %s, 回退配置兜底规则", e)
        if log_fn:
            log_fn("agents", f"Layer3 LLM 失败({type(e).__name__})，执行配置兜底规则")
        _resolve_hierarchy(registry, cfg, log_fn)
        _reconcile_weights(registry)
        return

    downgrades = data.get("downgrades", [])
    merges = data.get("merges", [])
    notes = str(data.get("notes", ""))[:80]

    if not isinstance(downgrades, list):
        downgrades = []
    if not isinstance(merges, list):
        merges = []

    # ── 3a. 执行 merge (Defect #5) ──
    merge_applied = 0
    for m in merges:
        if not isinstance(m, dict):
            continue
        keep_name = str(m.get("keep", "")).strip()
        discard_names = m.get("discard", [])
        if not isinstance(discard_names, list):
            discard_names = []
        reason = str(m.get("reason", "LLM合并"))[:40]
        if not keep_name:
            continue
        tgt = registry.entities.get(keep_name)
        if not tgt or tgt.decision != "KEEP":
            continue
        for dn in discard_names:
            dn = str(dn).strip()
            src = registry.entities.get(dn)
            if not src or src.decision != "KEEP":
                continue
            # 合并频次 + 描述 + 别名
            tgt.freq += src.freq
            if src.rich_description:
                tgt.rich_description += "；" + src.rich_description
            if dn not in tgt.aliases:
                tgt.aliases.append(dn)
            for a in src.aliases:
                if a not in tgt.aliases and a != keep_name:
                    tgt.aliases.append(a)
            src.decision = "DISCARD"
            src.reason = f"L3合并→{keep_name}({reason})"
            registry.kept -= 1
            registry.discarded += 1
            reason_key = f"L3合并({reason[:30]})"
            registry.discard_reasons[reason_key] = registry.discard_reasons.get(reason_key, 0) + 1
            merge_applied += 1
            logger.info("[Layer3 Merge] %s → %s (%s)", dn, keep_name, reason)

    # ── 3b. 执行 downgrade ──
    downgrade_applied = 0
    for item in downgrades:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        reason = str(item.get("reason", "LLM冗余检测"))[:40]
        if not name:
            continue
        entity = registry.entities.get(name)
        if entity and entity.decision == "KEEP":
            entity.decision = "DISCARD"
            entity.reason = f"L3({reason})"
            registry.kept -= 1
            registry.discarded += 1
            reason_key = f"L3({reason[:30]})"
            registry.discard_reasons[reason_key] = registry.discard_reasons.get(reason_key, 0) + 1
            downgrade_applied += 1
            logger.info("[Layer3 Downgrade] %s → %s", name, reason)

    # ── 3c. 组内去重：每一博弈单元最多 1 个 tier1 ──
    tier1_entities = [e for e in registry.entities.values() if e.tier == 1]
    # 收集 group 信息（L2 输出的 group 字段 + 从 tier1 实体名推断）
    group_map: dict[str, list[str]] = {}  # group_name → [entity_names]
    for e in tier1_entities:
        g = e.group.strip() if e.group else ""
        if not g:
            # 从类型推断归属：Person 类型 → 其描述/别名中可能含国家名
            # 无 group 则自成一族
            g = f"__{e.name}"
        group_map.setdefault(g, []).append(e.name)

    group_overrides = 0
    for group_name, members in group_map.items():
        if group_name.startswith("__"):
            continue  # 独立组，无需处理
        if len(members) <= 1:
            continue
        # 优先保留类型为 Country/Organization 的实体，降级其他
        prioritized: list[str] = []
        rest: list[str] = []
        for mname in members:
            ent = registry.entities.get(mname)
            if ent and ent.type in ("Country", "国家", "Organization", "组织", "政权"):
                prioritized.append(mname)
            else:
                rest.append(mname)
        keep_list = prioritized[:1] or [members[0]]
        for mname in members:
            if mname in keep_list:
                continue
            ent = registry.entities.get(mname)
            if ent and ent.tier == 1:
                ent.tier = 2
                ent.tier_evidence = f"L3组内去重({group_name})"
                group_overrides += 1

    if group_overrides:
        logger.info("[Layer3 Group] 组内去重: %d 个实体 tier1→2", group_overrides)

    # ── 4. 写缓存 (Defect #4) ──
    if cache_enabled:
        ck = _layer3_cache_key(registry)
        _layer3_decision_cache.set(ck, {
            "downgrades": downgrades,
            "merges": merges,
            "notes": notes,
        })

    # ── 5. 权重联动 (Defect #6) ──
    _reconcile_weights(registry)

    # ── 6. 方差日志 (Defect #8) ──
    _log_variance(total_kept, downgrades, merges, notes, domain)
    _persist_variance(entry=_layer3_variance_log[-1], log_file=cfg.get("log_file", ""))

    # ── 7. 日志 ──
    total_changes = merge_applied + downgrade_applied
    if total_changes > 0:
        parts = []
        if merge_applied:
            parts.append(f"合并 {merge_applied}")
        if downgrade_applied:
            parts.append(f"降级 {downgrade_applied}")
        log_msg = f"Layer3 交叉裁决: {total_kept} KEEP → {registry.kept} KEEP ({', '.join(parts)})"
        if notes:
            log_msg += f" [{notes}]"
        if log_fn:
            log_fn("agents", log_msg)
    else:
        if log_fn:
            log_fn("agents",
                   f"Layer3 交叉裁决: {total_kept} KEEP 无冗余"
                   f"{' — ' + notes if notes else ''}"
                   f"{' (≥' + str(warn_threshold) + '实体无冗余，建议人工复核)' if total_kept >= warn_threshold else ''}")
    logger.info(
        "[Layer3] 完成: %d KEEP → %d KEEP, 合并 %d, 降级 %d",
        total_kept, registry.kept, merge_applied, downgrade_applied,
    )


def _apply_layer3_result(registry: EntityRegistry, result: dict) -> None:
    """Defect #4: 从缓存结果应用到 registry (纯代码, 无 LLM)。"""
    downgrades = result.get("downgrades", [])
    merges = result.get("merges", [])

    # 执行 merge
    for m in merges:
        if not isinstance(m, dict):
            continue
        keep_name = str(m.get("keep", "")).strip()
        discard_names = m.get("discard", [])
        if not isinstance(discard_names, list):
            continue
        reason = str(m.get("reason", "缓存合并"))[:40]
        tgt = registry.entities.get(keep_name)
        if not tgt or tgt.decision != "KEEP":
            continue
        for dn in discard_names:
            dn = str(dn).strip()
            src = registry.entities.get(dn)
            if not src or src.decision != "KEEP":
                continue
            tgt.freq += src.freq
            if src.rich_description:
                tgt.rich_description += "；" + src.rich_description
            if dn not in tgt.aliases:
                tgt.aliases.append(dn)
            src.decision = "DISCARD"
            src.reason = f"L3合并→{keep_name}({reason})"
            registry.kept -= 1
            registry.discarded += 1

    # 执行 downgrade
    for item in downgrades:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        reason = str(item.get("reason", "缓存冗余"))[:40]
        entity = registry.entities.get(name)
        if entity and entity.decision == "KEEP":
            entity.decision = "DISCARD"
            entity.reason = f"L3({reason})"
            registry.kept -= 1
            registry.discarded += 1


def _resolve_hierarchy(
    registry: EntityRegistry,
    cfg: dict | None = None,
    log_fn: Any = None,
) -> None:
    """Defect #7: 配置驱动的兜底层级修正。

    cfg=None 时使用内置硬编码规则 (兼容旧调用)。
    """
    kept_entities = registry.get_kept()
    kept_names = {e.name for e in kept_entities}
    to_discard: list[tuple[RegisteredEntity, str]] = []

    # 读取配置的兜底规则
    fallback = (cfg or {}).get("fallback_rules", {}) if cfg else {}
    org_threshold = fallback.get("org_overlap_threshold", 3)
    custom_org_map: dict = fallback.get("org_members_map", {})
    custom_person_map: dict = fallback.get("person_country_map", {})

    # 合并内置 + 配置
    org_map = {**_ORG_MEMBERS}
    if isinstance(custom_org_map, dict):
        for k, v in custom_org_map.items():
            org_map[str(k)] = frozenset(str(x) for x in (v if isinstance(v, list) else []))

    person_map = {**_PERSON_COUNTRY}
    if isinstance(custom_person_map, dict):
        for k, v in custom_person_map.items():
            person_map[str(k)] = str(v)

    for e in kept_entities:
        if e.name in org_map:
            core = org_map[e.name]
            overlap = core & kept_names
            if len(overlap) >= org_threshold:
                to_discard.append((e, f"组织(成员国重叠:{len(overlap)}国)"))
        elif e.name in person_map:
            country = person_map[e.name]
            if country in kept_names:
                to_discard.append((e, f"人物(归入{country})"))

    for e, reason in to_discard:
        e.decision = "DISCARD"
        e.reason = reason
        registry.kept -= 1
        registry.discarded += 1

    if to_discard and log_fn:
        log_fn("agents", f"兜底层级修正: {len(to_discard)} 个重叠实体降级")
    if to_discard:
        logger.info("[EntityRegistry] 兜底修正: %d 个重叠实体降级", len(to_discard))


# ────────────────────────────────────────────────────────────
# 去重回退 (Layer 1 不可用时)
# ────────────────────────────────────────────────────────────

def _dedup_fallback(
    raw: list[dict],
    alias_map: dict[str, str],
) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for p in raw:
        name = p.get("name", "")
        std_name = alias_map.get(name, name)
        if std_name in seen:
            continue
        seen.add(std_name)
        d = {"name": std_name, "type": p.get("type", "Unknown"),
             "description": p.get("description", ""), "aliases": [],
             "id": p.get("id", "")}
        if std_name != name:
            d.setdefault("aliases", []).append(name)
        deduped.append(d)

    if len(deduped) > 1:
        names = [d["name"] for d in deduped]
        name_to_d = {d["name"]: d for d in deduped}
        merged: dict[str, str] = {}
        for i, short in enumerate(names):
            if not short or short in merged:
                continue
            for j, long in enumerate(names):
                if i == j or not long or long in merged:
                    continue
                if len(long) - len(short) >= 2 and short in long and (
                        long.startswith(short) or long.endswith(short)):
                    merged[short] = long
                    if short in name_to_d and long in name_to_d:
                        sd = name_to_d[short]
                        ld = name_to_d[long]
                        if sd.get("description") and not ld.get("description"):
                            ld["description"] = sd["description"]
                        ld.setdefault("aliases", [])
                        if short not in ld["aliases"]:
                            ld["aliases"].append(short)
                    break
        if merged:
            def _resolve(n: str) -> str:
                while n in merged and merged[n] != n:
                    n = merged[n]
                return n
            deduped = [
                name_to_d[_resolve(n)] for n in names
                if _resolve(n) not in {_resolve(m) for m in merged if m != _resolve(m)}
            ]
            deduped = list({d.get("name", ""): d for d in deduped if d is not None}.values())

    return deduped


# ────────────────────────────────────────────────────────────
# 构建入口
# ────────────────────────────────────────────────────────────

async def build_registry(
    graph: Any,
    preprocessor: Any = None,
    intel_list: list[dict] | None = None,
    source_material: str = "",
    domain: str = "",
    log_fn: Any = None,
) -> EntityRegistry:
    """多层 LLM 流水线构建实体注册表。

    流程：
      ≤12K 字: Layer 1 (单次归一化) → Layer 2 (逐批判定)
      >12K 字: 分片归一化 → 内存合并 → LLM 精修 → Layer 2
    """

    # ── 1. 从 Kuzu 读取所有实体 ──
    result = graph._conn.execute(
        f"MATCH (e:{graph.NODE_TABLE}) RETURN e.id, e.name, e.type, e.description"
    )
    raw_fragments: list[dict] = []
    while result.has_next():
        r = result.get_next()
        raw_fragments.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3]})

    if not raw_fragments:
        return EntityRegistry()

    # ── 2. 频次数据 ──
    freq_map: dict[str, int] = {}
    if preprocessor and getattr(preprocessor, "result", None):
        freq_map = getattr(preprocessor.result, "entity_frequencies", {}) or {}

    # ── 3. 构建别名映射 ──
    alias_to_std: dict[str, str] = {}
    if preprocessor and getattr(preprocessor, "result", None):
        for std, aliases in preprocessor.result.entity_aliases.items():
            alias_to_std[std] = std
            for a in aliases:
                alias_to_std[a] = std
    for e in (intel_list or []):
        canon = (e.get("name") or "").strip()
        if canon:
            alias_to_std[canon] = canon
            for a in e.get("aliases", []):
                a = str(a).strip()
                if a:
                    alias_to_std[a] = canon

    for name, std in alias_to_std.items():
        if name != std and name in freq_map and std not in freq_map:
            freq_map[std] = max(freq_map.get(std, 0), freq_map[name])

    # ── 4. Layer 1 ──
    from strategy_forge.core.config import config as _cfg
    use_llm = bool(_cfg.deduction_llm_review and source_material)
    source_len = len(source_material) if source_material else 0

    # P1-2: 分片阈值从配置读取
    l3cfg = _load_layer3_config(domain)
    shard_threshold = int(l3cfg.get("shard_threshold", _SHARD_THRESHOLD))
    use_shard_path = source_len > shard_threshold

    normalized = None
    # P1-3: Layer1 结果缓存 (domain加入cache key防止跨域误复用)
    l1_cache_enabled = bool(l3cfg.get("l1_cache_enabled", True))
    if l1_cache_enabled and source_material:
        l1_key = hashlib.md5(f"{source_material}|{domain}".encode("utf-8")).hexdigest()
        cached_l1 = _layer1_normalize_cache.get(l1_key)
        if cached_l1:
            normalized = list(cached_l1)  # type: list[dict]
            if log_fn:
                log_fn("agents", f"Layer1 缓存命中: 跳过归一化")

    if use_llm and normalized is None:
        try:
            if use_shard_path:
                normalized = await _layer1_shard_pipeline(
                    raw_fragments, source_material, freq_map, log_fn, domain
                )
            else:
                normalized = await _layer1_normalize(
                    raw_fragments, source_material, freq_map, log_fn
                )
            # 写 L1 缓存
            if l1_cache_enabled and source_material and normalized:
                l1_key = hashlib.md5(f"{source_material}|{domain}".encode("utf-8")).hexdigest()
                _layer1_normalize_cache.set(l1_key, normalized)
        except Exception as e:
            logger.warning("[Layer1] 归一化失败: %s, 回退到去重", e)
            if log_fn:
                log_fn("agents", f"Layer1 失败({type(e).__name__})，回退去重")
            normalized = _dedup_fallback(raw_fragments, alias_to_std)
    elif normalized is None:
        normalized = _dedup_fallback(raw_fragments, alias_to_std)

    # ── 5. 附件频次 + intel 信息 ──
    intel_map: dict[str, dict] = {}
    if intel_list:
        for e in intel_list:
            nm = (e.get("name") or "").strip()
            if nm:
                intel_map[nm] = e
            for a in e.get("aliases", []):
                a = str(a).strip()
                if a and a not in intel_map:
                    intel_map[a] = e

    # ── 5.0 从 raw_fragments 构建 name→id 映射，回填 Layer1 丢失的 Kuzu 实体 ID ──
    name_to_kuzu_id: dict[str, str] = {}
    alias_to_kuzu_id: dict[str, str] = {}
    for frag in raw_fragments:
        fid = (frag.get("id") or "").strip()
        fname = (frag.get("name") or "").strip()
        if fid and fname:
            name_to_kuzu_id[fname] = fid
    # 从 intel_map 扩增别名映射（解决 Layer1 查重后名称变更的归位问题）
    for nm, info in intel_map.items():
        if nm in name_to_kuzu_id:
            for a in info.get("aliases", []):
                a = str(a).strip()
                if a and a not in alias_to_kuzu_id:
                    alias_to_kuzu_id[a] = name_to_kuzu_id[nm]
        for a in info.get("aliases", []):
            a = str(a).strip()
            if a in name_to_kuzu_id and nm not in name_to_kuzu_id:
                name_to_kuzu_id[nm] = name_to_kuzu_id[a]

    entity_list: list[dict] = []
    for ne in normalized:
        nm = ne["name"]
        fm = freq_map.get(nm, 0)
        intel = intel_map.get(nm, {})
        aliases = list(ne.get("aliases", []))
        if not aliases and intel.get("aliases"):
            aliases = [str(a) for a in intel["aliases"] if a != nm]
        # 已保留的 ID（分片归一化合并时保留）
        raw_id = (ne.get("id") or "").strip()
        # 从 raw_fragments 回填（同名匹配 → 别名匹配 → 空）
        kuzu_id = raw_id or name_to_kuzu_id.get(nm) or ""
        if not kuzu_id:
            for a in [nm] + aliases:
                if a in name_to_kuzu_id:
                    kuzu_id = name_to_kuzu_id[a]
                    break
        if not kuzu_id:
            kuzu_id = alias_to_kuzu_id.get(nm) or ""
        entity_list.append({
            "name": nm,
            "type": ne.get("type", "Unknown"),
            "freq": fm,
            "aliases": aliases,
            "description": ne.get("description", ""),
            "parent": str(intel.get("parent") or ""),
            "id": kuzu_id,
        })

    # ── 5.1 描述富化层：从 Kuzu 原始碎片补全缺失的描述 ──
    # Layer1 JSON 解析失败回退 dedup 时描述会丢失——此步用 Kuzu 原始节点描述补救
    enriched = 0
    for e in entity_list:
        if e.get("description", "").strip():
            continue  # 已有富描述，跳过
        nm = e["name"]
        aliases = set(e.get("aliases", [])) | {nm}
        descs = []
        for frag in raw_fragments:
            fname = frag.get("name", "")
            if fname in aliases or (len(fname) >= 2 and fname in nm) or (len(nm) >= 2 and nm in fname):
                d = (frag.get("description") or "").strip()
                if d and d not in descs:
                    descs.append(d)
        if descs:
            e["description"] = "；".join(descs)[:300]
            enriched += 1
    if enriched and log_fn:
        log_fn("agents", f"描述富化: {enriched} 个实体补全了 Kuzu 原始描述")

    # ── 6. 构建 Registry 骨架 ──
    registry = EntityRegistry()
    registry.total = len(entity_list)

    if not use_llm:
        reg_entities = []
        for e in entity_list:
            reg_entities.append(RegisteredEntity(
                id=e.get("id", ""),
                name=e["name"], type=e["type"], freq=e["freq"],
                decision="PENDING", reason="",
                parent=e["parent"], aliases=e["aliases"],
                rich_description=e["description"],
            ))
        for re in reg_entities:
            registry.entities[re.name] = re
        _fallback_classify(registry, reg_entities, log_fn)
        _resolve_hierarchy(registry, log_fn)
        return registry

    # ── 7. Layer 2 ──
    try:
        decisions = await _layer2_classify_all(entity_list, source_material, batch_size=10, log_fn=log_fn)
    except Exception as e:
        logger.warning("[Layer2] 判定失败: %s, 回退兜底", e)
        if log_fn:
            log_fn("agents", f"Layer2 失败({type(e).__name__})，回退兜底")
        reg_entities = []
        for e in entity_list:
            reg_entities.append(RegisteredEntity(
                id=e.get("id", ""),
                name=e["name"], type=e["type"], freq=e["freq"],
                decision="PENDING", reason="",
                parent=e["parent"], aliases=e["aliases"],
                rich_description=e["description"],
            ))
        for re in reg_entities:
            registry.entities[re.name] = re
        _fallback_classify(registry, reg_entities, log_fn)
        _resolve_hierarchy(registry, log_fn)
        return registry

    # ── 8. 应用判定 ──
    # 白名单强制一级：_force_keep 中的实体无视 Layer 2 分类
    force_keep_list = _alias_data.get("_force_keep", {})
    force_base = set(force_keep_list.get("all", []))
    force_base.update(force_keep_list.get(domain, []))
    # 扩增：白名单条目 + 它们的别名 (从别名词典中获取)
    force_expanded: set[str] = set(force_base)
    alias_dict = _alias_data.get(domain, {})
    for fname in force_base:
        aliases = alias_dict.get(fname, [])
        force_expanded.update(str(a) for a in aliases if a)
    if force_expanded:
        overridden = 0
        ent_aliases: dict[str, set[str]] = {}
        for e in entity_list:
            nm = e["name"]
            aliases_set = set(e.get("aliases", []))
            ent_aliases[nm] = aliases_set
        for name in list(decisions.keys()):
            # 匹配 1: 精确匹配 expanded 集合
            matched = name in force_expanded
            # 匹配 2: 实体别名命中
            if not matched:
                for fa in ent_aliases.get(name, set()):
                    if fa in force_expanded:
                        matched = True
                        break
            # 匹配 3: 白名单条目是实体名的子串 (e.g. "中国" matches "中方")
            if not matched:
                for fname in force_base:
                    if len(fname) >= 2 and fname in name:
                        matched = True
                        break
            # 匹配 4: 逆: 实体名是白名单条目的子串 (e.g. "美" matches "美国")
            if not matched:
                for fname in force_base:
                    if len(name) >= 2 and name in fname:
                        matched = True
                        break
            if matched and decisions[name].get("tier", 3) != 1:
                decisions[name] = {"decision": "KEEP", "tier": 1, "reason": "白名单强制一级"}
                overridden += 1
        if overridden and log_fn:
            log_fn("agents", f"白名单覆盖: {overridden} 个实体强制 tier=1")

    missing_ids: list[str] = []
    for e in entity_list:
        d = decisions.get(e["name"], {"decision": "DISCARD", "tier": 3, "reason": "未判定"})
        tier = int(d.get("tier", 3))
        if tier not in (1, 2, 3):
            tier = 3
        kuzu_id = e.get("id", "").strip()
        if not kuzu_id and tier in (1, 2):
            missing_ids.append(e["name"])
        re = RegisteredEntity(
            id=kuzu_id,
            name=e["name"], type=e["type"], freq=e["freq"],
            decision=d["decision"], reason=d["reason"],
            parent=e["parent"], aliases=e["aliases"],
            rich_description=e["description"],
            tier=tier, tier_evidence=d.get("reason", "")[:80],
            group=d.get("group", ""),
        )
        registry.entities[e["name"]] = re
        if tier in (1, 2):
            registry.kept += 1
            if tier == 1:
                registry.tier1_count += 1
            else:
                registry.tier2_count += 1
        else:
            registry.discarded += 1
            reason_key = f"L2({re.reason[:30]})"
            registry.discard_reasons[reason_key] = registry.discard_reasons.get(reason_key, 0) + 1

    if log_fn:
        log_fn("agents",
               f"注册中心: tier1={registry.tier1_count}核心 tier2={registry.tier2_count}次级 "
               f"tier3={registry.discarded}丢弃 / {registry.total}总计")
    logger.info("[EntityRegistry] tier1=%d tier2=%d tier3=%d",
                registry.tier1_count, registry.tier2_count, registry.discarded)
    if missing_ids:
        logger.warning("[EntityRegistry] %d tier1/2 entities missing Kuzu ID: %s",
                       len(missing_ids), ", ".join(missing_ids[:10]))
        if log_fn:
            log_fn("agents", f"⚠ 实体注册中 {len(missing_ids)} 个实体缺失 Kuzu ID (关系反哺将跳过): "
                   + ", ".join(missing_ids[:8]))

    # ── 9. Layer 3: 交叉裁决 ──
    await _layer3_cross_validate(registry, source_material, log_fn, domain)

    if log_fn:
        log_fn("agents", f"EntityRegistry: {registry.total} total, "
               f"tier1={registry.tier1_count} tier2={registry.tier2_count} DISCARD={registry.discarded}")
    logger.info("[EntityRegistry] 完成: tier1=%d tier2=%d / %d",
                registry.tier1_count, registry.tier2_count, registry.total)
    return registry


# ────────────────────────────────────────────────────────────
# 分片流水线 (Layer 1 超长文本专用)
# ────────────────────────────────────────────────────────────

async def _layer1_shard_pipeline(
    raw_fragments: list[dict],
    source: str,
    freq_map: dict[str, int],
    log_fn: Any = None,
    domain: str = "",
) -> list[dict]:
    """超长文本分片路径：
    1. 滑动分片 7500+800
    2. 实体-分片文本匹配
    3. 每分片独立 LLM 保守归一化 (受控并发)
    4. 内存池合并 (代码, 毫秒级)
    5. LLM 全局精修 (仅裁定歧义)
    """

    shards = _shard_source(source)
    total = len(shards)
    if log_fn:
        log_fn("agents", f"Layer1 超长文本({len(source)}字) → {total} 分片 (每片{_SHARD_SIZE}字, 重叠{_SHARD_OVERLAP}字)")

    # 1. 实体-分片匹配
    shard_entities = _match_entities_to_shards(shards, raw_fragments)

    # 2. 并发控制：使用 FORGE_MAX_CONCURRENT 限制并行分片数
    from strategy_forge.core.providers import registry as _reg
    max_conc = max(1, _reg.max_concurrent)
    sem = asyncio.Semaphore(max_conc)
    if log_fn and max_conc < total:
        log_fn("agents", f"Layer1 分片并发控制: {total} 分片, 最大 {max_conc} 路并行")

    async def _run_shard(idx: int) -> list[dict]:
        async with sem:
            return await _layer1_shard_normalize(
                shards[idx], shard_entities[idx], idx + 1, total, log_fn
            )

    tasks = [_run_shard(i) for i in range(total)]
    all_batches = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. 收集到内存池
    pool: list[dict] = []
    for i, batch in enumerate(all_batches):
        if isinstance(batch, Exception):
            logger.warning("[Shard L1] 分片 %d 异常: %s", i + 1, batch)
            # 异常分片 → 保守保留原始碎片
            for e in shard_entities[i]:
                pool.append({
                    "name": e.get("name", ""),
                    "aliases": e.get("aliases", []),
                    "type": e.get("type", "Unknown"),
                    "description": (e.get("description") or "")[:150],
                    "id": e.get("id", ""),
                })
            continue
        pool.extend(batch)

    if log_fn:
        log_fn("agents", f"Layer1 内存池: {len(pool)} 局部实体 (来自 {total} 个分片)")

    # 4. 内存代码合并
    merged, conflicts = _memory_merge(pool, domain)
    if log_fn:
        log_fn("agents", f"Layer1 内存合并: {len(pool)} 局部 → {len(merged)} 预合并"
               f" (检测 {len(conflicts)} 对歧义)")

    # 5. LLM 精修
    try:
        final = await _layer1_global_refine(merged, conflicts, source, log_fn)
    except Exception as e:
        logger.warning("[Global Refine] LLM 精修失败: %s, 使用代码合并结果", e)
        if log_fn:
            log_fn("agents", f"Layer1 精修失败({type(e).__name__})，使用代码合并结果")
        final = merged

    return final


# ── 调试入口 ──
if __name__ == "__main__":
    import sys, os, asyncio
    from pathlib import Path
    if len(sys.argv) < 2:
        print("用法: python -m strategy_forge.engine.entity_registry <session_db> <graph_dir>")
        sys.exit(1)
    db = Path(sys.argv[1])
    graph_dir = Path(sys.argv[2])
    from strategy_forge.storage.graph_store import DeductionGraphStore
    graph = DeductionGraphStore(str(graph_dir))
    async def preview():
        registry = await build_registry(graph)
        print(registry.summary())
    asyncio.run(preview())
    graph.close()

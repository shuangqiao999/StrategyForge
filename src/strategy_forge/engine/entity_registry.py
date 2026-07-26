"""Entity Registry — 实体注册中心：多层 LLM 流水线式实体识别与归类。

架构：
  Layer 1: 实体归一化
    - 快速路径 (≤12K字): 单次 LLM 全量归一化
    - 分片路径 (>12K字):  滑动分片 → 保守归一化 → 内存合并 → LLM 精修
  Layer 2: 逐批角色判定 — 10 个一组并行判定 KEEP/DISCARD + 证据
  (Layer 3 预留): 交叉裁决 — 全局去冗余 + 层次修正

用法：
  registry = await build_registry(graph, preprocessor, intel_list, source_material=source)
  kept = registry.get_kept()
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import re as _re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── 元数据（保留供 Layer 3 使用）──
_ORG_MEMBERS: dict[str, frozenset[str]] = {
    "欧盟": frozenset({"法国", "德国", "意大利", "荷兰", "比利时", "西班牙"}),
    "北约": frozenset({"美国", "英国", "法国", "德国", "意大利", "加拿大"}),
    "G7":   frozenset({"美国", "日本", "德国", "英国", "法国", "意大利", "加拿大"}),
    "东盟": frozenset({"印度尼西亚", "马来西亚", "菲律宾", "新加坡", "泰国", "越南"}),
    "金砖": frozenset({"中国", "俄罗斯", "印度", "巴西", "南非"}),
}

_PERSON_COUNTRY: dict[str, str] = {
    "特朗普": "美国", "拜登": "美国", "习近平": "中国",
    "普京": "俄罗斯", "泽连斯基": "乌克兰", "内塔尼亚胡": "以色列",
    "马克龙": "法国",
}

_SHARD_SIZE = 7500
_SHARD_OVERLAP = 800
_SHARD_THRESHOLD = 12000  # 超过此字数启用分片路径

# Jaccard 阈值：地缘政治实体名较短(2-4字)，文学叙事实体名较长(3-6字)
_JACCARD_THRESHOLDS = {
    "geo_strategy": 0.40,
    "history": 0.50,
    "business": 0.45,
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


@dataclass
class EntityRegistry:
    entities: dict[str, RegisteredEntity] = field(default_factory=dict)
    total: int = 0
    kept: int = 0
    discarded: int = 0
    discard_reasons: dict[str, int] = field(default_factory=dict)

    def get_kept(self) -> list[RegisteredEntity]:
        return sorted(
            [e for e in self.entities.values() if e.decision == "KEEP"],
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
    ]:
        try:
            data = parser(s)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    raise ValueError(f"JSON parse failed. Raw(length={len(s)}):\n{s[:500]}")


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────

def _smart_sample(source: str, max_chars: int = 8000) -> str:
    """智能采样：头 + 均匀中段 + 尾，覆盖全文叙事弧。"""
    n = len(source)
    if n <= max_chars:
        return source
    head = source[:2000]
    tail = source[-2000:]
    remaining = max_chars - 4000
    if remaining <= 0:
        return head + "\n...(中段省略)...\n" + tail
    # 均匀采样中段
    mid_start = 2000
    mid_end = n - 2000
    mid_len = mid_end - mid_start
    if mid_len <= remaining:
        return source
    step = max(1, mid_len // 4)
    samples = []
    for i in range(4):
        pos = mid_start + i * step
        chunk = source[pos:pos + remaining // 4]
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
1. 识别同义异名实体（例如"崇祯"与"朱由检"是同一人，"美国"与"美方"是同一个国家）
2. 融合描述：将同一实体的所有描述片段合并为一段完整描述（100字以内）
3. 修正不一致的类型标签（如有的碎片标"国家"有的标"Organization"，统一为合理类型）
4. 输出规范实体列表

## 规则
- 同名实体在不同块出现，描述互补则融合，描述冲突则取多数
- 别名字符重叠≥2且语义相同→合并（如"特朗普"与"川普"）
- 上下级实体不合并（"国防部"≠"美国"）
- 二元关系词不合并（"中美关系"≠"中国"也不合并，单独保留）
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
    """Layer 1 快速路径：单次 LLM 全量归一化。"""

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

    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    client = DeductionLLMClient()
    resp = await client.chat(
        [Message(role="user", content=prompt)],
        system=_LAYER1_SYSTEM,
        temperature=0,
        max_tokens=4000,
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(b.text for b in content if hasattr(b, "text"))

    data = _parse_llm_json(str(content))
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
    resp = await client.chat(
        [Message(role="user", content=prompt)],
        system=_SHARD_SYSTEM,
        temperature=0,
        max_tokens=3000,
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
    conflicts = []
    merged_names = [m["name"] for m in merged]
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            sim = _char_jaccard(merged_names[i], merged_names[j])
            if sim > jaccard_threshold:
                ti = merged[i]["type"]
                tj = merged[j]["type"]
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
   - 高相似名但类型不同 → 需结合原文裁决（如"华为公司"vs"华为手机"可能是同一企业的主品牌和产品线）
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
    resp = await client.chat(
        [Message(role="user", content=prompt)],
        system=_REFINE_SYSTEM,
        temperature=0,
        max_tokens=4000,
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

_LAYER2_SYSTEM = """你是战略分析专家。你的任务是判断给定实体在原文中是否具有"独立战略决策权"——即是否为博弈参与者。

## 判定标准

**KEEP（具有独立战略决策权）：**
- 在原文中作为独立行动者出现（有独立立场、独立行为、影响格局）
- 包括：主权国家、国际组织、核心政治人物、跨国企业、军事联盟、反叛武装
- 低频但有关键行动（发动攻击、签署协议、被制裁、做出决策）→ KEEP

**DISCARD（不具有独立战略决策权）：**
- 纯地理概念（地名、海域）、纯经济指标、纯技术标准
- 纯下属部门、军队编制名、职务头衔
- 泛指集合概念、背景提及但无独立行为的实体
- 二元关系词（如"中美关系"）、协议/条约名

## 特别注意
- 一个人物如果原文详细描写其独立决策过程 → KEEP
- 一个组织如果仅是背景或研究机构 → DISCARD
- 频次仅作参考：1 次关键行动 > 10 次背景提及
- 地点型实体（如城市名）→ 大概率 DISCARD，除非明确作为政治主体行动

## 输出 JSON
{
  "results": [
    {"name": "实体名", "decision": "KEEP", "reason": "≤30字理由及证据"},
    {"name": "实体名", "decision": "DISCARD", "reason": "≤30字理由"}
  ]
}

只输出 JSON。"""

_LAYER2_USER = """## 原文全文
{source}

## 待判定实体（已归一化，含融合描述）
{batch}

请逐实体判定 KEEP 或 DISCARD。只输出 JSON。"""


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
    resp = await client.chat(
        [Message(role="user", content=prompt)],
        system=_LAYER2_SYSTEM,
        temperature=0,
        max_tokens=2000,
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(b.text for b in content if hasattr(b, "text"))

    data = _parse_llm_json(str(content))
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Layer 2 batch {batch_idx} bad format: {str(content)[:200]}")

    _tag = f"L2[{batch_idx}/{total_batches}]"
    keep = 0
    out = []
    for r in results:
        if isinstance(r, dict) and r.get("name"):
            d = str(r.get("decision", "")).upper().strip()
            if d not in ("KEEP", "DISCARD"):
                d = "DISCARD"
            if d == "KEEP":
                keep += 1
            out.append({
                "name": str(r["name"]).strip(),
                "decision": d,
                "reason": str(r.get("reason", ""))[:60],
            })
    if log_fn:
        log_fn("agents", f"  Layer2 批次{_tag}: {keep}/{len(out)} KEEP")
    logger.info("[Layer2] 批次%s: %d/%d KEEP", _tag, keep, len(out))
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
                log_fn("agents", f"  Layer2 批次{i+1} 失败: {result}")
            for e in batches[i]:
                decisions[e["name"]] = {"decision": "DISCARD", "reason": f"批次{i+1}LLM失败"}
            continue
        for r in result:
            decisions[r["name"]] = {"decision": r["decision"], "reason": r["reason"]}

    for e in entities:
        if e["name"] not in decisions:
            decisions[e["name"]] = {"decision": "DISCARD", "reason": "LLM未覆盖"}

    return decisions


# ────────────────────────────────────────────────────────────
# 兜底 + 层次修正
# ────────────────────────────────────────────────────────────

def _fallback_classify(
    registry: EntityRegistry,
    entities: list[RegisteredEntity],
    log_fn: Any = None,
) -> None:
    t = max(1, registry.total // 50)
    for e in entities:
        if e.type in ("Country", "国家", "Organization", "组织", "国际组织") and e.freq >= 1:
            e.decision = "KEEP"
            e.reason = "兜底(国家/组织≥1)"
            registry.kept += 1
        elif e.type in ("Person", "人物") and e.freq >= t:
            e.decision = "KEEP"
            e.reason = f"兜底(人物高≥{t})"
            registry.kept += 1
        elif e.freq >= t * 3:
            e.decision = "KEEP"
            e.reason = f"兜底(高频≥{t*3})"
            registry.kept += 1
        else:
            e.decision = "DISCARD"
            e.reason = "兜底排除"
            registry.discarded += 1
            registry.discard_reasons["兜底排除"] = registry.discard_reasons.get("兜底排除", 0) + 1
    if log_fn:
        log_fn("agents", f"兜底规则分类: {registry.kept} 保留 (LLM不可用)")


def _resolve_hierarchy(registry: EntityRegistry, log_fn: Any = None) -> None:
    kept_entities = registry.get_kept()
    kept_names = {e.name for e in kept_entities}
    to_discard: list[tuple[RegisteredEntity, str]] = []

    for e in kept_entities:
        if e.name in _ORG_MEMBERS:
            core = _ORG_MEMBERS[e.name]
            overlap = core & kept_names
            if len(overlap) >= 3:
                to_discard.append((e, f"组织(成员国重叠:{len(overlap)}国)"))
        elif e.name in _PERSON_COUNTRY:
            country = _PERSON_COUNTRY[e.name]
            if country in kept_names:
                to_discard.append((e, f"人物(归入{country})"))

    for e, reason in to_discard:
        e.decision = "DISCARD"
        e.reason = reason
        registry.kept -= 1
        registry.discarded += 1

    if to_discard and log_fn:
        log_fn("agents", f"层次修正: {len(to_discard)} 个重叠实体降级")
    if to_discard:
        logger.info("[EntityRegistry] 层次修正: %d 个重叠实体降级", len(to_discard))


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
             "description": p.get("description", ""), "aliases": []}
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
    use_shard_path = source_len > _SHARD_THRESHOLD

    if use_llm:
        try:
            if use_shard_path:
                normalized = await _layer1_shard_pipeline(
                    raw_fragments, source_material, freq_map, log_fn, domain
                )
            else:
                normalized = await _layer1_normalize(
                    raw_fragments, source_material, freq_map, log_fn
                )
        except Exception as e:
            logger.warning("[Layer1] 归一化失败: %s, 回退到去重", e)
            if log_fn:
                log_fn("agents", f"Layer1 失败({type(e).__name__})，回退去重")
            normalized = _dedup_fallback(raw_fragments, alias_to_std)
    else:
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

    entity_list: list[dict] = []
    for ne in normalized:
        nm = ne["name"]
        fm = freq_map.get(nm, 0)
        intel = intel_map.get(nm, {})
        aliases = list(ne.get("aliases", []))
        if not aliases and intel.get("aliases"):
            aliases = [str(a) for a in intel["aliases"] if a != nm]
        entity_list.append({
            "name": nm,
            "type": ne.get("type", "Unknown"),
            "freq": fm,
            "aliases": aliases,
            "description": ne.get("description", ""),
            "parent": str(intel.get("parent") or ""),
            "id": "",
        })

    # ── 6. 构建 Registry 骨架 ──
    registry = EntityRegistry()
    registry.total = len(entity_list)

    if not use_llm:
        reg_entities = []
        for e in entity_list:
            reg_entities.append(RegisteredEntity(
                id=raw_fragments[0].get("id", "") if raw_fragments else "",
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
    for e in entity_list:
        d = decisions.get(e["name"], {"decision": "DISCARD", "reason": "未判定"})
        re = RegisteredEntity(
            id=e.get("id", ""),
            name=e["name"], type=e["type"], freq=e["freq"],
            decision=d["decision"], reason=d["reason"],
            parent=e["parent"], aliases=e["aliases"],
            rich_description=e["description"],
        )
        registry.entities[e["name"]] = re
        if re.decision == "KEEP":
            registry.kept += 1
        else:
            registry.discarded += 1
            reason_key = f"L2({re.reason[:30]})"
            registry.discard_reasons[reason_key] = registry.discard_reasons.get(reason_key, 0) + 1

    # ── 9. 层次修正 ──
    _resolve_hierarchy(registry, log_fn)

    if log_fn:
        log_fn("agents", f"EntityRegistry: {registry.total} total, {registry.kept} KEEP, {registry.discarded} DISCARD")
    logger.info("[EntityRegistry] 完成: %d/%d KEEP", registry.kept, registry.total)
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

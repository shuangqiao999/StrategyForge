"""Entity Registry — 实体注册中心：全部博弈实体的唯一权威数据源。

架构：代码硬排除（100%准确）→ LLM 单次全量分类（温度 0）。

用法：
  registry = await build_registry(graph, preprocessor, intel_list, source_material=source)
  kept = registry.get_kept()
  for e in kept: print(e.name, e.type, e.reason)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── 硬排除规则（100% 准确，代码可枚举）──
_DISCARD_TYPES = frozenset({
    "地理区域", "地理位置", "地点", "天气", "气象",
    "文档", "协议", "合同", "批文", "文件",
    "概念", "现象", "事件", "日期", "时间",
    "设施", "基础设施", "建筑",
    "自然景观", "自然现象", "环境要素",
    "Location", "Document", "Concept", "Event",
    "Date", "Time", "Facility", "NaturalFeature",
})
_TITLE_SUFFIX = (
    "总统", "总理", "司令", "部长", "秘书长", "主席", "书记", "市长",
    "省长", "院长", "局长", "指挥官", "委员长", "主任", "将军", "上将",
    "特使", "代表", "发言人", "CEO", "董事长", "行长", "署长",
)
_MILITARY_SUFFIX = (
    "舰队", "战区", "司令部", "集团军", "师团", "旅", "营", "连", "军",
    "导弹旅", "驱逐舰", "航空母舰", "航母",
)
_DEPT_WORDS = (
    "国防部", "财政部", "外交部", "商务部", "内政部", "央行",
    "最高法院", "最高检", "议会", "参议院", "众议院", "国务院", "中央军委",
    "国会", "法院", "检察院", "监察院", "行政院", "白宫", "五角大楼",
)
_COLLECTIVE_SUFFIX = ("阵营", "群体", "板块", "民间", "大众", "行业", "同盟", "联盟国")
_KNOWN_DYADS = frozenset({
    "俄乌", "美伊", "中美", "美中", "巴以", "以巴",
    "印巴", "美俄", "俄美", "朝美", "美朝", "日菲",
})

_GEO_REGIONS = frozenset({
    "非洲", "亚洲", "欧洲", "美洲", "北美洲", "南美洲",
    "大洋洲", "拉丁美洲", "中东", "东南亚", "南亚", "东亚",
    "中亚", "西亚", "东欧", "西欧", "北欧", "南欧", "中欧",
    "北非", "西非", "东非", "中非", "南部非洲",
    "撒哈拉以南非洲", "印太地区", "亚太地区",
    "巴尔干", "高加索", "加勒比", "中美洲",
})

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


def _is_dyad(name: str) -> bool:
    for sep in ("与", "和", "及", "对", "vs", "vs.", "/", "-", "—",
                  "关系", "冲突", "战争", "会谈", "谈判", "对抗", "争端"):
        parts = name.split(sep)
        non_empty = [p for p in parts if p.strip()]
        if len(parts) >= 2 and len(non_empty) >= 2:
            return True
        if len(non_empty) >= 1 and name.endswith(sep) and len(name) - len(sep) >= 2:
            return True
    return name in _KNOWN_DYADS


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
        lines.append("  KEPT entities:")
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


# ── 构造函数 ──

def _extract_balanced(text: str, opener: str, closer: str) -> str | None:
    """返回从第一个 opener 起、括号配平的子串。"""
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


async def build_registry(
    graph: Any,
    preprocessor: Any = None,
    intel_list: list[dict] | None = None,
    source_material: str = "",
    domain: str = "",
    log_fn: Any = None,
) -> EntityRegistry:
    """从 Kuzu 图谱构建实体注册表。

    流程：硬排除（代码规则）→ LLM 全量分类 → fallback（LLM 失败时）。
    """
    # 1. 从 Kuzu 读取所有实体
    result = graph._conn.execute(
        f"MATCH (e:{graph.NODE_TABLE}) RETURN e.id, e.name, e.type, e.description"
    )
    raw: list[dict] = []
    while result.has_next():
        r = result.get_next()
        raw.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3]})

    # 2. 去重
    alias_to_std: dict[str, str] = {}
    if preprocessor and getattr(preprocessor, "result", None):
        for std, aliases in preprocessor.result.entity_aliases.items():
            alias_to_std[std] = std
            for a in aliases:
                alias_to_std[a] = std
    for e in (intel_list or []):
        canon = (e.get("name") or "").strip()
        if canon:
            for a in e.get("aliases", []):
                a = str(a).strip()
                if a:
                    alias_to_std[a] = canon
    seen: set[str] = set()
    deduped: list[dict] = []
    for p in raw:
        name = p.get("name", "")
        std_name = alias_to_std.get(name, name)
        if std_name in seen:
            continue
        seen.add(std_name)
        if std_name != name:
            p["name"] = std_name
        deduped.append(p)
    # 子串合并
    if len(deduped) > 1:
        names = [p.get("name", "") for p in deduped]
        name_to_p = {p.get("name", ""): p for p in deduped}
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
                    break
        if merged:
            def _resolve(n: str) -> str:
                while n in merged and merged[n] != n:
                    n = merged[n]
                return n
            deduped = [
                name_to_p.get(_resolve(n)) or name_to_p.get(n)
                for n in names
                if _resolve(n) not in {_resolve(m) for m in merged}
            ]
            deduped = list({p.get("name", ""): p for p in deduped if p is not None}.values())

    # 3. 频次数据 + sorter 补充信息
    freq_map: dict[str, int] = {}
    if preprocessor and getattr(preprocessor, "result", None):
        freq_map = getattr(preprocessor.result, "entity_frequencies", {}) or {}
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

    # 4. 硬排除：代码规则（100% 准确）
    registry = EntityRegistry()
    registry.total = len(deduped)
    hard_kept: list[RegisteredEntity] = []
    for p in deduped:
        pname = p.get("name", "")
        ptype = p.get("type", "") or ""
        fm = freq_map.get(pname, 0)
        discard_reason = None
        if ptype in _DISCARD_TYPES:
            discard_reason = "类型排除"
        elif _is_dyad(pname):
            discard_reason = "二元关系词"
        elif any(pname.endswith(s) for s in _TITLE_SUFFIX):
            discard_reason = "职务头衔"
        elif any(pname.endswith(s) for s in _COLLECTIVE_SUFFIX):
            discard_reason = "集合概念"
        elif any(pname.endswith(s) for s in _MILITARY_SUFFIX):
            discard_reason = "军队编制"
        elif any(w in pname for w in _DEPT_WORDS):
            discard_reason = "政府部门"
        elif pname in _GEO_REGIONS:
            discard_reason = "地理区域(非决策主体)"

        intel = intel_map.get(pname, {})
        entity = RegisteredEntity(
            id=p.get("id", ""), name=pname, type=ptype, freq=fm,
            decision="DISCARD" if discard_reason else "PENDING",
            reason=discard_reason or "待LLM分类",
            parent=str(intel.get("parent") or ""),
            aliases=list(intel.get("aliases", [])),
        )
        registry.entities[pname] = entity
        if discard_reason:
            registry.discarded += 1
            registry.discard_reasons[discard_reason] = registry.discard_reasons.get(discard_reason, 0) + 1
        else:
            hard_kept.append(entity)

    if not hard_kept:
        return registry

    # 5. LLM 单次全量分类
    from strategy_forge.core.config import config as _cfg
    if _cfg.deduction_llm_review and source_material:
        await _llm_classify(registry, hard_kept, source_material, domain, log_fn, preprocessor)
    else:
        _fallback_classify(registry, hard_kept, log_fn)

    # 6. 实体层次修正（组织-成员国、人物-国家重叠检测）
    _resolve_entity_hierarchy(registry, log_fn)

    return registry


# ── LLM 分类提示词（共享方法论，所有批次复用）──
_METHODOLOGY_PROMPT = """
## 分类方法论：三步检验法（按序应用）
对每个待分类实体，依次用以下三步检验。任何一步失败则 DISCARD，三步全通过则 KEEP。

### 检验一：归属检验 — 该实体的决策权来源于自身，还是另一个实体？
- 主权国家的决策权来源于自身主权 → 通过。
- 政治人物的决策权来源于其所代表的国家/政府 → 不通过，应归入该国。
- 跨国企业 CEO 的决策权来源于企业 → 若企业已保留则 CEO 不通过。
- 政府部门的决策权来源于国家 → 不通过。
- 军队编制的行动权来源于国家 → 不通过。
- 关键判据：如果上级实体消失，该实体是否还能独立做出同样的决策？

### 检验二：双重代表检验 — 保留该实体是否会造成"同一决策力量被重复计算"？
- 如果"法国"已保留，再保留"欧盟"等于法国的外交/经济决策权被算两次 → 不通过。
- 如果"美国"已保留，再保留"北约"等于美国的军事决策权被算两次 → 不通过。
- 关键判据：该实体做的决策，是否必须通过另一个实体（的投票/军队/资金）才能生效？若是，则决策权实质上属于后者。
- 例外：如果该组织内的主要成员国大多未被保留，且该组织有独立行动记录 → 可通过。

### 检验三：残留行动检验 — 该实体在种子材料中是否做出了具体的、独立的战略行动？
- "签署协议""发动攻击""实施制裁""被列入管制清单"是战略行动 → 通过。
- "作为背景被提及""统计数据""地理位置描述"不是战略行动 → 不通过。
- 大洲、地理区域永远无法"做出行动" → 不通过。
- 关键判据：能否为这个实体写出一个"X 做了 Y，影响了 Z"的句子？如果不能，它就不是决策者。
- 注意：频次低但行动关键（如1次被制裁）远强于频次高但无行动（如10次背景提及）。

## 正例（三步全通过，应当 KEEP）
  例1: [华为] type=科技企业 freq=3
      → 检验一：企业自主决策 ✓
      → 检验二：不与任何国家重叠 ✓
      → 检验三：被列入实体清单/受制裁 ✓
      → KEEP（独立企业，被制裁锚点）

  例2: [乌克兰] type=国家 freq=5
      → 检验一：主权国家 ✓
      → 检验二：不与任何已保留实体重叠 ✓
      → 检验三：发动攻击/签署协议/接受军援 ✓
      → KEEP（独立主权决策者）

  例3: [OPEC] type=国际组织 freq=2
      → 检验一：组织有独立决策机制 ✓
      → 检验二：核心成员国并未全部单独列出，不形成重复 ✓
      → 检验三：做出产量决策，直接影响油价 ✓
      → KEEP（有独立行动权的国际组织）

## 反例（某一步失败，应当 DISCARD）
  例1: [特朗普] type=人物 freq=6  — 假设"美国"已在待保留列表中
      → 检验一失败：决策权来源于美国，不是独立主权体
      → 检验二失败：与美国形成双重代表
      → DISCARD（政治人物，归入其所属国家）

  例2: [非洲] type=地理区域 freq=4
      → 检验三失败：不能做决策的地理概念，从未"签署"或"发动"任何行动
      → DISCARD（地理背景，非决策主体）

  例3: [G7] type=国际组织 freq=3  — 假设"美国""日本""法国"已在待保留列表中
      → 检验二失败：核心成员国美国/日本/法国均已独立保留，G7 无超出成员国的独立执行能力
      → 检验三可能失败：G7 的决议需要成员国各自落地执行
      → DISCARD（协调平台，决策权归于成员国）

## 边界案例（需结合种子材料判断）
  例4: [马斯克] type=人物 freq=3
      → 检验一：若他以 SpaceX CEO 身份独立行动，决策权来自企业自身 → 可能通过
      → 检验二：若 SpaceX 也在列表中 → 不通过；若不在 → 可能通过
      → 检验三：若他绕过政府管制独立决策 → 通过；若仅为政府顾问 → 不通过
      → 结论：取决于种子材料中该人物是否展现出超越国家的独立行动。

## 输出格式
严格输出 JSON，reasons 为 ≤20 字简述：
{"keep": ["实体名", ...], "discard": ["实体名", ...], "reasons": {"实体名": "简述原因"}}
未出现在 keep 或 discard 中的实体默认视为 discard。只输出 JSON。"""


async def _llm_classify(
    registry: EntityRegistry,
    pending: list[RegisteredEntity],
    source: str,
    domain: str,
    log_fn: Any = None,
    preprocessor: Any = None,
) -> None:
    """LLM 分批并行分类：将实体分组（每批≤15 个），各组独立调用 LLM 并并行执行。

    当 preprocessor 可用时，为每个实体检索 LanceDB 中的关联语义块作为证据上下文，
    替代固定窗口 source[:5000] 的随机截断。"""
    import time as _time
    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    import json as _json
    import re as _re

    # ── 为每个实体检索关联证据（共享，不重复检索）──
    entity_evidence: dict[str, list[str]] = {}
    seen_prefixes: set[str] = set()
    if preprocessor is not None:
        t0 = _time.time()
        for e in pending:
            ename = str(e.name or "").strip()
            if not ename or len(ename) < 2:
                continue
            try:
                must_set = {ename}
                for alias in (getattr(e, "aliases", None) or []):
                    a = str(alias).strip()
                    if a and len(a) >= 2:
                        must_set.add(a)
                must = must_set if ename and len(ename) >= 2 else None
                chunks = await asyncio.to_thread(
                    preprocessor.retrieve_for_entity, ename, top_k=3,
                    must_contain=must,
                )
                if chunks:
                    deduped: list[str] = []
                    for c in chunks:
                        c = str(c).strip()
                        if not c:
                            continue
                        prefix = c[:80]
                        if prefix in seen_prefixes:
                            continue
                        seen_prefixes.add(prefix)
                        deduped.append(c)
                        if len(deduped) >= 2:
                            break
                    if deduped:
                        entity_evidence[ename] = deduped
            except Exception:
                pass
        if log_fn and entity_evidence:
            dt = _time.time() - t0
            log_fn("agents", f"实体证据检索: {len(entity_evidence)}/{len(pending)} 实体有匹配段落 ({dt:.1f}s)")

    # ── 构建领域摘要（各组共享）──
    domain_preamble: list[str] = []
    if domain:
        from strategy_forge.core.rule_templates import get_domain_prompt
        dr = get_domain_prompt(domain, "intel_extra_rules")
        if dr:
            domain_preamble.append(f"## 当前领域：{domain}\n{dr}")

    # ── 分批（优先类型分组，每批≤15）──
    type_priority = {"国家": 0, "Country": 0, "国际组织": 1, "人物": 2, "Person": 2}
    sorted_pending = sorted(
        pending, key=lambda e: (type_priority.get(e.type, 9), -e.freq))
    batch_size = max(8, min(15, 100 // max(1, len(pending) // 3)))
    batches = [sorted_pending[i:i + batch_size] for i in range(0, len(sorted_pending), batch_size)]

    if log_fn:
        log_fn("agents", f"LLM 分批分类: {len(pending)} 实体 → {len(batches)} 批 (每批≤{batch_size})")

    # ── 并行处理各批次（Semaphore 控制并发 LLM 调用数）──
    from strategy_forge.core.providers import registry as _prov_reg
    sem = asyncio.Semaphore(max(1, _prov_reg.max_concurrent))

    async def _process_batch(batch_entities: list[RegisteredEntity], batch_idx: int) -> None:
        parts = ["你是实体分类员。判断以下实体在种子材料中是否具有独立战略决策权。"]
        parts.extend(domain_preamble)
        if entity_evidence:
            parts.append("\n## 待分类实体与关联证据")
            parts.append("每个实体下方附有其在种子材料中最相关的段落，请据此判断「残留行动」（检验三）。频次仅供参考——1 次关键行动 > 10 次背景提及。\n")
            for i, e in enumerate(batch_entities, 1):
                p = f" (上级: {e.parent})" if e.parent else ""
                parts.append(f"  {i}. {e.name}  type={e.type}  freq={e.freq}{p}")
                evidence = entity_evidence.get(e.name, [])
                for j, chunk in enumerate(evidence[:2], 1):
                    c = chunk[:280] + ("..." if len(chunk) > 280 else "")
                    parts.append(f"     | 证据{j}: {c}")
        else:
            parts.append("## 种子材料采样")
            parts.append(source[:5000])
            parts.append("\n## 待分类实体（频次仅供参考——1 次关键行动 > 10 次背景提及）")
            for i, e in enumerate(batch_entities, 1):
                p = f" (上级: {e.parent})" if e.parent else ""
                parts.append(f"  {i}. {e.name}  type={e.type}  freq={e.freq}{p}")
        parts.append(_METHODOLOGY_PROMPT)
        prompt = "\n".join(parts)

        _MAX_RETRIES = 1
        raw = ""
        for _attempt in range(_MAX_RETRIES + 1):
            try:
                client = DeductionLLMClient()
                async with sem:
                    resp = await client.chat(
                        [Message(role="user", content=prompt)],
                        system="你是实体分类员，只输出 JSON。",
                        temperature=0,
                        max_tokens=max(400, min(3000, 80 + len(batch_entities) * 40)),
                    )
                content = resp.content if hasattr(resp, "content") else str(resp)
                if isinstance(content, list):
                    content = "".join(b.text for b in content if hasattr(b, "text"))
                raw = str(content).strip()
                break
            except Exception as e:
                if _attempt < _MAX_RETRIES:
                    logger.warning("[EntityRegistry] 批次 %d 第%d次尝试失败: %s，重试中...",
                                   batch_idx, _attempt + 1, e)
                    await asyncio.sleep(0.5)
                else:
                    logger.warning("[EntityRegistry] 批次 %d 全部尝试失败: %s", batch_idx, e)
                    for e in batch_entities:
                        e.decision = "DISCARD"
                        e.reason = f"LLM(批次{batch_idx}调用失败)"
                        registry.discarded += 1
                        registry.discard_reasons[e.reason] = registry.discard_reasons.get(e.reason, 0) + 1
                    return

        # 多策略 JSON 解析
        data = None
        strategies = [
            ("direct", lambda s: _json.loads(s)),
            ("no_md",   lambda s: _json.loads(
                _re.sub(r'```(?:json)?\s*\n?', '', s).replace('```', '').strip())),
            ("greedy",  lambda s: _json.loads(
                (_re.search(r'\{[\s\S]*\}', s) or _re.search(r'\[[\s\S]*\]', s)).group(0)
                if (_re.search(r'\{[\s\S]*\}', s) or _re.search(r'\[[\s\S]*\]', s)) else "")),
            ("balanced", lambda s: _json.loads(
                _extract_balanced(s, '{', '}') or s)),
        ]
        for name, parser in strategies:
            try:
                data = parser(raw)
                if isinstance(data, dict):
                    break
            except Exception:
                continue
        if not isinstance(data, dict):
            logger.warning("[EntityRegistry] 批次 %d JSON 解析失败，整批排除", batch_idx)
            for e in batch_entities:
                e.decision = "DISCARD"
                e.reason = f"LLM(批次{batch_idx}解析失败)"
                registry.discarded += 1
                registry.discard_reasons[e.reason] = registry.discard_reasons.get(e.reason, 0) + 1
            return

        keep_set = set(str(n).strip() for n in data.get("keep", []))
        reasons = data.get("reasons", {}) or {}
        for e in batch_entities:
            if e.name in keep_set:
                e.decision = "KEEP"
                r = str(reasons.get(e.name, "LLM判定"))[:40]
                e.reason = f"LLM({r})"
                registry.kept += 1
            else:
                e.decision = "DISCARD"
                r = str(reasons.get(e.name, "LLM排除"))[:40]
                e.reason = f"LLM({r})"
                registry.discarded += 1
                registry.discard_reasons[e.reason] = registry.discard_reasons.get(e.reason, 0) + 1

    tasks = [_process_batch(b, idx + 1) for idx, b in enumerate(batches)]
    try:
        await asyncio.gather(*tasks)
    except Exception as ge:
        logger.warning("[EntityRegistry] 分批分类全局异常 (%s): %s", type(ge).__name__, str(ge))
        if log_fn:
            log_fn("agents", f"分批分类全局异常({type(ge).__name__})，回退规则兜底")

    # 全局兜底：仍有 PENDING 实体未分类 → fallback
    still_pending = [e for e in sorted_pending if e.decision == "PENDING"]
    if still_pending:
        logger.warning("[EntityRegistry] %d 实体仍为 PENDING，回退规则兜底", len(still_pending))
        if log_fn:
            log_fn("agents", f"分批未覆盖 {len(still_pending)} 个实体，回退规则兜底")
        _fallback_classify(registry, still_pending, log_fn)

    logger.info("[EntityRegistry] 分批分类完成: %d 批, %d KEEP / %d DISCARD",
                len(batches), registry.kept, registry.discarded)
    if log_fn:
        log_fn("agents", f"分批分类完成 ({len(batches)} 批): {registry.kept} 保留 / {registry.discarded} 排除")


def _fallback_classify(
    registry: EntityRegistry,
    pending: list[RegisteredEntity],
    log_fn: Any = None,
) -> None:
    """LLM 不可用时的简单兜底规则。阈值随实体总数自适应。"""
    t = max(1, registry.total // 50)
    for e in pending:
        if e.type in ("Person", "人物") and e.freq >= t:
            e.decision = "KEEP"
            e.reason = f"兜底(人物≥{t})"
            registry.kept += 1
        elif e.type in ("Country", "国家") and e.freq >= 1:
            e.decision = "KEEP"
            e.reason = "兜底(国家≥1)"
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
        log_fn("agents", f"兜底规则: {registry.kept} 保留")


def _resolve_entity_hierarchy(registry: EntityRegistry, log_fn: Any = None) -> None:
    """LLM 分类后修正：检测组织-成员国 / 人物-国家重叠，降级冗余实体。"""
    kept_entities = registry.get_kept()
    kept_names = {e.name for e in kept_entities}

    to_discard: list[RegisteredEntity] = []

    for e in kept_entities:
        if e.name in _ORG_MEMBERS:
            core = _ORG_MEMBERS[e.name]
            overlap = core & kept_names
            if overlap:
                to_discard.append((e, f"组织(成员国重叠:{','.join(sorted(overlap)[:3])})"))
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
        log_fn("agents", f"实体层次修正: {len(to_discard)} 个重叠实体降级")
    if to_discard:
        logger.info("[EntityRegistry] 层次修正: %d 个重叠实体降级", len(to_discard))


# ── 调试入口 ──
if __name__ == "__main__":
    import sys, os, json
    from pathlib import Path
    if len(sys.argv) < 3:
        print("用法: python -m strategy_forge.engine.entity_registry <session_db> <graph_dir>")
        sys.exit(1)
    if len(sys.argv) == 2:
        sid = sys.argv[1]
        db = Path(os.environ.get("FORGE_DATA_DIR", os.path.expandvars(
            "%LOCALAPPDATA%/StrategyForge/data"))) / "sessions.db"
        graph_dir = Path(os.environ.get("FORGE_DATA_DIR", os.path.expandvars(
            "%LOCALAPPDATA%/StrategyForge/data"))) / "graphs" / sid / "kuzu"
    else:
        db = Path(sys.argv[1])
        graph_dir = Path(sys.argv[2])
    from strategy_forge.storage.graph_store import DeductionGraphStore
    graph = DeductionGraphStore(str(graph_dir))
    import asyncio
    async def preview():
        registry = await build_registry(graph)
        print(registry.summary())
    asyncio.run(preview())
    graph.close()

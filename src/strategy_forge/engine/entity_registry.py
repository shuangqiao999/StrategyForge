"""Entity Registry — 实体注册中心：全部博弈实体的唯一权威数据源。

本模块是纯代码模块（零 LLM 调用）。它从 Kuzu 图谱读取所有实体，
按确定性代码规则进行分类（KEEP/DISCARD），输出标准化的注册表。
所有下游模块（agent_factory、reporter、optimizer）从此注册中心读取
实体数据，不再各自查询或分类。

用法：
  registry = build_registry(graph, preprocessor, intel_list)
  kept = registry.get_kept()
  for e in kept: print(e.name, e.type, e.reason)

调试入口（不需完整推演）：
  python -m strategy_forge.engine.entity_registry <session_db_path> <graph_path>
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── 类型常量 ──
_DISCARD_TYPES = frozenset({
    "地理区域", "地理位置", "地点", "天气", "气象",
    "文档", "协议", "合同", "批文", "文件",
    "概念", "现象", "事件", "日期", "时间",
    "设施", "基础设施", "建筑",
    "自然景观", "自然现象", "环境要素",
    "Location", "Document", "Concept", "Event",
    "Date", "Time", "Facility", "NaturalFeature",
})
_PERSON_TYPES = frozenset({"Person", "人物"})
_ORG_TYPES = frozenset({
    "Organization", "Party", "Company",
    "组织", "政党", "企业",
})
_COUNTRY_TYPES = frozenset({"Country", "国家", "国际组织"})
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


# ── Data Classes ──

@dataclass
class RegisteredEntity:
    id: str = ""
    name: str = ""
    type: str = ""
    freq: int = 0
    chunk_coverage: int = 0
    decision: str = ""   # "KEEP" | "DISCARD"
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

    def get_top_by_freq(self, n: int) -> list[RegisteredEntity]:
        return sorted(self.entities.values(), key=lambda e: -e.freq)[:n]

    def summary(self) -> str:
        lines = [f"EntityRegistry: {self.total} total, {self.kept} KEEP, {self.discarded} DISCARD"]
        if self.discard_reasons:
            detail = " | ".join(f"{k}:{v}" for k, v in
                                sorted(self.discard_reasons.items(), key=lambda x: -x[1]))
            lines.append(f"  DISCARD: {detail}")
        lines.append("  KEPT entities:")
        for e in self.get_kept()[:20]:
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

    def audit(self, name: str) -> str:
        e = self.find(name)
        if not e:
            return f"实体 '{name}' 不在注册表中"
        return (
            f"{e.name}  type={e.type}  freq={e.freq}  coverage={e.chunk_coverage}\n"
            f"  decision={e.decision}  reason={e.reason}\n"
            f"  parent={e.parent or '无'}  aliases={e.aliases or '无'}"
        )


# ── 分类逻辑（纯代码，零 LLM）──

def _is_dyad(name: str) -> bool:
    # 分隔符拆分："A与B" "A和B" "A-B" 等
    for sep in ("与", "和", "及", "对", "vs", "vs.", "/", "-", "—", "关系", "冲突", "战争",
                  "会谈", "谈判", "对抗", "争端"):
        parts = name.split(sep)
        non_empty = [p for p in parts if p.strip()]
        if len(parts) >= 2 and len(non_empty) >= 2:
            return True
        # 后缀型：像 "中美关系" "俄乌冲突" 等
        if len(non_empty) >= 1 and name.endswith(sep) and len(name) - len(sep) >= 2:
            return True
    # 紧凑二元词：两个国名/地区名并置（"俄乌" "中美" "印巴"），不误杀单国名
    _KNOWN_DYADS = frozenset({"俄乌", "美伊", "中美", "美中", "巴以", "以巴",
                               "印巴", "美俄", "俄美", "朝美", "美朝", "日菲"})
    if name in _KNOWN_DYADS:
        return True
    return False


def _classify_one(name: str, etype: str, freq: int, total: int,
                  person_types: frozenset | None = None,
                  org_types: frozenset | None = None,
                  country_types: frozenset | None = None,
                  extra_keep: frozenset | None = None,
                  extra_discard: frozenset | None = None,
                  threshold_factor: int = 50) -> tuple[bool, str]:
    """确定性实体分类：返回 (include_in_simulation, reason)。"""
    if person_types is None:
        person_types = _PERSON_TYPES
    if org_types is None:
        org_types = _ORG_TYPES
    if country_types is None:
        country_types = _COUNTRY_TYPES
    if extra_keep is None:
        extra_keep = frozenset()
    if extra_discard is None:
        extra_discard = frozenset()
    # 1. 硬排除规则（不可覆盖）
    if etype in _DISCARD_TYPES:
        return False, "类型排除"
    if _is_dyad(name):
        return False, "二元关系词"

    # 2. 领域专属保留词（优先于后续排斥规则，允许领域配置覆盖通用规则）
    if freq >= 1 and any(kw in name for kw in extra_keep):
        return True, "领域保留词"

    # 3. 软排除规则（可被 extra_keep 覆盖）
    if any(name.endswith(s) for s in _TITLE_SUFFIX):
        return False, "职务头衔"
    if any(name.endswith(s) for s in _MILITARY_SUFFIX):
        return False, "军队编制"
    if any(w in name for w in _DEPT_WORDS):
        return False, "政府部门"
    if any(name.endswith(s) for s in _COLLECTIVE_SUFFIX):
        return False, "集合概念"

    # 领域专属排除词
    if any(kw in name for kw in extra_discard):
        return False, "领域排除"

    # 3. 自适应阈值（领域可配置因子）
    threshold = max(1, total // threshold_factor)

    # 4. 保留规则
    if etype in person_types and freq >= threshold:
        return True, f"角色-{etype}(freq≥{threshold})"
    if etype in country_types and freq >= threshold:
        return True, f"国家-{etype}(freq≥{threshold})"
    if etype in org_types and freq >= threshold * 2:
        return True, f"组织-{etype}(freq≥{threshold*2})"
    if freq >= threshold * 5:
        return True, f"兜底(极高频≥{threshold*5})"

    return False, "低频/无类型"


# ── 构造函数 ──

async def build_registry(
    graph: Any,
    preprocessor: Any = None,
    intel_list: list[dict] | None = None,
    ontology: Any = None,
    source_material: str = "",
    domain: str = "",
    log_fn: Any = None,
) -> EntityRegistry:
    """从 Kuzu 图谱构建实体注册表。

    Args:
        graph: DeductionGraphStore 实例。
        preprocessor: DeductionPreprocessor 实例（提供频次和去重信息）。
        intel_list: sorter 输出（仅用于补充 parent/aliases，不参与决策）。
        ontology: Ontology 实例（用于动态推断实体类型的博弈属性，
                  当 LLM 生成新类型名如"国家/政治实体"时无需手动注册）。
    """
    # ── 动态类型推断：从 ontology 中自动识别"人物类"和"组织类"实体类型 ──
    dynamic_person: set[str] = set(_PERSON_TYPES)
    dynamic_org: set[str] = set(_ORG_TYPES)
    dynamic_country: set[str] = set(_COUNTRY_TYPES)
    if ontology is not None:
        for et in getattr(ontology, "entities", []) or []:
            tn = (getattr(et, "name", "") or "").strip()
            if not tn or tn in dynamic_person or tn in dynamic_org or tn in dynamic_country:
                continue
            # 关键词推断："人物/人/角色/Person/Actor"→person
            if any(kw in tn for kw in ("人物", "人", "角色", "Person", "Actor",
                                         "Player", "Individual")):
                dynamic_person.add(tn)
            # 关键词推断："国家/Country"→country（单倍阈值，同人物）
            elif any(kw in tn for kw in ("国家", "国家", "Country")):
                dynamic_country.add(tn)
            # 关键词推断："组织/企业/政党/公司/机构/Org/Party/Company"→org（双倍阈值）
            elif any(kw in tn for kw in ("组织", "企业", "政党", "公司",
                                           "机构", "集团", "Org",
                                           "Party", "Company", "Union")):
                dynamic_org.add(tn)
            elif any(kw in tn for kw in ("Location", "Document", "Event", "Concept",
                                           "Date", "Time", "Facility")):
                pass
            else:
                dynamic_org.add(tn)
    person_types = frozenset(dynamic_person)
    org_types = frozenset(dynamic_org)
    country_types = frozenset(dynamic_country)

    # ── 领域配置：读取 domain_prompts.json 的 registry_tweak 字段 ──
    threshold_factor = 50
    extra_keep = set()
    extra_discard = set()
    if domain:
        from strategy_forge.core.rule_templates import get_domain_prompt
        import json as _json
        raw_tweak = get_domain_prompt(domain, "registry_tweak")
        if raw_tweak:
            try:
                tweak = _json.loads(raw_tweak) if isinstance(raw_tweak, str) else raw_tweak
                if isinstance(tweak, dict):
                    threshold_factor = int(tweak.get("threshold_factor", 50)) or 50
                    extra_keep = set(tweak.get("extra_keep_words", []))
                    extra_discard = set(tweak.get("extra_discard_words", []))
                    logger.info("[EntityRegistry] 领域 %s 配置已加载 (threshold=%d, keep=%d, discard=%d)",
                                domain, threshold_factor, len(extra_keep), len(extra_discard))
                    if log_fn:
                        log_fn("agents", f"领域 {domain} 注册配置已加载 (阈值因子={threshold_factor}, 保留词={len(extra_keep)}, 排除词={len(extra_discard)})")
            except (_json.JSONDecodeError, ValueError, TypeError):
                pass

    # 1. 从 Kuzu 读取所有实体
    result = graph._conn.execute(
        f"MATCH (e:{graph.NODE_TABLE}) RETURN e.id, e.name, e.type, e.description"
    )
    raw: list[dict] = []
    while result.has_next():
        r = result.get_next()
        raw.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3]})

    # 2. 去重（预处理器别名 + 子串合并）
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

    # 3. 频次数据
    freq_map: dict[str, int] = {}
    cov_map: dict[str, int] = {}
    if preprocessor and getattr(preprocessor, "result", None):
        freq_map = getattr(preprocessor.result, "entity_frequencies", {}) or {}
        cov_map = getattr(preprocessor.result, "entity_chunk_coverage", {}) or {}

    # 4. sorter 补充信息（仅 parent/aliases，不用于决策）
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

    # 5. 逐个分类
    registry = EntityRegistry()
    registry.total = len(deduped)
    for p in deduped:
        pname = p.get("name", "")
        ptype = p.get("type", "") or ""
        fm = freq_map.get(pname, 0)
        cv = cov_map.get(pname, 0)

        keep, reason = _classify_one(pname, ptype, fm, registry.total,
                                        person_types, org_types, country_types,
                                        frozenset(extra_keep), frozenset(extra_discard),
                                        threshold_factor)

        intel = intel_map.get(pname, {})
        entity = RegisteredEntity(
            id=p.get("id", ""),
            name=pname,
            type=ptype,
            freq=fm,
            chunk_coverage=cv,
            decision="KEEP" if keep else "DISCARD",
            reason=reason,
            parent=str(intel.get("parent") or ""),
            aliases=list(intel.get("aliases", [])),
        )
        registry.entities[pname] = entity
        if keep:
            registry.kept += 1
        else:
            registry.discarded += 1
            registry.discard_reasons[reason] = registry.discard_reasons.get(reason, 0) + 1

    # 5. LLM 审核：代码规则定基线后，LLM 审核边缘实体
    from strategy_forge.core.config import config as _cfg
    if _cfg.deduction_llm_review and source_material:
        await _llm_review_borderline(registry, deduped, source_material, freq_map, cov_map, log_fn)
        await _llm_merge_and_supplement(registry, deduped, source_material, freq_map, log_fn)

    return registry


async def _llm_review_borderline(
    registry: EntityRegistry,
    deduped: list[dict],
    source: str,
    freq_map: dict[str, int],
    cov_map: dict[str, int],
    log_fn: Any = None,
) -> None:
    """LLM 审核边缘实体：代码规则定基线后，LLM 只看结构化错误。

    审核对象：
    - KEEP 实体中 freq=1 的（可能是噪音）
    - DISCARD 实体中 Person/freq≥1 的（可能是核心人物）
    - KEEP 实体中疑似军队编制/二元词/职务的（代码规则可能漏判）

    LLM 只做结构化纠正（这是军队编制吗？这是职务头衔吗？），不做模糊博弈分类。
    """
    borderline = []
    # 人名特征检测：2-4字中文，不含组织/地点/军队关键词
    _PERSON_NAME_KW = frozenset({"国", "党", "盟", "院", "部", "局", "军", "队", "省",
                                   "市", "组", "委", "会", "府", "署", "厅", "司",
                                   "社", "团", "联", "盟", "网", "报", "台", "站"})
    def _looks_like_person(name: str) -> bool:
        if not (2 <= len(name) <= 4):
            return False
        if not all('\u4e00' <= ch <= '\u9fff' for ch in name):
            return False
        return not any(kw in name for kw in _PERSON_NAME_KW)

    for p in deduped:
        pname = p.get("name", "")
        e = registry.entities.get(pname)
        if e is None:
            continue
        fm = freq_map.get(pname, 0)
        ptype = p.get("type", "") or ""
        # 边缘 KEEP：低频、疑似军队/职务/二元词
        if e.decision == "KEEP" and (fm <= 2
                or any(pname.endswith(s) for s in ("军", "舰队", "司令部", "总统", "总理"))
                or _is_dyad(pname)):
            borderline.append(e)
        # 边缘 DISCARD：人物类型、名字像人名、或非类型排除的未知类型实体（freq≥2）
        elif e.decision == "DISCARD" and fm >= 1 and (
                ptype in ("Person", "人物")
                or any(kw in ptype for kw in ("Person", "人物", "Actor"))
                or _looks_like_person(pname)
                or (ptype not in _DISCARD_TYPES and fm >= 2 and not _is_dyad(pname))):
            borderline.append(e)

    if not borderline:
        return
    # 去重、最多送 20 个实体给 LLM 审核
    seen = set()
    review = []
    for b in borderline[:20]:
        if b.name in seen:
            continue
        seen.add(b.name)
        review.append(b)

    if not review:
        return

    logger.info("[EntityRegistry] LLM 审核启动: %d 个边缘实体待审核", len(review))
    if log_fn:
        log_fn("agents", f"LLM 审核: {len(review)} 个边缘实体待审核")

    prompt_parts = ["你是实体分类审核员。检查以下实体是否被代码规则误分类。"]
    prompt_parts.append("## 种子材料（文本采样）")
    prompt_parts.append(source[:2000])
    prompt_parts.append("\n## 待审核实体（请逐条判断是否需要改写决策）")
    for i, e in enumerate(review, 1):
        prompt_parts.append(
            f"{i}. {e.name}  type={e.type}  freq={e.freq}  "
            f"当前判定={e.decision} 理由={e.reason}")
    prompt_parts.append("""
## 审核规则
1. 军队编制/番号（含"X军"如乌军、俄军、美军、"X舰队"、"X战区"）→ 改为 DISCARD
2. 职务头衔（含"总统""总理""主席""部长""司令"等）→ 改为 DISCARD
3. 二元关系/对抗词（如俄乌、中美、印巴、"A与B"等）→ 改为 DISCARD
4. 中文人名（2-4字，不含国/党/盟/院/部/局等组织词）且频次≥2且在源文本中有独立行动的 → 改为 KEEP
5. 被代码规则因"低频/无类型"排除但实际是重要博弈方（如"吕特"type=元首，"吕特"是北约秘书长）→ 改为 KEEP
6. 其余维持原判

## 输出 JSON
{"overrides": [{"name": "实体名", "decision": "KEEP|DISCARD", "reason": "≤20字理由"}]}
如果全部维持原判，输出 {"overrides": []}

只输出 JSON。""")
    prompt = "\n".join(prompt_parts)

    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    import json as _json
    try:
        client = DeductionLLMClient()
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是实体分类审核员，只输出 JSON。",
            temperature=0,
            max_tokens=500,
        )
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(b.text for b in content if hasattr(b, "text"))
        data = _json.loads(str(content).strip())
        if isinstance(data, dict) and "overrides" in data:
            overrides = data["overrides"]
            kept_to_discard = 0
            discard_to_keep = 0
            for ov in overrides:
                name = ov.get("name", "").strip()
                reason = ov.get("reason", "")[:80]
                new_decision = ov.get("decision", "").strip().upper()
                if not name or new_decision not in ("KEEP", "DISCARD"):
                    continue
                e = registry.find(name)
                if e is None:
                    continue
                old = e.decision
                if old != new_decision:
                    e.decision = new_decision
                    e.reason = f"LLM审核({reason})"
                    if old == "KEEP":
                        registry.kept -= 1
                        registry.discarded += 1
                        kept_to_discard += 1
                        registry.discard_reasons[e.reason] = registry.discard_reasons.get(e.reason, 0) + 1
                    else:
                        registry.kept += 1
                        registry.discarded -= 1
                        discard_to_keep += 1
            if overrides:
                logger.info("[EntityRegistry] LLM 审核完成: %d 条改写 (KEEP→DISCARD:%d, DISCARD→KEEP:%d)",
                            len(overrides), kept_to_discard, discard_to_keep)
                if log_fn:
                    log_fn("agents", f"LLM 审核: {len(overrides)} 条改写 (排除:{kept_to_discard} 恢复:{discard_to_keep})")
    except Exception as e:
        logger.warning("[EntityRegistry] LLM 审核失败，维持代码规则判定: %s", e)


async def _llm_merge_and_supplement(
    registry: EntityRegistry,
    deduped: list[dict],
    source: str,
    freq_map: dict[str, int],
    log_fn: Any = None,
) -> None:
    """LLM 合并审核：父子合并 + KEEP复核 + 从 DISCARD 中补充遗漏的核心博弈方。

    做三件事：
    1. 父子合并：子实体的 parent 也在 KEEP 中 → 将子合并入父
    2. KEEP 复核：全量 KEEP 列表中是否存在代码规则误判的无效博弈者
    3. 缺漏补充：从 DISCARD 列表中找出被代码规则遗漏的核心博弈方
    """
    kept = registry.get_kept()
    discarded = [e for e in registry.entities.values() if e.decision == "DISCARD"]
    if len(kept) < 2:
        return

    prompt_parts = ["你是实体治理审核员。检查代码规则筛选出的博弈实体是否合理。"]
    prompt_parts.append("## 种子材料采样")
    prompt_parts.append(source[:3000])

    prompt_parts.append("\n## 当前博弈实体 (KEEP) — 请逐条复核是否有误判")
    for e in kept[:25]:
        p = e.parent or "无"
        prompt_parts.append(f"  {e.name}  type={e.type}  freq={e.freq}  parent={p}  理由={e.reason}")

    # 找所有 DISCARD 实体（不限于高频——LLM 可以从全文判断重要性）
    high_freq_discard = sorted(
        [e for e in discarded if e.freq >= 1],
        key=lambda e: -e.freq)[:15]
    if high_freq_discard:
        prompt_parts.append("\n## 被排除的实体 (DISCARD) — 是否存在被代码规则错误排除的博弈方？")
        for e in high_freq_discard:
            prompt_parts.append(f"  {e.name}  type={e.type}  freq={e.freq}  排除理由={e.reason}")

    prompt_parts.append("""
## 任务
1. 父子合并：如果实体 parent 是上述某个 KEEP 实体 → 把子实体合并入父实体
2. KEEP 复核：KEEP 列表中是否存在应排除的（如"台海"是地理概念、"顿涅茨克"是地点、"欧洲"是泛指区域）→ 将其降级
3. 缺漏补充：从 DISCARD 列表中，找出被错误排除的核心博弈方（如"俄罗斯"全文多次出现但不在 KEEP → 应恢复）

## 输出 JSON
{"merge_into_parent": ["要合并的子实体名"],
 "demote_from_keep": ["要从KEEP降级的实体名"],
 "supplement_from_discard": ["要恢复的实体名"],
 "reasons": {"实体名": "≤20字理由"}}

只输出 JSON。""")
    prompt = "\n".join(prompt_parts)

    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    import json as _json
    try:
        client = DeductionLLMClient()
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是实体合并审核员，只输出 JSON。",
            temperature=0,
            max_tokens=500,
        )
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(b.text for b in content if hasattr(b, "text"))
        data = _json.loads(str(content).strip())

        if not isinstance(data, dict):
            return

        # 处理父子合并
        merged = 0
        reasons = data.get("reasons", {}) or {}
        for name in data.get("merge_into_parent", []):
            name = str(name).strip()
            if not name:
                continue
            e = registry.find(name)
            if e is None or e.decision != "KEEP":
                continue
            parent = e.parent
            if not parent:
                continue
            pe = registry.find(parent)
            if pe is None:
                continue
            # 合并：子实体降级为 DISCARD
            e.decision = "DISCARD"
            e.reason = f"合并入{parent}(LLM合并审核)"
            registry.kept -= 1
            registry.discarded += 1
            merged += 1

        # 处理 KEEP 复核降级
        demoted = 0
        for name in data.get("demote_from_keep", []):
            name = str(name).strip()
            if not name:
                continue
            e = registry.find(name)
            if e is None or e.decision != "KEEP":
                continue
            reason = str(reasons.get(name, "LLM复核降级"))[:40]
            e.decision = "DISCARD"
            e.reason = f"LLM复核({reason})"
            registry.kept -= 1
            registry.discarded += 1
            demoted += 1

        # 处理缺漏补充
        supplemented = 0
        for name in data.get("supplement_from_discard", []):
            name = str(name).strip()
            if not name:
                continue
            e = registry.find(name)
            if e is None or e.decision != "DISCARD":
                continue
            reason = str(reasons.get(name, "LLM补充"))[:40]
            e.decision = "KEEP"
            e.reason = f"LLM补充({reason})"
            registry.kept += 1
            registry.discarded -= 1
            supplemented += 1

        if merged or demoted or supplemented:
            logger.info("[EntityRegistry] LLM 合并审核: 合并%d 降级%d 补充%d", merged, demoted, supplemented)
            if log_fn:
                parts = []
                if merged: parts.append(f"合并{merged}个")
                if demoted: parts.append(f"降级{demoted}个")
                if supplemented: parts.append(f"补充{supplemented}个")
                log_fn("agents", f"LLM 合并审核: {', '.join(parts)}")
    except Exception as e:
        logger.warning("[EntityRegistry] LLM 合并审核失败: %s", e)
        if log_fn:
            log_fn("agents", f"LLM 合并审核失败（维持代码规则判定）")


# ── 调试入口 ──
if __name__ == "__main__":
    import sys, os, json
    from pathlib import Path

    if len(sys.argv) < 3:
        print("用法: python -m strategy_forge.engine.entity_registry <session_db> <graph_dir>")
        print("  或: python -m strategy_forge.engine.entity_registry <session_id>")
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
    registry = build_registry(graph)
    print(registry.summary())
    graph.close()

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
    "Organization", "Party", "Company", "Country",
    "组织", "政党", "企业", "国家", "国际组织",
})
_TITLE_SUFFIX = (
    "总统", "总理", "司令", "部长", "秘书长", "主席", "书记", "市长",
    "省长", "院长", "局长", "指挥官", "委员长", "主任", "将军", "上将",
    "特使", "代表", "发言人", "CEO", "董事长", "行长", "署长",
)
_MILITARY_SUFFIX = (
    "舰队", "战区", "司令部", "集团军", "师团", "旅", "营", "连",
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
                  org_types: frozenset | None = None) -> tuple[bool, str]:
    """确定性实体分类：返回 (include_in_simulation, reason)。"""
    if person_types is None:
        person_types = _PERSON_TYPES
    if org_types is None:
        org_types = _ORG_TYPES
    # 1. 排除规则（按优先级）
    if etype in _DISCARD_TYPES:
        return False, "类型排除"
    if _is_dyad(name):
        return False, "二元关系词"
    if any(name.endswith(s) for s in _TITLE_SUFFIX):
        return False, "职务头衔"
    if any(name.endswith(s) for s in _MILITARY_SUFFIX):
        return False, "军队编制"
    if any(w in name for w in _DEPT_WORDS):
        return False, "政府部门"
    if any(name.endswith(s) for s in _COLLECTIVE_SUFFIX):
        return False, "集合概念"

    # 2. 自适应阈值
    threshold = max(1, total // 50)

    # 3. 保留规则
    if etype in person_types and freq >= threshold:
        return True, f"角色-{etype}(freq≥{threshold})"
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
    if ontology is not None:
        for et in getattr(ontology, "entities", []) or []:
            tn = (getattr(et, "name", "") or "").strip()
            if not tn or tn in dynamic_person or tn in dynamic_org:
                continue
            # 关键词推断："人物/人/角色/Person/Actor"→person
            if any(kw in tn for kw in ("人物", "人", "角色", "Person", "Actor",
                                         "Player", "Individual")):
                dynamic_person.add(tn)
            # 关键词推断："国家/组织/企业/政党/公司/机构/Country/Org/Party/Company"
            elif any(kw in tn for kw in ("国家", "组织", "企业", "政党", "公司",
                                           "机构", "集团", "Country", "Org",
                                           "Party", "Company", "Union")):
                dynamic_org.add(tn)
            elif any(kw in tn for kw in ("Location", "Document", "Event", "Concept",
                                           "Date", "Time", "Facility")):
                pass  # 不加入任何博弈类型
            else:
                # 未知类型：如果 entity 中该类型实体的平均频次较高 → 视为 org
                dynamic_org.add(tn)
    person_types = frozenset(dynamic_person)
    org_types = frozenset(dynamic_org)

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
                                        person_types, org_types)

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
        await _llm_review_borderline(registry, deduped, source_material, freq_map, cov_map)

    return registry


async def _llm_review_borderline(
    registry: EntityRegistry,
    deduped: list[dict],
    source: str,
    freq_map: dict[str, int],
    cov_map: dict[str, int],
) -> None:
    """LLM 审核边缘实体：代码规则定基线后，LLM 只看结构化错误。

    审核对象：
    - KEEP 实体中 freq=1 的（可能是噪音）
    - DISCARD 实体中 Person/freq≥1 的（可能是核心人物）
    - KEEP 实体中疑似军队编制/二元词/职务的（代码规则可能漏判）

    LLM 只做结构化纠正（这是军队编制吗？这是职务头衔吗？），不做模糊博弈分类。
    """
    borderline = []
    for p in deduped:
        pname = p.get("name", "")
        e = registry.entities.get(pname)
        if e is None:
            continue
        fm = freq_map.get(pname, 0)
        ptype = p.get("type", "") or ""
        # 边缘 KEEP：低频且存在误判风险
        if e.decision == "KEEP" and (fm <= 2 or
                any(pname.endswith(s) for s in ("军", "舰队", "司令部",
                                                  "总统", "总理"))
                or _is_dyad(pname)):
            borderline.append(e)
        # 边缘 DISCARD：人物类型且有一定频次
        elif e.decision == "DISCARD" and fm >= 1 and (
                ptype in ("Person", "人物") or any(kw in ptype for kw in ("Person", "人物", "Actor"))):
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
1. 军队编制/番号（含X军、X舰队、X战区、X集团军、X导弹旅）→ 应排除
2. 职务头衔（含总统、总理、主席、部长、司令、秘书长）→ 应排除  
3. 二元关系/对抗词（含A与B、A和B、A/B、俄乌、中美、印巴等）→ 应排除
4. 核心人物（国家级领导人、组织领导人、关键角色，频次≥2且有独立描述）→ 应保留
5. 其余维持原判

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
            for ov in data["overrides"]:
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
                        registry.discard_reasons[e.reason] = registry.discard_reasons.get(e.reason, 0) + 1
                    else:
                        registry.kept += 1
                        registry.discarded -= 1
    except Exception as e:
        logger.warning("[EntityRegistry] LLM 审核失败，维持代码规则判定: %s", e)


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

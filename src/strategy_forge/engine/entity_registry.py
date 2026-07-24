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
    "国会", "法院", "检察院", "监察院", "行政院",
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
    for sep in ("与", "和", "及", "对", "vs", "vs.", "/", "-", "—"):
        parts = name.split(sep)
        if len(parts) == 2 and all(len(p.strip()) >= 1 for p in parts):
            return True
    return False


def _classify_one(name: str, etype: str, freq: int, total: int) -> tuple[bool, str]:
    """确定性实体分类：返回 (include_in_simulation, reason)。"""
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
    if etype in _PERSON_TYPES and freq >= threshold:
        return True, f"角色(人物+freq≥{threshold})"
    if etype in _ORG_TYPES and freq >= threshold * 2:
        return True, f"组织(freq≥{threshold*2})"
    if freq >= threshold * 5:
        return True, f"兜底(极高频≥{threshold*5})"

    return False, "低频/无类型"


# ── 构造函数 ──

def build_registry(
    graph: Any,
    preprocessor: Any = None,
    intel_list: list[dict] | None = None,
) -> EntityRegistry:
    """从 Kuzu 图谱构建实体注册表。

    Args:
        graph: DeductionGraphStore 实例。
        preprocessor: DeductionPreprocessor 实例（提供频次和去重信息）。
        intel_list: sorter 输出（仅用于补充 parent/aliases，不参与决策）。

    Returns:
        EntityRegistry 实例。
    """
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

        keep, reason = _classify_one(pname, ptype, fm, registry.total)

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

    return registry


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

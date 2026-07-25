"""Entity Registry — 实体注册中心：全部博弈实体的唯一权威数据源。

架构：代码硬排除（100%准确）→ LLM 单次全量分类（温度 0）。

用法：
  registry = await build_registry(graph, preprocessor, intel_list, source_material=source)
  kept = registry.get_kept()
  for e in kept: print(e.name, e.type, e.reason)
"""
from __future__ import annotations

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


# ── 构造函数 ──

async def build_registry(
    graph: Any,
    preprocessor: Any = None,
    intel_list: list[dict] | None = None,
    source_material: str = "",
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
        elif any(pname.endswith(s) for s in _MILITARY_SUFFIX):
            discard_reason = "军队编制"
        elif any(w in pname for w in _DEPT_WORDS):
            discard_reason = "政府部门"
        elif any(pname.endswith(s) for s in _COLLECTIVE_SUFFIX):
            discard_reason = "集合概念"

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
        await _llm_classify(registry, hard_kept, source_material, log_fn)
    else:
        _fallback_classify(registry, hard_kept, log_fn)

    return registry


async def _llm_classify(
    registry: EntityRegistry,
    pending: list[RegisteredEntity],
    source: str,
    log_fn: Any = None,
) -> None:
    """LLM 单次调用：对所有非硬排除实体做 KEEP/DISCARD 分类。"""

    prompt_parts = ["你是实体分类员。判断以下实体在种子材料中是否具有独立战略决策权。"]
    prompt_parts.append("## 种子材料采样")
    prompt_parts.append(source[:3000])
    prompt_parts.append("\n## 待分类实体")
    for i, e in enumerate(pending, 1):
        p = e.parent or "无"
        prompt_parts.append(f"  {i}. {e.name}  type={e.type}  freq={e.freq}  parent={p}")
    prompt_parts.append("""
## 判定标准
具有独立战略决策权：能独立做出影响格局的决策（国家、国际组织、大型企业、核心人物、政党等）
不具有独立战略决策权：地理概念、下属部门、军队编制、职务头衔、泛指集合、二元关系词

## 输出 JSON
{"keep": ["实体名", ...], "discard": ["实体名", ...], "reasons": {"实体名": "≤20字理由"}}
如果实体名未出现在 keep 或 discard 中，默认 discard。

只输出 JSON。""")

    prompt = "\n".join(prompt_parts)
    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    import json as _json
    try:
        client = DeductionLLMClient()
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是实体分类员，只输出 JSON。",
            temperature=0,
            max_tokens=800,
        )
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(b.text for b in content if hasattr(b, "text"))
        data = _json.loads(str(content).strip())
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        keep_set = set(str(n).strip() for n in data.get("keep", []))
        reasons = data.get("reasons", {}) or {}
        for e in pending:
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

        logger.info("[EntityRegistry] LLM 分类: %d KEEP / %d DISCARD", registry.kept, registry.discarded)
        if log_fn:
            log_fn("agents", f"LLM 全量分类: {registry.kept} 保留 / {registry.discarded} 排除")
    except Exception as e:
        logger.warning("[EntityRegistry] LLM 分类失败，回退 fallback: %s", e)
        if log_fn:
            log_fn("agents", f"LLM 分类失败，回退规则兜底")
        _fallback_classify(registry, pending, log_fn)


def _fallback_classify(
    registry: EntityRegistry,
    pending: list[RegisteredEntity],
    log_fn: Any = None,
) -> None:
    """LLM 不可用时的简单兜底规则。"""
    for e in pending:
        if e.type in ("Person", "人物") and e.freq >= 2:
            e.decision = "KEEP"
            e.reason = "兜底(人物≥2)"
            registry.kept += 1
        elif e.type in ("Country", "国家", "国际组织") and e.freq >= 1:
            e.decision = "KEEP"
            e.reason = "兜底(国家≥1)"
            registry.kept += 1
        elif e.freq >= 5:
            e.decision = "KEEP"
            e.reason = "兜底(高频≥5)"
            registry.kept += 1
        else:
            e.decision = "DISCARD"
            e.reason = "兜底排除"
            registry.discarded += 1
            registry.discard_reasons["兜底排除"] = registry.discard_reasons.get("兜底排除", 0) + 1
    if log_fn:
        log_fn("agents", f"兜底规则: {registry.kept} 保留")


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

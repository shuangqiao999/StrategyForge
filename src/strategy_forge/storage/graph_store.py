"""Graph store adapter — Kuzu embedded graph database.

Thread-safe: each thread should use its own Connection.
Primary key required on all node tables.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DeductionGraphStore:

    NODE_TABLE = "Entity"
    AGENT_TABLE = "Agent"
    EVENT_TABLE = "Event"

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Any = None
        self._init()

    def _init(self) -> None:
        import kuzu
        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()
        logger.info("[DeductionGraph] Kuzu database initialized: %s", self._db_path)

    def _init_schema(self) -> None:
        # Kuzu 对每个 NODE TABLE 的 PRIMARY KEY(id) 自动维护 hash 索引，
        # upsert_entity/upsert_relation 等按 id 的 MERGE/MATCH 均为 O(1) 主键查找；
        # 每个会话使用独立的 Kuzu 库目录，无需额外二级索引。
        with self._lock:
            self._conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Entity("
                "id STRING, name STRING, type STRING, description STRING, "
                "properties STRING, PRIMARY KEY(id))"
            )
            self._conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Agent("
                "id STRING, name STRING, persona STRING, background STRING, "
                "goals STRING, PRIMARY KEY(id))"
            )
            self._conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Event("
                "id STRING, description STRING, event_type STRING, "
                "timestamp STRING, agent_id STRING, round INT64, target_id STRING, "
                "PRIMARY KEY(id))"
            )
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS RELATES("
                "FROM Entity TO Entity, relation STRING, weight DOUBLE, evidence STRING)"
            )
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS ACTED("
                "FROM Agent TO Event, action STRING, timestamp STRING)"
            )
            # 因果链(硬档)：行动指向的目标 + 对目标各指标的精确数值影响
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS TARGETS(FROM Event TO Entity)"
            )
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS CAUSED("
                "FROM Event TO Entity, metric STRING, amount DOUBLE)"
            )

    def upsert_entity(self, entity_id: str, name: str, etype: str,
                      description: str = "", properties: str = "{}") -> None:
        # Kuzu 0.11.3 支持 $param 仅限 MERGE 节点匹配，不支持 MATCH..SET = $param。
        # 因此用参数化 MERGE + 内联转义 SET（已验证无 SQL 注入风险）。
        with self._lock:
            self._conn.execute(
                f"MERGE (e:{self.NODE_TABLE} {{id: $id}})",
                {"id": entity_id},
            )
            safe_name = name.replace("'", "\\'")
            safe_type = etype.replace("'", "\\'")
            safe_desc = description.replace("'", "\\'")
            safe_props = properties.replace("'", "\\'")
            self._conn.execute(
                f"MATCH (e:{self.NODE_TABLE} {{id: $id}}) "
                f"SET e.name = '{safe_name}', e.type = '{safe_type}', "
                f"e.description = '{safe_desc}', e.properties = '{safe_props}'",
                {"id": entity_id},
            )

    def upsert_relation(self, source_id: str, target_id: str,
                        relation: str, weight: float = 1.0, evidence: str = "") -> None:
        with self._lock:
            self._conn.execute(
                f"MATCH (a:{self.NODE_TABLE} {{id: $sid}}), (b:{self.NODE_TABLE} {{id: $tid}}) "
                "MERGE (a)-[r:RELATES {relation: $rel}]->(b) "
                "SET r.weight = $w, r.evidence = $ev",
                {"sid": source_id, "tid": target_id, "rel": relation,
                 "w": weight, "ev": evidence},
            )

    def upsert_agent_node(self, agent_id: str, name: str, persona: str,
                          background: str = "", goals: str = "[]") -> None:
        with self._lock:
            self._conn.execute(
                f"MERGE (a:{self.AGENT_TABLE} {{id: $id}}) "
                "SET a.name = $name, a.persona = $persona, a.background = $bg, a.goals = $goals",
                {"id": agent_id, "name": name, "persona": persona,
                 "bg": background, "goals": goals},
            )

    def add_event(self, event_id: str, description: str, event_type: str,
                  timestamp: str, agent_id: str = "", round_number: int = 0,
                  target_id: str = "") -> None:
        safe = {
            "id": event_id.replace("'", "\\'"),
            "desc": description.replace("'", "\\'")[:500],
            "type": event_type.replace("'", "\\'"),
            "ts": timestamp.replace("'", "\\'"),
            "aid": agent_id.replace("'", "\\'"),
            "tid": (target_id or "").replace("'", "\\'"),
        }
        rnd = int(round_number)
        with self._lock:
            self._conn.execute(
                f"CREATE (ev:{self.EVENT_TABLE} {{id: '{safe['id']}', "
                f"description: '{safe['desc']}', event_type: '{safe['type']}', "
                f"timestamp: '{safe['ts']}', agent_id: '{safe['aid']}', "
                f"round: {rnd}, target_id: '{safe['tid']}'}})"
            )

    def add_acted(self, agent_id: str, event_id: str, action: str, timestamp: str = "") -> None:
        with self._lock:
            self._conn.execute(
                f"MATCH (a:{self.AGENT_TABLE} {{id: $aid}}), (ev:{self.EVENT_TABLE} {{id: $eid}}) "
                "CREATE (a)-[:ACTED {action: $act, timestamp: $ts}]->(ev)",
                {"aid": agent_id, "eid": event_id, "act": action, "ts": timestamp},
            )

    def add_targets(self, event_id: str, target_id: str) -> None:
        """事件指向的目标实体（因果链：行动作用于谁）。"""
        with self._lock:
            self._conn.execute(
                f"MATCH (ev:{self.EVENT_TABLE} {{id: $eid}}), (e:{self.NODE_TABLE} {{id: $tid}}) "
                "CREATE (ev)-[:TARGETS]->(e)",
                {"eid": event_id, "tid": target_id},
            )

    def add_caused(self, event_id: str, target_id: str, metric: str, amount: float) -> None:
        """事件对目标某指标造成的精确数值影响（因果链：造成什么后果）。"""
        with self._lock:
            self._conn.execute(
                f"MATCH (ev:{self.EVENT_TABLE} {{id: $eid}}), (e:{self.NODE_TABLE} {{id: $tid}}) "
                "CREATE (ev)-[:CAUSED {metric: $m, amount: $a}]->(e)",
                {"eid": event_id, "tid": target_id, "m": metric, "a": float(amount)},
            )

    # ── Query helpers ──

    def query(self, cypher: str, params: dict | None = None) -> list[list[Any]]:
        result = self._conn.execute(cypher, params or {})
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def count_entities(self) -> int:
        result = self._conn.execute(f"MATCH (e:{self.NODE_TABLE}) RETURN count(e)")
        row = result.get_next()
        return row[0] if row else 0

    def count_relations(self) -> int:
        result = self._conn.execute("MATCH ()-[r:RELATES]->() RETURN count(r)")
        row = result.get_next()
        return row[0] if row else 0

    def get_entities_by_type(self, etype: str) -> list[dict[str, Any]]:
        result = self._conn.execute(
            f"MATCH (e:{self.NODE_TABLE}) WHERE e.type = $t RETURN e.id, e.name, e.type, e.description",
            {"t": etype},
        )
        rows: list[dict[str, Any]] = []
        while result.has_next():
            r = result.get_next()
            rows.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3]})
        return rows

    def get_entity_neighbors(self, entity_id: str, max_depth: int = 1) -> dict[str, Any]:
        """返回实体的关系邻居。

        1 跳返回带 relation/weight 的结构化邻居（用于盟友/对手识别与决策注入）；
        max_depth>1 时附带多跳邻居名称（仅作扩展上下文）。
        """
        rows = self.query(
            f"MATCH (e:{self.NODE_TABLE} {{id: $id}})-[r:RELATES]-(n:{self.NODE_TABLE}) "
            "RETURN r.relation, r.weight, n.id, n.name, n.type",
            {"id": entity_id},
        )
        neighbors = [
            {"relation": r[0] or "", "weight": (r[1] if r[1] is not None else 1.0),
             "id": r[2], "name": r[3] or "", "type": r[4] or ""}
            for r in rows
        ]
        extended: list[str] = []
        if max_depth > 1:
            erows = self.query(
                f"MATCH (e:{self.NODE_TABLE} {{id: $id}})-[:RELATES*2..{max_depth}]-(n:{self.NODE_TABLE}) "
                "RETURN DISTINCT n.name",
                {"id": entity_id},
            )
            extended = [r[0] for r in erows if r and r[0]]
        return {"entity_id": entity_id, "neighbors": neighbors, "extended": extended}

    def get_agent_timelines(self, limit_per_agent: int = 20) -> list[dict[str, Any]]:
        """按智能体聚合"行动—事件"时序（读取 Agent-[ACTED]->Event 时序行动图）。"""
        rows = self.query(
            f"MATCH (a:{self.AGENT_TABLE})-[r:ACTED]->(ev:{self.EVENT_TABLE}) "
            "RETURN a.id, a.name, r.action, r.timestamp, ev.description, ev.event_type "
            "ORDER BY r.timestamp"
        )
        grouped: dict[str, dict[str, Any]] = {}
        for r in rows:
            aid = r[0]
            bucket = grouped.setdefault(aid, {"agent_id": aid, "agent_name": r[1] or aid[:8],
                                              "actions": []})
            if len(bucket["actions"]) < limit_per_agent:
                bucket["actions"].append({
                    "action": r[2] or "", "timestamp": r[3] or "",
                    "description": r[4] or "", "event_type": r[5] or "",
                })
        return list(grouped.values())

    def get_event_sequence(self, limit: int = 40) -> list[dict[str, Any]]:
        """全局按时间排序的事件序列（供因果链分析）。"""
        rows = self.query(
            f"MATCH (a:{self.AGENT_TABLE})-[r:ACTED]->(ev:{self.EVENT_TABLE}) "
            "RETURN r.timestamp, a.name, r.action, ev.description, ev.event_type "
            "ORDER BY r.timestamp"
        )
        seq = [{
            "timestamp": r[0] or "", "agent_name": r[1] or "", "action": r[2] or "",
            "description": r[3] or "", "event_type": r[4] or "",
        } for r in rows]
        return seq[-limit:] if limit and len(seq) > limit else seq

    # ── 因果链（硬档）：基于 CAUSED 边的确定性归因 ──

    def get_outcome_attribution(self, entity_id: str) -> dict[str, Any]:
        """对某实体：按来源 agent 汇总 CAUSED 数值（负=致衰、正=助益），排名主因。"""
        rows = self.query(
            f"MATCH (a:{self.AGENT_TABLE})-[:ACTED]->(ev:{self.EVENT_TABLE})"
            f"-[c:CAUSED]->(e:{self.NODE_TABLE} {{id: $id}}) "
            "RETURN a.name, c.metric, sum(c.amount)",
            {"id": entity_id},
        )
        by_source: dict[str, dict[str, float]] = {}
        for r in rows:
            src = r[0] or "?"
            metric = r[1] or ""
            amt = float(r[2]) if r[2] is not None else 0.0
            m = by_source.setdefault(src, {})
            m[metric] = m.get(metric, 0.0) + amt
        contributors = []
        for src, metrics in by_source.items():
            harm = sum(v for v in metrics.values() if v < 0)
            benefit = sum(v for v in metrics.values() if v > 0)
            contributors.append({
                "source": src, "harm": round(harm, 2), "benefit": round(benefit, 2),
                "by_metric": {k: round(v, 2) for k, v in metrics.items()},
            })
        contributors.sort(key=lambda x: x["harm"])  # 最负(致衰最重)在前
        return {"entity_id": entity_id, "contributors": contributors}

    def get_causal_summary(self, limit: int = 15) -> list[dict[str, Any]]:
        """全局"源→目标 累计指标影响"摘要，按致衰程度排序（供报告确定性归因）。"""
        rows = self.query(
            f"MATCH (a:{self.AGENT_TABLE})-[:ACTED]->(ev:{self.EVENT_TABLE})"
            f"-[c:CAUSED]->(e:{self.NODE_TABLE}) "
            "RETURN a.name, e.name, c.metric, sum(c.amount)"
        )
        items = [{
            "source": r[0] or "?", "target": r[1] or "?", "metric": r[2] or "",
            "amount": round(float(r[3]), 1) if r[3] is not None else 0.0,
        } for r in rows]
        items.sort(key=lambda x: x["amount"])
        return items[:limit]

    def get_causal_subgraph(self, limit: int = 200) -> dict[str, Any]:
        """因果子图（Agent→Event→Entity，CAUSED 带 metric/amount），供前端因果视图。"""
        nodes: dict[str, dict[str, Any]] = {}
        links: list[dict[str, Any]] = []
        rows = self.query(
            f"MATCH (a:{self.AGENT_TABLE})-[:ACTED]->(ev:{self.EVENT_TABLE}) "
            "RETURN a.id, a.name, ev.id, ev.event_type, ev.round, ev.description "
            f"ORDER BY ev.round LIMIT {int(limit)}"
        )
        for r in rows:
            aid, aname, eid, etype, rnd, desc = r[0], r[1], r[2], r[3], r[4], r[5]
            nodes.setdefault(aid, {"id": aid, "kind": "agent", "label": aname or aid[:8]})
            nodes.setdefault(eid, {"id": eid, "kind": "event",
                                   "label": f"R{rnd or 0}:{etype or ''}",
                                   "desc": (desc or "")[:80]})
            links.append({"source": aid, "target": eid, "type": "ACTED", "label": etype or ""})
        crows = self.query(
            f"MATCH (ev:{self.EVENT_TABLE})-[c:CAUSED]->(e:{self.NODE_TABLE}) "
            f"RETURN ev.id, e.id, e.name, c.metric, c.amount LIMIT {int(limit) * 4}"
        )
        for r in crows:
            eid, tid, tname, metric, amount = r[0], r[1], r[2], r[3], r[4]
            nodes.setdefault(tid, {"id": tid, "kind": "entity", "label": tname or tid[:8]})
            lbl = f"{metric}{amount:+.0f}" if amount is not None else (metric or "")
            links.append({"source": eid, "target": tid, "type": "CAUSED", "label": lbl})
        return {"nodes": list(nodes.values()), "links": links}

    def export_graph_data(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        result = self._conn.execute(
            f"MATCH (e:{self.NODE_TABLE}) RETURN e.id, e.name, e.type, e.description"
        )
        while result.has_next():
            r = result.get_next()
            nodes.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3]})

        links: list[dict[str, Any]] = []
        result = self._conn.execute(
            "MATCH (a)-[r:RELATES]->(b) RETURN a.id, b.id, r.relation, r.weight"
        )
        while result.has_next():
            r = result.get_next()
            links.append({"source": r[0], "target": r[1], "relation": r[2], "weight": r[3]})

        return {"nodes": nodes, "links": links}

    def close(self) -> None:
        self._conn = None
        self._db = None
        logger.info("[DeductionGraph] Kuzu database closed")

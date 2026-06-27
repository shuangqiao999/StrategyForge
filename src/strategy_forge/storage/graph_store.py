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
    CHUNK_TABLE = "Chunk"
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
                "CREATE NODE TABLE IF NOT EXISTS Chunk("
                "id STRING, content STRING, source STRING, "
                "chunk_index INT64, PRIMARY KEY(id))"
            )
            self._conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Agent("
                "id STRING, name STRING, persona STRING, background STRING, "
                "goals STRING, PRIMARY KEY(id))"
            )
            self._conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Event("
                "id STRING, description STRING, event_type STRING, "
                "timestamp STRING, agent_id STRING, PRIMARY KEY(id))"
            )
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS RELATES("
                "FROM Entity TO Entity, relation STRING, weight DOUBLE, evidence STRING)"
            )
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS MENTIONS("
                "FROM Chunk TO Entity, confidence DOUBLE)"
            )
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS ACTED("
                "FROM Agent TO Event, action STRING, timestamp STRING)"
            )
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS PARTICIPATES("
                "FROM Agent TO Entity, role STRING)"
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

    def upsert_chunk(self, chunk_id: str, content: str, source: str, chunk_index: int = 0) -> None:
        with self._lock:
            self._conn.execute(
                f"MERGE (c:{self.CHUNK_TABLE} {{id: $id}}) "
                "SET c.content = $content, c.source = $source, c.chunk_index = $idx",
                {"id": chunk_id, "content": content, "source": source, "idx": chunk_index},
            )

    def add_mention(self, chunk_id: str, entity_id: str, confidence: float = 1.0) -> None:
        with self._lock:
            self._conn.execute(
                f"MATCH (c:{self.CHUNK_TABLE} {{id: $cid}}), (e:{self.NODE_TABLE} {{id: $eid}}) "
                "CREATE (c)-[:MENTIONS {confidence: $conf}]->(e)",
                {"cid": chunk_id, "eid": entity_id, "conf": confidence},
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
                  timestamp: str, agent_id: str = "") -> None:
        safe = {
            "id": event_id.replace("'", "\\'"),
            "desc": description.replace("'", "\\'")[:500],
            "type": event_type.replace("'", "\\'"),
            "ts": timestamp.replace("'", "\\'"),
            "aid": agent_id.replace("'", "\\'"),
        }
        with self._lock:
            self._conn.execute(
                f"CREATE (ev:{self.EVENT_TABLE} {{id: '{safe['id']}', "
                f"description: '{safe['desc']}', event_type: '{safe['type']}', "
                f"timestamp: '{safe['ts']}', agent_id: '{safe['aid']}'}})"
            )

    def add_acted(self, agent_id: str, event_id: str, action: str, timestamp: str = "") -> None:
        with self._lock:
            self._conn.execute(
                f"MATCH (a:{self.AGENT_TABLE} {{id: $aid}}), (ev:{self.EVENT_TABLE} {{id: $eid}}) "
                "CREATE (a)-[:ACTED {action: $act, timestamp: $ts}]->(ev)",
                {"aid": agent_id, "eid": event_id, "act": action, "ts": timestamp},
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

    def get_entity_neighbors(self, entity_id: str, max_depth: int = 2) -> dict[str, Any]:
        rows = self.query(
            f"MATCH (e:{self.NODE_TABLE} {{id: $id}})-[r:RELATES*1..{max_depth}]-(n) "
            "RETURN e.id, type(r), n.id, n.name, n.type",
            {"id": entity_id},
        )
        return {"entity_id": entity_id, "relations": rows}

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

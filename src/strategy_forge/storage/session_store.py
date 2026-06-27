"""Session store — SQLite-backed deduction session persistence."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionStore:

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deduction_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    source_material TEXT DEFAULT '',
                    status TEXT DEFAULT 'created',
                    phase TEXT DEFAULT 'ontology',
                    config_json TEXT DEFAULT '{}',
                    entity_count INTEGER DEFAULT 0,
                    relation_count INTEGER DEFAULT 0,
                    agent_count INTEGER DEFAULT 0,
                    current_round INTEGER DEFAULT 0,
                    total_rounds INTEGER DEFAULT 10,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT DEFAULT '',
                    report_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deduction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES deduction_sessions(id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_deduction_logs_session "
                "ON deduction_logs(session_id, timestamp)"
            )
            # 高效索引: get_logs 按 (session_id ORDER BY id)、delete 按 session_id
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_deduction_logs_session_id "
                "ON deduction_logs(session_id, id)"
            )
            # 高效索引: list_all 按 created_at DESC 分页
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_deduction_sessions_created "
                "ON deduction_sessions(created_at DESC)"
            )
            conn.commit()

    def create(self, session_id: str, title: str, source_material: str,
               config: dict[str, Any] | None = None) -> dict[str, Any]:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO deduction_sessions (id, title, source_material, config_json, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, title, source_material,
                 json.dumps(config or {}, ensure_ascii=False), now, now),
            )
            conn.commit()
        return self.get(session_id)

    def update(self, session_id: str, **kwargs: Any) -> dict[str, Any] | None:
        if not kwargs:
            return self.get(session_id)
        now = datetime.now().isoformat()
        set_parts = [f"{k} = ?" for k in kwargs]
        set_parts.append("updated_at = ?")
        values = list(kwargs.values()) + [now, session_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE deduction_sessions SET {', '.join(set_parts)} WHERE id = ?",
                values,
            )
            conn.commit()
        return self.get(session_id)

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM deduction_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["config_json"] = json.loads(d.get("config_json", "{}") or "{}")
        d["report_json"] = json.loads(d.get("report_json", "{}") or "{}")
        return d

    def list_all(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, status, phase, entity_count, relation_count, "
                "agent_count, current_round, total_rounds, created_at, updated_at "
                "FROM deduction_sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def append_log(self, session_id: str, phase: str, message: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO deduction_logs (session_id, phase, message, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (session_id, phase, message, now),
            )
            conn.commit()

    def get_logs(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, phase, message, timestamp FROM deduction_logs "
                "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM deduction_logs WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM deduction_sessions WHERE id = ?", (session_id,))
            conn.commit()

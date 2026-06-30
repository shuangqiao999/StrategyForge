"""Deduction Engine — main entry point.

Wires together: session store, Kuzu graph store, orchestrator.
Created once per Agent (like KnowledgeBaseManager).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from strategy_forge.core.config import config
from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.storage.session_store import SessionStore

from .models import DeductionSession, SessionStatus
from .orchestrator import DeductionOrchestrator

logger = logging.getLogger(__name__)


class DeductionEngine:

    def __init__(self, workspace_root: str | Path) -> None:
        ws = Path(workspace_root)
        data_dir = config.deduction_data_dir
        if not data_dir.is_absolute():
            data_dir = ws / data_dir
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self.session_store = SessionStore(self._data_dir / "sessions.db")
        self.graph: DeductionGraphStore | None = None
        self._graph_sid: str | None = None
        self._stream_events: dict[str, asyncio.Event] = {}
        self._round_data: dict[str, dict[str, int]] = {}

    def get_graph(self, session_id: str) -> DeductionGraphStore:
        # 句柄按 session_id 缓存；切换会话时先释放旧库句柄，避免多会话串库。
        if self.graph is not None and self._graph_sid == session_id:
            return self.graph
        self.close_graph()
        path = self._data_dir / "graphs" / session_id / "kuzu"
        self.graph = DeductionGraphStore(path)
        self._graph_sid = session_id
        return self.graph

    def close_graph(self) -> None:
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        self._graph_sid = None

    def create_session(self, title: str, source_material: str,
                       config: dict[str, Any] | None = None) -> DeductionSession:
        import uuid
        sid = uuid.uuid4().hex[:12]
        data = self.session_store.create(sid, title, source_material, config)
        return self._row_to_session(data)

    def get_session(self, session_id: str) -> DeductionSession | None:
        data = self.session_store.get(session_id)
        if data is None:
            return None
        return self._row_to_session(data)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.session_store.list_all(limit=limit)

    def delete_session(self, session_id: str, force: bool = False) -> None:
        existing = self.session_store.get(session_id)
        if existing and existing.get("status") in (
            "ontology_running", "graph_running", "agents_running",
            "simulating", "reporting", "optimizing",
        ):
            if not force:
                raise ValueError("推演/优化进行中，无法删除该会话（可传 force=true 强制删除）")
            logger.warning("[Engine] 强制删除进行中的会话: %s (status=%s)", session_id, existing.get("status"))
        self.close_graph()
        self.session_store.delete(session_id)
        # 清理 LanceDB 向量表 (物理回收磁盘空间)
        try:
            from .preprocessor import DeductionPreprocessor
            pp = DeductionPreprocessor(self._data_dir.parent.parent, session_id)
            pp.drop_tables()
        except Exception as e:
            logger.warning("[Engine] Failed to clean LanceDB for %s: %s", session_id, e)
        # 清理 Kuzu 物理文件夹
        import shutil
        kuzu_path = self._data_dir / "graphs" / session_id
        if kuzu_path.exists():
            shutil.rmtree(kuzu_path, ignore_errors=True)
            logger.info("[Engine] Removed Kuzu graph dir: %s", kuzu_path)

    def log(self, session_id: str, phase: str, message: str) -> None:
        self.session_store.append_log(session_id, phase, message)
        ev = self._stream_events.get(session_id)
        if ev:
            ev.set()

    def get_logs(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self.session_store.get_logs(session_id, limit=limit)

    def _ensure_event(self, session_id: str) -> asyncio.Event:
        if session_id not in self._stream_events:
            self._stream_events[session_id] = asyncio.Event()
        return self._stream_events[session_id]

    def get_stream_event(self, session_id: str) -> asyncio.Event:
        return self._ensure_event(session_id)

    def signal_round_complete(self, session_id: str, round_num: int, total_rounds: int) -> None:
        self._round_data[session_id] = {"round": round_num, "total": total_rounds}
        ev = self._stream_events.get(session_id)
        if ev:
            ev.set()

    def get_round_data(self, session_id: str) -> dict[str, int]:
        return self._round_data.get(session_id, {})

    def cleanup_events(self, session_id: str) -> None:
        self._stream_events.pop(session_id, None)
        self._round_data.pop(session_id, None)

    async def start(self, session_id: str, cancel_event=None) -> DeductionSession:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        if session.status in (SessionStatus.SIMULATING, SessionStatus.REPORTING):
            return session

        graph = self.get_graph(session_id)

        is_resume = session.status == SessionStatus.PAUSED
        resume_start_round = 0
        if is_resume:
            resume_start_round = max(session.current_round, 0)
            self.log(session_id, "orchestrator",
                      f"检测到暂停记录 (第 {resume_start_round} 轮), 从断点继续推演")

        orchestrator = DeductionOrchestrator(
            session=session,
            graph=graph,
            session_store=self.session_store,
            logger_fn=lambda phase, msg: self.log(session_id, phase, msg),
            cancel_event=cancel_event,
            round_callback=lambda rnd, total: self.signal_round_complete(session_id, rnd, total),
            resume_start_round=resume_start_round,
        )

        await orchestrator.run()

        # Persist token stats
        import json
        from strategy_forge.core.token_counter import accumulator
        stats = accumulator.get_session_stats(session_id)
        if stats:
            self.session_store.update(session_id,
                                      token_json=json.dumps(stats, ensure_ascii=False))

        updated = self.get_session(session_id)
        if updated is None:
            raise RuntimeError("Session lost after pipeline")
        return updated

    @staticmethod
    def _row_to_session(data: dict[str, Any]) -> DeductionSession:
        config = data.get("config_json", {}) or {}
        return DeductionSession(
            id=data["id"],
            title=data.get("title", ""),
            source_material=data.get("source_material", ""),
            status=SessionStatus(data.get("status", "created")),
            entity_count=data.get("entity_count", 0),
            relation_count=data.get("relation_count", 0),
            agent_count=data.get("agent_count", 0),
            current_round=data.get("current_round", 0),
            total_rounds=config.get("total_rounds", data.get("total_rounds", 10)),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            error=data.get("error", ""),
        )

    def close(self) -> None:
        self.close_graph()

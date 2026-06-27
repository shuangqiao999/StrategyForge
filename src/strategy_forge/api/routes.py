"""StrategyForge API routes — REST + SSE streaming.

All endpoints under /api/forge/.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/forge", tags=["strategy-forge"])

_MAX_UPLOAD = 20 * 1024 * 1024
_ALLOWED_EXT = {
    ".txt", ".md", ".markdown", ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".c", ".cpp", ".h",
    ".pdf", ".docx", ".csv", ".log", ".rst", ".html", ".htm",
}


def _extract_text_from_file(file_path: str, suffix: str) -> str:
    path = Path(file_path)
    text_exts = {
        ".txt", ".md", ".markdown", ".json", ".yaml", ".yml",
        ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".c", ".cpp",
        ".h", ".csv", ".log", ".rst", ".html", ".htm",
    }
    if suffix in text_exts:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:100_000]
        except Exception:
            return path.read_text(encoding="gbk", errors="replace")[:100_000]
    if suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)[:100_000]
        except ImportError:
            raise HTTPException(501, "PDF parsing requires PyPDF2 (pip install PyPDF2)")
    if suffix == ".docx":
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)[:100_000]
        except ImportError:
            raise HTTPException(501, "DOCX parsing requires python-docx (pip install python-docx)")
    raise HTTPException(400, f"Unsupported file type: {suffix}")


# ── Request models ──

class CreateSessionRequest(BaseModel):
    title: str = Field(default="", description="会话标题")
    source_material: str = Field(default="", description="种子材料/原文")
    config: dict[str, Any] = Field(default_factory=dict)


class InterventionRequest(BaseModel):
    content: str = Field(default="", description="用户干预内容")
    scope: str = Field(default="during", description="pre | during")
    round_number: int | None = Field(None, description="指定生效轮次")


class PreGoalRequest(BaseModel):
    content: str = Field(default="", description="推演前的愿景/目标")


# ── File upload ──

@router.post("/upload")
async def upload_source_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型: {suffix}")
    if not file.filename:
        raise HTTPException(400, "文件名为空")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        total = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_UPLOAD:
                raise HTTPException(400, "文件超过 20MB 限制")
            os.write(fd, chunk)
    finally:
        os.close(fd)
    try:
        text = _extract_text_from_file(tmp_path, suffix)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"文本提取失败: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return {
        "status": "ok", "filename": file.filename,
        "size": total, "text_content": text,
    }


# ── Session CRUD ──

@router.post("/session")
async def create_session(req: CreateSessionRequest, request: Request):
    engine = _get_engine(request)
    session = engine.create_session(req.title, req.source_material, req.config)
    return {
        "id": session.id, "title": session.title,
        "status": session.status.value, "created_at": session.created_at,
    }


@router.get("/session/{session_id}")
async def get_session(session_id: str, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {
        "id": session.id, "title": session.title,
        "status": session.status.value, "phase": session.phase.value,
        "entity_count": session.entity_count, "relation_count": session.relation_count,
        "agent_count": session.agent_count, "current_round": session.current_round,
        "total_rounds": session.total_rounds,
        "created_at": session.created_at, "error": session.error,
    }


@router.get("/sessions")
async def list_sessions(limit: int = Query(50, ge=1, le=200), request: Request = None):
    engine = _get_engine(request)
    return engine.list_sessions(limit=limit)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    engine = _get_engine(request)
    try:
        engine.delete_session(session_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"deleted": session_id}


# ── Pipeline control ──

@router.post("/session/{session_id}/start")
async def start_deduction(session_id: str, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    try:
        updated = await engine.start(session_id)
        return {
            "session_id": updated.id,
            "status": updated.status.value,
            "report": {
                "summary": updated.report.summary if updated.report else "",
            } if updated.report else None,
        }
    except Exception as e:
        logger.exception("[StrategyForge] start failed")
        raise HTTPException(500, str(e))


@router.post("/session/{session_id}/intervene")
async def intervene_session(session_id: str, req: InterventionRequest, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    round_num = req.round_number or (session.current_round + 1)
    try:
        from strategy_forge.engine.preprocessor import DeductionPreprocessor
        preprocessor = getattr(request.app.state, f"_pp_{session_id}", None)
        if preprocessor is None:
            preprocessor = DeductionPreprocessor(
                engine._data_dir.parent.parent, session_id)
            setattr(request.app.state, f"_pp_{session_id}", preprocessor)
        preprocessor.add_event_memory(
            content=req.content, agent_id="system_user",
            round_number=round_num,
            event_type="user_intervention", priority=1.0,
        )
        engine.log(session_id, "intervene", f"用户干预: {req.content[:100]}")
        return {"session_id": session_id, "injected": True, "round_number": round_num}
    except Exception as e:
        raise HTTPException(500, f"干预注入失败: {e}")


@router.post("/session/{session_id}/pre-goal")
async def set_pre_goal(session_id: str, req: PreGoalRequest, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    data = engine.session_store.get(session_id)
    config = (data or {}).get("config_json", {}) or {}
    if isinstance(config, str):
        config = json.loads(config)
    pre_goals = config.get("pre_goals", [])
    pre_goals.append(req.content)
    config["pre_goals"] = pre_goals
    engine.session_store.update(session_id, config_json=json.dumps(config, ensure_ascii=False))
    engine.log(session_id, "pre-goal", f"推演前目标: {req.content[:100]}")
    return {"session_id": session_id, "pre_goals": pre_goals}


# ── Data export ──

@router.get("/session/{session_id}/graph")
async def get_graph_data(session_id: str, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    graph = engine.get_graph(session_id)
    return graph.export_graph_data()


@router.get("/session/{session_id}/report")
async def get_report(session_id: str, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    data = engine.session_store.get(session_id)
    if data is None:
        raise HTTPException(404, "Session data not found")
    report_json = data.get("report_json", {}) or {}
    return {
        "session_id": session_id,
        "status": session.status.value,
        "report": report_json if isinstance(report_json, dict) else json.loads(report_json),
    }


@router.get("/session/{session_id}/logs")
async def get_logs(session_id: str, limit: int = Query(200), request: Request = None):
    engine = _get_engine(request)
    return engine.get_logs(session_id, limit=limit)


# ── SSE Stream ──

@router.get("/session/{session_id}/stream")
async def stream_deduction(session_id: str, request: Request):
    async def event_generator():
        engine = _get_engine(request)
        last_log_id = 0
        while True:
            logs = engine.get_logs(session_id, limit=50)
            new_logs = [l for l in logs if l.get("id", 0) > last_log_id]
            for log_entry in new_logs:
                last_log_id = max(last_log_id, log_entry.get("id", 0))
                yield f"data: {json.dumps(log_entry, ensure_ascii=False)}\n\n"
            session = engine.get_session(session_id)
            if session and session.status.value in ("complete", "failed", "paused"):
                yield f"data: {json.dumps({'type': 'status', 'status': session.status.value}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _get_engine(request: Request):
    """Lazy-init the DeductionEngine on the FastAPI app state."""
    engine = getattr(request.app.state, "forge_engine", None)
    if engine is None:
        from strategy_forge.core.config import config
        from strategy_forge.engine.engine import DeductionEngine
        engine = DeductionEngine(config.project_root)
        request.app.state.forge_engine = engine
    return engine

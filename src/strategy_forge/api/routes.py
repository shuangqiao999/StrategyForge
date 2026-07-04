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
    # Cleanup accumulator to free memory
    from strategy_forge.core.token_counter import accumulator
    accumulator.remove_session(session_id)
    return {"deleted": session_id}


@router.delete("/session/{session_id}/force")
async def force_delete_session(session_id: str, request: Request):
    engine = _get_engine(request)
    engine.delete_session(session_id, force=True)
    return {"deleted": session_id}


# ── Pipeline control ──

def _ded_cancel_state(app):
    if not hasattr(app.state, "ded_cancels"):
        app.state.ded_cancels = {}
    return app.state.ded_cancels


@router.post("/session/{session_id}/start")
async def start_deduction(session_id: str, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    cancels = _ded_cancel_state(request.app)
    if session_id in cancels and not cancels[session_id].is_set():
        raise HTTPException(409, "该会话的推演任务正在运行中")
    cancel_event = asyncio.Event()
    cancels[session_id] = cancel_event
    try:
        # 后台异步执行推演，立即返回——前端通过轮询状态感知"推演中"并显示取消按钮
        asyncio.create_task(_run_deduction(engine, session_id, cancel_event, cancels))
        return {"session_id": session_id, "status": "started"}
    except Exception as e:
        logger.exception("[StrategyForge] start failed")
        cancels.pop(session_id, None)
        raise HTTPException(500, str(e))


async def _run_deduction(engine, session_id: str, cancel_event, cancels: dict):
    try:
        await engine.start(session_id, cancel_event=cancel_event)
    except Exception:
        logger.exception("[StrategyForge] background deduction failed")
    finally:
        cancels.pop(session_id, None)


@router.post("/session/{session_id}/start/cancel")
async def cancel_deduction(session_id: str, request: Request):
    cancels = _ded_cancel_state(request.app)
    ev = cancels.get(session_id)
    if ev is not None:
        ev.set()
        return {"cancelled": True}
    return {"cancelled": False}


@router.post("/session/{session_id}/pause")
async def pause_deduction(session_id: str, request: Request):
    """暂停推演：保存当前进度到快照，状态变为 paused。"""
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status.value not in _RUNNING_STATUSES:
        raise HTTPException(409, f"无法暂停：当前状态为 {session.status.value}")
    cancels = _ded_cancel_state(request.app)
    ev = cancels.get(session_id)
    if ev is not None:
        ev.set()
        engine.log(session_id, "control", "用户暂停推演")
        return {"session_id": session_id, "status": "pausing"}
    return {"session_id": session_id, "status": "idle"}


@router.post("/session/{session_id}/resume")
async def resume_deduction(session_id: str, request: Request):
    """继续推演：从 paused 断点恢复。"""
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status != SessionStatus.PAUSED:
        raise HTTPException(409, f"无法继续：当前状态为 {session.status.value}")
    engine.log(session_id, "control", "用户继续推演")
    # Re-use /start logic which handles paused→resume
    cancel_event = asyncio.Event()
    cancels = _ded_cancel_state(request.app)
    cancels[session_id] = cancel_event
    asyncio.create_task(_run_deduction(engine, session_id, cancel_event, cancels))
    return {"session_id": session_id, "status": "resuming"}


# ── 策略优化器 (蒙特卡洛多方案对比) ──

class OptimizeRequest(BaseModel):
    scenarios: list[dict] = Field(default_factory=list)
    win_condition: str = ""
    iterations: int = 20
    objective: str = "balanced"
    max_concurrent: int | None = None


class SettingsRequest(BaseModel):
    enable_multi_action: bool = False
    max_actions: int = 3
    weather: str = ""
    terrain: str = ""


def _opt_state(app):
    if not hasattr(app.state, "optimize_tasks"):
        app.state.optimize_tasks = {}
    if not hasattr(app.state, "optimize_cancel"):
        app.state.optimize_cancel = {}
    if not hasattr(app.state, "optimize_progress"):
        app.state.optimize_progress = {}
    return app.state.optimize_tasks, app.state.optimize_cancel, app.state.optimize_progress


@router.post("/session/{session_id}/optimize")
async def run_optimization(session_id: str, body: OptimizeRequest, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    tasks, cancels, progress = _opt_state(request.app)
    existing = tasks.get(session_id)
    if existing is not None and not existing.done():
        raise HTTPException(409, "该会话的优化任务正在运行中")
    if session.status.value in ("simulating", "reporting", "optimizing"):
        raise HTTPException(409, "推演/优化进行中，请等待完成")

    # 方案校验（至少 1 个；允许无指令的默认基线策略）
    scenarios: list[dict] = []
    for idx, s in enumerate(body.scenarios or []):
        directive = str(s.get("directive", "")).strip()
        name = str(s.get("name", "")).strip() or f"方案 {idx + 1}"
        scenarios.append({"name": name, "directive": directive,
                          "win_target": s.get("win_target")})
    if not scenarios:
        scenarios = [{"name": "默认策略", "directive": ""}]

    # 胜利条件回退：请求 > 会话 pre_goals > 报错
    win_condition = (body.win_condition or "").strip()
    if not win_condition:
        data = engine.session_store.get(session_id)
        cfg = (data or {}).get("config_json", {}) or {}
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        pre_goals = cfg.get("pre_goals", [])
        if pre_goals:
            win_condition = "；".join(str(g) for g in pre_goals)
        else:
            raise HTTPException(
                400,
                "未指定胜利条件(win_condition)。请在优化器面板填写胜利条件，"
                "或先为该会话设定推演前目标(pre-goal)。",
            )

    iterations = max(1, min(int(body.iterations or 20), 200))

    # 多动作设置以会话 config_json 为单一真值源（由 /settings 写入），优化器与主推演统一读取

    from strategy_forge.engine.optimizer import StrategyOptimizer
    optimizer = StrategyOptimizer(engine)
    cancel_event = asyncio.Event()
    cancels[session_id] = cancel_event
    progress[session_id] = {"done": 0, "total": len(scenarios) * iterations,
                            "current": "", "best_win": 0.0}

    def progress_cb(done, total, current, outcome):
        prev = progress.get(session_id, {})
        progress[session_id] = {
            "done": done, "total": total, "current": current,
            "best_win": max(prev.get("best_win", 0.0), outcome.win_score),
        }

    async def _task():
        engine.session_store.update(session_id, status="optimizing")
        try:
            report = await optimizer.run_monte_carlo(
                session_id=session_id, scenarios=scenarios, win_condition=win_condition,
                iterations=iterations, objective=body.objective,
                max_concurrent=body.max_concurrent, cancel_event=cancel_event,
                progress_cb=progress_cb,
            )
            engine.session_store.update(
                session_id,
                optimization_report_json=json.dumps(report, ensure_ascii=False),
                status="complete",
            )
        except Exception as e:
            logger.exception("[StrategyForge] optimize failed")
            engine.session_store.update(session_id, status="failed", error=str(e)[:500])
            engine.log(session_id, "optimize", f"优化失败：{e}")
        finally:
            cancels.pop(session_id, None)

    tasks[session_id] = asyncio.create_task(_task())
    return {"status": "started", "total_runs": len(scenarios) * iterations}


@router.post("/session/{session_id}/optimize/cancel")
async def cancel_optimization(session_id: str, request: Request):
    _, cancels, _ = _opt_state(request.app)
    ev = cancels.get(session_id)
    if ev is not None:
        ev.set()
        return {"cancelled": True}
    return {"cancelled": False}


@router.get("/session/{session_id}/optimize/result")
async def optimization_result(session_id: str, request: Request):
    engine = _get_engine(request)
    tasks, _, progress = _opt_state(request.app)
    task = tasks.get(session_id)
    running = task is not None and not task.done()
    data = engine.session_store.get(session_id)
    if data is None:
        raise HTTPException(404, "Session not found")
    report = data.get("optimization_report_json", {}) or {}
    if isinstance(report, str):
        report = json.loads(report or "{}")
    return {
        "running": running,
        "status": data.get("status", ""),
        "progress": progress.get(session_id, {}),
        "report": report,
    }


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


@router.post("/session/{session_id}/settings")
async def update_settings(session_id: str, req: SettingsRequest, request: Request):
    """更新推演级设置（多动作资源分配），供普通推演与优化器统一读取。"""
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    data = engine.session_store.get(session_id)
    config = (data or {}).get("config_json", {}) or {}
    if isinstance(config, str):
        config = json.loads(config)
    config["enable_multi_action"] = bool(req.enable_multi_action)
    config["max_actions"] = max(1, int(req.max_actions or 3))
    config["weather"] = (req.weather or "").strip()
    config["terrain"] = (req.terrain or "").strip()
    engine.session_store.update(session_id, config_json=json.dumps(config, ensure_ascii=False))
    return {"session_id": session_id, "enable_multi_action": config["enable_multi_action"],
            "max_actions": config["max_actions"], "weather": config["weather"], "terrain": config["terrain"]}


@router.get("/domains")
async def list_domains_route():
    """返回所有已加载规则包的 domain / name 列表（内置 + 自定义）。"""
    from strategy_forge.core.rule_templates import list_domains as get_domains
    return {"domains": get_domains()}


class RulesUpload(BaseModel):
    domain: str
    content: str  # JSON 文本（前端 readAsText 或直接传字符串）


@router.post("/rules/upload")
async def upload_rules(body: RulesUpload, request: Request):
    """上传/覆盖自定义规则包 JSON。保存到 data/rule/custom/{domain}.json 并重载缓存。"""
    dom = (body.domain or "").strip()
    if not dom:
        raise HTTPException(400, "domain 不能为空")
    try:
        rule = json.loads(body.content)
    except json.JSONDecodeError:
        raise HTTPException(400, "content 无效 JSON")
    if "domain" not in rule:
        rule["domain"] = dom
    from strategy_forge.core.config import config
    from strategy_forge.core.rule_templates import reload_rules
    out_dir = Path(config.deduction_data_dir) / "rule" / "custom"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{dom}.json").write_text(json.dumps(rule, indent=2, ensure_ascii=False), encoding="utf-8")
    reload_rules()
    return {"status": "ok", "domain": dom}


# ── Data export ──

@router.get("/session/{session_id}/graph")
async def get_graph_data(session_id: str, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    graph = engine.get_graph(session_id)
    return graph.export_graph_data()


@router.get("/session/{session_id}/timeline")
async def get_timeline(session_id: str, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    graph = engine.get_graph(session_id)
    try:
        return {
            "timelines": graph.get_agent_timelines(),
            "sequence": graph.get_event_sequence(),
        }
    except Exception as e:
        logger.warning("[StrategyForge] timeline query failed: %s", e)
        return {"timelines": [], "sequence": []}


@router.get("/session/{session_id}/causal")
async def get_causal(session_id: str, request: Request):
    engine = _get_engine(request)
    session = engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    graph = engine.get_graph(session_id)
    try:
        sub = graph.get_causal_subgraph()
        sub["summary"] = graph.get_causal_summary()
        return sub
    except Exception as e:
        logger.warning("[StrategyForge] causal query failed: %s", e)
        return {"nodes": [], "links": [], "summary": []}


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


@router.get("/session/{session_id}/tokens")
async def get_tokens(session_id: str, request: Request):
    engine = _get_engine(request)
    from strategy_forge.core.token_counter import accumulator
    live = accumulator.get_session_stats(session_id)
    data = engine.session_store.get(session_id)
    stored: dict = {}
    if data:
        raw = data.get("token_json", {}) or {}
        if isinstance(raw, str):
            raw = json.loads(raw)
        stored = raw
    # Prefer live accumulator data; fall back to stored
    stats = live if (live and live.get("total_tokens", 0) > 0) else stored
    return {"session_id": session_id, "stats": stats}


# ── SSE Stream ──

_RUNNING_STATUSES = frozenset({
    "ontology_running", "graph_running", "agents_running",
    "simulating", "reporting", "optimizing",
})

_TERMINAL_STATUSES = frozenset({"complete", "failed", "paused"})


@router.get("/session/{session_id}/stream")
async def stream_deduction(session_id: str, request: Request):

    async def event_generator():
        engine = _get_engine(request)
        last_log_id = 0
        last_round = 0
        last_status = ""

        session = engine.get_session(session_id)
        if session is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'session not found'}, ensure_ascii=False)}\n\n"
            return
        if session.status.value not in _RUNNING_STATUSES and session.status.value not in _TERMINAL_STATUSES:
            yield f"data: {json.dumps({'type': 'status', 'status': session.status.value}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Push existing logs on connect so the frontend can catch up
        existing = engine.get_logs(session_id, limit=200)
        for log_entry in existing:
            last_log_id = max(last_log_id, log_entry.get("id", 0))
            yield f"data: {json.dumps(log_entry, ensure_ascii=False)}\n\n"

        # Push initial status so frontend syncs immediately
        if session.status.value:
            last_status = session.status.value
            yield f"data: {json.dumps({'type': 'status', 'status': last_status}, ensure_ascii=False)}\n\n"

        ev = engine.get_stream_event(session_id)
        ev.clear()

        while True:
            try:
                await asyncio.wait_for(ev.wait(), timeout=5.0)
            except TimeoutError:
                pass
            ev.clear()

            # ── push new log entries ──
            logs = engine.get_logs(session_id, limit=100)
            new_logs = [l for l in logs if l.get("id", 0) > last_log_id]
            for log_entry in new_logs:
                last_log_id = max(last_log_id, log_entry.get("id", 0))
                yield f"data: {json.dumps(log_entry, ensure_ascii=False)}\n\n"

            # ── push round-complete event (triggers graph/timeline/dashboard refresh) ──
            rd = engine.get_round_data(session_id)
            cr = rd.get("round", 0)
            if cr > last_round:
                last_round = cr
                payload: dict[str, Any] = {"type": "round", "round": cr, "total": rd.get("total", 0)}
                snap = engine.get_round_snapshot(session_id)
                if snap:
                    payload["snapshot"] = snap
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # ── push status changes (every transition, not just terminal) ──
            session = engine.get_session(session_id)
            if session and session.status.value != last_status:
                last_status = session.status.value
                yield f"data: {json.dumps({'type': 'status', 'status': last_status}, ensure_ascii=False)}\n\n"

            # ── check terminal status ──
            if session and session.status.value in _TERMINAL_STATUSES:
                yield "data: [DONE]\n\n"
                engine.cleanup_events(session_id)
                return

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

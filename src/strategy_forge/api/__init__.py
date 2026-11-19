"""StrategyForge API application.

Creates the FastAPI app with CORS middleware and deduction routes.
"""
import os as _os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config_routes import router as config_router
from .routes import router as forge_router

app = FastAPI(
    title="StrategyForge",
    description="专职战略决策推演工具",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forge_router)
app.include_router(config_router)

# 前端静态文件（打包模式自动服务；开发模式 Vite 代理到本机 5173）
_frontend_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), "frontend")
if _os.path.isdir(_frontend_dir):
    app.mount("/assets", StaticFiles(directory=_os.path.join(_frontend_dir, "assets")), name="assets")

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def _spa_fallback(full_path: str = ""):
        index_path = _os.path.join(_frontend_dir, "index.html")
        if _os.path.isfile(index_path):
            return FileResponse(index_path)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("StrategyForge API", status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "StrategyForge"}

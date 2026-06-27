"""StrategyForge API application.

Creates the FastAPI app with CORS middleware and deduction routes.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "StrategyForge"}

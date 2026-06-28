"""StrategyForge config API — delegates to providers.registry."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from strategy_forge.core.providers import _mask_key, registry

router = APIRouter(prefix="/api/forge/config", tags=["config"])


class LLMConfigUpdate(BaseModel):
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    provider_slug: str = ""
    llm_temperature: float = 0.3


class EmbedConfigUpdate(BaseModel):
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_model_name: str = ""
    provider_slug: str = ""


class ModelListRequest(BaseModel):
    base_url: str = ""
    api_key: str = ""


# ── Provider catalog ──
@router.get("/providers")
async def list_providers():
    return {"providers": registry.get_providers()}


# ── LLM config ──
@router.get("/llm")
async def get_llm_config():
    return {
        "provider_slug": registry.llm_provider_slug,
        "llm_base_url": registry.llm_base_url,
        "llm_model": registry.llm_model,
        "llm_api_key": _mask_key(registry.llm_api_key),
        "llm_temperature": registry.llm_temperature,
    }

@router.post("/llm")
async def update_llm(body: LLMConfigUpdate):
    if body.llm_base_url: registry.llm_base_url = body.llm_base_url.rstrip("/")
    if body.llm_api_key and body.llm_api_key != "••••••••": registry.llm_api_key = body.llm_api_key
    if body.llm_model: registry.llm_model = body.llm_model
    if body.provider_slug: registry.llm_provider_slug = body.provider_slug
    registry.llm_temperature = body.llm_temperature
    registry.save()
    return {"status": "ok"}


# ── Embedding config ──
@router.get("/embedding")
async def get_embed_config():
    return {
        "provider_slug": registry.embed_provider_slug,
        "embedding_api_base": registry.embedding_api_base,
        "embedding_model_name": registry.embedding_model_name,
        "embedding_api_key": _mask_key(registry.embedding_api_key),
    }

@router.post("/embedding")
async def update_embed(body: EmbedConfigUpdate):
    if body.embedding_api_base: registry.embedding_api_base = body.embedding_api_base.rstrip("/")
    if body.embedding_api_key and body.embedding_api_key != "••••••••": registry.embedding_api_key = body.embedding_api_key
    if body.embedding_model_name: registry.embedding_model_name = body.embedding_model_name
    if body.provider_slug: registry.embed_provider_slug = body.provider_slug
    registry.save()
    return {"status": "ok"}


# ── Model listing + test ──
def _real_key_or(req_key: str) -> str:
    """脱敏串(含 * / •)、空、或 'local' 时回退使用 registry 中存储的真实 Key，
    避免把脱敏 Key 当真实 Key 塞进 HTTP 头(非 ASCII 字符会导致编码错误)。"""
    k = (req_key or "").strip()
    if not k or k.lower() == "local" or "*" in k or "•" in k:
        return registry.llm_api_key or ""
    return k


@router.post("/list-models")
async def list_models(body: ModelListRequest):
    return await registry.list_models(body.base_url, _real_key_or(body.api_key))


@router.post("/test-connection")
async def test_connection(body: ModelListRequest):
    return await registry.test_connection(body.base_url, _real_key_or(body.api_key))


# ── Reload ──
@router.post("/reload")
async def reload():
    registry.reload()
    return {"status": "ok"}

"""Unified provider registry — single source of truth for LLM and embedding endpoints.

Resolution priority: forge_config.json > FORGE_* env vars > provider catalog > empty
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProviderDef:
    slug: str
    name: str
    api_type: str = "openai"
    default_llm_base_url: str = ""
    default_llm_model: str = ""
    default_embed_base_url: str = ""
    default_embed_model: str = ""
    note: str = ""
    is_local: bool = False


PROVIDER_CATALOG: dict[str, ProviderDef] = {
    "openai": ProviderDef("openai", "OpenAI", "openai",
                          "https://api.openai.com/v1", "gpt-4o", "", "text-embedding-3-small"),
    "anthropic": ProviderDef("anthropic", "Anthropic (Claude)", "anthropic",
                             "https://api.anthropic.com"),
    "dashscope": ProviderDef("dashscope", "阿里云 DashScope", "openai",
                             "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "dashscope-intl": ProviderDef("dashscope-intl", "DashScope (International)", "openai",
                                  "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "kimi-cn": ProviderDef("kimi-cn", "Kimi 月之暗面 (中国区)", "openai",
                           "https://api.moonshot.cn/v1"),
    "kimi-int": ProviderDef("kimi-int", "Kimi (International)", "openai",
                            "https://api.moonshot.ai/v1"),
    "minimax-cn": ProviderDef("minimax-cn", "MiniMax (中国区)", "openai",
                              "https://api.minimaxi.com/v1"),
    "minimax-int": ProviderDef("minimax-int", "MiniMax (International)", "openai",
                               "https://api.minimax.io/v1"),
    "deepseek": ProviderDef("deepseek", "DeepSeek", "openai",
                            "https://api.deepseek.com/v1", "deepseek-chat"),
    "openrouter": ProviderDef("openrouter", "OpenRouter", "openai",
                              "https://openrouter.ai/api/v1"),
    "siliconflow": ProviderDef("siliconflow", "硅基流动 (SiliconFlow·中国)", "openai",
                               "https://api.siliconflow.cn/v1"),
    "siliconflow-intl": ProviderDef("siliconflow-intl", "SiliconFlow (International)", "openai",
                                    "https://api.siliconflow.com/v1"),
    "volcengine": ProviderDef("volcengine", "火山引擎 (豆包)", "openai",
                              "https://ark.cn-beijing.volces.com/api/v3"),
    "zhipu-cn": ProviderDef("zhipu-cn", "智谱AI GLM (中国区)", "openai",
                            "https://open.bigmodel.cn/api/paas/v4"),
    "zhipu-int": ProviderDef("zhipu-int", "Zhipu AI (International)", "openai",
                             "https://api.z.ai/api/paas/v4"),
    "qianfan": ProviderDef("qianfan", "百度千帆", "openai",
                           "https://qianfan.baidubce.com/v2"),
    "hunyuan": ProviderDef("hunyuan", "腾讯混元", "openai",
                           "https://api.hunyuan.cloud.tencent.com/v1"),
    "gemini": ProviderDef("gemini", "Google Gemini", "openai",
                           "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "xai": ProviderDef("xai", "xAI (Grok)", "openai",
                       "https://api.x.ai/v1"),
    "mistral": ProviderDef("mistral", "Mistral AI", "openai",
                           "https://api.mistral.ai/v1"),
    "nvidia-nim": ProviderDef("nvidia-nim", "NVIDIA NIM", "openai",
                              "https://integrate.api.nvidia.com/v1"),
    "groq": ProviderDef("groq", "Groq", "openai",
                        "https://api.groq.com/openai/v1"),
    "together": ProviderDef("together", "Together AI", "openai",
                            "https://api.together.xyz/v1"),
    "fireworks": ProviderDef("fireworks", "Fireworks AI", "openai",
                             "https://api.fireworks.ai/inference/v1"),
    "cohere": ProviderDef("cohere", "Cohere", "openai",
                          "https://api.cohere.ai/compatibility/v1"),
    "yunwu": ProviderDef("yunwu", "云雾 API", "openai",
                         "https://yunwu.ai/v1"),
    "longcat": ProviderDef("longcat", "美团 LongCat", "openai",
                           "https://api.longcat.chat/openai"),
    "iflow": ProviderDef("iflow", "心流 iFlow", "openai",
                         "https://apis.iflow.cn/v1"),
    "xfyun": ProviderDef("xfyun", "讯飞星辰 MaaS", "openai",
                         "https://maas-api.cn-huabei-1.xf-yun.com/v2"),
    "ollama": ProviderDef("ollama", "Ollama (本地)", "openai",
                          "http://localhost:11434/v1", is_local=True, note="本地运行，无需API Key"),
    "lmstudio": ProviderDef("lmstudio", "LM Studio (本地)", "openai",
                            "http://127.0.0.1:1234/v1", "",
                            "", "text-embedding-embeddinggemma-300m-qat",
                            is_local=True, note="本地运行，无需API Key"),
    "custom": ProviderDef("custom", "自定义 OpenAI 兼容", "openai"),
}


class ProviderRegistry:
    def __init__(self, config_dir: Path | None = None) -> None:
        if config_dir is None:
            from strategy_forge.core.config import _get_data_dir
            config_dir = _get_data_dir()
        self._config_file = Path(config_dir) / "forge_config.json"
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._config_file.exists():
            try:
                self._data = json.loads(self._config_file.read_text("utf-8"))
            except Exception as e:
                logger.warning("[providers] Failed to load config: %s", e)
        self._migrate_dead_temperature()

    def _migrate_dead_temperature(self) -> None:
        """一次性迁移：llm_temperature 曾是"死配置"（从未接入生成调用），旧存的 0.3 只是
        表单默认而非有意选择。既然现在它真正接入模拟决策温度，为避免决策温度从旧的
        有效值(0.7)意外跌到 0.3，此处移除旧 0.3 使其回落到新默认(0.6)。仅执行一次
        （_temp_migrated 标记），此后用户显式设置的任何值（含 0.3）都正常保留。
        """
        if self._data.get("_temp_migrated"):
            return
        try:
            legacy = float(self._data.get("llm_temperature", "") or 0.0)
        except (TypeError, ValueError):
            legacy = 0.0
        changed = False
        if abs(legacy - 0.3) < 1e-9:
            self._data.pop("llm_temperature", None)
            changed = True
        self._data["_temp_migrated"] = True
        if self._config_file.exists() or changed:
            try:
                self.save()
            except Exception as e:
                logger.debug("[providers] temp migration save skipped: %s", e)

    def save(self) -> None:
        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        self._config_file.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def reload(self) -> None:
        self._load()

    @property
    def llm_provider_slug(self) -> str:
        return self._data.get("llm_provider", "") or os.getenv("FORGE_PROVIDER", "")

    @llm_provider_slug.setter
    def llm_provider_slug(self, v: str) -> None:
        self._data["llm_provider"] = v

    @property
    def embed_provider_slug(self) -> str:
        return self._data.get("embed_provider", "") or os.getenv("FORGE_EMBED_PROVIDER", "")

    @embed_provider_slug.setter
    def embed_provider_slug(self, v: str) -> None:
        self._data["embed_provider"] = v

    def _resolve(self, prefix: str, provider_slug: str) -> dict[str, str]:
        pd = PROVIDER_CATALOG.get(provider_slug)
        # Explicit None/empty check: allow UI to clear fields by storing ""
        # even when env vars have old values ("" or chain would skip to env).
        base = self._data.get(f"{prefix}_base_url")
        if base is None:
            base = os.getenv(f"FORGE_{prefix.upper()}_BASE", "") or (pd.default_llm_base_url if pd else "")
        else:
            base = base or os.getenv(f"FORGE_{prefix.upper()}_BASE", "") or (pd.default_llm_base_url if pd else "")
        key = self._data.get(f"{prefix}_api_key")
        if key is None:
            key = os.getenv(f"FORGE_{prefix.upper()}_KEY", "")
        else:
            key = key or os.getenv(f"FORGE_{prefix.upper()}_KEY", "")
        model = self._data.get(f"{prefix}_model")
        if prefix == "llm":
            if model is None:
                model = os.getenv("FORGE_LLM_MODEL", "") or (pd.default_llm_model if pd else "")
            else:
                model = model or os.getenv("FORGE_LLM_MODEL", "") or (pd.default_llm_model if pd else "")
        else:
            if model is None:
                model = self._data.get("embedding_model_name", "") or os.getenv("FORGE_EMBED_MODEL", "") or (pd.default_embed_model if pd else "")
            else:
                model = model or self._data.get("embedding_model_name", "") or os.getenv("FORGE_EMBED_MODEL", "") or (pd.default_embed_model if pd else "")
        return {"api_base": base.rstrip("/") if base else "", "api_key": key, "model": model}

    def resolve_for_llm_client(self) -> dict[str, str]:
        return self._resolve("llm", self.llm_provider_slug)

    def resolve_for_embedding(self) -> dict[str, str]:
        r = self._resolve("embed", self.embed_provider_slug)
        r["model_name"] = r.pop("model")
        return r

    @property
    def llm_base_url(self) -> str:
        return self.resolve_for_llm_client()["api_base"]
    @llm_base_url.setter
    def llm_base_url(self, v: str) -> None: self._data["llm_base_url"] = v

    @property
    def llm_api_key(self) -> str:
        return self.resolve_for_llm_client()["api_key"]
    @llm_api_key.setter
    def llm_api_key(self, v: str) -> None: self._data["llm_api_key"] = v

    @property
    def llm_model(self) -> str:
        return self.resolve_for_llm_client()["model"]
    @llm_model.setter
    def llm_model(self, v: str) -> None: self._data["llm_model"] = v

    @property
    def embedding_api_base(self) -> str:
        r = self.resolve_for_embedding(); return r.get("api_base", "")
    @embedding_api_base.setter
    def embedding_api_base(self, v: str) -> None: self._data["embed_base_url"] = v

    @property
    def embedding_api_key(self) -> str:
        r = self.resolve_for_embedding(); return r.get("api_key", "")
    @embedding_api_key.setter
    def embedding_api_key(self, v: str) -> None: self._data["embed_api_key"] = v

    @property
    def embedding_model_name(self) -> str:
        r = self.resolve_for_embedding(); return r.get("model_name", "")
    @embedding_model_name.setter
    def embedding_model_name(self, v: str) -> None: self._data["embedding_model_name"] = v

    @property
    def llm_temperature(self) -> float:
        from .config import config
        return float(self._data.get("llm_temperature", "") or config.deduction_llm_temperature)
    @llm_temperature.setter
    def llm_temperature(self, v: float) -> None: self._data["llm_temperature"] = v

    @staticmethod
    async def list_models(base_url: str, api_key: str) -> dict:
        if not base_url: return {"error": "未配置 API 地址", "models": []}
        headers: dict[str, str] = {}
        if api_key: headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                r = await c.get(f"{base_url.rstrip('/')}/models", headers=headers)
                r.raise_for_status()
                return {"models": sorted(m.get("id","") for m in r.json().get("data",[]) if m.get("id"))}
        except Exception as e:
            return {"error": _friendly(e), "models": []}

    @staticmethod
    async def test_connection(base_url: str, api_key: str) -> dict:
        if not base_url: return {"ok": False, "status": 0, "error": "未配置 API 地址"}
        headers: dict[str, str] = {}
        if api_key: headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{base_url.rstrip('/')}/models", headers=headers)
                return {"ok": r.status_code < 500, "status": r.status_code}
        except Exception as e:
            return {"ok": False, "status": 0, "error": str(e)}

    @staticmethod
    def get_providers() -> list[dict]:
        return [{"slug": p.slug, "name": p.name, "api_type": p.api_type, "default_llm_base_url": p.default_llm_base_url, "default_llm_model": p.default_llm_model, "default_embed_model": p.default_embed_model, "note": p.note, "is_local": p.is_local} for p in PROVIDER_CATALOG.values()]


def _friendly(e: Exception) -> str:
    m = str(e).lower()
    if "connect" in m or "refused" in m: return "无法连接服务商，请检查地址"
    if "401" in m: return "API Key无效"
    if "404" in m: return "不支持模型列表查询，请手动输入"
    if "timeout" in m: return "请求超时"
    return str(e)[:150]

def _mask_key(k: str) -> str:
    return k[:4] + "*"*(len(k)-8) + k[-4:] if k and len(k)>=8 else k


registry = ProviderRegistry()

"""StrategyForge functional test — full pipeline against LM Studio.

Tests: config API, provider catalog, model listing, connection test,
       session creation, deduction pipeline, graph export, report, logs.

Usage:  pytest tests/functional/test_strategyforge_e2e.py -v -s

Requires:
  - LM Studio running at http://127.0.0.1:1234
  - StrategyForge backend running: python run.py (port 8000)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import requests

API_BASE = "http://127.0.0.1:8000/api/forge"
LMSTUDIO_BASE = "http://127.0.0.1:1234/v1"
CHAT_MODEL = os.environ.get("TEST_CHAT_MODEL", "qwen/qwen3.5-9b")
EMBED_MODEL = os.environ.get("TEST_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")

TEST_SOURCE = """
2026年6月，中国科技公司星辰科技（StarTech）宣布成功研发新一代量子计算芯片"天枢"。
公司CEO李明在发布会上表示，该芯片将大幅提升AI训练速度，预计2027年量产。
竞争对手华光半导体（HuaGuang）同日宣布获得政府50亿元补贴，加速光子芯片研发。
行业分析师张三认为，量子计算与光子计算的路线之争将进入白热化阶段。
欧洲监管机构EUTech表示将对中国芯片企业开展反垄断调查。
星辰科技CTO王五回应称，公司始终遵守国际规则，欢迎公平竞争。
下游企业智能云（SmartCloud）宣布将率先采购天枢芯片建设AI训练中心。
"""

_sid: str = ""


@pytest.fixture(scope="session")
def setup():
    """Check LM Studio + backend, configure providers."""
    # Check LM Studio
    try:
        r = requests.get(f"{LMSTUDIO_BASE}/models", timeout=5)
        models = {m["id"] for m in r.json()["data"]}
        assert CHAT_MODEL in models, f"Chat model {CHAT_MODEL} not loaded: {sorted(models)}"
        assert EMBED_MODEL in models, f"Embed model {EMBED_MODEL} not loaded"
        print(f"\n  LM Studio OK: chat={CHAT_MODEL}, embed={EMBED_MODEL}")
    except requests.RequestException:
        pytest.skip("LM Studio not running at 127.0.0.1:1234")

    # Check backend
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        assert r.status_code == 200
        print(f"  Backend OK: {API_BASE}")
    except requests.RequestException:
        pytest.skip("Backend not running at port 8000. Start: python run.py")

    # Configure
    print("\n  Configuring...")
    requests.post(f"{API_BASE}/config/llm", json={
        "llm_base_url": LMSTUDIO_BASE, "llm_api_key": "lm-studio",
        "llm_model": CHAT_MODEL, "provider_slug": "lmstudio", "llm_temperature": 0.3,
    })
    requests.post(f"{API_BASE}/config/embedding", json={
        "embedding_api_base": LMSTUDIO_BASE, "embedding_api_key": "lm-studio",
        "embedding_model_name": EMBED_MODEL, "provider_slug": "lmstudio",
    })
    requests.post(f"{API_BASE}/config/reload")
    yield
    # Cleanup
    global _sid
    if _sid:
        requests.delete(f"{API_BASE}/session/{_sid}")
        _sid = ""


@pytest.mark.usefixtures("setup")
class TestPipeline:

    def test_01_config_readback(self):
        r = requests.get(f"{API_BASE}/config/llm")
        d = r.json()
        assert d["llm_base_url"] == LMSTUDIO_BASE
        print(f"    LLM config OK: {d['llm_base_url']} model={d['llm_model']}")

    def test_02_providers(self):
        r = requests.get(f"{API_BASE}/config/providers")
        providers = r.json()["providers"]
        assert len(providers) >= 30
        slugs = {p["slug"] for p in providers}
        for s in ("openai", "deepseek", "lmstudio", "dashscope", "zhipu-cn"):
            assert s in slugs, f"Missing: {s}"
        print(f"    {len(providers)} providers listed")

    def test_03_list_models(self):
        r = requests.post(f"{API_BASE}/config/list-models", json={
            "base_url": LMSTUDIO_BASE, "api_key": "lm-studio",
        })
        d = r.json()
        assert not d.get("error"), f"Error: {d.get('error')}"
        assert len(d["models"]) >= 1
        assert CHAT_MODEL in d["models"]
        print(f"    {len(d['models'])} models found, includes {CHAT_MODEL}")

    def test_04_test_connection(self):
        r = requests.post(f"{API_BASE}/config/test-connection", json={
            "base_url": LMSTUDIO_BASE, "api_key": "lm-studio",
        })
        d = r.json()
        assert d["ok"], f"Connection test failed: {d}"
        print(f"    Connection OK (HTTP {d['status']})")

    def test_05_embedding(self):
        r = requests.post(f"{LMSTUDIO_BASE}/embeddings", json={
            "model": EMBED_MODEL, "input": "测试嵌入",
        }, timeout=30)
        vec = r.json()["data"][0]["embedding"]
        assert len(vec) >= 64
        print(f"    Embed dim: {len(vec)}")

    def test_06_create_session(self):
        global _sid
        r = requests.post(f"{API_BASE}/session", json={
            "title": "E2E 推演测试", "source_material": TEST_SOURCE,
        })
        d = r.json()
        _sid = d["id"]
        assert d["status"] == "created"
        print(f"    Session: {_sid}")

    def test_07_start_deduction(self):
        global _sid
        assert _sid, "No session — run test_06 first"
        print(f"    Starting deduction {_sid}...")
        r = requests.post(f"{API_BASE}/session/{_sid}/start", timeout=600)

        for i in range(300):
            r = requests.get(f"{API_BASE}/session/{_sid}")
            s = r.json()
            if s["status"] in ("complete", "failed"):
                break
            if i % 20 == 0:
                print(f"      round={s.get('current_round',0)}/{s.get('total_rounds',0)} status={s['status']}")
            time.sleep(2)

        s = requests.get(f"{API_BASE}/session/{_sid}").json()
        status = s["status"]
        print(f"    Done: {s['entity_count']} entities, {s['agent_count']} agents, {s['current_round']} rounds, status={status}")
        assert status in ("complete", "failed"), f"Still running after 10min"
        # Don't assert complete — report/graph tests will verify data

    def test_08_graph(self):
        global _sid
        assert _sid, "No session"
        r = requests.get(f"{API_BASE}/session/{_sid}/graph")
        d = r.json()
        assert len(d["nodes"]) >= 2
        print(f"    Graph: {len(d['nodes'])} nodes, {len(d['links'])} links")

    def test_09_report(self):
        global _sid
        assert _sid, "No session"
        r = requests.get(f"{API_BASE}/session/{_sid}/report")
        d = r.json()
        report = d.get("report", {}) or {}
        summary = report.get("summary", "") if isinstance(report, dict) else ""
        assert summary, "Empty report summary"
        print(f"    Report: {summary[:100]}...")

    def test_10_logs(self):
        global _sid
        assert _sid, "No session"
        r = requests.get(f"{API_BASE}/session/{_sid}/logs")
        logs = r.json()
        assert len(logs) >= 5
        phases = {l["phase"] for l in logs}
        for p in ("ontology", "graph", "agents", "simulation", "report"):
            assert p in phases, f"Missing phase: {p}"
        print(f"    {len(logs)} log entries, phases: {sorted(phases)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""P0 缺陷回归测试：_smart_sample、_batch_apply_deltas 兜底、agent_factory 无注册中心兜底。"""
import asyncio
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_EMBED_PROVIDER", "lmstudio")


# ── P0-1 ──

def test_smart_sample_long_source_does_not_crash():
    from strategy_forge.engine.entity_registry import _smart_sample
    src = "这是测试材料。" * 20000  # > 8000 chars
    out = _smart_sample(src, 8000)
    assert isinstance(out, str)
    assert 0 < len(out) < len(src)


def test_smart_sample_short_source_returns_unchanged():
    from strategy_forge.engine.entity_registry import _smart_sample
    assert _smart_sample("短文本", 8000) == "短文本"


# ── P0-2 ──

def test_batch_apply_deltas_py_clamps():
    from strategy_forge.engine._jit_utils import _batch_apply_deltas_py
    metrics = np.array([[10.0, 50.0], [90.0, 5.0]])
    deltas = np.array([[5.0, -10.0], [30.0, -20.0]])
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 100.0])
    _batch_apply_deltas_py(metrics, deltas, lo, hi)
    assert np.allclose(metrics, [[15.0, 40.0], [100.0, 0.0]])


def test_batch_apply_deltas_fallback_dispatch_no_numba():
    import strategy_forge.engine._jit_utils as ju
    metrics = np.array([[10.0], [90.0]])
    deltas = np.array([[5.0], [30.0]])
    lo = np.array([0.0])
    hi = np.array([100.0])
    old = ju._HAS_NUMBA
    ju._HAS_NUMBA = False
    try:
        ju.batch_apply_deltas(metrics, deltas, lo, hi)
    finally:
        ju._HAS_NUMBA = old
    assert np.allclose(metrics, [[15.0], [100.0]])


# ── P0-3 ──

class _GraphStub:
    def upsert_agent_node(self, *args, **kwargs):
        pass


def _fake_chat(messages, system, temperature):
    return '{"persona": "测试人格", "background": "背景", "goals": ["目标1"]}'


def test_agent_factory_without_registry_fallback():
    from strategy_forge.engine.agent_factory import create_agents_from_graph

    async def _run():
        return await create_agents_from_graph(
            graph=_GraphStub(),
            source_material="测试材料",
            log_fn=lambda p, m: None,
            intel_list=[
                {"name": "A国", "type": "国家", "id": "a", "freq": 3, "aliases": ["甲国"]},
                {"name": "B国", "type": "国家", "id": "b", "freq": 2},
            ],
            domain="geo_strategy",
            entity_registry=None,
            chat_fn=_fake_chat,
        )

    agents = asyncio.run(_run())
    assert {a.name for a in agents} == {"A国", "B国"}

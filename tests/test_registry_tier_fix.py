"""P0-1 / P1-1 回归测试：Layer3 降级回写 tier 计数 + 准则 Jaccard 去重。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.entity_registry import EntityRegistry, RegisteredEntity
from strategy_forge.engine.simulator import SimulationEngine


def test_mark_discard_updates_tier_counts():
    reg = EntityRegistry()
    e1 = RegisteredEntity(name="A", type="国家", freq=10, decision="KEEP", tier=1)
    e2 = RegisteredEntity(name="B", type="人物", freq=5, decision="KEEP", tier=2)
    reg.entities["A"] = e1
    reg.entities["B"] = e2
    reg.kept = 2
    reg.tier1_count = 1
    reg.tier2_count = 1

    assert reg._mark_discard(e1, "L3降级") is True
    assert e1.decision == "DISCARD" and e1.tier == 3
    assert reg.tier1_count == 0
    assert reg.kept == 1 and reg.discarded == 1
    # get_tier1 不再返回被降级实体
    assert reg.get_tier1() == []
    assert [e.name for e in reg.get_tier2()] == ["B"]


def test_mark_discard_idempotent():
    reg = EntityRegistry()
    e1 = RegisteredEntity(name="A", type="国家", freq=10, decision="KEEP", tier=1)
    reg.entities["A"] = e1
    reg.kept = 1
    reg.tier1_count = 1
    assert reg._mark_discard(e1, "第一次") is True
    assert reg._mark_discard(e1, "第二次") is False  # 已 DISCARD，不重复计数
    assert reg.kept == 0 and reg.discarded == 1 and reg.tier1_count == 0


def test_char_jaccard_dedup():
    eng = object.__new__(SimulationEngine)
    assert eng._char_jaccard("收缩战线保核心区", "收缩战线保核心区域") > 0.8
    assert eng._char_jaccard("收缩战线保核心区", "主动进攻扩大战果") < 0.3

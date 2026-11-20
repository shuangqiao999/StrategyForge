"""反思机制重构回归测试：规则注册表 + persona 进化读取。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.models import DeductionAgentProfile
from strategy_forge.engine.strategic_reasoner import StrategicReasoner


def test_agent_has_rule_registry_default():
    a = DeductionAgentProfile(entity_id="e1", name="A", persona="P")
    assert a._rule_registry == []


def test_persona_evolution_reads_rule_registry_by_charge():
    a = DeductionAgentProfile(entity_id="e1", name="A", persona="核心人格")
    a._rule_registry = [
        {"text": "被背叛后复仇心切", "source": "emotional", "charge": 0.9,
         "round_added": 1, "effectiveness": 0, "target": "", "mode": "narrative"},
        {"text": "资源优先自保", "source": "strategic", "charge": 0.0,
         "round_added": 1, "effectiveness": 0, "target": "", "mode": "quantified"},
    ]
    out = StrategicReasoner._persona_with_evolution(a)
    assert "核心人格" in out
    assert "【当前强烈倾向】 被背叛后复仇心切" in out
    assert "【长期原则】 资源优先自保" in out


def test_persona_evolution_falls_back_to_shadow():
    a = DeductionAgentProfile(entity_id="e1", name="A", persona="核心人格")
    a.system_prompt_extra = "旧准则一；旧准则二"
    out = StrategicReasoner._persona_with_evolution(a)
    assert "旧准则一" in out
    assert "行为准则" in out

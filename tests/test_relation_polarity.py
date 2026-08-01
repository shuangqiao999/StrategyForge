"""Unit tests for relation polarity (Layer A/B/C) — no LLM connectivity required.

Covers:
  - relation_polarity.infer_polarity / merge_polarity_map / normalize_polarity
  - ontology relation polarity parsing fallback
  - simulator._classify_relation three-layer priority (A > C > B)
"""
import os
import sys

sys.path.insert(0, "src")


class TestInferPolarity:
    def test_foe_business_terms(self):
        from strategy_forge.engine.relation_polarity import infer_polarity

        for rel in ("市场竞争", "抢占份额", "争夺流量", "技术对抗"):
            assert infer_polarity(rel) == "foe", rel

    def test_ally_business_terms(self):
        from strategy_forge.engine.relation_polarity import infer_polarity

        for rel in ("战略合作", "联合研发", "投资", "供应链合作"):
            assert infer_polarity(rel) == "ally", rel

    def test_neutral_terms(self):
        from strategy_forge.engine.relation_polarity import infer_polarity

        for rel in ("技术适配", "营收贡献", "位于", "参与", "观察"):
            assert infer_polarity(rel) == "neutral", rel

    def test_semantic_terms_need_layer_ac(self):
        # 「占据市场份额」等语义性竞争词不含显式 foe 关键字，
        # 靠 Layer A（结构化映射）或 Layer C（LLM 精判）覆盖，关键字兜底归 neutral
        from strategy_forge.engine.relation_polarity import infer_polarity

        assert infer_polarity("占据市场份额") == "neutral"

    def test_english(self):
        from strategy_forge.engine.relation_polarity import infer_polarity

        assert infer_polarity("compete") == "foe"
        assert infer_polarity("partnership") == "ally"
        assert infer_polarity("report") == "neutral"


class TestMergePolarityMap:
    def test_override_order(self):
        from strategy_forge.engine.relation_polarity import merge_polarity_map

        m = merge_polarity_map({"占据市场份额": "foe"}, {"占据市场份额": "ally"})
        assert m["占据市场份额"] == "ally"  # 后者覆盖

    def test_ignores_invalid(self):
        from strategy_forge.engine.relation_polarity import merge_polarity_map

        m = merge_polarity_map({"a": "foe", "b": "banana", "c": "neutral"})
        assert m["a"] == "foe"
        assert "b" not in m
        assert "c" not in m  # neutral 不进入映射


class TestOntologyPolarity:
    def test_parse_with_polarity(self):
        from strategy_forge.engine.ontology import _parse_ontology

        raw = ('{"entities": [{"name": "企业", "description": "d"}], '
               '"relations": [{"name": "竞争", "description": "x", "polarity": "foe"}, '
               '{"name": "合作", "description": "y"}]}')
        onto = _parse_ontology(raw)
        by_name = {r.name: r.polarity for r in onto.relations}
        assert by_name["竞争"] == "foe"
        # 缺失 polarity → 静态兜底
        assert by_name["合作"] == "ally"

    def test_default_ontology_polarity(self):
        from strategy_forge.engine.ontology import _default_ontology

        onto = _default_ontology()
        by_name = {r.name: r.polarity for r in onto.relations}
        assert by_name["opposes"] == "foe"
        assert by_name["supports"] == "ally"
        assert by_name["works_for"] == "neutral"


class TestSimulatorClassify:
    def _make_engine(self, polarity_map=None):
        from strategy_forge.engine.simulator import SimulationEngine

        class FakeGraph:
            def get_entity_neighbors(self, *a, **k):
                return {"neighbors": []}

        class FakeReasoner:
            def seed_trust(self, *a, **k):
                pass

        eng = object.__new__(SimulationEngine)
        eng._relation_polarity = polarity_map or {}
        eng._relation_llm_overrides = {}
        eng._relation_llm_overrides = {}
        return eng

    def test_layer_a_priority(self):
        eng = self._make_engine({"占据市场份额": "foe"})
        assert eng._classify_relation("占据市场份额") == "foe"

    def test_layer_b_fallback(self):
        eng = self._make_engine({})
        assert eng._classify_relation("技术对抗") == "foe"
        assert eng._classify_relation("营收贡献") == "neutral"

    def test_layer_c_priority_over_b(self):
        eng = self._make_engine({})
        # Layer C 精判缓存：某关系被覆盖为 ally，即使关键字会判 foe
        eng._relation_llm_overrides["竞争"] = "ally"
        assert eng._classify_relation("竞争") == "ally"


class TestEventVisibility:
    def test_private_only_participants(self):
        from strategy_forge.engine.simulator import _is_event_visible_to

        priv = {"visibility": "private", "participants": "A|B", "agent": "A"}
        assert _is_event_visible_to("A", "AgentA", priv) is True
        assert _is_event_visible_to("B", "AgentB", priv) is True
        assert _is_event_visible_to("C", "AgentC", priv) is False

    def test_private_actor_visible(self):
        from strategy_forge.engine.simulator import _is_event_visible_to

        priv = {"visibility": "private", "participants": "", "agent": "X"}
        assert _is_event_visible_to("X", "AgentX", priv) is True
        assert _is_event_visible_to("Y", "AgentY", priv) is False

    def test_public_visible_to_all(self):
        from strategy_forge.engine.simulator import _is_event_visible_to

        pub = {"visibility": "public", "participants": "", "agent": "A"}
        assert _is_event_visible_to("C", "AgentC", pub) is True


class TestStateSnapshotSig:
    def test_stable_unchanged(self):
        from strategy_forge.engine.simulator import _state_snapshot_sig

        class St:
            def __init__(self, m, h):
                self.metrics = m
                self.history = h

        s1 = {"e1": St({"a": 1.0, "b": 2.0}, [])}
        s2 = {"e1": St({"a": 1.0, "b": 2.0}, [])}
        assert _state_snapshot_sig(s1, ["e1"]) == _state_snapshot_sig(s2, ["e1"])

    def test_changes_on_metric(self):
        from strategy_forge.engine.simulator import _state_snapshot_sig

        class St:
            def __init__(self, m, h):
                self.metrics = m
                self.history = h

        s1 = {"e1": St({"a": 1.0, "b": 2.0}, [])}
        s3 = {"e1": St({"a": 9.0, "b": 2.0}, [])}
        assert _state_snapshot_sig(s1, ["e1"]) != _state_snapshot_sig(s3, ["e1"])


class TestUnknownHeuristic:
    """C: 语义中介层 Unknown 启发式兜底（领域无关，覆盖静态表外的动态类型）。"""

    def _map(self, t):
        from strategy_forge.engine.semantic_mediator import map_to_base_type
        from strategy_forge.engine.domain_adapter import get_adapter

        return map_to_base_type(t, get_adapter("narrative"))

    def test_agent_terms(self):
        for t in ("车企", "平台", "品牌", "银行", "芯片厂商", "独角兽"):
            assert self._map(t) == "Agent", t

    def test_subordinate_terms(self):
        for t in ("人物", "官员", "创始人", "总裁"):
            assert self._map(t) == "Subordinate", t

    def test_resource_terms(self):
        for t in ("芯片", "产品型号", "武器", "基础设施", "战略资源"):
            assert self._map(t) == "Resource", t

    def test_concept_terms(self):
        for t in ("市场份额", "经济指标", "政策", "数据指标"):
            assert self._map(t) == "Concept", t

    def test_geography_terms(self):
        for t in ("地理区域", "城市", "海域"):
            assert self._map(t) == "Geography", t

    def test_known_types_unchanged(self):
        # 静态表命中优先于启发式
        assert self._map("国家") == "Agent"
        assert self._map("企业") == "Agent"

    def test_truly_unknown(self):
        assert self._map("未知类型XYZ") == "Unknown"


if __name__ == "__main__":
    import traceback

    failed = 0
    for cls in (TestInferPolarity, TestMergePolarityMap, TestOntologyPolarity,
                TestSimulatorClassify, TestEventVisibility, TestStateSnapshotSig,
                TestUnknownHeuristic):
        for name in dir(cls):
            if not name.startswith("test_"):
                continue
            try:
                getattr(cls(), name)()
                print(f"PASS {cls.__name__}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {cls.__name__}.{name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)

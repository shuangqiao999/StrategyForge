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


if __name__ == "__main__":
    import traceback

    failed = 0
    for cls in (TestInferPolarity, TestMergePolarityMap, TestOntologyPolarity, TestSimulatorClassify):
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

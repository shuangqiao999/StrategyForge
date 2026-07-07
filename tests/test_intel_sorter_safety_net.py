"""Unit tests for intel_sorter safety net functions.

Tests _any_name_matches and _apply_safety_net without requiring LLM connectivity.
"""
import os
import sys

sys.path.insert(0, "src")

from strategy_forge.engine.intel_sorter import _any_name_matches, _apply_safety_net


class TestAnyNameMatches:
    def test_substring_chinese_dept(self):
        e = {"name": "国防部", "aliases": []}
        from strategy_forge.engine.intel_sorter import _DEPT_WORDS

        assert _any_name_matches(e, _DEPT_WORDS, mode="substring")

    def test_substring_english_dept(self):
        e = {"name": "Department of Defense", "aliases": []}
        from strategy_forge.engine.intel_sorter import _DEPT_WORDS_EN

        assert _any_name_matches(e, _DEPT_WORDS_EN, mode="substring")

    def test_substring_via_alias(self):
        e = {"name": "Pentagon", "aliases": ["U.S. Department of Defense"]}
        from strategy_forge.engine.intel_sorter import _DEPT_WORDS_EN

        assert _any_name_matches(e, _DEPT_WORDS_EN, mode="substring")

    def test_substring_case_insensitive(self):
        e = {"name": "DOD", "aliases": []}
        from strategy_forge.engine.intel_sorter import _DEPT_WORDS_EN

        assert _any_name_matches(e, _DEPT_WORDS_EN, mode="substring")

    def test_substring_no_match(self):
        e = {"name": "特斯拉", "aliases": ["Tesla"]}
        from strategy_forge.engine.intel_sorter import _DEPT_WORDS

        assert not _any_name_matches(e, _DEPT_WORDS, mode="substring")

    def test_suffix_chinese_title(self):
        e = {"name": "秘书长", "aliases": []}
        from strategy_forge.engine.intel_sorter import _TITLE_SUFFIX

        assert _any_name_matches(e, _TITLE_SUFFIX, mode="suffix")

    def test_suffix_english_title(self):
        e = {"name": "Secretary-General", "aliases": []}
        from strategy_forge.engine.intel_sorter import _TITLE_SUFFIX_EN

        assert _any_name_matches(e, _TITLE_SUFFIX_EN, mode="suffix")

    def test_suffix_chinese_unit(self):
        e = {"name": "第七舰队", "aliases": []}
        from strategy_forge.engine.intel_sorter import _UNIT_SUFFIX

        assert _any_name_matches(e, _UNIT_SUFFIX, mode="suffix")

    def test_suffix_english_unit(self):
        e = {"name": "7th Fleet", "aliases": []}
        from strategy_forge.engine.intel_sorter import _UNIT_SUFFIX_EN

        assert _any_name_matches(e, _UNIT_SUFFIX_EN, mode="suffix")

    def test_exact_chinese_dyad(self):
        e = {"name": "俄乌", "aliases": []}
        from strategy_forge.engine.intel_sorter import _DYAD_WORDS

        assert _any_name_matches(e, _DYAD_WORDS, mode="exact")

    def test_exact_english_dyad(self):
        e = {"name": "us-china", "aliases": []}
        from strategy_forge.engine.intel_sorter import _DYAD_WORDS_EN

        assert _any_name_matches(e, _DYAD_WORDS_EN, mode="exact")

    def test_exact_via_alias(self):
        e = {"name": "US-China Relations", "aliases": ["us-china"]}
        from strategy_forge.engine.intel_sorter import _DYAD_WORDS_EN

        assert _any_name_matches(e, _DYAD_WORDS_EN, mode="exact")


class TestApplySafetyNet:
    def test_chinese_dept_demoted(self):
        e = {"name": "国防部", "aliases": [], "include_in_simulation": True, "role": ""}
        demoted = _apply_safety_net([e])
        assert not e["include_in_simulation"]
        assert "安全网降级" in e["role"]
        assert demoted == 1

    def test_english_dept_demoted(self):
        e = {"name": "DoD", "aliases": [], "include_in_simulation": True, "role": ""}
        demoted = _apply_safety_net([e])
        assert not e["include_in_simulation"]
        assert demoted == 1

    def test_english_title_demoted(self):
        e = {"name": "Prime Minister", "aliases": [], "include_in_simulation": True, "role": ""}
        demoted = _apply_safety_net([e])
        assert not e["include_in_simulation"]
        assert demoted == 1

    def test_english_unit_demoted(self):
        e = {"name": "U.S. Pacific Fleet", "aliases": [], "include_in_simulation": True, "role": ""}
        demoted = _apply_safety_net([e])
        assert not e["include_in_simulation"]
        assert demoted == 1

    def test_english_dyad_demoted(self):
        e = {"name": "russia-ukraine", "aliases": [], "include_in_simulation": True, "role": ""}
        demoted = _apply_safety_net([e])
        assert not e["include_in_simulation"]
        assert demoted == 1

    def test_alias_demoted(self):
        e = {
            "name": "U.S. DoD",
            "aliases": ["国防部", "Department of Defense"],
            "include_in_simulation": True,
            "role": "",
        }
        demoted = _apply_safety_net([e])
        assert not e["include_in_simulation"]
        assert demoted == 1

    def test_normal_entity_not_demoted(self):
        e = {"name": "特斯拉", "aliases": ["Tesla"], "include_in_simulation": True, "role": "核心博弈者"}
        demoted = _apply_safety_net([e])
        assert e["include_in_simulation"]
        assert demoted == 0

    def test_safety_net_disabled(self):
        os.environ["FORGE_INTEL_SAFETY_NET"] = "0"
        try:
            e = {"name": "国防部", "aliases": [], "include_in_simulation": True, "role": ""}
            demoted = _apply_safety_net([e])
            assert e["include_in_simulation"]
            assert demoted == 0
        finally:
            del os.environ["FORGE_INTEL_SAFETY_NET"]

    def test_multiple_entities(self):
        result = [
            {"name": "美国", "aliases": [], "include_in_simulation": True, "role": ""},
            {"name": "国防部", "aliases": ["DoD"], "include_in_simulation": True, "role": ""},
            {"name": "Secretary-General", "aliases": [], "include_in_simulation": True, "role": ""},
            {"name": "第七舰队", "aliases": [], "include_in_simulation": True, "role": ""},
            {"name": "特斯拉", "aliases": ["Tesla"], "include_in_simulation": True, "role": ""},
        ]
        demoted = _apply_safety_net(result)
        assert result[0]["include_in_simulation"]
        assert not result[1]["include_in_simulation"]
        assert not result[2]["include_in_simulation"]
        assert not result[3]["include_in_simulation"]
        assert result[4]["include_in_simulation"]
        assert demoted == 3

    def test_case_insensitive(self):
        e = {"name": "SECRETARY", "aliases": [], "include_in_simulation": True, "role": ""}
        demoted = _apply_safety_net([e])
        assert not e["include_in_simulation"]
        assert demoted == 1

    def test_empty_name_skipped(self):
        e = {"name": "", "aliases": [], "include_in_simulation": True, "role": ""}
        demoted = _apply_safety_net([e])
        assert e["include_in_simulation"]
        assert demoted == 0

    def test_already_demoted_not_counted_twice(self):
        e = {"name": "国防部", "aliases": [], "include_in_simulation": False, "role": ""}
        demoted = _apply_safety_net([e])
        assert not e["include_in_simulation"]
        assert demoted == 0

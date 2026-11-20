"""召回片段定位回归测试：按实体名截取附近片段，而非粗暴取块头。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.preprocessor import DeductionPreprocessor


def _snip(content, name, width=600):
    inst = object.__new__(DeductionPreprocessor)
    return inst._snippet_around(content, name, width)


def test_snippet_centers_entity_name():
    content = "A" * 500 + "实体X" + "B" * 500
    s = _snip(content, "实体X", 600)
    assert "实体X" in s
    assert s.index("实体X") <= 600 // 3 + 3, "实体名应位于片段前 1/3 处"


def test_snippet_not_found_falls_back_to_head():
    content = "C" * 1000
    s = _snip(content, "不存在", 600)
    assert s == content[:600]

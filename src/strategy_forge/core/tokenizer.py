"""
共享中文分词工具模块

提供统一的 jieba 分词入口，懒加载 + 回退策略。
项目中所有需要中文分词的地方应统一调用此模块，避免分散实现。
推演引擎扩展: posseg 词性标注、自定义古文词典、实体提取、关键词标签压缩。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

_jieba_mod: Any = None
_jieba_checked = False
_dict_loaded = False

_logger = logging.getLogger(__name__)

# 古文自定义词典路径 (红楼梦/三国等人物名)，可手动编辑
_CLASSIC_DICT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "custom_dict" / "classic_names.txt"
)


def _ensure_jieba() -> Any:
    global _jieba_mod, _jieba_checked  # noqa: PLW0603
    if not _jieba_checked:
        try:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            _jieba_mod = jieba
        except ImportError:
            _logger.debug("jieba not installed, falling back to regex tokenizer")
        _jieba_checked = True
    return _jieba_mod


def _ensure_classic_dict() -> None:
    """加载古文专用自定义词典，防止 袭人→袭/人 等误切。"""
    global _dict_loaded  # noqa: PLW0603
    if _dict_loaded:
        return
    _dict_loaded = True
    jieba = _ensure_jieba()
    if jieba is None:
        return
    if not _CLASSIC_DICT_PATH.exists():
        _CLASSIC_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CLASSIC_DICT_PATH.write_text(
            "# 古文专用词典 — 每行一个词 (词频 词性)\n"
            "# 格式: 词 频次 词性 (频次越高越优先)\n"
            "# 若不需要词性，只写词即可\n",
            encoding="utf-8",
        )
        _logger.debug("Classic dict placeholder created at %s", _CLASSIC_DICT_PATH)
        return
    try:
        jieba.load_userdict(str(_CLASSIC_DICT_PATH))
        _logger.debug("Classic dictionary loaded: %s", _CLASSIC_DICT_PATH)
    except Exception as e:
        _logger.debug("Failed to load classic dict: %s", e)


def tokenize_words(text: str) -> set[str]:
    """将文本分词为词集合 (用于去重/匹配/Jaccard/关键词提取)。

    使用 jieba cut_for_search 分词，回退到英文单词 + CJK 双字提取。
    过滤长度 < 2 的 token。
    """
    if not text or not text.strip():
        return set()
    lowered = text.lower()
    _jb = _ensure_jieba()
    if _jb is not None:
        return {w for w in _jb.cut_for_search(lowered) if len(w) >= 2}
    en = set(re.findall(r"[a-zA-Z]\w+", lowered))
    cjk = set(re.findall(r"[\u4e00-\u9fff]{2,}", lowered))
    return en | cjk


def segment_text(text: str) -> str:
    """将文本分词为空格连接的字符串 (用于 FTS5 索引/SimHash)。

    使用 jieba cut_for_search 分词，回退到原文。
    """
    if not text:
        return ""
    _jb = _ensure_jieba()
    if _jb is not None:
        return " ".join(_jb.cut_for_search(text))
    return text


def extract_keywords(text: str, top_k: int = 5) -> list[str]:
    """从文本中提取关键词，按长度降序返回。

    使用 jieba 分词后取最长的 top_k 个词。
    """
    if not text or not text.strip():
        return []
    words = tokenize_words(text)
    ranked = sorted(words, key=len, reverse=True)
    return ranked[:top_k]


# ═══════════════════════════════════════════════════════════════════
# 推演引擎扩展 — posseg 实体提取 + 关键词标签压缩
# ═══════════════════════════════════════════════════════════════════

def extract_named_entities(
    text: str, top_k: int = 100, min_freq: int = 2,
) -> dict[str, set[str]]:
    """从文本中提取高频专有名词 (人名/地名/机构名) 及其别名组。

    使用 jieba.posseg 词性标注过滤 nr/ns/nz 词性。
    降级: posseg 不可用时用 cut + 2-4字重复模式做粗糙 NER。
    先加载自定义古文词典以保证 袭人→袭人 (非 袭/人)。

    Returns:
        {标准名: {别名1, 别名2, ...}}，标准名取出现频次最高的词。
    """
    if not text or not text.strip():
        return {}

    _ensure_classic_dict()
    jieba = _ensure_jieba()
    if jieba is None:
        return _fallback_entity_extract(text, top_k, min_freq)

    try:
        import jieba.posseg as pseg
        words = pseg.cut(text)
        tagged = [(w.word.strip(), w.flag) for w in words
                  if w.flag in ("nr", "ns", "nz", "nt", "ng") and len(w.word.strip()) >= 2]
    except Exception:
        _logger.debug("jieba.posseg unavailable, falling back to regex NER")
        return _fallback_entity_extract(text, top_k, min_freq)

    if not tagged:
        return {}

    # 频率统计
    freq: Counter[str] = Counter(w for w, _ in tagged)
    # 过滤低频
    qualified = {w for w, c in freq.items() if c >= min_freq}
    if not qualified:
        qualified = {w for w, _ in freq.most_common(min(top_k, len(freq)))}

    # 构建 标准名 → 别名映射
    entity_map: dict[str, set[str]] = {}
    sorted_entities = sorted(qualified, key=lambda w: freq[w], reverse=True)

    for entity in sorted_entities[:top_k]:
        # 寻找同现别名: 2-3 字, 与该实体有公共字符, 相同 POS 标签
        aliases: set[str] = {entity}
        entity_chars = set(entity)
        for other, flag in tagged:
            if other != entity and len(other) >= 2:
                other_chars = set(other)
                shared = entity_chars & other_chars
                if shared and len(shared) >= 1 and len(entity) >= 3:
                    if len(other) == 2 and entity.endswith(other):
                        aliases.add(other)
                    elif len(other) == 3 and len(entity) == 3:
                        if shared == entity_chars or shared == other_chars:
                            aliases.add(other)
        entity_map[entity] = aliases

    return entity_map


def _fallback_entity_extract(text: str, top_k: int, min_freq: int) -> dict[str, set[str]]:
    """posseg 不可用时的回退 NER: 正则匹配 2-4 字连续汉字 + 频率过滤。"""
    jieba = _ensure_jieba()
    if jieba is not None:
        words = list(jieba.cut(text))
    else:
        words = []
    # 过滤: 2-4 字纯汉字
    pattern = re.compile(r"^[\u4e00-\u9fff]{2,4}$")
    candidates = [w.strip() for w in words if pattern.match(w.strip())]
    if not candidates:
        return {}
    freq = Counter(candidates)
    qualified = {w for w, c in freq.items() if c >= min_freq}
    if len(qualified) < 5:
        qualified = {w for w, _ in freq.most_common(min(top_k, len(freq)))}
    sorted_entities = sorted(qualified, key=lambda w: freq[w], reverse=True)[:top_k]
    entity_map: dict[str, set[str]] = {}
    for entity in sorted_entities:
        aliases: set[str] = {entity}
        for other in sorted_entities:
            if other != entity and (entity in other or other in entity):
                aliases.add(other)
        entity_map[entity] = aliases
    return entity_map


def compress_to_keywords(text: str, top_k: int = 20) -> list[str]:
    """从文本提取高频关键词作为标签 (不破坏原文)。

    仅返回标签字符串列表，供 Prompt 末尾附加使用。
    使用 jieba.posseg 优先保留名词/动词/形容词。
    posseg 不可用时退化为 extract_keywords。

    Returns:
        关键词标签列表 (按词频降序)
    """
    if not text or not text.strip():
        return []

    jieba = _ensure_jieba()
    if jieba is None:
        return extract_keywords(text, top_k=top_k)

    try:
        import jieba.posseg as pseg
        words = pseg.cut(text)
        # 保留 n/v/a 词性, 长度 >= 2
        content_words = [w.word.strip() for w in words
                         if w.flag.startswith(("n", "v", "a")) and len(w.word.strip()) >= 2]
    except Exception:
        return extract_keywords(text, top_k=top_k)

    if not content_words:
        return extract_keywords(text, top_k=top_k)

    freq = Counter(content_words)
    return [w for w, _ in freq.most_common(top_k)]


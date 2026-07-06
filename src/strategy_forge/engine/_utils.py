"""Shared utilities for the deduction pipeline."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(raw: str) -> Any:
    """健壮地从 LLM 文本里提取 JSON（dict 或 list），失败返回 None。

    逐级尝试，越靠后越宽容，全部无损（只为救回同一次调用的输出）：
      1) 直接 json.loads
      2) 去 markdown 围栏后再 loads
      3) 贪婪匹配最外层 {...} / [...] 再 loads
      4) 花括号/方括号配平扫描（抗前后杂文、深层嵌套、字符串内的括号）
      5) 部分救回：输出被截断时，逐个收集已完整的顶层 {...} 对象
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()

    # 1) 直接解析
    obj = _try_loads(s)
    if obj is not None:
        return obj

    # 2) 去 markdown 围栏
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", s)
    cleaned = re.sub(r"\n?```", "", cleaned).strip()
    if cleaned != s:
        obj = _try_loads(cleaned)
        if obj is not None:
            return obj

    # 3) 贪婪匹配最外层对象/数组
    for pat in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pat, cleaned)
        if m:
            obj = _try_loads(m.group(0))
            if obj is not None:
                return obj

    # 4) 配平扫描第一个平衡的 {...} 或 [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        balanced = _extract_balanced(cleaned, opener, closer)
        if balanced is not None:
            obj = _try_loads(balanced)
            if obj is not None:
                return obj

    # 5) 部分救回：从截断文本里逐个抠出完整的顶层 {...} 对象
    salvaged = _salvage_objects(cleaned)
    if salvaged:
        return salvaged

    return None


def _try_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_balanced(text: str, opener: str, closer: str) -> str | None:
    """返回从第一个 opener 起、括号配平的子串；考虑字符串与转义，避免被字符串内的括号误导。"""
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _salvage_objects(text: str) -> list | None:
    """从（可能被截断的）文本里逐个抠出配平的顶层 {...} 对象，返回对象列表。

    专治"输出到一半被 max_tokens 截断"——已完整生成的实体对象仍可救回。
    """
    objs: list = []
    idx = 0
    n = len(text)
    while idx < n:
        nxt = text.find("{", idx)
        if nxt < 0:
            break
        chunk = _extract_balanced(text[nxt:], "{", "}")
        if chunk is None:
            # 从这里起不配平（外层未闭合或被截断）——跳过这个 { 继续找下一个可能完整的对象
            idx = nxt + 1
            continue
        parsed = _try_loads(chunk)
        if isinstance(parsed, dict):
            objs.append(parsed)
        idx = nxt + len(chunk)
    return objs or None


def extract_text(response) -> str:
    """Extract text content from various LLM response formats."""
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        c = response.content
        if isinstance(c, list):
            from strategy_forge.core.llm_client import TextBlock
            return "".join(b.text for b in c if isinstance(b, TextBlock))
        return str(c)
    if isinstance(response, dict):
        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        return str(response)
    return str(response)

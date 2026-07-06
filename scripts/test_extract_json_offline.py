"""离线单元测试：engine._utils.extract_json 的健壮性（无需 LLM/联网）。

验证种子/情报解析从"非贪婪易返回空"升级为"贪婪+配平+部分救回"后，
能正确解析或救回 6 类脏输出；并对照旧非贪婪写法证明原来会失败。

用法（项目根目录）：
    python scripts/test_extract_json_offline.py
"""
from __future__ import annotations

import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine._utils import extract_json  # noqa: E402

NESTED = '{"entities": [{"name": "特斯拉", "metrics": {"strength": 85, "cash_flow": 60}}]}'


def _old_nongreedy(raw: str):
    """旧实现：非贪婪 + 只接受 dict 的等价行为，用于对照。"""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", raw)
    cleaned = re.sub(r"\n?```", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    for pat in (r"\{[\s\S]*?\}", r"\[[\s\S]*?\]"):
        m = re.search(pat, cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def main() -> int:
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  \u2713 {name} {detail}")
        else:
            failed += 1
            print(f"  \u2717 {name} FAILED {detail}")

    def names(obj):
        ents = obj.get("entities", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
        return [e.get("name") for e in ents if isinstance(e, dict)]

    print("=== extract_json 健壮性离线单测 ===")

    # 1) 干净嵌套 JSON
    o = extract_json(NESTED)
    check("干净嵌套 JSON", isinstance(o, dict) and names(o) == ["特斯拉"], f"{names(o)}")

    # 2) markdown 围栏包裹
    o = extract_json("```json\n" + NESTED + "\n```")
    check("markdown 围栏", names(o) == ["特斯拉"])

    # 3) 前后有解说文字
    o = extract_json("好的，分析如下：\n" + NESTED + "\n以上为结果。")
    check("前后杂文", names(o) == ["特斯拉"])

    # 4) 顶层数组（省略 entities 外壳）
    o = extract_json('[{"name": "华为", "metrics": {"tech": 90}}]')
    check("顶层数组", isinstance(o, list) and names(o) == ["华为"], f"{names(o)}")

    # 5) 内层含中文与引号（配平扫描不被字符串内括号误导）
    tricky = '{"entities": [{"name": "O\'Brien", "metrics": {"note_use": 50}, "role": "含 } 和 { 符号"}]}'
    o = extract_json(tricky)
    check("字符串内含花括号", names(o) == ["O'Brien"], f"{names(o)}")

    # 6) 被 max_tokens 截断的半个 JSON —— 应救回已完整的实体对象
    truncated = ('{"entities": [{"name": "甲", "metrics": {"a": 1}}, '
                 '{"name": "乙", "metrics": {"a": 2}}, {"name": "丙", "metr')  # 丙 被截断
    o = extract_json(truncated)
    salvaged = names(o)
    check("截断救回已完整实体", "甲" in salvaged and "乙" in salvaged and "丙" not in salvaged,
          f"救回={salvaged}")

    # 对照：旧非贪婪在"带前后杂文的嵌套 JSON"上会失败/返回非 dict（证明这是真 bug）
    wrapped = "分析：\n" + NESTED + "\n完毕。"
    old = _old_nongreedy(wrapped)
    old_ok = isinstance(old, dict) and old.get("entities")
    check("对照-旧非贪婪确实解析失败（应为 True=已复现 bug）", not old_ok,
          f"old_result_type={type(old).__name__}")
    # 同一 wrapped 输入，新实现应正确解析
    check("新实现解析 wrapped 嵌套 JSON", names(extract_json(wrapped)) == ["特斯拉"])

    print(f"\n结果: {passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

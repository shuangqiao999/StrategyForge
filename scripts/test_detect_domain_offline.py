"""离线单元测试：RuleEngine.detect_domain（domain="auto" 领域识别）。

不联网、不需要云端或 LM Studio —— 用假的 chat_client 注入固定 JSON 响应，
验证修复后 detect_domain 不再因 prompt 内 JSON 花括号触发 KeyError('"domain"')
或 list_domains 键名错误 KeyError('display_name')，能正常返回识别到的领域。

用法（项目根目录）：
    python scripts/test_detect_domain_offline.py
"""
from __future__ import annotations

import asyncio
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://test-offline/v1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.rule_engine import RuleEngine  # noqa: E402


class FakeResp:
    def __init__(self, content: str):
        self.text = content
        self.content = content
        self.choices: list = []


class FakeChat:
    """记录收到的 prompt，返回预设 JSON 内容。"""
    def __init__(self, content: str):
        self.content = content
        self.last_prompt = ""

    async def chat(self, messages, system: str = "", **kwargs):
        self.last_prompt = messages[0].content if messages else ""
        return FakeResp(self.content)


async def main() -> int:
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  \u2713 {name} {detail}")
        else:
            failed += 1
            print(f"  \u2717 {name} FAILED {detail}")

    print("=== detect_domain 离线单测 ===")

    # 1) 正常识别：LLM 返回高置信 military → 应返回 'military'（内置域，get_template 命中）
    fc = FakeChat('{"domain": "military", "confidence": 0.92}')
    dom = await RuleEngine.detect_domain("两军在雪原对峙，粮草与士气决定胜负。", fc)
    check("识别 military（不再抛 KeyError）", dom == "military", f"got={dom}")
    check("prompt 正常拼出(含文本注入)", "雪原" in fc.last_prompt and "可选领域" in fc.last_prompt)
    check("prompt 保留 JSON 示例花括号", '{"domain"' in fc.last_prompt)

    # 2) 低置信 → 回退 narrative
    fc2 = FakeChat('{"domain": "military", "confidence": 0.2}')
    dom2 = await RuleEngine.detect_domain("一些文字", fc2, confidence_floor=0.6)
    check("低置信回退 narrative", dom2 == "narrative", f"got={dom2}")

    # 3) 未知领域 → 回退 narrative（get_template 未命中）
    fc3 = FakeChat('{"domain": "does_not_exist", "confidence": 0.99}')
    dom3 = await RuleEngine.detect_domain("一些文字", fc3)
    check("未知领域回退 narrative", dom3 == "narrative", f"got={dom3}")

    # 4) LLM 返回垃圾 → 不抛，回退 narrative
    fc4 = FakeChat("not a json at all")
    dom4 = await RuleEngine.detect_domain("一些文字", fc4)
    check("非法响应回退 narrative", dom4 == "narrative", f"got={dom4}")

    print(f"\n结果: {passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

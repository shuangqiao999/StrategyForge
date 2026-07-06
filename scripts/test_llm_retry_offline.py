"""离线单元测试：DeductionLLMClient 的 429/5xx 重试与 4xx 不重试逻辑。

不联网、不需要云端或 LM Studio —— 用假的 http 客户端注入到 _http，
直接驱动 _request_with_retry，验证退避重试行为。

用法（项目根目录）：
    python scripts/test_llm_retry_offline.py
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
os.environ.setdefault("FORGE_LLM_MODEL", "test-model")
# 加速：极小退避
os.environ["FORGE_LLM_RETRY_BASE"] = "0.001"
os.environ["FORGE_LLM_RETRY_CAP"] = "0.01"
os.environ["FORGE_LLM_MAX_RETRIES"] = "3"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import httpx  # noqa: E402

from strategy_forge.core.llm_client import DeductionLLMClient  # noqa: E402


class FakeResp:
    def __init__(self, status: int, headers: dict | None = None):
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://test-offline/v1/chat/completions"),
                response=httpx.Response(self.status_code),
            )


class FakeHttp:
    """按预设序列逐次返回响应；支持抛传输错误（用 Exception 实例占位）。"""
    def __init__(self, sequence: list):
        self.sequence = list(sequence)
        self.calls = 0

    async def post(self, url, json=None):
        self.calls += 1
        item = self.sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _mk_client(sequence: list) -> DeductionLLMClient:
    c = DeductionLLMClient(api_base="http://test-offline/v1", api_key="", model="test-model")
    c._http = FakeHttp(sequence)  # 绕过 _ensure_client，注入假客户端
    return c


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

    print("=== LLM 重试离线单测 ===")

    # 1) 429 一次 → 200：应重试1次并成功，共2次调用
    c = _mk_client([FakeResp(429), FakeResp(200)])
    resp = await c._request_with_retry({"model": "m"})
    check("429→200 重试成功", resp.status_code == 200 and c._http.calls == 2,
          f"calls={c._http.calls}")

    # 2) 503 两次 → 200：应重试2次，共3次调用
    c = _mk_client([FakeResp(503), FakeResp(503), FakeResp(200)])
    resp = await c._request_with_retry({"model": "m"})
    check("503×2→200 重试成功", resp.status_code == 200 and c._http.calls == 3,
          f"calls={c._http.calls}")

    # 3) 400：永久错误，不重试，立即抛，仅1次调用
    c = _mk_client([FakeResp(400), FakeResp(200)])
    raised = False
    try:
        await c._request_with_retry({"model": "m"})
    except httpx.HTTPStatusError:
        raised = True
    check("400 不重试立即抛", raised and c._http.calls == 1, f"calls={c._http.calls}")

    # 4) 传输错误一次 → 200：应重试1次
    c = _mk_client([httpx.ConnectError("boom"), FakeResp(200)])
    resp = await c._request_with_retry({"model": "m"})
    check("传输错误→200 重试成功", resp.status_code == 200 and c._http.calls == 2,
          f"calls={c._http.calls}")

    # 5) 持续 429 超过上限：应抛（max_retries=3 → 共 4 次调用后抛）
    c = _mk_client([FakeResp(429)] * 5)
    raised = False
    try:
        await c._request_with_retry({"model": "m"})
    except httpx.HTTPStatusError:
        raised = True
    check("429 超上限后抛出", raised and c._http.calls == 4, f"calls={c._http.calls}")

    # 6) Retry-After 头被遵循（值取 0.005s，仍应重试成功）
    c = _mk_client([FakeResp(429, {"Retry-After": "0.005"}), FakeResp(200)])
    resp = await c._request_with_retry({"model": "m"})
    check("遵循 Retry-After 头", resp.status_code == 200 and c._http.calls == 2,
          f"calls={c._http.calls}")

    print(f"\n结果: {passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

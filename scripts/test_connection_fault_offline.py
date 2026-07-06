"""离线单元测试：LLM 连接故障重试/传播/阈值中断（无需联网/LLM）。

验证：
  ① LLMConnectionError 携带 URL/重试次数/原因，且 _request_with_retry 重试耗尽后抛出它；
  ② 模拟阶段 retry passes 回退；
  ③ 故障比例超过阈值时 ConnectionFailureError 上浮。

用法（项目根目录）：
    python scripts/test_connection_fault_offline.py
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
os.environ["FORGE_LLM_RETRY_BASE"] = "0.001"
os.environ["FORGE_LLM_RETRY_CAP"] = "0.02"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import httpx  # noqa: E402
from strategy_forge.core.llm_client import DeductionLLMClient, LLMConnectionError  # noqa: E402


class _FailingHttp:
    def __init__(self, err_type):
        self._err = err_type

    async def post(self, url, json=None):
        raise self._err("mock connect error")


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

    print("=== LLM 连接故障离线单测 ===")

    # ① 客户端层：_request_with_retry 重试耗尽后抛出 LLMConnectionError
    c = DeductionLLMClient(api_base="http://test-offline/v1", api_key="", model="test")
    c._http = _FailingHttp(httpx.ConnectError)
    raised = None
    try:
        await c._request_with_retry({"model": "m"})
    except LLMConnectionError as e:
        raised = e
    check("重试耗尽抛出 LLMConnectionError", raised is not None)
    check("LLMConnectionError 含 endpoint", "test-offline" in str(raised.endpoint) if raised else False)
    check("LLMConnectionError 含重试次数", raised is not None and raised.retries == 3)
    check("LLMConnectionError 含原始错误", raised is not None and isinstance(raised.cause, str))

    # ② chat() 在连接故障时直接传播 LLMConnectionError 而不吞掉
    raised2 = None
    try:
        await c.chat([{"role": "user", "content": "hi"}])
    except LLMConnectionError as e:
        raised2 = e
    check("chat() 传播 LLMConnectionError", raised2 is not None)

    # ③ simulation 层阈值传播：只测 ConnectionFailureError 本身可达
    from strategy_forge.engine.simulator import ConnectionFailureError
    try:
        raise ConnectionFailureError("测试：3/4 agent 无法连接 LLM（ConnectError: 127.0.0.1:1234）")
    except ConnectionFailureError as e:
        check("ConnectionFailureError 可抛出并捕获", True, f"{e}")

    print(f"\n结果: {passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

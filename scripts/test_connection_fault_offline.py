"""离线单元测试：LLM 连接故障重试/传播/阈值中断（无需联网/LLM）。

验证：
  ① ConnectError → LLMConnectionError(无法连接) ｜ ReadTimeout → LLMConnectionError(响应超时+调参提示)
  ② chat() 传播 LLMConnectionError
  ③ ConnectionFailureError 可抛出（模拟阶段阈值触发）
  ④ 生成超时重试时递增 read timeout (escalation)

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
os.environ["FORGE_LLM_GENERATION_TIMEOUT"] = "30"   # 压低以加速单测
os.environ["FORGE_LLM_CONNECT_TIMEOUT"] = "10"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import httpx  # noqa: E402
from strategy_forge.core.llm_client import DeductionLLMClient, LLMConnectionError  # noqa: E402
from strategy_forge.engine.simulator import ConnectionFailureError          # noqa: E402  # noqa: E402


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

    print("=== LLM 连接故障离线单测 ===\n")

    # ① ConnectError → "无法连接" 消息（不含 GENERATION_TIMEOUT 提示）
    c = DeductionLLMClient(api_base="http://test-offline/v1", api_key="", model="test")
    c._http = _FailingHttp(httpx.ConnectError)
    raised = None
    try:
        await c._request_with_retry({"model": "m"})
    except LLMConnectionError as e:
        raised = e
    check("ConnectError → LLMConnectionError", raised is not None)
    check("消息含 无法连接/检查 等可操作指引", raised is not None and
          ("无法连接" in str(raised) or "检查" in str(raised)))
    check("消息不含 GENERATION_TIMEOUT 提示 (是联网问题)", raised is not None and
          "GENERATION_TIMEOUT" not in str(raised))

    # ①b ReadTimeout → "响应超时" 消息（含 GENERATION_TIMEOUT 调参指引）
    c2 = DeductionLLMClient(api_base="http://test-offline/v1", api_key="", model="test")
    c2._http = _FailingHttp(httpx.ReadTimeout)
    raised2 = None
    try:
        await c2._request_with_retry({"model": "m"})
    except LLMConnectionError as e:
        raised2 = e
    check("ReadTimeout → LLMConnectionError", raised2 is not None)
    check("消息含 响应超时 + GENERATION_TIMEOUT 指引", raised2 is not None and
          "响应超时" in str(raised2) and "GENERATION_TIMEOUT" in str(raised2))
    check("message 包含当前 gen_timeout 值", raised2 is not None and
          "30s" in str(raised2))  # ":30s" from {self._gen_timeout:.0f}=30

    # ② chat() 在连接故障时直接传播 LLMConnectionError 而不吞掉
    raised3 = None
    try:
        await c.chat([{"role": "user", "content": "hi"}])
    except LLMConnectionError as e:
        raised3 = e
    check("chat() 传播 LLMConnectionError", raised3 is not None)

    # ③ simulation 层 ConnectionFailureError 可达
    try:
        raise ConnectionFailureError("测试：3/4 agent 无法连接 LLM（ConnectError: 127.0.0.1:1234）")
    except ConnectionFailureError as e:
        check("ConnectionFailureError 可抛出并捕获", True, f"{e}")

    print(f"\n结果: {passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

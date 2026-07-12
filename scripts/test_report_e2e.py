"""端到端报告输出验证 — 连接本地 LM Studio 全流程推演 + 报告生成。

验证目标: 推演完成后的 _REPORT_PROMPT 能否产出有效报告。
用法: python scripts/test_report_e2e.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import json as _json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 必须在所有 import 前设置环境变量
os.environ["FORGE_PROVIDER"] = "lmstudio"
os.environ["FORGE_LLM_BASE"] = "http://127.0.0.1:1234/v1"
os.environ["FORGE_LLM_MODEL"] = "qwen/qwen3.5-9b"
os.environ["FORGE_EMBED_BASE"] = "http://127.0.0.1:1234/v1"
os.environ["FORGE_EMBED_MODEL"] = "text-embedding-embeddinggemma-300m-qat"
os.environ["FORGE_DEFAULT_ROUNDS"] = "3"
os.environ["FORGE_MAX_CONCURRENT"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.engine import DeductionEngine

SEED_TEXT = """2025年，红方与蓝方在边境地区发生军事对峙。红方集结3个机械化师向蓝方防线推进，
蓝方依托山地地形组织梯次防御。绿方作为中立方进行外交斡旋，呼吁双方回到谈判桌。
红方补给线较长，蓝方拥有地形优势，绿方掌握关键情报渠道。"""

DOMAIN = "military"
ROUNDS = 3
PRE_GOALS = ["红方力求速战速决，蓝方力求消耗对手，绿方力求避免战火蔓延"]


async def main():
    print("=" * 60)
    print(f"  端到端报告输出验证 (领域: {DOMAIN}, {ROUNDS}轮)")
    print(f"  LLM: {os.environ.get('FORGE_LLM_MODEL','?')}  @  {os.environ.get('FORGE_LLM_BASE','?')}")
    print("=" * 60)

    # 创建临时工作空间
    ws = tempfile.mkdtemp(prefix="sf_e2e_")
    print(f"\n工作目录: {ws}")

    path = os.path.join(ws, ".forge")
    os.environ["FORGE_DATA_DIR"] = path

    engine = DeductionEngine(ws)

    # 创建会话
    session = engine.create_session(
        title="端到端报告测试",
        source_material=SEED_TEXT,
        config={"domain": DOMAIN, "total_rounds": ROUNDS, "pre_goals": PRE_GOALS},
    )
    print(f"会话创建: {session.id}")

    # 运行全流程
    print("\n启动推演流程...")
    try:
        result = await engine.start(session.id)
    except Exception as e:
        print(f"\n!!! 推演失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 从 session store 获取报告
    data = engine.session_store.get(session.id)
    report_json_raw = data.get("report_json", "") if data else ""
    report = report_json_raw if isinstance(report_json_raw, dict) else (_json.loads(report_json_raw) if report_json_raw else {})
    print(f"\nreport_json 类型: {type(report_json_raw).__name__}, 已解析: {bool(report)}")

    # 验证报告字段
    print("\n" + "─" * 40)
    print("  报告验证")
    print("─" * 40)

    checks = []

    narrative = report.get("summary", "")
    checks.append(("narrative 非空", len(narrative) > 50,
                   f"长度={len(narrative)}"))
    print(f"  {'PASS' if len(narrative) > 50 else 'FAIL'} narrative: {len(narrative)}字符")
    if narrative:
        print(f"    预览: {narrative[:120]}...")

    risk_alerts = report.get("risk_alerts", [])
    checks.append(("risk_alerts >= 2条", len(risk_alerts) >= 2,
                   f"实际={len(risk_alerts)}"))
    print(f"  {'PASS' if len(risk_alerts) >= 2 else 'FAIL'} risk_alerts: {len(risk_alerts)}条")
    for ra in risk_alerts[:3]:
        if isinstance(ra, str):
            print(f"    - {ra[:80]}")
        elif isinstance(ra, dict):
            print(f"    - {str(ra)[:80]}")

    recommendations = report.get("recommendations", [])
    checks.append(("recommendations >= 2条", len(recommendations) >= 2,
                   f"实际={len(recommendations)}"))
    print(f"  {'PASS' if len(recommendations) >= 2 else 'FAIL'} recommendations: {len(recommendations)}条")
    for rec in recommendations[:3]:
        print(f"    - {rec[:80]}")

    conclusion = report.get("conclusion", "")
    checks.append(("conclusion 非空", len(conclusion) > 30,
                   f"长度={len(conclusion)}"))
    print(f"  {'PASS' if len(conclusion) > 30 else 'FAIL'} conclusion: {len(conclusion)}字符")
    print(f"    内容: {conclusion[:100]}...")

    has_dim = "###" in narrative
    checks.append(("narrative 含 ### 维度标题", has_dim))
    print(f"  {'PASS' if has_dim else 'FAIL'} ### 维度标题: {'有' if has_dim else '无'}")

    has_causal = "→" in narrative
    checks.append(("narrative 含 → 因果链", has_causal))
    print(f"  {'PASS' if has_causal else 'FAIL'} → 因果链: {'有' if has_causal else '无'}")

    has_event = "[事件" in narrative
    checks.append(("narrative 含 [事件N] 引用", has_event))
    print(f"  {'PASS' if has_event else 'FAIL'} [事件N]引用: {'有' if has_event else '无'}")

    has_although = conclusion.startswith("虽然") if conclusion else False
    checks.append(("conclusion 以'虽然'开头", has_although))
    print(f"  {'PASS' if has_although else 'FAIL'} 虽然: {'是' if has_although else '否'}")

    # 汇总
    passed = sum(1 for c in checks if c[1])
    total = len(checks)
    print(f"\n{'='*60}")
    print(f"  结果: {passed}/{total} 通过")
    print(f"{'='*60}")

    # 清理
    engine.close()
    import shutil
    shutil.rmtree(ws, ignore_errors=True)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

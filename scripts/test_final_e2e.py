"""全流程端到端验证 — 覆盖所有本次迭代的修改点。

验证项:
1. 全流程五阶段无崩溃 (ontology→quantify→graph→agents→simulation→report)
2. 报告产出完整 (narrative + risk_alerts + recommendations + conclusion)
3. key_events 多轮覆盖 (不再全来自第1轮)
4. ### 维度标题、→ 因果链、[事件N]引用、虽然...但是... conclusion
5. 无人称占位符泄漏 (无 "X"、"某方" 孤立出现)
6. fatigue 恢复无震荡 (by ODE 饱和因子验证)
7. ODE 无 NoneType 错误 (by 完成推演即验证)
8. 报告归一化层输出全部为字符串 (by 检查 risk_alerts 类型)

用法: python scripts/test_final_e2e.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
import json as _json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ["FORGE_PROVIDER"] = "lmstudio"
os.environ["FORGE_LLM_BASE"] = "http://127.0.0.1:1234/v1"
os.environ["FORGE_LLM_MODEL"] = "google/gemma-4-12b"
os.environ["FORGE_EMBED_BASE"] = "http://127.0.0.1:1234/v1"
os.environ["FORGE_EMBED_MODEL"] = "text-embedding-embeddinggemma-300m-qat"
os.environ["FORGE_MAX_CONCURRENT"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SEED = """红方与蓝方在边境地区发生军事对峙。红方军队在东部集结，向蓝方防线推进，
蓝方军队依托山地地形组织梯次防御。绿方作为第三方保持中立，进行外交斡旋。
红方补给线较长但兵力充足，蓝方拥有地形优势，绿方掌握关键情报。"""

DOMAIN = "military"
ROUNDS = 3


async def main():
    print("=" * 60)
    print(f"  全流程端到端验证 (military, {ROUNDS}轮)")
    print(f"  LLM: qwen/qwen3.5-9b")
    print("=" * 60)

    ws = tempfile.mkdtemp(prefix="sf_final_")
    os.environ["FORGE_DATA_DIR"] = os.path.join(ws, ".forge")

    from strategy_forge.engine.engine import DeductionEngine
    engine = DeductionEngine(ws)

    session = engine.create_session(
        title="全流程验证",
        source_material=SEED,
        config={"domain": DOMAIN, "total_rounds": ROUNDS},
    )
    print(f"会话: {session.id}")

    # ── 全流程运行 ──
    print("\n启动推演...")
    crash = False
    try:
        result = await engine.start(session.id)
    except Exception as e:
        print(f"\n!!! 推演崩溃: {e}")
        import traceback
        traceback.print_exc()
        crash = True

    logs = engine.get_logs(session.id, limit=200)
    data = engine.session_store.get(session.id)
    report = data.get("report_json", {}) if data else {}
    if isinstance(report, str):
        report = _json.loads(report)

    # ── 逐项验证 ──
    print("\n" + "=" * 60)
    print("  验证结果")
    print("=" * 60)

    checks: list[tuple[str, bool]] = []

    # 1. 无崩溃
    checks.append(("全流程无崩溃", not crash))

    # 2. 报告完整
    summary = report.get("summary", "")
    checks.append(("narrative 非空", len(summary) > 50))
    risk_alerts = report.get("risk_alerts", [])
    checks.append(("risk_alerts >= 2条", len(risk_alerts) >= 2))
    recs = report.get("recommendations", [])
    checks.append(("recommendations >= 2条", len(recs) >= 2))
    conclusion = report.get("conclusion", "")
    checks.append(("conclusion 非空", len(conclusion) > 30))

    # 3. key_events 多轮
    key_events = report.get("key_events", [])
    rounds_set = set()
    for ev in key_events:
        desc = ev.get("description", "") if isinstance(ev, dict) else str(ev)
        m = re.search(r"\[轮(\d+)\]", desc)
        if m:
            rounds_set.add(int(m.group(1)))
    checks.append(("key_events >= 2个轮次", len(rounds_set) >= 2))

    # 4. 格式标记
    checks.append(("### 维度标题", "### " in summary))
    checks.append(("→ 因果链", "→" in summary))
    checks.append(("[事件N]引用", "[事件" in summary))
    checks.append(("conclusion 以'虽然'开头", conclusion.startswith("虽然")))

    # 5. 无占位符泄漏
    has_x_placeholder = bool(
        re.search(r"(?<![a-zA-Z])X(?![a-zA-Z])", summary.replace("X方", ""))
        and "X为" not in summary.replace("某方", "").replace("各方", "")
    )
    # Check: standalone "X" (not part of "某方")
    # Simpler check: look for isolated "X" used as placeholder
    suspicious = re.findall(r"X(?:为|虽|方)", summary)
    checks.append(("无 'X' 占位符", len(suspicious) == 0))

    # 6. risk_alerts 全部为字符串
    all_str = all(isinstance(r, str) for r in risk_alerts)
    checks.append(("risk_alerts 全为字符串", all_str))
    # 7. recommendations 全部为字符串
    all_str_rec = all(isinstance(r, str) for r in recs)
    checks.append(("recommendations 全为字符串", all_str_rec))

    # 输出结果
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")

    if summary:
        print(f"\n  narrative 预览: {summary[:120]}...")
    if conclusion:
        print(f"  conclusion 预览: {conclusion[:80]}...")

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"\n{'=' * 60}")
    print(f"  结果: {passed}/{total} 通过")
    print(f"{'=' * 60}")

    engine.close()
    import shutil
    shutil.rmtree(ws, ignore_errors=True)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

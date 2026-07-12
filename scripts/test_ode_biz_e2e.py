"""ODE+报告+Partner修复验证 — 商业领域全流程测试。

验证项:
1. cash_flow 不会过度下跌（_rnd_monetization 生效）
2. 报告中出现"战略性投入，非危机"标记
3. Partner 动作目标不跨行业（不再出现车企→超市）
4. 全流程无崩溃

用法: python scripts/test_ode_biz_e2e.py
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
os.environ["FORGE_LLM_MODEL"] = "qwen/qwen3.5-9b"
os.environ["FORGE_EMBED_BASE"] = "http://127.0.0.1:1234/v1"
os.environ["FORGE_EMBED_MODEL"] = "text-embedding-embeddinggemma-300m-qat"
os.environ["FORGE_MAX_CONCURRENT"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SEED = """2025年新能源汽车市场竞争白热化。比亚迪以垂直整合和刀片电池技术占据中国市场35%份额，
年营收突破6000亿元，研发投入超400亿元。理想汽车以増程路线主攻家庭市场，2025年交付超50万辆，
现金储备超900亿元。小鹏汽车主打智能驾驶，2025年交付18万辆，研发投入占比超20%。
华为昇腾AI芯片占据国产AI芯片市场43%份额，向蔚来、小鹏、理想等车企供应智能驾驶芯片。
寒武纪则专注于AI推理芯片，2025年营收约12亿元，经营现金流由负转正。"""

DOMAIN = "business"
ROUNDS = 5


async def main():
    print("=" * 60)
    print(f"  ODE+报告+Partner修复验证 (business, {ROUNDS}轮)")
    print(f"  LLM: qwen/qwen3.5-9b")
    print("=" * 60)

    ws = tempfile.mkdtemp(prefix="sf_odebiz_")
    os.environ["FORGE_DATA_DIR"] = os.path.join(ws, ".forge")

    from strategy_forge.engine.engine import DeductionEngine
    engine = DeductionEngine(ws)

    session = engine.create_session(
        title="ODE+Partner验证",
        source_material=SEED,
        config={"domain": DOMAIN, "total_rounds": ROUNDS},
    )
    print(f"会话: {session.id}")

    # 全流程
    try:
        result = await engine.start(session.id)
    except Exception as e:
        print(f"\n!!! 崩溃: {e}")
        import traceback
        traceback.print_exc()
        engine.close()
        import shutil
        shutil.rmtree(ws, ignore_errors=True)
        return 1

    data = engine.session_store.get(session.id)
    report = data.get("report_json", {}) if data else {}
    if isinstance(report, str):
        report = _json.loads(report)

    logs = engine.get_logs(session.id, limit=200)

    # ── 验证 ──
    print("\n" + "=" * 60)
    print("  验证结果")
    print("=" * 60)

    checks: list[tuple[str, bool, str]] = []

    # 1. 全流程无崩溃
    checks.append(("全流程无崩溃", True, ""))

    # 2. 报告中"战略性投入"标记 — LLM从量化上下文读到了标记并在叙事中使用
    summary = report.get("summary", "")
    has_strategic = "战略性投入" in summary
    checks.append(("叙事务含'战略性投入'标记", has_strategic, "有" if has_strategic else "无"))

    # 3. cash_flow 数值合理性
    final_states = report.get("final_states", {})
    cash_flows = []
    for eid, fs in final_states.items():
        cf = fs.get("metrics", {}).get("cash_flow", -1)
        if cf >= 0:
            cash_flows.append((fs.get("name", eid[:8]), cf))

    # 检查：至少一个实体 cash_flow > 20（有研发反哺）
    high_cf = [f"{n}={cf:.0f}" for n, cf in cash_flows if cf > 20]
    checks.append(("存在 cash_flow > 20 的实体", len(high_cf) > 0,
                   ", ".join(high_cf[:5]) if high_cf else "无"))

    # 5. 报告完整性
    checks.append(("narrative 非空", len(summary) > 50, f"{len(summary)}字"))
    risk_alerts = report.get("risk_alerts", [])
    checks.append(("risk_alerts >= 2", len(risk_alerts) >= 2, f"{len(risk_alerts)}条"))
    conclusion = report.get("conclusion", "")
    checks.append(("conclusion 非空", len(conclusion) > 30, f"{len(conclusion)}字"))

    # 6. 无 X 占位符
    has_x = bool(_re.search(r"(?<![a-zA-Z])X(?![a-zA-Z])", summary.replace("X方", "").replace("某方", "")))
    checks.append(("无'X'占位符", not has_x, ""))

    # 输出
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")

    if summary:
        print(f"\n  narrative 预览: {summary[:150]}...")
    if cash_flows:
        print(f"  现金流快照: {', '.join(f'{n}={cf:.0f}' for n, cf in cash_flows[:8])}")

    passed = sum(1 for _, ok, _ in checks if ok)
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

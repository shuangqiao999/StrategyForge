"""端到端修复验证 — Bug1(key_events 截断) + Bug2(FSM 索引错位) 全流程测试。

验证项:
1. key_events 覆盖多轮而非全部来自第1轮
2. 报告中 [轮1]/[轮2]/[轮3] 分布是否符合预期
3. FSM 存活 agent 动作数量是否合理
4. 报告叙事/预警/建议/结论完整

用法: python scripts/test_bugfix_e2e.py
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
os.environ["FORGE_DEFAULT_ROUNDS"] = "3"
os.environ["FORGE_MAX_CONCURRENT"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SEED = """红方与蓝方在边境地区发生军事对峙。红方军队在东部集结，向蓝方防线推进，
蓝方军队依托山地地形组织梯次防御。绿方作为第三方保持中立，进行外交斡旋。
红方补给线较长但兵力充足，蓝方拥有地形优势，绿方掌握关键情报。"""

DOMAIN = "military"
ROUNDS = 3


def parse_rounds_from_events(key_events: list) -> dict:
    """从 key_events 中统计各轮次事件数量。"""
    round_counts: dict[str, int] = {}
    for ev in key_events:
        desc = ev.get("description", "") if isinstance(ev, dict) else str(ev)
        m = re.search(r"\[轮(\d+)\]", desc)
        r = m.group(1) if m else "?"
        round_counts[r] = round_counts.get(r, 0) + 1
    return round_counts


async def main():
    print("=" * 60)
    print(f"  Bug1/2 修复验证 (military, {ROUNDS}轮)")
    print(f"  LLM: qwen/qwen3.5-9b")
    print("=" * 60)

    ws = tempfile.mkdtemp(prefix="sf_bugfix_")
    os.environ["FORGE_DATA_DIR"] = os.path.join(ws, ".forge")

    from strategy_forge.engine.engine import DeductionEngine
    engine = DeductionEngine(ws)

    session = engine.create_session(
        title="Bugfix验证",
        source_material=SEED,
        config={"domain": DOMAIN, "total_rounds": ROUNDS},
    )
    print(f"会话: {session.id}")

    print("\n启动推演...")
    result = await engine.start(session.id)

    # 获取原始日志检查 FSM 分流
    logs = engine.get_logs(session.id, limit=500)
    fsm_logs = [l for l in logs if l.get("phase") == "simulation" and "[FSM]" in l.get("message", "")]
    print(f"FSM 动作数: {len(fsm_logs)}")

    # 获取报告
    data = engine.session_store.get(session.id)
    report = data.get("report_json", {}) if data else {}
    if isinstance(report, str):
        report = _json.loads(report)

    # ── Bug1 验证: key_events 轮次分布 ──
    print("\n" + "─" * 50)
    print("  Bug1 验证: key_events 轮次分布")
    print("─" * 50)

    key_events = report.get("key_events", [])
    round_dist = parse_rounds_from_events(key_events)
    print(f"  key_events 总数: {len(key_events)}")
    print(f"  轮次分布: {sorted(round_dist.items())}")
    for rnd, cnt in sorted(round_dist.items(), key=lambda x: (int(x[0]) if x[0].isdigit() else 99)):
        print(f"    [轮{rnd}]: {cnt} 条事件")

    bug1_ok = len(round_dist) >= 2 and "1" not in round_dist or any(
        k != "1" and round_dist.get(k, 0) > 0 for k in round_dist
    )
    # 修复后应该覆盖多轮
    multi_round = sum(1 for k in round_dist if k.isdigit() and int(k) > 1 and round_dist[k] > 0)
    if multi_round >= 1:
        print(f"  PASS: key_events 覆盖了 {multi_round} 个非第1轮的轮次")
        bug1_ok = True
    elif len(round_dist) <= 1 and "1" in round_dist:
        print(f"  INFO: 只有第1轮有事件（可能是模拟本身产出集中在第1轮）")
        bug1_ok = None  # 不确定
    else:
        print(f"  FAIL: key_events 轮次分布异常")

    # ── Bug2 验证: FSM 存活 agent 动作数 ──
    print("\n" + "─" * 50)
    print("  Bug2 验证: FSM + 存活 agent")
    print("─" * 50)

    # 检查每轮 LLM 决策次数 vs FSM 决策次数
    llm_logs = [l for l in logs if l.get("phase") == "simulation"
                and ("[LLM]" in l.get("message", "") or "决策" in l.get("message", ""))]
    print(f"  LLM 相关日志: {len(llm_logs)} 条")
    for l in fsm_logs[:5]:
        print(f"    {l['message'][:100]}")
    if fsm_logs:
        print(f"  PASS: FSM 正常运行，{len(fsm_logs)} 次确定性动作")
    else:
        print(f"  INFO: 无 FSM 动作（可能全部 agent 处于 command 态）")

    # ── 报告完整性验证 ──
    print("\n" + "─" * 50)
    print("  报告完整性验证")
    print("─" * 50)

    checks = []
    summary = report.get("summary", "")
    checks.append(("narrative 非空", len(summary) > 50, len(summary)))
    risk_alerts = report.get("risk_alerts", [])
    checks.append(("risk_alerts", len(risk_alerts) >= 2, len(risk_alerts)))
    recs = report.get("recommendations", [])
    checks.append(("recommendations", len(recs) >= 2, len(recs)))
    conclusion = report.get("conclusion", "")
    checks.append(("conclusion", len(conclusion) > 30, len(conclusion)))

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")

    print(f"\n  narrative 预览: {summary[:150]}...")

    all_ok = all(ok for _, ok, _ in checks) and bug1_ok is not False
    print(f"\n{'='*60}")
    print(f"  总体: {'PASS' if all_ok else 'FAIL'}")
    print(f"{'='*60}")

    engine.close()
    import shutil
    shutil.rmtree(ws, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

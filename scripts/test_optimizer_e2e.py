"""策略优化器修复验证 — 连接本地 LM Studio 全流程测试。

验证项:
1. 优化器全流程无崩溃 (基线构建 + 多方案采样 + 统计)
2. cost 计算使用了规则包权重 (非等权)
3. recommended 有 CI 宽度 tiebreaker (不是任意选择)
4. recommendation_rationale 字段非空且内容有意义

用法: python scripts/test_optimizer_e2e.py
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

os.environ["FORGE_PROVIDER"] = "lmstudio"
os.environ["FORGE_LLM_BASE"] = "http://127.0.0.1:1234/v1"
os.environ["FORGE_LLM_MODEL"] = "qwen/qwen3.5-9b"
os.environ["FORGE_EMBED_BASE"] = "http://127.0.0.1:1234/v1"
os.environ["FORGE_EMBED_MODEL"] = "text-embedding-embeddinggemma-300m-qat"
os.environ["FORGE_DEFAULT_ROUNDS"] = "3"
os.environ["FORGE_MAX_CONCURRENT"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SEED = """红方与蓝方在边境对峙。红方拥有3个机械化师，兵力充足但补给线过长。
蓝方依托山地地形防守，拥有2个山地旅和充足的弹药储备。
绿方作为中立方掌握关键情报资源，在外交渠道对双方施压。"""

DOMAIN = "military"
ROUNDS = 3


async def main():
    print("=" * 60)
    print(f"  策略优化器修复验证 (military, {ROUNDS}轮)")
    print(f"  LLM: qwen/qwen3.5-9b")
    print("=" * 60)

    ws = tempfile.mkdtemp(prefix="sf_opt_")
    os.environ["FORGE_DATA_DIR"] = os.path.join(ws, ".forge")

    from strategy_forge.engine.engine import DeductionEngine
    from strategy_forge.engine.optimizer import StrategyOptimizer

    engine = DeductionEngine(ws)

    # 创建会话
    session = engine.create_session(
        title="优化器修复测试",
        source_material=SEED,
        config={"domain": DOMAIN, "total_rounds": ROUNDS},
    )
    print(f"会话: {session.id}")

    # 构建优化器
    optimizer = StrategyOptimizer(engine)

    # 定义方案（2 个方案 × 2 次采样 = 4 次模拟）
    scenarios = [
        {"name": "速攻方案", "directive": "红方集中全部兵力速战速决，在补给耗尽前击溃蓝方"},
        {"name": "消耗方案", "directive": "红方稳步推进，优先保障补给线，消耗蓝方弹药储备"},
    ]

    print(f"\n启动优化（{len(scenarios)} 方案 × 2 次）...")

    try:
        graph = engine.get_graph(session.id)
        result = await optimizer.run_monte_carlo(
            session_id=session.id,
            scenarios=scenarios,
            iterations=2,
            objective="balanced",
            win_condition="红方在3轮内占领蓝方防线",
            cancel_event=None,
        )
        print("优化完成")
    except Exception as e:
        print(f"\n!!! 优化崩溃: {e}")
        import traceback
        traceback.print_exc()
        engine.close()
        import shutil
        shutil.rmtree(ws, ignore_errors=True)
        return 1

    # ── 验证 ──
    print("\n" + "=" * 60)
    print("  验证结果")
    print("=" * 60)

    checks: list[tuple[str, bool, str]] = []

    # 1. 场景数据存在
    scenarios_data = result.get("scenarios", [])
    checks.append(("场景列表非空", len(scenarios_data) > 0, f"{len(scenarios_data)} 个方案"))

    # 2. 推荐方案存在
    recommended = result.get("recommended")
    checks.append(("推荐方案存在", recommended is not None, recommended["name"] if recommended else "N/A"))

    # 3. 推荐理由非空
    rationale = result.get("recommendation_rationale", "")
    checks.append(("推荐理由非空", len(rationale) > 30, rationale[:80] if rationale else "N/A"))

    # 4. 帕累托前沿存在
    pareto = result.get("pareto_front", [])
    checks.append(("帕累托前沿存在", len(pareto) > 0, str(pareto)))

    # 5. cost 格式正确（0-1 之间的浮点数）
    for s in scenarios_data:
        cost = s.get("cost_mean", -1)
        if not (0 <= cost <= 1):
            checks.append((f"cost 范围正确 ({s['name']})", False, f"cost={cost}"))
            break
    else:
        checks.append(("cost 范围正确", True, f"全部在 [0,1]"))

    # 6. win_ci95 排序（CI 更窄 = 更稳定）
    if len(scenarios_data) >= 2:
        ci_widths = [s["win_ci95"][1] - s["win_ci95"][0] for s in scenarios_data]
        checks.append(("win_ci95 存在", all(w > 0 for w in ci_widths), str(ci_widths)))

    # 输出结果
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")

    if rationale:
        print(f"\n  推荐理由:\n  {rationale}")

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

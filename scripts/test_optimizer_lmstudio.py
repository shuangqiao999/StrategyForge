"""策略优化器功能测试 — 连接本地 LM Studio 实际跑通蒙特卡洛多方案对比。

用法（项目根目录）：
    python scripts/test_optimizer_lmstudio.py

前置条件：
    - 本地 LM Studio 监听 127.0.0.1:1234，并已加载对话模型与嵌入模型。
    - 可用环境变量覆盖默认模型，例如：
        set FORGE_LLM_MODEL=qwen/qwen3.5-9b

本脚本验证：自建基线(Phase1-3) → M×N 隔离蒙特卡洛 → LLM 结局评估 →
统计(胜率/置信区间/成功率/成本) → 帕累托前沿 → 推荐方案 → 报告持久化。
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile

# 确保输出为 UTF-8（避免 Windows GBK 控制台对中文/emoji 编码报错）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 环境：临时数据目录 + LM Studio 配置（均可被外部环境变量覆盖） ──
_TMP = os.path.join(tempfile.gettempdir(), "opencode", "opt_func_test")
shutil.rmtree(_TMP, ignore_errors=True)
os.makedirs(_TMP, exist_ok=True)
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen3.5-2b")  # 小模型加速测试
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.engine import DeductionEngine  # noqa: E402
from strategy_forge.engine.optimizer import StrategyOptimizer  # noqa: E402

TEXT = (
    "东京八十万禁军教头林冲为人忠厚。高俅之子高衙内见林冲娘子貌美心生歹意。"
    "高俅设计陷害林冲，命陆谦诱骗林冲带刀误入白虎堂，林冲被发配沧州。"
    "陆谦奉高俅之命一路追杀林冲，鲁智深暗中保护林冲，在野猪林救下林冲性命。"
    "林冲在沧州看守草料场，陆谦放火欲烧死林冲。林冲风雪山神庙杀死陆谦，雪夜上梁山。"
    "鲁智深本是渭州提辖鲁达，三拳打死镇关西郑屠，逃到五台山出家，后与林冲结为兄弟。"
    "高俅奸诈，林冲悲愤，陆谦阴险，鲁智深豪爽，众人最终在梁山聚义。"
) * 2


def _progress(done, total, current, outcome):
    print(f"  [进度] {done}/{total} {current} 胜分={outcome.win_score:.2f} 成本={outcome.cost:.2f}")


async def main() -> int:
    print("=== 策略优化器功能测试（LM Studio）===")
    print("LLM:", os.environ["FORGE_LLM_MODEL"], "| EMBED:", os.environ["FORGE_EMBED_MODEL"])

    engine = DeductionEngine(_TMP)
    session = engine.create_session("优化器测试", TEXT, {"total_rounds": 2})
    sid = session.id
    print("会话:", sid, "（每方案 2 次 × 2 轮，2 个方案）")

    optimizer = StrategyOptimizer(engine)
    report = await optimizer.run_monte_carlo(
        session_id=sid,
        scenarios=[
            {"name": "拒绝招安", "directive": "坚决反对招安，主张独立发展势力、对抗朝廷"},
            {"name": "接受招安", "directive": "接受朝廷招安，争取正统地位与高位"},
        ],
        win_condition="梁山核心势力长期存续且主要头领得以善终",
        iterations=2,
        objective="balanced",
        max_concurrent=2,
        progress_cb=_progress,
    )

    print("\n=== 优化报告 ===")
    print("完成:", report["completed_runs"], "/", report["total_runs"], "| 取消:", report["cancelled"])
    print("帕累托前沿:", report["pareto_front"])
    print("推荐方案:", report["recommended"])
    for s in report["scenarios"]:
        print(f"  方案[{s['name']}] 胜率={s['win_mean']:.2f} CI={s['win_ci95']} "
              f"成功率={s['success_rate']:.2f} 成本={s['cost_mean']:.2f} 次数={s['runs']} 帕累托={s['is_pareto']}")

    # ── 断言 ──
    assert report["completed_runs"] >= 1, "未产生任何有效结果"
    assert report["scenarios"], "无方案统计"
    assert report["recommended"] is not None, "未给出推荐方案"
    for s in report["scenarios"]:
        assert 0.0 <= s["win_mean"] <= 1.0, "胜率超出范围"
        assert 0.0 <= s["cost_mean"] <= 1.0, "成本超出范围"
    assert any(s["is_pareto"] for s in report["scenarios"]), "帕累托前沿为空"

    # ── 持久化验证（optimization_report_json 列 + 读回） ──
    engine.session_store.update(sid, optimization_report_json=json.dumps(report, ensure_ascii=False))
    d = engine.session_store.get(sid)
    assert isinstance(d["optimization_report_json"], dict), "优化报告列读回类型错误"
    assert d["optimization_report_json"].get("recommended"), "优化报告未正确持久化"

    engine.close()
    shutil.rmtree(_TMP, ignore_errors=True)
    print("\n[OK] 优化器功能测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

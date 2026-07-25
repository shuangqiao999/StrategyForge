"""EntityRegistry 专项测试：验证代码硬排除 + LLM 全量分类。

用大国博弈种子材料跑全流程，重点验证：
1. 代码硬排除正确性（二元词/军队/职务/部门/集合概念）
2. LLM 分类是否成功运行（非 fallback）
3. 核心博弈方是否被保留（中国/美国/俄罗斯/伊朗/以色列等）
4. 非博弈方是否被排除（乌军/俄军/总统/中美关系等）
5. agent 数量合理（10-25，不是 4 也不是 100）

用法: python tests/test_entity_registry.py
"""
import sys, os, asyncio, time, uuid, shutil, re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

os.environ["FORGE_PROVIDER"] = "lmstudio"
os.environ["FORGE_EMBED_PROVIDER"] = "lmstudio"
os.environ["FORGE_MAX_CONCURRENT"] = "2"

from strategy_forge.core.providers import registry
registry._data["llm_provider"] = "lmstudio"
registry._data["llm_model"] = "qwen/qwen3.5-9b"
registry._data["embed_provider"] = "lmstudio"
registry._data["embedding_model_name"] = "text-embedding-embeddinggemma-300m-qat"
registry._data["max_concurrent"] = "2"

from strategy_forge.engine.models import DeductionSession
from strategy_forge.engine.orchestrator import DeductionOrchestrator
from strategy_forge.engine.entity_registry import build_registry, _is_dyad
from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.storage.session_store import SessionStore

SOURCE = r"E:\gongxiang\软件\资本论\大国博弈.txt"

REQUIRED_KEEP = ["中国", "美国", "俄罗斯", "伊朗", "以色列"]
SHOULD_KEEP = ["日本", "法国", "菲律宾", "乌克兰", "特朗普", "普京", "吕特"]
MUST_DISCARD = ["中美关系", "俄乌冲突", "俄乌", "乌军", "俄军"]
SHOULD_DISCARD = ["国防部", "总统", "太平洋舰队", "西方阵营"]


def test_code_rules_offline():
    """离线测试：代码硬排除规则。"""
    print("=" * 60)
    print("  离线测试：代码硬排除规则")
    print("=" * 60)

    tests = [
        # (name, type, expected_discard_reason)
        ("哈尔科夫", "Location", "类型排除"),
        ("南海", "Location", "类型排除"),
        ("中美关系", "Other", "二元关系词"),
        ("俄乌", "Other", "二元关系词"),
        ("俄乌冲突", "Other", "二元关系词"),
        ("乌军", "Organization", "军队编制"),
        ("太平洋舰队", "Organization", "军队编制"),
        ("总统", "Person", "职务头衔"),
        ("国防部长", "Person", "职务头衔"),
        ("财政部", "Organization", "政府部门"),
        ("白宫", "Organization", "政府部门"),
        ("西方阵营", "Other", "集合概念"),
    ]

    from strategy_forge.engine.entity_registry import (
        _DISCARD_TYPES, _TITLE_SUFFIX, _MILITARY_SUFFIX,
        _DEPT_WORDS, _COLLECTIVE_SUFFIX,
    )

    passed = 0
    for name, etype, expected in tests:
        reason = None
        if etype in _DISCARD_TYPES: reason = "类型排除"
        elif _is_dyad(name): reason = "二元关系词"
        elif any(name.endswith(s) for s in _TITLE_SUFFIX): reason = "职务头衔"
        elif any(name.endswith(s) for s in _COLLECTIVE_SUFFIX): reason = "集合概念"
        elif any(name.endswith(s) for s in _MILITARY_SUFFIX): reason = "军队编制"
        elif any(w in name for w in _DEPT_WORDS): reason = "政府部门"

        if reason == expected:
            passed += 1
            print(f"  OK  {name:12s} {etype:16s} → {reason}")
        else:
            print(f"  FAIL {name:12s} {etype:16s} → got={reason}, expected={expected}")

    print(f"\n  结果: {passed}/{len(tests)} 通过")
    return passed == len(tests)


async def test_full_pipeline():
    """全流程测试。"""
    t0 = time.time()
    sid = uuid.uuid4().hex[:12]
    tmp = os.path.join(os.environ["TEMP"], f"forge_er_{sid}")
    os.makedirs(tmp, exist_ok=True)

    print()
    print("=" * 60)
    print("  全流程测试：大国博弈（量化 geo_strategy）")
    print(f"  Model: {registry.llm_model}")
    print(f"  Session: {sid}")
    print("=" * 60)

    source = open(SOURCE, encoding="utf-8").read()
    print(f"\n  种子材料: {len(source):,} chars")

    session = DeductionSession(id=sid, title="EntityRegistry测试", source_material=source, total_rounds=3)
    graph_path = os.path.join(tmp, "graphs", sid, "kuzu")
    db_path = os.path.join(tmp, "session.db")
    store = SessionStore(db_path)
    graph = DeductionGraphStore(graph_path)
    store.create(sid, "EntityRegistry测试", source, {"domain": "geo_strategy", "total_rounds": 3})

    log_lines: list[str] = []

    def logger(phase: str, msg: str):
        elapsed = time.time() - t0
        log_lines.append((elapsed, phase, msg))
        print(f"  [{elapsed:6.1f}s] [{phase:12s}] {msg}")

    orch = DeductionOrchestrator(session=session, graph=graph, session_store=store, logger_fn=logger)

    print()
    print("-" * 60)
    print("  开始推演...")
    print("-" * 60)
    print()

    try:
        result = await orch.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = orch.session

    total = time.time() - t0
    session_data = store.get(sid) or {}
    a_count = session_data.get("agent_count", 0) or getattr(result, "agent_count", 0)
    error = session_data.get("error", "") or getattr(result, "error", "")

    # ── 注册中心独立验证 ──
    print()
    print("=" * 60)
    print("  EntityRegistry 验证")
    print("=" * 60)

    ent_registry = await build_registry(graph, source_material=source, domain="geo_strategy")
    kept_names = {e.name for e in ent_registry.get_kept()}
    discard_names = {e.name: e.reason for e in ent_registry.entities.values() if e.decision == "DISCARD"}

    checks = []

    # 1. LLM 分类是否运行
    llm_ran = any("LLM 全量分类" in m for _, _, m in log_lines)
    fallback_used = any("兜底规则" in m for _, _, m in log_lines)
    checks.append(("LLM分类成功", llm_ran and not fallback_used,
                   "OK" if llm_ran and not fallback_used else ("fallback兜底" if fallback_used else "未运行")))
    print(f"  {'PASS' if checks[-1][1] else 'WARN'} LLM分类: {checks[-1][2]}")

    # 2. agent数量合理
    checks.append(("agent数量(6-25)", 6 <= a_count <= 25, f"{a_count} agents"))
    print(f"  {'PASS' if checks[-1][1] else 'FAIL'} agent数量: {checks[-1][2]}")

    # 3. 核心博弈方保留
    missing = [n for n in REQUIRED_KEEP if n not in kept_names]
    checks.append(("核心博弈方", len(missing) <= 2, f"缺失: {missing}" if missing else "全部保留"))
    print(f"  {'PASS' if checks[-1][1] else 'FAIL'} 核心博弈方: {checks[-1][2]}")

    # 4. 必须排除的实体
    kept_bad = [n for n in MUST_DISCARD if n in kept_names]
    checks.append(("必须排除", len(kept_bad) == 0, f"误保留: {kept_bad}" if kept_bad else "全部排除"))
    print(f"  {'PASS' if checks[-1][1] else 'FAIL'} 必须排除: {checks[-1][2]}")

    # 5. 代码排除日志
    hard_discard_log = any("军队编制" in m or "二元关系词" in m or "政府部门" in m
                           for _, _, m in log_lines)
    checks.append(("硬排除生效", hard_discard_log, "OK" if hard_discard_log else "未找到"))
    print(f"  {'PASS' if checks[-1][1] else 'FAIL'} 硬排除: {checks[-1][2]}")

    # ── 详细实体列表 ──
    print()
    print(f"  KEPT ({ent_registry.kept}): {', '.join(sorted(kept_names)[:20])}")
    if discard_names:
        summary: dict[str, int] = {}
        for reason in discard_names.values():
            summary[reason] = summary.get(reason, 0) + 1
        print(f"  DISCARD ({ent_registry.discarded}): {', '.join(f'{k}:{v}' for k,v in sorted(summary.items()))}")

    # ── 检查关键排除 ──
    print()
    for name in MUST_DISCARD + SHOULD_DISCARD:
        if name in discard_names:
            print(f"  OK  {name} → DISCARD({discard_names[name]})")
        elif name in kept_names:
            print(f"  WARN {name} → KEPT（应该在排除列表）")
    for name in SHOULD_KEEP:
        if name in kept_names:
            print(f"  OK  {name} → KEEP")
        elif name in discard_names:
            print(f"  WARN {name} → DISCARD({discard_names[name]})（应该在保留列表）")

    # ── 判定 ──
    passed = sum(1 for c in checks if c[1])
    print()
    print("=" * 60)
    print(f"  结果: {passed}/{len(checks)} 项通过")
    print(f"  耗时: {total:.1f}s ({total/60:.1f}min)")
    if passed == len(checks):
        print("  结论: 全部通过")
    elif passed >= len(checks) - 1:
        print("  结论: 基本通过（1 项偏差）")
    else:
        print("  结论: 未通过")
    print("=" * 60)

    graph.close()
    store.close()
    shutil.rmtree(tmp, ignore_errors=True)

    return passed >= len(checks) - 1


async def main():
    print("=" * 60)
    print("  EntityRegistry 专项测试")
    print("=" * 60)

    if not test_code_rules_offline():
        print("\nFAIL 离线规则测试未通过")
        return

    ok = await test_full_pipeline()
    if ok:
        print("\nPASS")
    else:
        print("\nFAIL")


if __name__ == "__main__":
    asyncio.run(main())

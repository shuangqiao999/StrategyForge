"""架构重构专项测试：EntityRegistry + 代码规则 + jieba纯提取。

用大国博弈种子材料（~9000 chars）跑全流程推演，验证：
1. 同一材料多次推演实体列表一致（确定性）
2. EntityRegistry 分类规则生效（排除二元词/职务/部门/集合概念）
3. agent 数量合理（不再 4 或 29 的极端漂移）
4. sorter 降级后不冲突
5. 无崩溃/无 RuntimeError

用法: python tests/test_arch_refactor.py
"""
import sys, os, asyncio, time, uuid, json, shutil

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
from strategy_forge.engine.entity_registry import build_registry, _classify_one
from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.storage.session_store import SessionStore

SOURCE = r"E:\gongxiang\软件\资本论\大国博弈.txt"


def test_classification_logic():
    """离线测试：验证代码规则的分类逻辑。不调用 LLM。"""
    print("=" * 60)
    print("  单元测试：_classify_one 分类逻辑")
    print("=" * 60)
    tests = [
        # (name, type, freq, total, expected_keep)
        ("普京", "Person", 5, 100, True),
        ("特朗普", "Person", 4, 100, True),
        ("吕特", "Person", 3, 100, True),
        ("赖清德", "Person", 3, 100, True),
        ("某个小角色", "Person", 1, 100, False),  # freq < threshold(2)
        ("白宫", "政府部门", 20, 100, False),      # 政府部门排除
        ("财政部", "Organization", 15, 100, False), # 政府部门排除
        ("北约", "Organization", 8, 100, True),     # freq >= 4
        ("联合国", "Organization", 5, 100, False),  # freq=5 < 4? No, 5 >= 4 → True. Wait, 100/50=2, threshold*2=4
        ("中美关系", "Other", 20, 100, False),      # 二元关系词
        ("西方阵营", "Other", 10, 100, False),      # 集合概念
        ("太平洋舰队", "军队编制", 8, 100, False),  # 军队编制
        ("总统", "Person", 50, 100, False),         # 职务头衔
        ("美国", "Country", 30, 100, True),         # freq >= 10
        ("南海", "Location", 15, 100, False),       # 类型排除
        ("俄乌", "Other", 20, 100, False),          # 二元关系词
        ("哈尔科夫", "Location", 5, 100, False),    # 类型排除
    ]
    threshold = max(1, 100 // 50)
    print(f"  threshold={threshold} (total=100)")
    correct = 0
    wrong = []
    for name, etype, freq, total, expected in tests:
        keep, reason = _classify_one(name, etype, freq, total)
        status = "OK" if keep == expected else "FAIL"
        if keep == expected:
            correct += 1
        else:
            wrong.append(f"  {status} {name:12s} {etype:16s} freq={freq:<3} → {str(keep):5s} ({reason}) expected {expected}")
        print(f"  {status} {name:12s} {etype:16s} freq={freq:<3} → {str(keep):5s} ({reason})")

    print(f"\n  结果: {correct}/{len(tests)} 通过")
    if wrong:
        for w in wrong:
            print(w)
    return correct == len(tests)


async def test_full_pipeline(rounds=3):
    """全流程测试：大国博弈种子材料，验证 agent 一致性。"""
    t0 = time.time()
    sid = uuid.uuid4().hex[:12]
    tmp = os.path.join(os.environ["TEMP"], f"forge_arch_{sid}")
    os.makedirs(tmp, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  全流程测试: 大国博弈 ({rounds} 轮)")
    print(f"  Model: {registry.llm_model}")
    print(f"  Session: {sid}")
    print("=" * 60)

    source = open(SOURCE, encoding="utf-8").read()
    print(f"\n  种子材料: {len(source):,} chars")

    session = DeductionSession(id=sid, title="架构重构测试", source_material=source, total_rounds=rounds)
    graph_path = os.path.join(tmp, "graphs", sid, "kuzu")
    db_path = os.path.join(tmp, "session.db")
    store = SessionStore(db_path)
    graph = DeductionGraphStore(graph_path)
    store.create(sid, "架构重构测试", source, {"domain": "narrative", "total_rounds": rounds})

    log_lines: list[str] = []

    def logger(phase: str, msg: str):
        elapsed = time.time() - t0
        log_lines.append(f"[{elapsed:6.1f}s] [{phase:12s}] {msg}")
        print(f"  [{elapsed:6.1f}s] [{phase:12s}] {msg}")

    orch = DeductionOrchestrator(session=session, graph=graph, session_store=store, logger_fn=logger)

    print("\n" + "-" * 60)
    print("  开始推演...")
    print("-" * 60 + "\n")

    try:
        result = await orch.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  [CRASH] {e}")
        result = orch.session

    total = time.time() - t0
    session_data = store.get(sid) or {}
    e_count = graph.count_entities()
    a_count = session_data.get("agent_count", 0) or getattr(result, "agent_count", 0)
    error = session_data.get("error", "") or getattr(result, "error", "")

    # ── 关键验证 ──
    print("\n" + "=" * 60)
    print("  架构验证")
    print("=" * 60)

    checks = []

    # 1. 无崩溃
    checks.append(("无崩溃", not error, error[:80] if error else "OK"))
    print(f"  {'OK' if checks[-1][1] else 'XX'} 无崩溃: {checks[-1][2]}")

    # 2. 有实体产出
    checks.append(("实体产出", e_count > 0, f"{e_count} entities"))
    print(f"  {'OK' if checks[-1][1] else 'XX'} 实体产出: {checks[-1][2]}")

    # 3. agent 数量合理（大国博弈应该 > 3）
    checks.append(("agent数量(>3)", a_count > 3, f"{a_count} agents"))
    print(f"  {'OK' if checks[-1][1] else 'XX'} agent数量(>3): {checks[-1][2]}")

    # 4. agent 数量不过激（< 100）
    checks.append(("agent数量(<100)", a_count < 100, f"{a_count} agents"))
    print(f"  {'OK' if checks[-1][1] else 'XX'} agent数量(<100): {checks[-1][2]}")

    # 5. 代码规则过滤日志
    has_filter_log = any("代码规则过滤" in l for l in log_lines)
    checks.append(("代码规则日志", has_filter_log, "OK" if has_filter_log else "未找到"))
    print(f"  {'OK' if checks[-1][1] else 'XX'} 代码规则日志: {checks[-1][2]}")

    # 6. 注册中心日志
    has_registry_log = any("注册中心" in l for l in log_lines)
    checks.append(("注册中心日志", has_registry_log, "OK" if has_registry_log else "未找到"))
    print(f"  {'OK' if checks[-1][1] else 'XX'} 注册中心日志: {checks[-1][2]}")

    # 7. 没有 sorter active/passive 旧日志
    has_old_sorter_log = any("核心博弈者" in l for l in log_lines)
    checks.append(("无旧sorter日志", not has_old_sorter_log, "OK" if not has_old_sorter_log else "仍有旧格式"))
    print(f"  {'OK' if checks[-1][1] else 'XX'} 无旧sorter日志: {checks[-1][2]}")

    # ── 判定 ──
    passed = sum(1 for c in checks if c[1])
    print(f"\n  结果: {passed}/{len(checks)} 项通过")
    print(f"  耗时: {total:.1f}s ({total/60:.1f}min)")

    # ── 注册中心详情 ──
    if e_count > 0:
        registry = build_registry(graph)
        kept = registry.get_kept()
        print(f"\n  EntityRegistry: {registry.kept}/{registry.total} KEEP")
        print(f"  DISCARD reasons: {', '.join(f'{k}:{v}' for k,v in sorted(registry.discard_reasons.items()))}")
        if kept:
            print(f"  Top agents: {', '.join(e.name for e in kept[:10])}")

    graph.close()
    store.close()
    shutil.rmtree(tmp, ignore_errors=True)

    return passed == len(checks)


async def test_determinism():
    """确定性测试：两次运行 agent 列表应该完全一致。"""
    sid1 = uuid.uuid4().hex[:8]
    sid2 = uuid.uuid4().hex[:8]
    tmp = os.path.join(os.environ["TEMP"], f"forge_det_{sid1}")
    os.makedirs(tmp, exist_ok=True)

    print("\n" + "=" * 60)
    print("  确定性测试：两次运行 agent 列表对比")
    print("=" * 60)

    source = open(SOURCE, encoding="utf-8").read()

    agents_list = []
    for i, sid in enumerate([sid1, sid2]):
        graph_path = os.path.join(tmp, f"graphs_{sid}", sid, "kuzu")
        db_path = os.path.join(tmp, f"session_{sid}.db")
        store = SessionStore(db_path)
        graph = DeductionGraphStore(graph_path)

        session = DeductionSession(id=sid, title=f"确定性测试{i+1}", source_material=source)
        store.create(sid, f"确定性测试{i+1}", source, {"domain": "narrative", "total_rounds": 1})

        orch = DeductionOrchestrator(session=session, graph=graph, session_store=store)
        try:
            await orch.run()
        except Exception:
            pass

        registry = build_registry(graph)
        agents = sorted(e.name for e in registry.get_kept())
        agents_list.append(agents)
        print(f"  运行{i+1}: {len(agents)} agents → {', '.join(agents[:8])}{'...' if len(agents) > 8 else ''}")

        graph.close()
        store.close()

    # Compare
    if agents_list[0] == agents_list[1]:
        print(f"\n  OK 完全一致：两次运行 {len(agents_list[0])} 个 agent 名单相同")
        shutil.rmtree(tmp, ignore_errors=True)
        return True
    else:
        only_first = set(agents_list[0]) - set(agents_list[1])
        only_second = set(agents_list[1]) - set(agents_list[0])
        print(f"\n  XX 不一致！仅运行1有: {only_first}, 仅运行2有: {only_second}")
        shutil.rmtree(tmp, ignore_errors=True)
        return False


async def main():
    print("=" * 60)
    print("  StrategyForge 架构重构专项测试")
    print("=" * 60)

    # 1. 单元测试（离线，秒级）
    if not test_classification_logic():
        print("\nFAIL 单元测试未通过，跳过集成测试")
        return

    # 2. 确定性测试（需跑两次 LLM 推演，每轮约 15-30s）
    # Note: 确定性依赖于 jieba 产出完全相同的实体列表
    # 如果两次推演的 jieba 分词一致，agent 列表应该一致
    det_ok = await test_determinism()
    if not det_ok:
        print("\nWARN 确定性测试显示差异——jieba 分词或 chunk-pass 仍有方差")

    # 3. 全流程测试
    full_ok = await test_full_pipeline(rounds=3)

    print("\n" + "=" * 60)
    if full_ok and det_ok:
        print("  结论: 全部通过 OK")
    elif full_ok:
        print("  结论: 部分通过（全流程OK，确定性有方差）")
    else:
        print("  结论: 未通过 XX")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

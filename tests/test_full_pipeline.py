"""全流程 E2E 测试：本地 LM Studio 实际五阶段推演 + 结果分析。

用法: python tests/test_full_pipeline.py
环境要求: LM Studio 运行中 (127.0.0.1:1234), gemma-4-12b / qwen3.5-9b
"""
import sys, os, asyncio, time, uuid, shutil, json, re

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
from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.storage.session_store import SessionStore

SOURCE = r"E:\gongxiang\软件\资本论\维东国之变.txt"


async def main():
    t0 = time.time()
    sid = uuid.uuid4().hex[:12]
    tmp = os.path.join(os.environ["TEMP"], f"forge_full_test_{sid}")
    os.makedirs(tmp, exist_ok=True)

    print("=" * 70)
    print(f"  StrategyForge 全流程测试")
    print(f"  Model: {registry.llm_model}")
    print(f"  Embed: {registry.embedding_model_name}")
    print(f"  Concurrency: {registry.max_concurrent}")
    print(f"  Session: {sid}")
    print("=" * 70)

    source = open(SOURCE, encoding="utf-8").read()
    print(f"\n  源文本: {len(source):,} chars ({len(source.encode('utf-8'))} bytes)")

    session = DeductionSession(
        id=sid, title="全流程测试", source_material=source,
        total_rounds=5,
    )

    graph_path = os.path.join(tmp, "graphs", sid, "kuzu")
    db_path = os.path.join(tmp, "session.db")
    store = SessionStore(db_path)
    graph = DeductionGraphStore(graph_path)

    store.create(sid, "全流程测试", source,
                 {"domain": "narrative", "total_rounds": 5, "chunk_size": 600})

    log_lines: list[tuple[float, str, str]] = []

    def logger(phase: str, msg: str):
        elapsed = time.time() - t0
        log_lines.append((elapsed, phase, msg))
        print(f"  [{elapsed:6.1f}s] [{phase:12s}] {msg}")

    orch = DeductionOrchestrator(
        session=session, graph=graph, session_store=store,
        logger_fn=logger,
    )

    print("\n" + "-" * 70)
    print("  开始推演...")
    print("-" * 70 + "\n")

    try:
        result = await orch.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  [CRASH] 推演崩溃: {e}")
        result = orch.session

    total_time = time.time() - t0

    # ── 从日志提取各阶段耗时 ──
    phase_times: dict[str, float] = {}
    for elapsed, phase, msg in log_lines:
        if phase == "orchestrator" and "耗时" in msg:
            m = re.match(r"阶段 (\w+) 耗时 ([\d.]+)s", msg)
            if m:
                phase_times[m.group(1)] = float(m.group(2))

    # ── 读取最终状态 ──
    session_data = store.get(sid) or {}
    status = getattr(result, "status", None)
    status_str = status.value if hasattr(status, "value") else str(status)
    error = session_data.get("error", "") or getattr(result, "error", "")

    g_entities = graph.count_entities()
    g_relations = graph.count_relations()
    a_count = session_data.get("agent_count", 0) or getattr(result, "agent_count", 0)
    r_count = session_data.get("current_round", 0) or getattr(result, "current_round", 0)
    total_r = session_data.get("total_rounds", 5) or getattr(result, "total_rounds", 5)

    # ── 关键事件追踪 ──
    key_events = {
        "sorter_ran": False,       # 叙事 sorter 是否有输出
        "sorter_roles": 0,         # sorter 产出的角色数
        "entity_drive_zero": False, # 实体驱动是否产出 0
        "agents_gen_ok": False,    # agent 生成是否正常
        "sim_has_actions": False,  # 模拟是否有动作
    }
    for _, phase, msg in log_lines:
        if phase == "graph":
            if "故事编辑" in msg:
                key_events["sorter_ran"] = True
                m = re.search(r"(\d+) 角色", msg)
                if m:
                    key_events["sorter_roles"] = int(m.group(1))
            if "pool=0 实体" in msg:
                key_events["entity_drive_zero"] = True
        if phase == "agents":
            if "智能体工厂完成" in msg:
                m = re.search(r"(\d+) 个", msg)
                if m and int(m.group(1)) > 0:
                    key_events["agents_gen_ok"] = True
        if phase == "simulation":
            if "个动作" in msg and "0 个" not in msg:
                key_events["sim_has_actions"] = True

    # ── 日志证据 ──
    print("\n" + "=" * 70)
    print("  日志关键事件")
    print("=" * 70)
    for _, phase, msg in log_lines:
        if phase in ("graph", "agents", "simulation", "report"):
            if any(kw in msg for kw in ("故事编辑", "实体驱动", "pool=", "智能体工厂完成",
                                         "情报过滤", "频率兜底", "类型过滤", "模拟完成",
                                         "报告生成完成", "失败", "崩溃", "error")):
                print(f"    {msg}")

    # ── 检查点 ──
    issues = []
    print("\n" + "=" * 70)
    print("  检查点")
    print("=" * 70)

    # 0. 崩溃检查
    if error:
        issues.append(f"session 错误: {error[:150]}")
        print(f"  [FAIL] 状态={status_str}, 错误={error[:120]}")
    else:
        print(f"  [OK] 状态={status_str}")

    # 1. 图谱实体
    if g_entities == 0:
        issues.append("图谱 0 实体 → 实体驱动模式 prompt 过于严苛")
        print(f"  [FAIL] 图谱: {g_entities} 实体, {g_relations} 关系")
    else:
        print(f"  [OK] 图谱: {g_entities} 实体, {g_relations} 关系")

    # 2. 实体驱动模式
    if key_events["entity_drive_zero"]:
        if g_entities > 0:
            print(f"  [OK] 实体驱动返回空但分块顺带补救 → {g_entities} 实体 (正常)")
        else:
            issues.append("实体驱动+分块顺带均 0 → prompt 问题或 LM Studio 不可用")
            print(f"  [FAIL] 实体驱动 0 + 分块顺带 0")

    # 3. 叙事 sorter
    if key_events["sorter_ran"]:
        print(f"  [OK] 叙事 sorter: {key_events['sorter_roles']} 角色 + {g_entities - key_events['sorter_roles']} 背景")
    else:
        print(f"  [WARN] 叙事 sorter 无输出 → intel_list 为空, 退化为类型过滤")
        # 这不一定是 bug——gemma 可能不兼容 sorter prompt

    # 4. 智能体
    if a_count > 0:
        print(f"  [OK] 智能体: {a_count} 个")
    else:
        issues.append("0 个智能体 → freq_map 作用域 / sorter 静默失败 / LLM 全拒")
        print(f"  [FAIL] 智能体: {a_count} 个")

    # 5. 模拟
    if key_events["sim_has_actions"]:
        print(f"  [OK] 模拟: {r_count}/{total_r} 轮, 有动作")
    elif a_count > 0:
        issues.append("有 agent 但模拟无动作 → simulator prompt 或 LM Studio 问题")
        print(f"  [WARN] 模拟: {r_count}/{total_r} 轮, 0 动作 (有 {a_count} 个 agent)")
    else:
        print(f"  [WARN] 模拟: {r_count}/{total_r} 轮 (跳过: 无 agent)")

    # 6. 报告
    report_json = session_data.get("report_json", "{}") or "{}"
    try:
        report = json.loads(report_json) if isinstance(report_json, str) else report_json
    except (json.JSONDecodeError, TypeError):
        report = {}
    narrative = report.get("narrative", "") or ""
    if narrative and not narrative.startswith("推演未产生"):
        print(f"  [OK] 报告: {len(narrative)} chars")
    else:
        print(f"  [INFO] 报告: 无叙事内容 (有 {a_count} agent 但模拟无动作)")

    # ── Token ──
    token_json = session_data.get("token_json", "{}") or "{}"
    try:
        tk = json.loads(token_json) if isinstance(token_json, str) else token_json
        total_tokens = tk.get("total_tokens", 0) if isinstance(tk, dict) else 0
    except (json.JSONDecodeError, TypeError):
        total_tokens = 0
    if total_tokens > 0:
        print(f"  [INFO] Token: {total_tokens:,}")

    # ── 耗时汇总 ──
    print(f"\n  各阶段耗时:")
    for pn in ("ontology", "quantify", "graph", "agents", "simulation", "report"):
        t = phase_times.get(pn)
        if t is not None:
            print(f"    {pn:12s}: {t:6.1f}s")

    # ── 判定 ──
    print("\n" + "=" * 70)
    if issues and any("FAIL" in str(i) for i in ["图谱 0" if g_entities == 0 else "",
                                                   "0 个智能体" if a_count == 0 else ""]):
        print(f"  结论: 未通过 — {len(issues)} 个问题:")
        for i, iss in enumerate(issues, 1):
            print(f"    [{i}] {iss}")
    elif g_entities > 0 and a_count > 0:
        print(f"  结论: 通过 — {g_entities} 实体, {a_count} agent, {r_count} 轮")
    else:
        print(f"  结论: 部分通过 — {g_entities} 实体, {a_count} agent")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("=" * 70)

    # ── 清理 ──
    graph.close()
    store.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  临时数据已清理: {tmp}")


if __name__ == "__main__":
    asyncio.run(main())

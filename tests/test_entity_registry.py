"""EntityRegistry 专项测试：验证实体识别归类的准确性。

用法: python tests/test_entity_registry.py
"""
import sys, os, asyncio, time, uuid, shutil, argparse

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

SOURCES = {
    "daiguo": r"E:\gongxiang\软件\资本论\大国博弈.txt",
    "chongzhen": r"E:\gongxiang\软件\资本论\崇祯十五年.txt",
}

EXPECTED = {
    "daiguo": {
        "required": {"中国", "美国", "俄罗斯", "伊朗", "以色列"},
        "should_keep": {"日本", "法国", "菲律宾", "乌克兰"},
        "must_discard": {"中美关系", "俄乌冲突", "俄乌", "乌军", "俄军"},
        "domain": "geo_strategy",
    },
    "chongzhen": {
        "required": {"大明", "崇祯", "朱由检", "周延儒", "温体仁", "李自成", "皇太极", "张献忠"},
        "should_keep": {"杨嗣昌", "黄宗羲", "陈子龙", "阮大铖"},
        "must_discard": {"秦淮河", "北京城", "紫禁城", "流寇", "官军", "辽东", "河南", "陕西", "湖广"},
        "domain": "history",
    },
}


async def test_classify(material: str):
    """全流程实体识别归类测试。"""
    info = EXPECTED[material]
    source_path = SOURCES[material]
    t0 = time.time()
    sid = uuid.uuid4().hex[:12]
    tmp = os.path.join(os.environ["TEMP"], f"forge_er_{sid}")
    os.makedirs(tmp, exist_ok=True)

    print("=" * 60)
    print(f"  实体识别归类测试 — {material}")
    print(f"  Model: {registry.llm_model}")
    print("=" * 60)

    source = open(source_path, encoding="utf-8").read()
    print(f"  种子材料: {len(source):,} chars")

    session = DeductionSession(id=sid, title="ER测试", source_material=source, total_rounds=2)
    graph_path = os.path.join(tmp, "graphs", sid, "kuzu")
    db_path = os.path.join(tmp, "session.db")
    store = SessionStore(db_path)
    graph = DeductionGraphStore(graph_path)
    domain = info.get("domain", "narrative")
    store.create(sid, "ER测试", source, {"domain": domain, "total_rounds": 2})

    agent_names: list[str] = []

    def logger(phase: str, msg: str):
        elapsed = time.time() - t0
        print(f"  [{elapsed:6.1f}s] [{phase:12s}] {msg}")
        if phase == "agents":
            import re
            m = re.search(r'\] (\S+):', msg)
            if m:
                agent_names.append(m.group(1))

    orch = DeductionOrchestrator(session=session, graph=graph, session_store=store, logger_fn=logger)

    print("\n" + "-" * 60)
    print("  开始推演...")
    print("-" * 60 + "\n")

    try:
        await orch.run()
    except Exception as e:
        import traceback
        traceback.print_exc()

    total = time.time() - t0

    required = info["required"]
    should_keep = info["should_keep"]
    must_discard = info["must_discard"]

    kept = set(agent_names)
    missing = required - kept
    extra = must_discard & kept
    bonus = should_keep & kept

    print()
    print("=" * 60)
    print("  归类结果")
    print("=" * 60)
    print(f"  Agent({len(agent_names)}): {', '.join(agent_names)}")
    print(f"  核心方(需保留 {len(required)}): {'OK' if not missing else 'MISS: ' + ', '.join(missing)}")
    print(f"  次要方(应保留 {len(should_keep)}): {len(bonus)}/{len(should_keep)} → {', '.join(bonus) if bonus else 'none'}")
    print(f"  排除词: {'OK' if not extra else 'LEAK: ' + ', '.join(extra)}")
    print(f"  耗时: {total:.1f}s ({total/60:.1f}min)")

    ok = len(missing) <= 2 and len(extra) == 0 and len(agent_names) >= 3
    print(f"\n  结论: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)

    graph.close()
    store.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("material", nargs="?", choices=["daiguo", "chongzhen"], default="chongzhen",
                        help="测试素材: daiguo(大国博弈) | chongzhen(崇祯十五年)")
    args = parser.parse_args()
    asyncio.run(test_classify(args.material))

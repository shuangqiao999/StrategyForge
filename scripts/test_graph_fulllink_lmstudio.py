"""Kuzu 图库充分利用 功能测试 — 连接本地 LM Studio (9B 对话)。

用法（项目根目录）：
    python scripts/test_graph_fulllink_lmstudio.py

覆盖：
  A. 关系反哺：get_entity_neighbors 返回 relation/weight/name；SimulationEngine 开局
     从 Kuzu 预取盟友/对手 → _rel_context；并播种 reasoner._trust_matrix（定性）。
  B. 量化补写：量化轮(persist_events=True)写入 Kuzu Event 节点 + ACTED 边；
     优化器隔离(persist_events=False)不写。
  C. 死代码清理：新建库 schema 无 Chunk/MENTIONS；add_mention/upsert_chunk 已移除。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_graph_test_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _check_server() -> bool:
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/models", timeout=5).read()
        return True
    except Exception as e:
        print(f"[致命] 无法连接 LM Studio: {base}/models -> {e}")
        return False


def _discover_chat_model() -> str:
    if os.environ.get("FORGE_LLM_MODEL"):
        return os.environ["FORGE_LLM_MODEL"]
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        chat = [m for m in ids if "embed" not in m.lower()]
        pick = next((m for m in chat if "9b" in m.lower()), None) or (chat[0] if chat else "local-9b")
        os.environ["FORGE_LLM_MODEL"] = pick
        return pick
    except Exception:
        os.environ.setdefault("FORGE_LLM_MODEL", "local-9b")
        return os.environ["FORGE_LLM_MODEL"]


_AGENTS = [("A", "甲军团", "勇猛激进的统帅"),
           ("B", "乙军团", "老成持重的统帅"),
           ("C", "丙军团", "机动灵活的统帅")]


def _build_graph(subdir: str):
    from strategy_forge.storage.graph_store import DeductionGraphStore
    g = DeductionGraphStore(os.path.join(_TMP, "graphs", subdir, "kuzu"))
    for eid, name, _ in _AGENTS:
        g.upsert_entity(eid, name, "Person", f"{name}的描述")
        g.upsert_agent_node(eid, name, "persona", "bg", "[]")
    # 甲-盟友->乙；甲-敌对->丙
    g.upsert_relation("A", "B", "盟友", weight=3.0, evidence="并肩作战")
    g.upsert_relation("A", "C", "敌对", weight=2.0, evidence="长期交战")
    return g


def _mk_agents():
    from strategy_forge.engine.models import DeductionAgentProfile
    return [DeductionAgentProfile(entity_id=eid, name=name, persona=persona,
                                  background="", goals=["击败对手，保全本部"])
            for eid, name, persona in _AGENTS]


def _new_engine(graph, persist: bool, rounds: int):
    from strategy_forge.engine.rule_engine import RuleEngine
    from strategy_forge.engine.simulator import SimulationEngine
    re_engine = RuleEngine.from_domain("military")
    agents = _mk_agents()
    states = {a.entity_id: re_engine.init_state(a.entity_id, a.name) for a in agents}
    eng = SimulationEngine(
        agents=agents, graph=graph, total_rounds=rounds, log_fn=lambda p, m: None,
        preprocessor=None, pre_goals=["攻守兼顾"],
        seed=11, temperature=0.6, persist_events=persist, max_concurrent=2,
        rule_engine=re_engine, states=states, enable_narrate=False,
        enable_multi_action=False,
    )
    return eng


def _stage_a() -> None:
    print("\n=== 阶段A：关系反哺推理 + 信任播种 ===")
    g = _build_graph("a")
    try:
        nb = g.get_entity_neighbors("A", max_depth=1)
        names = {x["name"]: x["relation"] for x in nb["neighbors"]}
        assert "乙军团" in names and "丙军团" in names, f"邻居缺失: {names}"
        assert names["乙军团"] == "盟友" and names["丙军团"] == "敌对", f"关系错误: {names}"
        print(f"  get_entity_neighbors(结构化) OK: {names}")

        eng = _new_engine(g, persist=False, rounds=1)
        rc = eng._rel_context.get("A", {})
        assert "乙军团" in rc.get("allies", []), f"盟友识别失败: {rc}"
        assert "丙军团" in rc.get("opponents", []), f"对手识别失败: {rc}"
        assert rc.get("summary"), "关系摘要为空"
        print(f"  _rel_context[甲]: {rc['summary']}")
        assert eng.reasoner.get_trust("A", "乙军团") > 0, "盟友信任未播种为正"
        assert eng.reasoner.get_trust("A", "丙军团") < 0, "对手信任未播种为负"
        print(f"  信任播种 OK: 甲→乙={eng.reasoner.get_trust('A','乙军团'):+.1f}, "
              f"甲→丙={eng.reasoner.get_trust('A','丙军团'):+.1f}")
        print("  [OK] 阶段A 通过")
    finally:
        g.close()


def _stage_c() -> None:
    print("\n=== 阶段C：死代码/冗余表清理 ===")
    g = _build_graph("c")
    try:
        chunk_exists = True
        try:
            g.query("MATCH (c:Chunk) RETURN count(c)")
        except Exception:
            chunk_exists = False
        assert not chunk_exists, "Chunk 表仍存在（未清理）"
        part_exists = True
        try:
            g.query("MATCH (:Agent)-[r:PARTICIPATES]->(:Entity) RETURN count(r)")
        except Exception:
            part_exists = False
        assert not part_exists, "PARTICIPATES 表仍存在（未清理）"
        assert not hasattr(g, "add_mention"), "add_mention 未删除"
        assert not hasattr(g, "upsert_chunk"), "upsert_chunk 未删除"
        print("  [OK] schema 无 Chunk/MENTIONS/PARTICIPATES；add_mention/upsert_chunk 已移除")
    finally:
        g.close()


def _stage_d() -> None:
    """因果链确定性归因（无 LLM，精确核对数值真值）。"""
    from strategy_forge.engine.rule_engine import RuleEngine
    print("\n=== 阶段D：因果链确定性归因（无 LLM）===")
    g = _build_graph("d")
    try:
        re_engine = RuleEngine.from_domain("military")
        a = re_engine.init_state("A", "甲军团")
        b = re_engine.init_state("B", "乙军团")
        snap = {"A": a, "B": b}
        n2i = {"甲军团": "A", "乙军团": "B"}
        decisions = [{"actor_id": "A", "action_type": "attack", "intensity": 1.0, "target": "乙军团"}]
        deltas, interactions = re_engine.resolve_round(
            snap, decisions, n2i, None, collect_interactions=True)
        assert interactions, "未产生交互归因"
        it = interactions[0]
        assert it["actor"] == "A" and it["target"] == "B" and it["action"] == "attack", it
        exp = {k: v * 1.0 for k, v in re_engine.pack["target_effects"]["attack"].items()}
        assert it["deltas"] == exp, (it["deltas"], exp)
        print(f"  resolve_round 交互归因正确: 甲→乙 {it['deltas']}")
        eid = "evt-det1"
        g.add_event(eid, "甲军团进攻乙军团", "attack", "2026-01-01T00:00:00", "A",
                    round_number=1, target_id="B")
        g.add_acted("A", eid, "attack", "2026-01-01T00:00:00")
        g.add_targets(eid, "B")
        for m, amt in it["deltas"].items():
            g.add_caused(eid, "B", m, float(amt))
        attr = g.get_outcome_attribution("B")
        contrib = {c["source"]: c for c in attr["contributors"]}
        assert "甲军团" in contrib, attr
        exp_harm = round(sum(v for v in exp.values() if v < 0), 2)
        assert abs(contrib["甲军团"]["harm"] - exp_harm) < 1e-6, (contrib["甲军团"], exp_harm)
        print(f"  确定性归因校验: 乙的致衰主因=甲军团 harm={contrib['甲军团']['harm']}（=数值真值 {exp_harm}）")
        summ = g.get_causal_summary()
        assert any(s["source"] == "甲军团" and s["target"] == "乙军团" for s in summ), summ
        sub = g.get_causal_subgraph()
        assert any(n["kind"] == "event" for n in sub["nodes"]), sub
        assert any(l["type"] == "CAUSED" for l in sub["links"]), sub
        print(f"  因果摘要/子图 OK: summary={len(summ)} 条, 子图节点={len(sub['nodes'])}, 链接={len(sub['links'])}")
        print("  [OK] 阶段D 通过")
    finally:
        g.close()


async def _stage_b() -> None:
    print("\n=== 阶段B：量化补写 Event/ACTED + 优化器隔离 ===")
    # persist=True → 写入
    g1 = _build_graph("b_persist")
    try:
        eng = _new_engine(g1, persist=True, rounds=2)
        for rnd in range(1, 3):
            await eng.run_round(rnd)
        acted = g1.query("MATCH (:Agent)-[r:ACTED]->(:Event) RETURN count(r)")[0][0]
        events = g1.query("MATCH (e:Event) RETURN count(e)")[0][0]
        assert acted > 0 and events > 0, f"量化轮未写 Event/ACTED: acted={acted}, events={events}"
        print(f"  persist=True 写入生效：Event={events}, ACTED={acted}")
        tls = g1.get_agent_timelines()
        seq = g1.get_event_sequence()
        assert tls and sum(len(t["actions"]) for t in tls) > 0, "get_agent_timelines 读取方失效"
        assert len(seq) > 0, "get_event_sequence 读取方失效"
        print(f"  读取方生效：时间线 agent={len(tls)}, 事件序列={len(seq)} 条")
        caused = g1.query("MATCH (:Event)-[c:CAUSED]->(:Entity) RETURN count(c)")[0][0]
        print(f"  因果边 CAUSED={caused}（量化轮目标行动的确定性归因）")
    finally:
        g1.close()
    # persist=False → 隔离不写
    g2 = _build_graph("b_isolated")
    try:
        eng2 = _new_engine(g2, persist=False, rounds=2)
        for rnd in range(1, 3):
            await eng2.run_round(rnd)
        acted2 = g2.query("MATCH (:Agent)-[r:ACTED]->(:Event) RETURN count(r)")[0][0]
        assert acted2 == 0, f"隔离模式不应写 ACTED，但 acted={acted2}"
        assert len(g2.get_event_sequence()) == 0, "隔离模式事件序列应为空"
        print(f"  persist=False 隔离保持：ACTED={acted2}（未写入），事件序列=0")
    finally:
        g2.close()
    print("  [OK] 阶段B 通过")


async def main() -> int:
    print("=== Kuzu 图库充分利用 测试（LM Studio）===")
    if not _check_server():
        return 2
    print("对话模型:", _discover_chat_model())
    _stage_a()
    _stage_c()
    _stage_d()
    await _stage_b()
    print("\n[全部通过] Kuzu 图库全链路测试 OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

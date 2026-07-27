"""测试 entity_id 保留 + 关系反哺 —— 绕过 LLM 层直接验证核心逻辑。

用本地 LM Studio gemma-4-12b 跑关系分类关键词 + 图邻居查询验证。
"""
import asyncio
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_MODEL", "google/gemma-4-12b-qat")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategy_forge.storage.graph_store import DeductionGraphStore

_pass, _fail = 0, 0

def check(desc: str, condition: bool, detail: str = ""):
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  [PASS] {desc}" + (f"  ({detail})" if detail else ""))
    else:
        _fail += 1
        print(f"  [FAIL] {desc}" + (f"  ({detail})" if detail else ""))

def summary():
    total = _pass + _fail
    print(f"\n{'='*50}")
    print(f"  Result: {_pass}/{total} passed, {_fail} failed")
    if _fail:
        print(f"  FAILED: {_fail} checks failed")
    else:
        print(f"  ALL PASSED!")


# ── 测试素材 ──
# 故意用复杂的别名关系：同一个人有多个称呼（米国=美国，睡王=拜登，毛熊=俄罗斯）
# 关系类型用中文，测试关键词分类是否能正确识别盟友/对手
TEST_TEXT = """国际局势风云变幻。美国（又称米国）总统拜登（人称睡王）与俄罗斯（又称毛熊）
总统普京在中东博弈。拜登联合法国总统马克龙、英国首相约翰逊，组成西方联盟，共同制裁俄罗斯。
普京则与伊朗最高领袖哈梅内伊结盟，对抗西方压力。中国（又称中方）周旋于各方之间，
既不结盟也不对抗，保持战略自主。金正恩的朝鲜（又称北韩）多次试射导弹，挑衅美韩同盟。
泽连斯基的乌克兰（又称基辅政权）在东线与俄军激战，同时寻求西方更多军事援助。"""

def _make_id(name: str, etype: str = "") -> str:
    return hashlib.md5(f"{name}:{etype}".encode()).hexdigest()[:12]


def build_test_graph(tmpdir: Path) -> DeductionGraphStore:
    graph = DeductionGraphStore(tmpdir / "kuzu")

    entities = [
        # name, type, description
        ("美国", "Country", "世界超级大国，北约领袖"),
        ("俄罗斯", "Country", "横跨欧亚的军事强国"),
        ("拜登", "Leader", "美国总统，决策者"),
        ("普京", "Leader", "俄罗斯总统，强人领袖"),
        ("马克龙", "Leader", "法国总统，欧洲协调者"),
        ("约翰逊", "Leader", "英国首相，美国盟友"),
        ("哈梅内伊", "Leader", "伊朗最高领袖"),
        ("伊朗", "Country", "中东什叶派大国"),
        ("中国", "Country", "世界第二大经济体，战略自主"),
        ("金正恩", "Leader", "朝鲜最高领导人"),
        ("朝鲜", "Country", "东北亚威权国家"),
        ("泽连斯基", "Leader", "乌克兰总统"),
        ("乌克兰", "Country", "东欧反俄前线国家"),
    ]
    # build entity type map for correct relation ID matching
    entity_types = {name: etype for name, etype, _ in entities}
    for name, etype, desc in entities:
        graph.upsert_entity(_make_id(name, etype), name, etype, desc)

    relations = [
        # src, tgt, relation, evidence
        # 盟友关系
        ("拜登", "马克龙", "结盟", "北约+西方联盟"),
        ("拜登", "约翰逊", "同盟", "美英特殊关系"),
        ("美国", "法国", "盟友", "北约"),
        ("美国", "英国", "支持", "英美同盟"),
        ("普京", "哈梅内伊", "结盟", "俄伊战略协作"),
        ("俄罗斯", "伊朗", "盟友", "军事合作"),
        ("伊朗", "哈梅内伊", "效忠", "最高领袖"),
        ("美国", "泽连斯基", "支持", "援乌抗俄"),
        ("美国", "乌克兰", "合作", "北约东扩"),
        # 对手关系
        ("拜登", "普京", "对抗", "美俄冲突"),
        ("美国", "俄罗斯", "对手", "大国竞争"),
        ("美国", "朝鲜", "敌对", "美朝对立"),
        ("金正恩", "拜登", "威胁", "核导挑衅"),
        ("俄罗斯", "乌克兰", "冲突", "俄乌战争"),
        ("普京", "泽连斯基", "敌对", "领土争端"),
        ("朝鲜", "韩国", "对抗", "朝韩对立"),
        # 中立/其他
        ("中国", "美国", "竞争", "战略竞争但不结盟不对抗"),
        ("中国", "俄罗斯", "合作", "全面战略协作伙伴"),
        ("中国", "伊朗", "贸易", "石油贸易"),
        ("朝鲜", "中国", "盟友", "传统友谊"),
    ]
    for src, tgt, rel, ev in relations:
        stype = entity_types.get(src, "")
        ttype = entity_types.get(tgt, "")
        sid = _make_id(src, stype) if src in entity_types else _make_id(src, "")
        tid = _make_id(tgt, ttype) if tgt in entity_types else _make_id(tgt, "")
        try:
            graph.upsert_relation(sid, tid, rel, evidence=ev)
        except Exception as e:
            print(f"  WARNING: relation {src}->{tgt} failed: {e}")

    e_count = graph.count_entities()
    r_count = graph.count_relations()
    print(f"  Built graph: {e_count} entities, {r_count} relations")
    check("Graph entities created", e_count >= 13, str(e_count))
    check("Graph relations created", r_count >= 15, str(r_count))
    return graph


# ── 测试1: entity_id 在 registry 中是否正确回填 ──
def test_registry_id_backfill(graph: DeductionGraphStore):
    """模拟 build_registry 中的 entity_list 构建逻辑，验证 ID 回填正确。"""
    # 从 Kuzu 读取原始 fragment（模拟 build_registry 第一步）
    result = graph._conn.execute(
        f"MATCH (e:{graph.NODE_TABLE}) RETURN e.id, e.name, e.type, e.description"
    )
    raw_fragments = []
    while result.has_next():
        r = result.get_next()
        raw_fragments.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3]})

    print(f"\n  Raw fragments from Kuzu: {len(raw_fragments)}")

    # 构建 name->id 映射（修复后的逻辑）
    name_to_id = {}
    for frag in raw_fragments:
        fid = (frag.get("id") or "").strip()
        fname = (frag.get("name") or "").strip()
        if fid and fname:
            name_to_id[fname] = fid

    # 模拟 Layer1 输出（LLM 不返回 id，只返回 name/type/description）
    mock_l1_output = [
        {"name": "美国", "aliases": ["米国", "USA"], "type": "Country",
         "description": "世界超级大国"},
        {"name": "俄罗斯", "aliases": ["毛熊", "俄国"], "type": "Country",
         "description": "军事强国"},
        {"name": "拜登", "aliases": ["睡王"], "type": "Leader",
         "description": "美国总统"},
        {"name": "中国", "aliases": ["中方"], "type": "Country",
         "description": "战略自主"},
    ]

    # 回填 ID（修复后的 entity_list 逻辑）
    for e in mock_l1_output:
        nm = e["name"]
        # 1) 同名匹配
        kuzu_id = name_to_id.get(nm, "")
        # 2) 别名匹配
        if not kuzu_id:
            for a in e.get("aliases", []):
                if a in name_to_id:
                    kuzu_id = name_to_id[a]
                    break
        e["id"] = kuzu_id

    # 验证
    for e in mock_l1_output:
        has_id = bool(e.get("id", "").strip())
        check(f"  {e['name']}: has Kuzu ID", has_id, e.get("id", "MISSING")[:12])


# ── 测试2: 关系分类关键词 ──
def test_relation_keywords():
    from strategy_forge.engine.simulator import SimulationEngine
    classify = SimulationEngine._classify_relation

    tests = [
        # Ally
        ("同盟", "ally"), ("盟友", "ally"), ("结盟", "ally"),
        ("支持", "ally"), ("合作", "ally"), ("效忠", "ally"),
        ("部下", "ally"), ("下属", "ally"), ("追随", "ally"),
        ("友军", "ally"), ("同盟关系", "ally"),
        # Foe
        ("敌对", "foe"), ("敌人", "foe"), ("对抗", "foe"),
        ("对手", "foe"), ("竞争", "foe"), ("冲突", "foe"),
        ("背叛", "foe"), ("攻击", "foe"), ("威胁", "foe"),
        ("死敌", "foe"), ("竞争关系", "foe"),
        # Neutral
        ("制裁", "neutral"), ("贸易", "neutral"), ("父子", "neutral"),
        ("位于", "neutral"), ("隶属于", "neutral"), ("战略协作", "neutral"),
    ]

    failed = []
    for rel, expected in tests:
        result = classify(rel)
        if result != expected:
            failed.append((rel, result, expected))

    if failed:
        print(f"  Failed {len(failed)} items:")
        for rel, got, exp in failed:
            print(f"    FAIL: '{rel}' -> {got} (expected {exp})")

    ok = len(tests) - len(failed)
    check(f"Keyword classification ({ok}/{len(tests)})", len(failed) == 0)


# ── 测试3: _build_relationship_context 完整流程 ──
def test_relationship_context(graph: DeductionGraphStore):
    from strategy_forge.engine.simulator import SimulationEngine
    from strategy_forge.engine.models import DeductionAgentProfile
    from strategy_forge.engine.strategic_reasoner import StrategicReasoner
    from strategy_forge.core.providers import registry as _reg

    # 构建 agent 列表（模拟 tier1 实体）
    agent_data = [
        ("拜登", "Leader", "美国总统，西方联盟领袖"),
        ("普京", "Leader", "俄罗斯总统"),
        ("美国", "Country", "世界超级大国"),
        ("俄罗斯", "Country", "欧亚军事强国"),
        ("中国", "Country", "战略自主的大国"),
        ("金正恩", "Leader", "朝鲜最高领导人"),
    ]

    entity_types = {
        "拜登": "Leader", "普京": "Leader", "美国": "Country",
        "俄罗斯": "Country", "中国": "Country", "金正恩": "Leader",
    }

    agents = []
    for name, etype, desc in agent_data:
        eid = _make_id(name, etype)
        agents.append(DeductionAgentProfile(
            entity_id=eid, name=name, persona=desc,
            background=desc, goals=[], entity_type=etype,
        ))

    print(f"\n  Agent count: {len(agents)}")

    # 创建模拟引擎（不跑全流程，只测 _build_relationship_context）
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.graph = graph
    engine.agents = agents
    engine._log = lambda p, m: print(f"  [{p}] {m[:120]}")
    engine._rel_context = {}
    engine.reasoner = StrategicReasoner(candidate_count=_reg.candidate_count)
    engine._states = {}

    engine._build_relationship_context()

    total = len(engine._rel_context)
    with_rels = sum(1 for v in engine._rel_context.values() if v.get("summary"))
    with_allies = sum(1 for v in engine._rel_context.values() if v.get("allies"))
    with_foes = sum(1 for v in engine._rel_context.values() if v.get("opponents"))

    print(f"\n  Relationship feedback: {total} agents, {with_rels} with graph relations")
    print(f"    With allies: {with_allies}, With opponents: {with_foes}")

    for a in agents:
        ctx = engine._rel_context.get(a.entity_id, {})
        summary = ctx.get("summary", "(none)")
        print(f"    {a.name}: {summary}")

    check("Relationship feedback active", with_rels > 0, f"{with_rels} agents")
    check("Ally detection", with_allies > 0, f"{with_allies} agents")
    check("Foe detection", with_foes > 0, f"{with_foes} agents")

    # 具体关系验证
    # 美国/拜登 应有对手: 俄罗斯/普京
    # 拜登 应有盟友: 马克龙, 约翰逊
    biden_id = _make_id("拜登", "Leader")
    putin_id = _make_id("普京", "Leader")
    us_id = _make_id("美国", "Country")
    ru_id = _make_id("俄罗斯", "Country")

    score = 0

    # 拜登 -> 对手: 普京
    biden_opps = engine._rel_context.get(biden_id, {}).get("opponents", [])
    if "普京" in biden_opps:
        score += 1
        print(f"    [OK] Biden -> opponent -> Putin: {biden_opps}")
    else:
        print(f"    [FAIL] Biden opponents={biden_opps} (expected: Putin)")

    # 普京 -> 对手: 拜登
    putin_opps = engine._rel_context.get(putin_id, {}).get("opponents", [])
    if "拜登" in putin_opps:
        score += 1
        print(f"    [OK] Putin -> opponent -> Biden: {putin_opps}")
    else:
        print(f"    [FAIL] Putin opponents={putin_opps} (expected: Biden)")

    # 美国 -> 对手: 俄罗斯
    us_opps = engine._rel_context.get(us_id, {}).get("opponents", [])
    if "俄罗斯" in us_opps:
        score += 1
        print(f"    [OK] US -> opponent -> Russia: {us_opps}")
    else:
        print(f"    [FAIL] US opponents={us_opps} (expected: Russia)")

    # 中国应为中立(无极化的盟友/对手，除非朝鲜盟友)
    china_id = _make_id("中国", "Country")
    china_allies = engine._rel_context.get(china_id, {}).get("allies", [])
    china_opps = engine._rel_context.get(china_id, {}).get("opponents", [])
    if "朝鲜" in china_allies:
        score += 1
        print(f"    [OK] China -> ally -> North Korea")
    else:
        print(f"    [INFO] China allies={china_allies}, opponents={china_opps}")

    check(f"Specific relations correct", score >= 3, f"{score}/4 correct")


# ── 测试4: _seed_polarization_relations 极化补全 ──
def test_polarization_relations(graph: DeductionGraphStore):
    """验证：即使无图谱关系，polarization 互补逻辑也能正确划分阵营。"""
    from strategy_forge.engine.simulator import SimulationEngine
    from strategy_forge.engine.models import DeductionAgentProfile, EntityState
    from strategy_forge.engine.strategic_reasoner import StrategicReasoner
    from strategy_forge.core.providers import registry as _reg

    # 创建 agent（无图谱关系的那些，靠 polarization 区分）
    agent_data = [
        ("特朗普", "Leader", 0.8),      # +极化 → 和拜登同阵营
        ("哈里斯", "Leader", 0.9),
        ("梅德韦杰夫", "Leader", -0.7),  # -极化 → 和普京同阵营
        ("朔尔茨", "Leader", 0.5),       # +极化
        ("欧尔班", "Leader", -0.3),      # -极化
    ]

    agents = []
    states = {}
    for name, etype, polar in agent_data:
        eid = _make_id(name, etype)
        agents.append(DeductionAgentProfile(
            entity_id=eid, name=name, persona="test",
            background="", goals=[], entity_type=etype,
        ))
        states[eid] = EntityState(
            id=eid, name=name, domain="geo_strategy",
            metrics={"polarization": polar}, history=[])

    engine = SimulationEngine.__new__(SimulationEngine)
    engine.graph = graph
    engine.agents = agents
    engine._states = states
    engine._log = lambda p, m: print(f"  [{p}] {m[:120]}")
    engine._rel_context = {}
    engine.reasoner = StrategicReasoner(candidate_count=_reg.candidate_count)

    # 只跑极化补全（不跑图谱关系预取）
    engine._seed_polarization_relations(graph_seeded=0)

    for a in agents:
        ctx = engine._rel_context.get(a.entity_id, {})
        print(f"    {a.name} (polar={states[a.entity_id].metrics.get('polarization',0):.1f}): "
              f"allies={ctx.get('allies',[])}, foes={ctx.get('opponents',[])}")

    # 验证：同极化应结盟
    trump_id = next((a.entity_id for a in agents if a.name == "特朗普"), "")
    harris_id = next((a.entity_id for a in agents if a.name == "哈里斯"), "")
    medved_id = next((a.entity_id for a in agents if a.name == "梅德韦杰夫"), "")

    trump_allies = engine._rel_context.get(trump_id, {}).get("allies", [])
    trump_foes = engine._rel_context.get(trump_id, {}).get("opponents", [])
    check("Same-polar alliance (Trump+Harris)", "哈里斯" in trump_allies)
    check("Opposite-polar foe (Trump+Medvedev)", "梅德韦杰夫" in trump_foes)


# ── 主流程 ──
async def main():
    print("=" * 60)
    print("Entity ID Preservation + Relationship Feedback Test Suite")
    print(f"Model: {os.environ.get('FORGE_LLM_MODEL')}")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="forge_test_id_"))
    try:
        # 1. 建图
        print("\n-- Stage 1: Build test graph --")
        graph = build_test_graph(tmpdir)

        # 2. 关键词分类
        print("\n-- Stage 2: Keyword classification --")
        test_relation_keywords()

        # 3. ID 回填逻辑
        print("\n-- Stage 3: Entity ID backfill logic --")
        test_registry_id_backfill(graph)

        # 4. 关系反哺 (核心)
        print("\n-- Stage 4: Relationship feedback (graph-based) --")
        test_relationship_context(graph)

        # 5. 极化补全
        print("\n-- Stage 5: Polarization complement --")
        test_polarization_relations(graph)

        graph.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    summary()
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

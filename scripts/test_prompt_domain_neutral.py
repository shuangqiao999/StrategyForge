"""领域中性提示词全流程验证 — 连接本地 LM Studio 9B 模型。

验证目标：
1. 所有环节的提示词在军事/商业/生态三个领域中均不产生领域锚定偏差
2. 本体生成不硬编码 Person/Organization 等商业命名范式
3. 情报排序不泄露特定领域机构名
4. 报告生成不预设特定维度标题
5. 定性动作类型已泛化为 initiate/respond/collaborate/compete/observe

用法（项目根目录）：
    python scripts/test_prompt_domain_neutral.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import json
import re

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("FORGE_DATA_DIR", os.path.join(os.environ.get("TEMP", "."), "sf_neutral_test"))
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_LLM_MODEL", "qwen/qwen3.5-9b")
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-all-minilm-l6-v2")
os.environ.setdefault("FORGE_DEFAULT_ROUNDS", "3")
os.environ.setdefault("FORGE_MAX_CONCURRENT", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.core.config import config
from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient, Message
from strategy_forge.engine.ontology import generate_ontology
from strategy_forge.engine.rule_engine import RuleEngine

# ── 三个领域的测试种子材料 ──
TEST_CASES = {
    "military": {
        "display": "军事博弈",
        "text": (
            "2025年，红方军队在东部战线集结了3个机械化师和2个航空旅，发起代号为'黎明行动'的大规模进攻。"
            "蓝方军队依托山地防线组织防御，第7山地旅和第12装甲团作为预备队部署在二线。"
            "绿方武装力量在南部边境保持中立但进行动员。联合国安理会召开紧急会议讨论局势，"
            "呼吁各方回到谈判桌。莫斯科向红方提供后勤支援，华盛顿则加大对蓝方的军事援助。"
        ),
    },
    "business": {
        "display": "商业竞争",
        "text": (
            "NovaTech公司是智能手机芯片市场的领先者，2025年Q2市场份额达35%。该公司宣布投资50亿美元"
            "在越南建设新工厂以降低供应链风险。主要竞争对手ArcSilicon紧随其后，宣布与台积电合作开发"
            "下一代3纳米制程。市场监管机构对此启动反垄断调查。第三方研究机构Gartner发布报告预测"
            "全球芯片市场2026年将达6000亿美元规模。"
        ),
    },
    "ecology": {
        "display": "生态博弈",
        "text": (
            "亚马逊雨林2025年Q2的毁林率同比上升22%，主要驱动因素是非法采矿和农业扩张。巴西政府宣布"
            "新的保护计划投入8亿美元，并派遣环保执法部队进入帕拉州和马托格罗索州。国际环保组织"
            "'亚马逊观察'批评该计划执行力度不足。欧盟碳边境调整机制(CBAM)将巴西大豆出口列为重点"
            "监控对象。土著社区领导人要求获得更多话语权参与保护决策。"
        ),
    },
}


def extract_json(raw: str) -> dict:
    """从 LLM 原始输出中提取 JSON。"""
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def check_domain_leak(text: str, domain: str) -> list[str]:
    """检测输出中是否包含其他领域的特定术语泄漏。"""
    violations: list[str] = []
    domain_keywords = {
        "military": [
            (r"Person\b|Organization\b|works_for", "商业本体术语泄漏"),
            (r"特斯拉|比亚迪|芯片|BMS|DeepSeek|现金流", "商业/科技实体名泄漏"),
            (r"SEC\b|证监会|标普|穆迪", "监管机构名泄漏"),
            (r"G7\b|G20\b|OECD\b|WEF\b", "论坛机构名泄漏"),
        ],
        "business": [
            (r"军事力量|军队编制|第[一二三四五六七八九十]+舰队|战区|集团军", "军事术语泄漏"),
            (r"占领|歼灭|炮击|导弹", "军事动作泄漏"),
            (r"士兵|连长|营长|师长", "军职泄漏"),
        ],
        "ecology": [
            (r"军事力量|军队编制|第[一二三四五六七八九十]+舰队", "军事术语泄漏"),
            (r"市场份额|现金流|cash.flow|反垄断|芯片", "商业术语泄漏"),
            (r"胜利条件|win_score.*胜利", "军事评估术语泄漏"),
        ],
    }
    for pattern, desc in domain_keywords.get(domain, []):
        if re.search(pattern, text):
            violations.append(f"[{desc}] {pattern}")
    return violations


class TestResult:
    def __init__(self, step: str):
        self.step = step
        self.passed = True
        self.issues: list[str] = []
        self.data: dict = {}

    def fail(self, msg: str):
        self.passed = False
        self.issues.append(msg)

    def add(self, key: str, value):
        self.data[key] = value

    def __repr__(self):
        status = "  PASS" if self.passed else "  FAIL"
        return f"{status} {self.step}" + (f" ({len(self.issues)} issues)" if self.issues else "")


async def test_domain_detection(domain_key: str, text: str, client: LLMClient) -> TestResult:
    """Phase 0: 领域检测 — 使用实际引擎的 detect_domain 方法。"""
    r = TestResult("领域检测")
    try:
        detected = await RuleEngine.detect_domain(text, client)
        r.add("detected", detected)
        r.add("expected", domain_key)
        if detected != domain_key:
            r.fail(f"领域检测错误：期望 {domain_key}，实际 {detected}")
    except Exception as e:
        r.fail(f"领域检测调用失败: {e}")
    return r


async def test_ontology(client: LLMClient, text: str, domain: str) -> TestResult:
    """Phase 1: 本体生成 — 验证不产生固定领域命名。"""
    r = TestResult("本体生成")
    try:
        ontology = await generate_ontology(text[:5000])
        entity_types = [e.name for e in ontology.entities]
        relation_types = [r.name for r in ontology.relations]
        r.add("entity_types", entity_types)
        r.add("relation_types", relation_types)

        full_output = str(entity_types) + str(relation_types)
        for v in check_domain_leak(full_output, domain):
            r.fail(v)

        if domain == "military":
            if not any("军" in e or "部" in e or "旅" in e or "师" in e or "国家" in e for e in entity_types):
                r.issues.append("[信息] 军事本体可能过于泛化，未出现军事相关类型")
        elif domain == "business":
            if not any("企业" in e or "公司" in e or "市场" in e for e in entity_types):
                r.issues.append("[信息] 商业本体可能过于泛化，未出现商业相关类型")
        elif domain == "ecology":
            if not any("生态" in e or "环境" in e or "政府" in e or "社区" in e for e in entity_types):
                r.issues.append("[信息] 生态本体可能过于泛化，未出现生态相关类型")
    except Exception as e:
        r.fail(f"本体生成失败: {e}")
    return r


async def test_quantified_sim(re_engine: RuleEngine, domain: str) -> TestResult:
    """Phase 4: 量化模拟 — 运行 3 轮并验证动作和叙事无领域泄漏。"""
    r = TestResult("量化模拟")
    from strategy_forge.engine.models import DeductionAgentProfile
    from strategy_forge.engine.simulator import SimulationEngine
    import uuid

    agents_map = {
        "military": [
            ("红方指挥部", "激进派统帅，信奉先发制人"),
            ("蓝方指挥部", "保守派统帅，依托防御消耗对手"),
            ("绿方指挥部", "务实派，倾向外交斡旋避免直接冲突"),
        ],
        "business": [
            ("NovaTech", "技术激进派，以研发突破为核心战略"),
            ("ArcSilicon", "成本优先派，通过规模效应压制对手"),
            ("MidChip", "差异化派，专注利基市场避开正面竞争"),
        ],
        "ecology": [
            ("巴西环保部", "执行强制保护政策，但有资源不足的困境"),
            ("亚马逊观察", "激进环保组织，推动国际舆论施压"),
            ("土著社区联盟", "维护原住民权益，寻求参与决策"),
        ],
    }
    names = agents_map.get(domain, agents_map["military"])
    agents = [
        DeductionAgentProfile(
            entity_id=uuid.uuid4().hex[:8], name=n, persona=p,
            background="", goals=["达成核心战略目标"],
        )
        for n, p in names
    ]
    states = {a.entity_id: re_engine.init_state(a.entity_id, a.name) for a in agents}

    engine = SimulationEngine(
        agents=agents, graph=None, total_rounds=3,
        log_fn=lambda p, m: None, preprocessor=None,
        pre_goals=["测试目标"], seed=42, temperature=0.6,
        persist_events=False, max_concurrent=1,
        rule_engine=re_engine, states=states, enable_narrate=True,
    )
    try:
        for round_num in range(1, 4):
            rd = await engine.run_round(round_num)
            for act in rd.actions:
                at = act.action_type
                r.add(f"R{round_num}_{act.agent_id[:4]}_action", at)
            if rd.state_delta.get("narration"):
                narration = rd.state_delta["narration"]
                r.add(f"R{round_num}_narration", narration[:120])
                for v in check_domain_leak(narration, domain):
                    r.fail(v)
    except Exception as e:
        r.fail(f"模拟失败: {e}")
    return r


async def test_report(client: LLMClient, domain: str) -> TestResult:
    """Phase 5: 报告生成 — 使用实际 _REPORT_PROMPT 模板验证。"""
    r = TestResult("报告生成")
    from string import Template
    from strategy_forge.engine.reporter import _REPORT_PROMPT

    prompt = Template(_REPORT_PROMPT).substitute(
        title="领域中性测试",
        domain=domain,
        round_count=3,
        agent_count=3,
        immutable_goals="测试目标",
        agent_overview="- 甲方: 主导方，寻求扩大优势\n- 乙方: 挑战方，寻求打破现状\n- 丙方: 中立方，维持平衡",
        key_relations="- 甲方 --[竞争]--> 乙方 (高权重)\n- 丙方 --[斡旋]--> 甲方",
        key_events="[事件1] 甲方: 主动出击 — 投入主要力量向前推进\n[事件2] 乙方: 防守反击 — 消耗对手后发动突袭\n[事件3] 丙方: 协调斡旋 — 寻求避免全面升级",
        action_timeline="- 甲方: 第1轮出击，第2轮巩固防线\n- 乙方: 第1轮防守，第2轮反击",
        quantified_context="甲方: 力量(高位·趋稳) 补给(中位·下降) 士气(高位·上升)\n乙方: 力量(偏低·下降) 补给(低位承压·下降) 士气(中位·趋稳)",
        causal_attribution="- 甲方 → 乙方: 力量 大幅削弱（-15）\n- 乙方 → 甲方: 补给 中度削弱（-5）",
        turning_points="- [R2] 乙方 → 力量: 骤降20",
    )
    try:
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是推演分析专家，撰写自然语言推演报告。只输出 JSON。",
            temperature=0.4,
            max_tokens=config.deduction_report_max_tokens,
        )
        raw = str(resp)
        data = extract_json(raw)
        # 调试: 如果解析失败，打印原始响应开头以便诊断
        if not data:
            r.add("raw_response_head", str(resp)[:300])
        narrative = data.get("narrative", "")
        risk_alerts = data.get("risk_alerts", [])
        conclusion = data.get("conclusion", "")
        recommendations = data.get("recommendations", [])

        r.add("narrative_preview", narrative[:150])
        r.add("risk_count", len(risk_alerts))
        r.add("conclusion_start", conclusion[:50])
        r.add("rec_count", len(recommendations))

        full_output = narrative + str(risk_alerts) + conclusion + str(recommendations)
        for v in check_domain_leak(full_output, domain):
            r.fail(v)

        if "### " not in narrative:
            r.fail("narrative 缺少 ### 维度标题")
        if len(risk_alerts) < 3:
            r.fail(f"risk_alerts 不足 3 条（实际 {len(risk_alerts)}）")
        if not conclusion.startswith("虽然"):
            r.fail(f"conclusion 未以'虽然'开头（实际: {conclusion[:30]}）")
        if len(recommendations) < 3:
            r.fail(f"recommendations 不足 3 条（实际 {len(recommendations)}）")
        for ra in risk_alerts:
            if isinstance(ra, dict):
                if not ("风险标题" in ra or ra.get("标题")):
                    r.fail(f"risk_alert dict 缺少风险标题: {str(ra)[:60]}")
            elif isinstance(ra, str) and ra.count("|") < 2:
                r.fail(f"risk_alert 格式不正确（缺少 | 分隔符）: {ra[:60]}")
    except Exception as e:
        r.fail(f"报告生成失败: {e}")
    return r


async def test_action_prompt(client: LLMClient) -> TestResult:
    """验证定性动作 prompt 泛化 — 确认 action 类型是泛化的5种。"""
    r = TestResult("定性动作泛化")
    prompt = """你是一个战略模拟中的智能体「侦察兵」，位于两军对峙的前线。

## 你的当前状态
你发现敌军正在前方500米处集结兵力。你的上级需要情报来决定下一步行动。

## 你的目标
获取准确情报并安全返回。

## 输出 JSON
{
  "action": "initiate|respond|collaborate|compete|observe",
  "target": "目标实体名或留空",
  "content": "行动描述 (20-60字)"
}
只返回 JSON，不要解释。"""
    try:
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是模拟智能体，根据局势选择行动。只输出 JSON。",
            temperature=0.3,
            max_tokens=200,
        )
        data = extract_json(str(resp))
        if not data:
            r.add("raw_response_head", str(resp)[:200])
        action = data.get("action", "")
        r.add("action", action)
        r.add("content", data.get("content", "")[:60])
        valid = ("initiate", "respond", "collaborate", "compete", "observe")
        if action not in valid:
            r.fail(f"定性动作类型不正确: '{action}'，期望 {valid} 之一")
    except Exception as e:
        r.fail(f"动作测试失败: {e}")
    return r


async def test_eval_prompt(client: LLMClient) -> TestResult:
    """验证优化器评估 prompt 不包含"胜利条件"等军事术语。"""
    r = TestResult("优化器评估术语")
    prompt = """你是推演结果评估专家。请依据"目标条件"评估本次推演结局，并量化打分。

## 目标条件（唯一判定标准）
市场份额进入前三

## 输出 JSON
{"success": true 或 false, "win_score": 0.0~1.0, "cost": 0.0~1.0, "rationale": "30字以内理由"}
只输出 JSON。"""
    try:
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是推演评估专家，只输出 JSON。",
            temperature=0.1,
        )
        data = extract_json(str(resp))
        rationale = data.get("rationale", "")
        r.add("rationale", rationale)
        if "胜利" in rationale:
            r.fail(f"输出中包含'胜利'一词（中性化失败）: {rationale}")
    except Exception as e:
        r.fail(f"评估测试失败: {e}")
    return r


async def main() -> int:
    print("=" * 60)
    print("  StrategyForge 领域中性提示词全流程验证")
    print(f"  LLM: {os.environ['FORGE_LLM_MODEL']}  @  {os.environ['FORGE_LLM_BASE']}")
    print("=" * 60)

    client = LLMClient()
    all_results: list[TestResult] = []

    # ═══ 步骤 1: 定性动作泛化 ═══
    print("\n── 1. 定性动作泛化验证 ──")
    r = await test_action_prompt(client)
    all_results.append(r)
    print(r)

    # ═══ 步骤 2: 优化器评估术语 ═══
    print("\n── 2. 优化器评估术语验证 ──")
    r = await test_eval_prompt(client)
    all_results.append(r)
    print(r)

    # ═══ 步骤 3-7: 按领域循环 ═══
    for domain_key, case in TEST_CASES.items():
        display = case["display"]
        text = case["text"]
        print(f"\n{'='*60}")
        print(f"  [{display}] 领域测试")
        print(f"{'='*60}")

        re_engine = RuleEngine.from_domain(domain_key)
        print(f"规则包: {re_engine.pack['display_name']} | 指标: {re_engine.metrics()}")

        print(f"\n── 3.{domain_key} 领域检测 ──")
        r = await test_domain_detection(domain_key, text, client)
        all_results.append(r)
        print(r)

        print(f"\n── 4.{domain_key} 本体生成 ──")
        r = await test_ontology(client, text, domain_key)
        all_results.append(r)
        print(r)
        print(f"   实体类型: {r.data.get('entity_types', [])}")
        print(f"   关系类型: {r.data.get('relation_types', [])}")

        print(f"\n── 5.{domain_key} 量化模拟 ──")
        r = await test_quantified_sim(re_engine, domain_key)
        all_results.append(r)
        print(r)
        action_keys = [k for k in r.data if k.endswith("_action")]
        for k in action_keys[:9]:
            print(f"   {k}: {r.data[k]}")

        print(f"\n── 6.{domain_key} 报告生成 ──")
        r = await test_report(client, domain_key)
        all_results.append(r)
        print(r)
        print(f"   风险预警数: {r.data.get('risk_count', 0)}")
        print(f"   建议数: {r.data.get('rec_count', 0)}")
        print(f"   conclusion开头: {r.data.get('conclusion_start', '')}")
        narr = r.data.get("narrative_preview", "")
        print(f"   narrative预览: {narr[:120]}")

    # ═══ 汇总 ═══
    print(f"\n{'='*60}")
    print("  测试汇总")
    print(f"{'='*60}")
    passed = sum(1 for r in all_results if r.passed)
    failed = len(all_results) - passed
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.step}")
        for issue in r.issues:
            print(f"         {issue}")
    print(f"\n  通过: {passed}/{len(all_results)}  失败: {failed}/{len(all_results)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

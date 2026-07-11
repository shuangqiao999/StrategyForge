"""Phase 5: Report Generation — analyze simulation results, produce structured report."""
from __future__ import annotations

import logging
from collections.abc import Callable
from string import Template
from typing import Any

from strategy_forge.storage.graph_store import DeductionGraphStore

from ._utils import extract_text
from .models import DeductionReport, DeductionSession, SimulationRound

logger = logging.getLogger(__name__)

_REPORT_PROMPT = """你是一位高级战略分析专家。你面前的资料，来自一线情报与模拟推演的交叉验证——涵盖军事、商业、科技与政治博弈。你的任务不是罗列数据，而是从这些纷繁的迹象中，剥离出最精准、最克制、最具指向性的方向性判断。言必有据，不夸大，不臆测。

## 核心写作法则（决定报告质量，逐条强制执行）
1. **因果链闭环 + 事件锚定**：每一条战略判断必须引用下方「关键事件序列」中的至少一个[事件N]，并以「动作→机制→结果→反制」的因果链展开。核心判断格式："[事件N]中A的[动作] → 直接导致B的[领域]发生[变化] → 迫使B转而[反制]"。次要事件可简写为"[事件N]中A的[动作] → 导致[结果]，进而使B陷入[困境]"。严禁"A施压B"的扁平句。
2. **段落即战略维度**：正文必须按博弈维度/主题逻辑分段（如军事对抗、经济博弈、科技竞赛、政治角力、联盟重组），每段用 `### 维度名` 作为小标题。每段聚焦1-2个主要行为体。这是输出的**强制格式要求**，不是提示词的元指令。
3. **必须写出"战略困境"**：至少为 3 个不同阵营的行为体写出得损失衡分析。句式模板："X为获取[收益]，不得不承受[代价/风险]"，或"X虽在[领域A]占优，但在[领域B]已逼近临界点"。
4. **语气与节奏**：克制、事实稠密，但允许使用"倒逼""对冲""临界点"等有张力的动词。每段控制在6行以内，避免堆砌形容词。
5. **多方平衡视角**：避免单向"某方全面胜出/衰落"的叙事。各方均有优势与制约，必须给出对手视角。
6. **区分事实与推测**：只有推演数据/因果归因支持的才作为判断陈述；不确定的用"可能/或将/存在风险"表述，不要写成既成事实。
7. **避免极端断言**：禁止出现"完全孤立""社会崩溃""彻底失败"等表述——此类只能作为风险情景谨慎提及，不作为结论。

## 推演基础信息
- 标题: $title · 领域: $domain · 轮次: $round_count
- 智能体数量: $agent_count · 不可变目标: $immutable_goals

## 智能体概览
$agent_overview

## 关键关系网络（图谱按权重）
$key_relations

## 关键事件序列（精选·必须引用）
$key_events

## 行动时序（Agent→事件，跨轮）
$action_timeline

## 全局态势数据（每实体各指标的定性档位与趋势，及淘汰线参考）
$quantified_context

## 确定性因果归因（源→目标 的影响方向与强度）
$causal_attribution

## 重点转折事件（方向性变化最剧烈的3个事件）
$turning_points

---

## 输出格式（严格遵守 JSON）
返回 JSON，包含四个字段：

1. **"narrative"**：完整推演报告（800-2000字）。
   - **硬性要求**：正文必须至少引用 3 条下方「关键事件序列」中的[事件N]编号，作为因果链锚点。
   - **硬性要求**：正文每个维度段落**必须**以 `### 维度名` 开头（如 `### 科技竞赛`、`### 区域安全`），禁止写成无标题的平铺段落。
   - 开篇（150字内）：作为全文第一个 `###` 段落之前的引入段，直接点明全局核心矛盾与主要博弈轴线。禁用"报告显示""推演表明""第X轮推演中"等废话。
   - 正文：按战略维度分段，每段以 `### ` 开头。因果句格式：`[事件N]中A的[动作] → B的[领域]发生[变化] → 迫使B[反制]`。
   - 结尾（150字内）：总览系统级风险与胜负手临界点。
   - 严禁输出任何具体数值/评分，不得出现 JSON、表格或项目符号列表。

   正确正文格式示例：
   ```
   开篇引入段（无标题，直接进入分析）...

   ### 科技竞赛
   [事件4]中美国的转向科技投资突破封锁 → 直接导致中国产业链面临更严峻的断供压力 → 迫使DeepSeek深化合作巩固战略后方（[事件7]），试图以商业韧性对冲技术围堵。中国虽在现金流高位趋稳，但在高端制造领域已逼近临界点，不得不承受供应链重构的代价。

   ### 区域安全
   [事件2]中真主党的打破僵化防守 → 直接导致以色列的军事优势被稀释 → 迫使伊朗以攻代守打击核心目标（[事件10]）...
   ```

2. **"risk_alerts"**：输出 3~5 条。格式：`{风险标题} | {具体触发机制/路径} | {受影响方}`。narrative 中已分析过的风险场景必须在 risk_alerts 中逐一展开触发链，不得遗漏。必须写明"如何触发"的因果链条，而非重复"存在风险"或"导致受损"的表象。
   - 错误示例："供应链风险 | 连续削弱 | 中国"（只写了影响，不是触发路径）
   - 正确示例："芯片断供风险 | 出口管制扩至成熟制程 → 切断BMS芯片供应 → 产能瘫痪 | 某实体"
   - 直接陈述事实，不前置"可能/或将"。

3. **"recommendations"**：最多5条。格式：`{针对方}→{具体动作}→{预期机制与效果}`。不写"建议""应""需要"等虚词，每条不超过40字。

4. **"conclusion"**：150-250字。首句必须以"虽然…但是…"结构开头，点明全局胜负手与临界变量。不得照抄 narrative 的句子，是更高层的凝练总结。不含具体数值，区分事实与推测。

只返回纯 JSON，不要 markdown 代码块，不要注释。"""


def _level_label(v: float) -> str:
    """将指标值映射为定性档位（不输出具体数值，避免过度戏剧化极端值）。"""
    if v >= 70:
        return "高位"
    if v >= 45:
        return "中位"
    if v >= 20:
        return "偏低"
    return "低位承压"


def _trend_label(d: float) -> str:
    """将累计变化量映射为趋势词。"""
    if d > 15:
        return "显著上升"
    if d > 3:
        return "上升"
    if d < -15:
        return "显著下降"
    if d < -3:
        return "下降"
    return "趋稳"


def _build_quantified_summary(
    rounds: list[SimulationRound],
    states: dict[str, Any] | None,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Build a structured qualitative trajectory summary for the report prompt.

    每个实体输出其关键指标的快照：档位·趋势·累计变化·淘汰线参考。
    数值仅注入 prompt 供 LLM 理解方向性——LLM 仍以叙事输出趋势，不复制数字。
    """
    if not states:
        return "（叙事模式，无量化指标数据）"

    # 辅助：指标中文化（与 simulator._METRIC_NAME 一致）
    _MN: dict[str, str] = {
        "strength": "军力", "morale": "士气", "supply": "补给", "fatigue": "疲劳度",
        "leadership": "领导力", "market_share": "市场份额", "cash_flow": "现金流",
        "brand": "品牌", "rnd": "研发", "supply_chain": "供应链",
        "support_rate": "支持率", "economy": "经济", "unity": "团结度",
        "intl_relations": "国际关系", "legislative_power": "立法权",
        "population": "人口", "resources": "资源", "pollution": "污染",
        "biodiversity": "生物多样性", "stability": "稳定性",
        "employment": "就业", "infrastructure": "基础设施", "finance": "财政",
        "satisfaction": "满意度", "tech_lead": "技术领先", "chip_stock": "芯片储备",
        "talent_pool": "人才池", "patent_barrier": "专利壁垒",
        "commercialization": "商业化", "narrative_dominance": "舆情主导",
        "public_trust": "公信力", "polarization": "极化度", "media_reach": "媒体触达",
    }
    _mn = _MN.get
    thresholds = thresholds or {}

    parts: list[str] = []
    for eid, st in states.items():
        name = getattr(st, "name", eid[:8])
        metrics = getattr(st, "metrics", {})
        if not metrics:
            parts.append(f"{name}: 无指标")
            continue
        history = getattr(st, "history", []) or []
        deltas_by_metric: dict[str, float] = {}
        for h in history[-30:]:
            m = h.get("metric", "")
            d = h.get("delta", 0)
            if m:
                deltas_by_metric[m] = deltas_by_metric.get(m, 0) + float(d)

        # 选取最关键的指标（按与淘汰线的逼近程度排序）
        scored = []
        for k, v in metrics.items():
            th = thresholds.get(k, 0)
            proximity = float(v) / max(float(th), 1.0) if th > 0 else 10.0
            scored.append((proximity, k, v, th))
        scored.sort()

        segs: list[str] = []
        for proximity, k, v, th in scored[:5]:
            label = _mn(k, k)
            level = _level_label(v)
            trend = _trend_label(deltas_by_metric.get(k, 0.0))
            cum_delta = deltas_by_metric.get(k, 0.0)
            th_text = f" 淘汰线={th:.0f}" if th > 0 else ""
            near_thresh = " ⚠逼近淘汰线" if th > 0 and v <= th * 1.3 else ""
            segs.append(f"{label}({level}·{trend} Δ{cum_delta:+.0f}{th_text}{near_thresh})")

        parts.append(f"{name}: {'; '.join(segs) if segs else '无关键指标'}")
    return "\n".join(parts)


async def generate_report(
    session: DeductionSession,
    graph: DeductionGraphStore,
    rounds: list[SimulationRound],
    log_fn: Callable[[str, str], None],
    preprocessor: Any = None,
    pre_goals: list[str] | None = None,
    states: dict[str, Any] | None = None,
) -> DeductionReport:
    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    from strategy_forge.core.llm_client import Message

    # Collect key events from all rounds (not just last 5)
    key_events: list[str] = []
    agent_trajectories: dict[str, list[str]] = {}
    # Track per-round deltas for turning point detection
    all_deltas: list[tuple[int, str, str, float]] = []  # (round, agent, metric, delta)

    def _agent_name(aid: str) -> str:
        if states and aid in states:
            st = states[aid]
            return getattr(st, "name", "") or aid[:8]
        return aid[:8]

    for rnd in rounds:
        for action in rnd.actions:
            brief = action.content[:60].split("，")[0]
            key_events.append(f"[轮{rnd.round_number}] "
                               f"{_agent_name(action.agent_id)}: "
                               f"{action.action_type} — {brief}")
            agent_trajectories.setdefault(action.agent_id, []).append(action.content[:60])
            if hasattr(action, "metadata") and isinstance(action.metadata, dict):
                for m, v in action.metadata.get("deltas", {}).items():
                    if abs(float(v)) > 3:
                        all_deltas.append((rnd.round_number, action.agent_id[:8], m, float(v)))

    # 跨轮语义召回：从 LanceDB events 表按场景主题召回最相关事件，补足"只看最近5轮"的盲区
    if preprocessor is not None:
        try:
            query = (session.title or session.source_material[:200] or "关键转折与冲突").strip()
            recalled = preprocessor.retrieve_dynamic_events(query, max(config.deduction_retrieve_top_k, 10),
                min_similarity=config.deduction_similarity_threshold)
            for c in recalled:
                line = f"[语义召回] {c[:100]}"
                if line not in key_events:
                    key_events.append(line)
            if recalled:
                log_fn("report", f"LanceDB 语义召回 {len(recalled)} 条跨轮关键事件")
        except Exception as e:
            logger.debug("[Reporter] 语义召回关键事件失败: %s", e)

    if not key_events:
        return DeductionReport(
            session_id=session.id,
            summary="推演未产生足够事件数据以生成报告。",
            raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
        )

    # 从知识图谱取关键关系(按权重)丰富报告
    key_relations = "（无显著关系）"
    if graph is not None:
        try:
            rows = graph.query(
                "MATCH (a:Entity)-[r:RELATES]->(b:Entity) "
                "RETURN a.name, r.relation, b.name, r.weight "
                "ORDER BY r.weight DESC LIMIT 15"
            )
            rels = [f"- {r[0]} --[{r[1]}]--> {r[2]}"
                    for r in rows if r and r[0] and r[2]]
            if rels:
                key_relations = "\n".join(rels)
                log_fn("report", f"图谱关键关系 {len(rels)} 条注入报告")
        except Exception as e:
            logger.debug("[Reporter] 关系查询失败: %s", e)

    # 从 Kuzu 时序行动图(Agent-[ACTED]->Event)取全局事件序列，供因果链分析
    action_timeline = "（无行动时序记录）"
    if graph is not None:
        try:
            seq = graph.get_event_sequence(limit=30)
            lines = [f"- [{e['timestamp'][:19]}] {e['agent_name']} {e['action']}: {e['description'][:60]}"
                     for e in seq if e.get("agent_name")]
            if lines:
                action_timeline = "\n".join(lines)
                log_fn("report", f"Kuzu 行动时序 {len(lines)} 条注入报告")
        except Exception as e:
            logger.debug("[Reporter] 行动时序查询失败: %s", e)

    # 确定性因果归因：从 Kuzu CAUSED 边汇总"源→目标 影响方向"，并附加轮次和变化量级
    causal_attribution = "（无确定性因果数据）"
    if graph is not None:
        try:
            summary = graph.get_causal_summary(limit=15)

            def _causal_dir(amt: float) -> str:
                mag = "大幅" if abs(amt) >= 10 else ("中度" if abs(amt) >= 3 else "小幅")
                return ("助益" if amt > 0 else "削弱") + f"({mag})"

            clines = [f"- {s['source']} → {s['target']}: {s['metric']} {_causal_dir(float(s['amount']))}"
                       f"（{s['amount']:+.0f}）"
                       for s in summary if s.get("metric")]
            if clines:
                causal_attribution = "\n".join(clines)
                log_fn("report", f"Kuzu 确定性因果归因 {len(clines)} 条注入报告")
        except Exception as e:
            logger.debug("[Reporter] 因果归因查询失败: %s", e)

    # ── 重点转折事件：方向性变化最大的 3 个事件 ──
    turning_points = "（无显著转折事件）"
    if all_deltas:
        top_3 = sorted(all_deltas, key=lambda x: abs(x[3]), reverse=True)[:3]
        tp_lines = [f"- [R{r}] {agent} → {metric}: {'激增' if delta > 0 else '骤降'}{abs(delta):.0f}"
                     for r, agent, metric, delta in top_3]
        if tp_lines:
            turning_points = "\n".join(tp_lines)

    # 推演设定上下文
    domain_text = "叙事模式（无量化）"
    if states:
        try:
            first = next(iter(states.values()))
            d = getattr(first, "domain", "")
            if d and d != "generic":
                domain_text = d
        except (StopIteration, AttributeError):
            pass
    agent_overview = "（无智能体数据）"
    if graph is not None:
        try:
            agents = graph.query(
                f"MATCH (a:{graph.AGENT_TABLE}) RETURN a.name, a.persona ORDER BY a.name")
            if agents:
                agent_overview = "\n".join(
                    f"- {r[0]}: {r[1][:60]}" for r in agents[:12] if r[0])
                log_fn("report", f"智能体总览 {len(agents)} 个注入报告")
        except Exception:
            pass

    client = LLMClient()
    # Extract thresholds from states if available
    _thresholds: dict[str, float] = {}
    if states and hasattr(next(iter(states.values())), "metrics"):
        try:
            from strategy_forge.engine.rule_engine import RuleEngine
        except Exception:
            pass
        else:
            # thresholds come from the domain rule pack; pass them through
            # to _build_quantified_summary for elimination-line context
            pass
    quantified_context = _build_quantified_summary(rounds, states, _thresholds)
    immutable_goals = "；".join(pre_goals) if pre_goals else "（无）"
    system = "你是推演分析专家，撰写自然语言推演报告。只输出 JSON。"
    numbered = [f"[事件{i+1}] {e}" for i, e in enumerate(key_events[-20:])]
    messages = [Message(role="user", content=Template(_REPORT_PROMPT).substitute(
        title=session.title or "推演会话",
        domain=domain_text,
        immutable_goals=immutable_goals,
        agent_count=session.agent_count,
        round_count=session.current_round,
        agent_overview=agent_overview,
        key_relations=key_relations,
        key_events="\n".join(numbered),
        action_timeline=action_timeline,
        quantified_context=quantified_context,
        causal_attribution=causal_attribution,
        turning_points=turning_points,
    ))]

    default_report = DeductionReport(
        session_id=session.id,
        summary="推演完成，请查看详细事件记录。",
        key_events=[{"description": e} for e in key_events[:10]],
        agent_trajectories=agent_trajectories,
        raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
    )

    try:
        response = await client.chat(messages, system=system, temperature=0.4,
                                     max_tokens=config.deduction_report_max_tokens)
        content = extract_text(response)
        report_data = _parse_report_json(content)
    except Exception as e:
        logger.warning("[Deduction] Report LLM failed, using defaults: %s", e)
        return default_report

    log_fn("report", "报告 LLM 生成完成")

    return DeductionReport(
        session_id=session.id,
        summary=report_data.get("narrative", "") or report_data.get("summary", default_report.summary),
        key_events=default_report.key_events,
        agent_trajectories=default_report.agent_trajectories,
        risk_alerts=report_data.get("risk_alerts", []),
        recommendations=report_data.get("recommendations", []),
        causal_summary=report_data.get("causal_summary", []),
        stage_narratives=report_data.get("stage_narratives", []),
        deviation_analysis=report_data.get("deviation_analysis", []),
        conclusion=report_data.get("conclusion", ""),
        raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
    )


def _parse_report_json(raw: str) -> dict[str, Any]:
    from ._utils import extract_json
    data = extract_json(raw)
    return data if isinstance(data, dict) else {}

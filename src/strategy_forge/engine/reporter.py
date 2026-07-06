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

_REPORT_PROMPT = """你是一位资深战略分析师。根据以下推演数据，撰写一份自然语言的推演报告。

报告应像《经济学人》或战略研究机构的风格——流畅叙事，将数据融入行文之中，而非罗列表格或项目符号。

## 推演基础信息
- 标题: $title · 领域: $domain · 轮次: $round_count
- 智能体数量: $agent_count · 不可变目标: $immutable_goals

## 智能体概览
$agent_overview

## 关键关系网络（图谱按权重）
$key_relations

## 关键事件序列
$key_events

## 行动时序（Agent→事件，跨轮）
$action_timeline

## 全局态势数据（每实体各指标的定性档位与趋势，无具体数值）
$quantified_context

## 确定性因果归因（源→目标 的影响方向与强度，无具体数值）
$causal_attribution

## 输出要求
返回 JSON，必须包含以下四个字段：

1. "narrative": 完整推演报告的自然语言文本（800-2000字）。写作要求：
   - 用"开头概括、中间叙事、结尾总结"的三段式结构
   - 本引擎定位战略推演，聚焦事物走向与趋势判断：用自然语言描述各方态势的方向性变化（如"北约防御力量维持高位、财政持续承压"），准确刻画趋势即可
   - 严禁输出任何具体数值/评分（不得写"技术领导力89分""信任度跌至个位数""strength=100"之类），也不要凭空编造精确数字
   - 避免战术层面的定量建议；策略判断以方向、力度、态势为主
   - 不要出现 JSON 格式的痕迹、不要出现表格、不要出现项目符号列表
   - 事实稠密、判断克制的非虚构叙事风格
   - 多方视角平衡：避免单向"某方全面胜出/衰落"的叙事，各方均有优势与制约，给出对手视角
   - 区分"事实"与"推测"：只有推演数据/因果归因支持的才作为判断陈述；不确定的用"可能/或将/存在风险"表述，不要写成既成事实
   - 避免不符常识的极端断言（如"完全孤立""社会崩溃""彻底失败"）；此类只能作为风险情景谨慎提及，不作为结论

2. "risk_alerts": 风险预警列表（最多5条字符串，描述趋势性风险，不含具体数值）

3. "recommendations": 策略建议列表（最多5条字符串，偏战略方向，不含具体数值）

4. "conclusion": 收束性结论（150-250字）。要求：提炼全局判断与关键变量，**不得照抄 narrative 的句子**，是更高层的凝练总结；同样不含具体数值、区分事实与推测。

只返回 JSON，不要 markdown 标记。"""


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
) -> str:
    """Build a QUALITATIVE trajectory summary (档位+趋势, 不含具体数值) for the
    narrative prompt. 战略推演聚焦走向趋势，不向报告注入裸数值以免锚定极端读数。
    """
    if not states:
        return "（叙事模式，无量化指标数据）"

    parts: list[str] = []
    for eid, st in states.items():
        name = getattr(st, "name", eid[:8])
        # 累计各指标近期变化
        history = getattr(st, "history", []) or []
        deltas_by_metric: dict[str, float] = {}
        for h in history[-30:]:
            m = h.get("metric", "")
            d = h.get("delta", 0)
            if m:
                deltas_by_metric[m] = deltas_by_metric.get(m, 0) + float(d)
        # 每个指标输出 "指标(档位·趋势)"，不写具体数字
        segs: list[str] = []
        for k, v in st.metrics.items():
            trend = _trend_label(deltas_by_metric.get(k, 0.0))
            segs.append(f"{k}({_level_label(float(v))}·{trend})")
        parts.append(f"{name}: {', '.join(segs) if segs else '无指标'}")
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

    # Collect key events
    key_events: list[str] = []
    agent_trajectories: dict[str, list[str]] = {}
    for rnd in rounds[-5:]:
        for action in rnd.actions:
            key_events.append(f"[轮{action.timestamp[:10] if action.timestamp else rnd.round_number}] "
                              f"{action.agent_id[:8]}: {action.action_type} — {action.content[:80]}")
            agent_trajectories.setdefault(action.agent_id, []).append(action.content[:60])

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

    # 确定性因果归因：从 Kuzu CAUSED 边汇总"源→目标 影响方向"，校正 LLM 软推断（定性，不输出数值）
    causal_attribution = "（无确定性因果数据）"
    if graph is not None:
        try:
            summary = graph.get_causal_summary(limit=15)

            def _causal_dir(amt: float) -> str:
                mag = "强" if abs(amt) >= 10 else ("中" if abs(amt) >= 3 else "弱")
                return ("助益" if amt > 0 else "致衰") + f"（{mag}）"

            clines = [f"- {s['source']} → {s['target']}: {s['metric']} {_causal_dir(float(s['amount']))}"
                      for s in summary if s.get("metric")]
            if clines:
                causal_attribution = "\n".join(clines)
                log_fn("report", f"Kuzu 确定性因果归因 {len(clines)} 条注入报告")
        except Exception as e:
            logger.debug("[Reporter] 因果归因查询失败: %s", e)

    # 推演设定上下文
    domain_text = "叙事模式（无量化）"
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
        try:
            dom = graph.query(
                f"MATCH (a:{graph.AGENT_TABLE}) RETURN a.name LIMIT 1")
            if dom:
                # 从 agent area 推断 domain（有限）
                pass
        except Exception:
            pass

    client = LLMClient()
    quantified_context = _build_quantified_summary(rounds, states)
    immutable_goals = "；".join(pre_goals) if pre_goals else "（无）"
    system = "你是资深战略分析师，撰写自然语言推演报告。只输出 JSON。"
    messages = [Message(role="user", content=Template(_REPORT_PROMPT).substitute(
        title=session.title or "推演会话",
        domain=domain_text,
        immutable_goals=immutable_goals,
        agent_count=session.agent_count,
        round_count=session.current_round,
        agent_overview=agent_overview,
        key_relations=key_relations,
        key_events="\n".join(key_events[-20:]),
        action_timeline=action_timeline,
        quantified_context=quantified_context,
        causal_attribution=causal_attribution,
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

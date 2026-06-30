"""Phase 5: Report Generation — analyze simulation results, produce structured report."""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from strategy_forge.storage.graph_store import DeductionGraphStore

from ._utils import extract_text
from .models import DeductionReport, DeductionSession, SimulationRound

logger = logging.getLogger(__name__)

_REPORT_PROMPT = """你是一个推演分析专家。基于以下推演数据，生成一份结构化的推演报告。返回 JSON。

## 推演概览
- 会话标题: {title}
- 推演领域: {domain}
- 智能体数量: {agent_count}
- 模拟轮数: {round_count}
- 图谱实体数: {entity_count}, 关系数: {relation_count}
- 不可变目标: {immutable_goals}

## 智能体总览
{agent_overview}

## 关键事件（最近 20 个）
{key_events}

## 推演原文背景
{source_snippet}

## 关键关系（知识图谱：实体—关系—实体）
{key_relations}

## 行动时序与因果链（Kuzu 时序行动图：按时间排列的"谁—做了什么"）
{action_timeline}

## 因果归因（确定性·来自数值真值：源 → 目标 累计指标影响，负值=致衰）
{causal_attribution}

## 量化指标轨迹（每轮各实体关键指标变化，含具体数值）
{quantified_context}

## 输出 JSON
```json
{{
  "summary": "推演总结 (150-300字)",
  "key_events": [
    {{"round": 1, "description": "事件描述", "significance": "高/中/低"}}
  ],
  "agent_trajectories": {{
    "agent_id": ["行动1", "行动2"]
  }},
  "risk_alerts": ["风险预警1", "风险预警2"],
  "recommendations": ["策略建议1", "策略建议2"],
  "causal_summary": [
    "→ 因果链1：A做了什么 → B发生了什么 → 最终导致C（引用具体轮次与数值变化，如+12/-8）",
    "→ 因果链2：..."
  ],
  "stage_narratives": [
    {{
      "stage": "阶段名称（如试探期/对抗期/决战期）",
      "round_range": "第X-Y轮",
      "start_state": "阶段起始状态（含关键指标值）",
      "key_decisions": "核心决策与行动",
      "causal_logic": "因果逻辑描述（为什么A导致B）",
      "end_state": "阶段终点与为下一阶段埋下的伏笔"
    }},
    ...
  ],
  "deviation_analysis": [
    {{
      "round": 1,
      "agent": "行为体名称",
      "decision": "具体决策描述",
      "deviation_level": "显著/轻微",
      "reason": "偏离不可变目标的原因分析"
    }}
  ],
  "conclusion": "整体结论与启示（100-200字）"
}}
```

- causal_summary: 识别推演中最重要的3-5条因果链，用箭头式表述，**必须引用具体轮次和量化变化值**（如"第3轮宋江决策→民心**+12**→第5轮获得新兵源"）
- stage_narratives: 将整个推演按局势转折分为2-4个阶段，每阶段描述起因-经过-结果的完整逻辑链
- deviation_analysis: 识别推演中哪些决策**偏离了不可变目标的方向**，分析原因。若无偏离则返回空数组
- conclusion: 对推演整体规律的提炼，特别关注不可变目标的达成情况及其偏离原因

只返回 JSON，不要解释。"""


def _build_quantified_context(
    rounds: list[SimulationRound],
    states: dict[str, Any] | None,
) -> str:
    """Build structured quantified trajectory text for the LLM prompt."""
    if not states:
        return "（叙事模式，无量化指标数据）"

    parts: list[str] = []
    # Per-entity metric snapshots at key rounds (first, last, and every ~3rd round)
    sample_rounds = set()
    if rounds:
        total = max(r.round_number for r in rounds)
        sample_rounds = {r.round_number for r in rounds
                         if r.round_number in (1, total)
                         or r.round_number % max(1, total // 4) == 0}
    for eid, st in states.items():
        name = getattr(st, "name", eid[:8])
        history = getattr(st, "history", []) or []
        # Group history by round for per-round summaries
        by_round: dict[int, list[dict]] = {}
        for h in history:
            rnd = h.get("round", 0)
            if rnd not in by_round:
                by_round[rnd] = []
            by_round[rnd].append(h)
        # Show key round snapshots
        snapshots: list[str] = []
        for rnd in sorted(by_round.keys()):
            if rnd in sample_rounds or len(snapshots) < 3:
                deltas = ", ".join(
                    f"{h.get('metric','?')}{h.get('delta',0):+.1f}"
                    for h in by_round[rnd][:6]
                )
                snapshots.append(f"  R{rnd}: {deltas}")
        if snapshots:
            parts.append(f"- {name}:"
                         f"\n{'  '.join(snapshots[:8])}")
    if not parts:
        parts.append("- （无量化轨迹数据）")
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
            recalled = preprocessor.retrieve_dynamic_events(query, top_k=10, min_similarity=0.2)
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

    # 确定性因果归因：从 Kuzu CAUSED 边汇总"源→目标 累计指标影响"，校正 LLM 软推断
    causal_attribution = "（无确定性因果数据）"
    if graph is not None:
        try:
            summary = graph.get_causal_summary(limit=15)
            clines = [f"- {s['source']} → {s['target']}: {s['metric']}{s['amount']:+.1f}（累计）"
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
    quantified_context = _build_quantified_context(rounds, states)
    immutable_goals = "；".join(pre_goals) if pre_goals else "（无）"
    system = "你是推演分析专家，生成结构化推演报告。只输出 JSON。"
    messages = [Message(role="user", content=_REPORT_PROMPT.format(
        title=session.title or "推演会话",
        domain=domain_text,
        immutable_goals=immutable_goals,
        agent_count=session.agent_count,
        round_count=session.current_round,
        entity_count=session.entity_count,
        relation_count=session.relation_count,
        agent_overview=agent_overview,
        key_events="\n".join(key_events[-20:]),
        source_snippet=session.source_material[:1000],
        key_relations=key_relations,
        action_timeline=action_timeline,
        causal_attribution=causal_attribution,
        quantified_context=quantified_context,
    ))]

    default_report = DeductionReport(
        session_id=session.id,
        summary="推演完成，请查看详细事件记录。",
        key_events=[{"description": e} for e in key_events[:10]],
        agent_trajectories=agent_trajectories,
        raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
    )

    try:
        response = await client.chat(messages, system=system, temperature=0.3)
        content = extract_text(response)
        report_data = _parse_report_json(content)
    except Exception as e:
        logger.warning("[Deduction] Report LLM failed, using defaults: %s", e)
        return default_report

    log_fn("report", "报告 LLM 生成完成")

    return DeductionReport(
        session_id=session.id,
        summary=report_data.get("summary", default_report.summary),
        key_events=report_data.get("key_events", default_report.key_events),
        agent_trajectories=report_data.get("agent_trajectories", default_report.agent_trajectories),
        risk_alerts=report_data.get("risk_alerts", []),
        recommendations=report_data.get("recommendations", []),
        causal_summary=report_data.get("causal_summary", []),
        stage_narratives=report_data.get("stage_narratives", []),
        deviation_analysis=report_data.get("deviation_analysis", []),
        conclusion=report_data.get("conclusion", ""),
        raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
    )


def _parse_report_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

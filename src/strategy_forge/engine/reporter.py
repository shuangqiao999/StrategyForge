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
- 智能体数量: {agent_count}
- 模拟轮数: {round_count}
- 图谱实体数: {entity_count}, 关系数: {relation_count}

## 关键事件（最近 20 个）
{key_events}

## 推演原文背景
{source_snippet}

## 输出 JSON
```json
{{
  "summary": "推演总结 (100-200字)",
  "key_events": [
    {{"round": 1, "description": "事件描述", "significance": "高/中/低"}}
  ],
  "agent_trajectories": {{
    "agent_id": ["行动1", "行动2"]
  }},
  "risk_alerts": ["风险预警1", "风险预警2"],
  "recommendations": ["策略建议1", "策略建议2"]
}}
```

只返回 JSON，不要解释。"""


async def generate_report(
    session: DeductionSession,
    graph: DeductionGraphStore,
    rounds: list[SimulationRound],
    log_fn: Callable[[str, str], None],
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

    if not key_events:
        return DeductionReport(
            session_id=session.id,
            summary="推演未产生足够事件数据以生成报告。",
            raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
        )

    client = LLMClient()
    system = "你是推演分析专家，生成结构化推演报告。只输出 JSON。"
    messages = [Message(role="user", content=_REPORT_PROMPT.format(
        title=session.title or "推演会话",
        agent_count=session.agent_count,
        round_count=session.current_round,
        entity_count=session.entity_count,
        relation_count=session.relation_count,
        key_events="\n".join(key_events[-20:]),
        source_snippet=session.source_material[:1000],
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

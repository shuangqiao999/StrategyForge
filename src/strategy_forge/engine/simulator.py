"""Phase 4: Parallel Simulation — multi-agent with dual-path LanceDB memory recall.

Dual-path retrieval:
  Path A (static): retrieval from deduction_chunks table — original source material
  Path B (dynamic): retrieval from deduction_events table — simulation-generated events
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import uuid
from collections.abc import Callable
from typing import Any

from strategy_forge.storage.graph_store import DeductionGraphStore

from ._utils import extract_text
from .models import DeductionAgentProfile, SimulationAction, SimulationRound
from .preprocessor import DeductionPreprocessor

logger = logging.getLogger(__name__)


_ACTION_PROMPT = """你是一个推演模拟中的智能体。根据你的角色设定和当前世界状态，决定你的下一步行动。

## 你的固定人格（基于原文）
{persona}

## 你的背景
{background}

## 你的目标
{goals}

## 当前轮次
第 {round_number} 轮

## 近期模拟动态事件（重要！以下是其他角色刚刚做过的事）
{dynamic_memory}

## 你的原著背景参考（仅供参考）
{static_knowledge}

## 近期世界缓存
{recent_events}

## 输出 JSON — 选择一种行动
```json
{{
  "action": "post|reply|interact|observe",
  "target": "目标实体名或留空",
  "content": "行动内容 (30-100字)"
}}
```

只返回 JSON，不要解释。"""


class SimulationEngine:
    """多智能体并行模拟引擎 — 双路语义记忆。

    决策上下文优先级:
      1. 动态事件表 (LanceDB deduction_events) — 模拟中生成的事件, 语义检索
      2. 静态原文表 (LanceDB deduction_chunks) — 原著背景, 语义检索
      3. 近期缓存 (event_history[-5:]) — 最近 5 条全局事件
      4. 智能体自身设定 (persona / background / goals)
    """

    def __init__(
        self,
        agents: list[DeductionAgentProfile],
        graph: DeductionGraphStore,
        total_rounds: int = 10,
        log_fn: Callable[[str, str], None] | None = None,
        preprocessor: DeductionPreprocessor | None = None,
        chat_fn: Any = None,
        pre_goals: list[str] | None = None,
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        persist_events: bool = True,
        max_concurrent: int | None = None,
        rule_engine: Any = None,
        states: dict[str, Any] | None = None,
        enable_narrate: bool = True,
        env: dict[str, str] | None = None,
        enable_multi_action: bool = False,
        max_actions: int = 3,
    ) -> None:
        self.agents = agents
        self.graph = graph
        self.total_rounds = total_rounds
        self._log = log_fn or (lambda p, m: None)
        self._event_history: list[dict[str, Any]] = []
        self._preprocessor = preprocessor
        self._chat_fn = chat_fn
        self._immutable_goals: list[str] = list(pre_goals or [])
        # 蒙特卡洛隔离与可控性参数
        self._persist_events = persist_events
        self._temperature = temperature
        self._rng = random.Random(seed)
        # 量化模式参数（rule_engine 非空即进入量化模式）
        self._rule_engine = rule_engine
        self._states: dict[str, Any] = states or {}
        self._quantified = rule_engine is not None
        self._enable_narrate = enable_narrate
        self._env = env
        self._enable_multi_action = enable_multi_action
        self._max_actions = max(1, int(max_actions))
        from strategy_forge.core.config import config

        self._max_concurrent = (
            max_concurrent if max_concurrent is not None
            else config.deduction_max_concurrent
        )

        from .strategic_reasoner import StrategicReasoner
        self.reasoner = StrategicReasoner(
            candidate_count=config.deduction_candidate_count,
            preprocessor=preprocessor,
            chat_fn=chat_fn,
            immutable_goals=self._immutable_goals,
            temperature=temperature,
            enable_multi_action=self._enable_multi_action,
            max_actions=self._max_actions,
        )

    async def run_round(self, round_number: int) -> SimulationRound:
        if self._quantified:
            return await self._run_round_quantified(round_number)

        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient

        sim_round = SimulationRound(round_number=round_number)
        client = LLMClient()

        ordered = list(self.agents)
        self._rng.shuffle(ordered)

        sem = asyncio.Semaphore(self._max_concurrent)

        async def process_agent(agent: DeductionAgentProfile) -> SimulationAction | None:
            async with sem:
                return await self._agent_decide(client, agent, round_number)

        tasks = [process_agent(a) for a in ordered]
        results = await asyncio.gather(*tasks)

        for action in results:
            if action is not None:
                sim_round.actions.append(action)
                self._event_history.append({
                    "agent": action.agent_id,
                    "agent_name": getattr(
                        next((a for a in self.agents if a.entity_id == action.agent_id), None),
                        "name", action.agent_id[:8],
                    ),
                    "action": action.action_type,
                    "content": action.content,
                    "round": round_number,
                    "timestamp": action.timestamp,
                })

        if len(self._event_history) > 200:
            self._event_history = self._event_history[-200:]

        # Write round events to Kuzu graph + LanceDB dynamic event table
        # 蒙特卡洛隔离模式 (persist_events=False): 不落盘、不写向量库，仅保留内存事件历史，
        # 保证 M×N 次模拟相互隔离、可并发，且不污染主会话数据。
        if self._persist_events:
            for action in sim_round.actions:
                event_id = f"evt-{uuid.uuid4().hex[:8]}"
                self.graph.add_event(
                    event_id, action.content[:200], action.action_type,
                    action.timestamp, action.agent_id,
                )
                self.graph.add_acted(action.agent_id, event_id, action.action_type, action.timestamp)

                # ★ 动态事件写入 LanceDB (下一轮决策即可语义召回)
                if self._preprocessor is not None:
                    try:
                        self._preprocessor.add_event_memory(
                            content=action.content,
                            agent_id=action.agent_id,
                            round_number=round_number,
                            event_type=action.action_type,
                        )
                    except Exception as e:
                        logger.warning("[Simulator] Event memory write failed for %s: %s",
                                     action.agent_id, e)

        return sim_round

    async def _agent_decide(
        self, client: Any, agent: DeductionAgentProfile, round_number: int
    ) -> SimulationAction | None:
        # ── 近期事件 (最近 5 条) ──
        recent = self._event_history[-5:]
        recent_text = "\n".join(
            f"- [{e.get('round', '?')}] {e.get('agent_name', e.get('agent', '?'))}: "
            f"{e.get('content', '')[:80]}"
            for e in recent
        ) or "无近期事件"

        # ── Path A: 静态原著背景检索 ──
        static_text = "无特定背景"
        if self._preprocessor and self._preprocessor.result:
            try:
                static_frags = await asyncio.to_thread(
                    self._preprocessor.retrieve_for_entity,
                    agent.name, top_k=2,
                    must_contain={agent.name} if agent.name else None,
                )
                if static_frags:
                    static_text = "\n---\n".join(f[:300] for f in static_frags)
            except Exception as e:
                logger.warning("[Simulator] Static recall failed for %s: %s", agent.name, e)

        # ── Path B: 动态模拟事件检索 ──
        dynamic_text = "无近期模拟事件"
        if self._persist_events and self._preprocessor is not None:
            try:
                from strategy_forge.core.config import config
                aliases: set[str] = set()
                if self._preprocessor.result:
                    aliases = self._preprocessor.result.high_freq_entities.get(agent.name, set())
                    aliases.update(
                        self._preprocessor.result.low_freq_entities.get(agent.name, set()))
                query = agent.name + " " + " ".join(aliases - {agent.name})
                dynamic_frags = await asyncio.to_thread(
                    self._preprocessor.retrieve_dynamic_events,
                    query, top_k=3, min_similarity=config.deduction_similarity_threshold,
                )
                if dynamic_frags:
                    dynamic_text = "\n---\n".join(dynamic_frags)
            except Exception as e:
                logger.warning("[Simulator] Dynamic recall failed for %s: %s", agent.name, e)
        elif not self._persist_events:
            # 隔离模式(蒙特卡洛): 仅用内存事件历史, 不触碰 LanceDB
            mem = [e for e in self._event_history[-20:]
                   if agent.name in e.get("content", "") or e.get("agent") == agent.entity_id]
            if mem:
                dynamic_text = "\n".join(f"- {e.get('content', '')[:80]}" for e in mem[-3:])

        # ── Strategic Reasoning (primary path) ──
        world = {"recent_events": recent_text, "static_knowledge": static_text,
                  "dynamic_memory": dynamic_text}
        try:
            decision = await self.reasoner.reason(agent, world, round_number, client=client)
            sel = decision.get("selected", {})
            action_data = {"action": sel.get("action", "observe"),
                           "target": sel.get("target", ""),
                           "content": sel.get("content", f"{agent.name}观察着周围环境")}
            # Update trust matrix from selected action
            if sel.get("target"):
                self.reasoner.record_interaction(
                    agent.entity_id, sel["target"], action_data["action"], action_data["content"])
        except Exception as e:
            logger.warning("[Simulator] Reasoner failed for %s, using inline prompt: %s", agent.name, e)
            # ── Fallback: inline prompt ──
            from strategy_forge.core.llm_client import Message
            system = "你是推演模拟中的角色，根据角色设定和历史事件做出合理的下一步行动。只输出 JSON。"
            messages = [Message(role="user", content=_ACTION_PROMPT.format(
                persona=agent.persona, background=agent.background,
                goals=", ".join(agent.goals) if agent.goals else "参与互动",
                round_number=round_number, recent_events=recent_text,
                static_knowledge=static_text, dynamic_memory=dynamic_text,
            ))]
            try:
                if self._chat_fn is not None:
                    response = await asyncio.to_thread(self._chat_fn, messages, system, 0.7)
                    content = response
                else:
                    response = await client.chat(messages, system=system, temperature=0.7)
                    content = extract_text(response)
                action_data = _parse_action_json(content)
            except Exception as e2:
                logger.warning("[Deduction] Agent %s decision failed: %s", agent.name, e2)
                return None

        from datetime import datetime
        return SimulationAction(
            agent_id=agent.entity_id,
            action_type=action_data.get("action", "observe"),
            target_id=action_data.get("target", ""),
            content=action_data.get("content", f"{agent.name}观察着周围环境"),
            timestamp=datetime.now().isoformat(),
        )

    # ── 量化模式：决策 → 快照交互解算 → 批量应用 → 阈值淘汰 → 可选解读 ──
    async def _run_round_quantified(self, round_number: int) -> SimulationRound:
        from datetime import datetime

        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient

        sim_round = SimulationRound(round_number=round_number)
        re_engine = self._rule_engine
        states = self._states
        client = LLMClient()

        alive_agents = [a for a in self.agents
                        if a.entity_id in states and re_engine.is_alive(states[a.entity_id])]
        if not alive_agents:
            return sim_round

        ordered = list(alive_agents)
        self._rng.shuffle(ordered)

        recent = "\n".join(
            f"- [{e.get('round','?')}] {e.get('agent_name','?')}: {e.get('content','')[:80]}"
            for e in self._event_history[-5:]
        ) or "（无近期事件）"

        def others_ctx(self_id: str) -> str:
            return "\n".join(
                states[a.entity_id].to_prompt_context()
                for a in alive_agents if a.entity_id != self_id
            ) or "（无其他参与方）"

        sem = asyncio.Semaphore(self._max_concurrent)

        async def decide(agent: DeductionAgentProfile) -> dict[str, Any]:
            async with sem:
                d = await self.reasoner.reason_quantified(
                    agent, states[agent.entity_id], re_engine,
                    recent_events=recent, other_context=others_ctx(agent.entity_id),
                    round_number=round_number, client=client,
                )
                d["actor_id"] = agent.entity_id
                return d

        decisions = await asyncio.gather(*[decide(a) for a in ordered])

        # 轮初快照(批量应用语义) + 交互解算
        name_to_id = {a.name: a.entity_id for a in self.agents}
        deltas = re_engine.resolve_round(states, decisions, name_to_id, self._env)
        ranges = re_engine.ranges()
        for eid, d in deltas.items():
            if eid in states:
                states[eid].apply_deltas(d, round_number, ranges)

        # 构造行动 + 内存事件历史
        for dec in decisions:
            actor = dec["actor_id"]
            agent = next((a for a in self.agents if a.entity_id == actor), None)
            nm = agent.name if agent else actor[:8]
            d_applied = deltas.get(actor, {})
            delta_txt = ", ".join(f"{k}{v:+.1f}" for k, v in d_applied.items())
            alloc = dec.get("actions") or None
            alloc_txt = ""
            if alloc:
                alloc_txt = ", ".join(
                    f"{a.get('action_type', '')}{float(a.get('weight', 0)):.2f}"
                    + (f"→{a.get('target')}" if a.get("target") else "")
                    for a in alloc
                )
                content = dec.get("rationale", "") or f"{nm} 资源分配: {alloc_txt}"
            else:
                content = dec.get("rationale", "") or f"{nm} 执行 {dec['action_type']}"
            meta: dict[str, Any] = {
                "intensity": dec.get("intensity", dec.get("budget", 0.5)),
                "deltas": d_applied,
                "metrics": dict(states[actor].metrics) if actor in states else {},
            }
            if alloc:
                meta["budget"] = dec.get("budget", dec.get("intensity", 0.5))
                meta["allocation"] = alloc
            sim_round.actions.append(SimulationAction(
                agent_id=actor, action_type=dec["action_type"],
                target_id=dec.get("target", ""), content=content,
                timestamp=datetime.now().isoformat(),
                metadata=meta,
            ))
            evt_suffix = (f"［{alloc_txt}］" if alloc_txt else "") + (f"（{delta_txt}）" if delta_txt else "")
            self._event_history.append({
                "agent": actor, "agent_name": nm, "action": dec["action_type"],
                "content": content + evt_suffix,
                "round": round_number,
            })
        if len(self._event_history) > 200:
            self._event_history = self._event_history[-200:]

        # 轮末快照(供报告/趋势) + 可选叙事解读
        sim_round.state_delta["states"] = {
            a.entity_id: {"name": a.name, "metrics": dict(states[a.entity_id].metrics),
                          "alive": re_engine.is_alive(states[a.entity_id])}
            for a in self.agents if a.entity_id in states
        }
        if self._enable_narrate:
            try:
                narration = await self._narrate_round(client, round_number, decisions, deltas)
                if narration:
                    sim_round.state_delta["narration"] = narration
            except Exception as e:
                logger.warning("[Simulator] 轮末叙事失败: %s", e)

        return sim_round

    async def _narrate_round(self, client: Any, round_number: int,
                             decisions: list[dict], deltas: dict) -> str:
        from strategy_forge.core.llm_client import Message

        from ._utils import extract_text
        lines = []
        for dec in decisions:
            actor = dec["actor_id"]
            agent = next((a for a in self.agents if a.entity_id == actor), None)
            nm = agent.name if agent else actor[:8]
            d = deltas.get(actor, {})
            chg = ", ".join(f"{k}{v:+.1f}" for k, v in d.items()) or "无显著变化"
            alloc = dec.get("actions") or None
            if alloc:
                budget = float(dec.get("budget", dec.get("intensity", 0.5)))
                act_txt = "资源分配 " + ", ".join(
                    f"{a.get('action_type', '')}{float(a.get('weight', 0)):.0%}"
                    + (f"(→{a.get('target')})" if a.get("target") else "")
                    for a in alloc
                ) + f"，总投入{budget:.1f}"
            else:
                act_txt = (f"采取 {dec['action_type']}(强度{dec.get('intensity', 0.5):.1f}) "
                           f"目标:{dec.get('target') or '—'}")
            lines.append(f"{nm} {act_txt}，数值变化: {chg}")
        prompt = (
            f"将第 {round_number} 轮量化推演结果改写为一段生动简洁的战局叙事（100 字以内）。\n\n"
            "## 本轮各方行动与数值变化\n" + "\n".join(lines) + "\n\n只输出叙事段落，不要解释或列表。"
        )
        resp = await client.chat([Message(role="user", content=prompt)],
                                 system="你是推演解说员，把数值变化翻译成简洁叙事。", temperature=0.5)
        return extract_text(resp).strip()[:300]


def _parse_action_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

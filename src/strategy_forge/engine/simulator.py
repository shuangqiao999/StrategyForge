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


# 关系→盟友/对手的关键词启发式（中英），用于从 Kuzu RELATES 关系反哺决策与信任。
_REL_ALLY_KW = ("盟", "同盟", "结盟", "联盟", "支持", "合作", "友", "部下", "下属",
                "效忠", "追随", "ally", "allied", "support", "friend", "cooperat",
                "subordinate", "loyal")
_REL_FOE_KW = ("敌", "对立", "对抗", "对手", "竞争", "冲突", "背叛", "仇", "攻击",
               "威胁", "rival", "enemy", "hostil", "oppos", "compet", "conflict",
               "betray", "threat")


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
        cancel_event: Any = None,
    ) -> None:
        self.agents = agents
        self.graph = graph
        self.total_rounds = total_rounds
        self._log = log_fn or (lambda p, m: None)
        self._event_history: list[dict[str, Any]] = []
        self._preprocessor = preprocessor
        self._chat_fn = chat_fn
        self._immutable_goals: list[str] = list(pre_goals or [])
        self._cancel = cancel_event
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

        # A. 关系反哺：开局一次性从 Kuzu 预取盟友/对手并播种信任(关系在一次推演内静态)
        self._rel_context: dict[str, dict] = {}
        self._build_relationship_context()

    @staticmethod
    def _classify_relation(relation: str) -> str:
        r = (relation or "").lower()
        if any(k in r for k in _REL_FOE_KW):
            return "foe"
        if any(k in r for k in _REL_ALLY_KW):
            return "ally"
        return "neutral"

    def _build_relationship_context(self) -> None:
        """开局一次性从 Kuzu 预取各 agent 的盟友/对手(关系静态)，缓存并播种信任矩阵。

        顺序执行(非并发)，规避 Kuzu 单连接线程安全问题；运行中只读缓存，
        不在并发 decide() 里查图。量化经 relationship_context 注入 Prompt，
        定性额外经 seed_trust 影响打分/信任摘要。
        """
        if self.graph is None or not self.agents:
            return
        for a in self.agents:
            allies: list[str] = []
            foes: list[str] = []
            try:
                data = self.graph.get_entity_neighbors(a.entity_id, max_depth=1)
            except Exception as e:
                logger.debug("[Simulator] 关系预取失败 %s: %s", a.name, e)
                continue
            for nb in data.get("neighbors", []):
                nm = nb.get("name", "")
                if not nm or nm == a.name:
                    continue
                kind = self._classify_relation(nb.get("relation", ""))
                if kind == "ally" and nm not in allies:
                    allies.append(nm)
                elif kind == "foe" and nm not in foes:
                    foes.append(nm)
            parts = []
            if allies:
                parts.append("盟友: " + "、".join(allies[:6]))
            if foes:
                parts.append("对手: " + "、".join(foes[:6]))
            self._rel_context[a.entity_id] = {
                "allies": allies, "opponents": foes, "summary": "；".join(parts)}
            if allies or foes:
                self.reasoner.seed_trust(a.entity_id, allies, foes)
        seeded = sum(1 for v in self._rel_context.values() if v["summary"])
        if seeded:
            self._log("simulation", f"关系反哺：{seeded} 个智能体注入图谱盟友/对手并播种信任")

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
                  "dynamic_memory": dynamic_text,
                  "relationship_context": self._rel_context.get(agent.entity_id, {}).get("summary", "")}
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

        from strategy_forge.core.config import config as _cfg
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _recall(agent: DeductionAgentProfile) -> tuple[str, str]:
            """量化轮的 LanceDB 语义召回：Path A 原著静态(只读，优化器也启用) + Path B 动态事件。"""
            static_text, dynamic_text = "", ""
            pp = self._preprocessor
            if pp is not None and getattr(pp, "result", None):
                try:
                    frags = await asyncio.to_thread(
                        pp.retrieve_for_entity, agent.name, 2,
                        {agent.name} if agent.name else None)
                    if frags:
                        static_text = "\n---\n".join(f[:300] for f in frags)
                except Exception as e:
                    logger.debug("[Simulator] 量化静态召回失败 %s: %s", agent.name, e)
            if self._persist_events and pp is not None:
                try:
                    aliases: set[str] = set()
                    if pp.result:
                        aliases = set(pp.result.high_freq_entities.get(agent.name, set()))
                        aliases.update(pp.result.low_freq_entities.get(agent.name, set()))
                    query = (agent.name + " " + " ".join(aliases - {agent.name})).strip()
                    frags = await asyncio.to_thread(
                        pp.retrieve_dynamic_events, query, 3,
                        _cfg.deduction_similarity_threshold)
                    if frags:
                        dynamic_text = "\n---\n".join(frags)
                except Exception as e:
                    logger.debug("[Simulator] 量化动态召回失败 %s: %s", agent.name, e)
            elif not self._persist_events:
                # 隔离模式(蒙特卡洛)：仅用内存事件历史，不触碰 LanceDB 动态表
                mem = [e for e in self._event_history[-20:]
                       if agent.name in e.get("content", "") or e.get("agent") == agent.entity_id]
                if mem:
                    dynamic_text = "\n".join(f"- {e.get('content', '')[:80]}" for e in mem[-3:])
            return static_text, dynamic_text

        async def decide(agent: DeductionAgentProfile) -> dict[str, Any]:
            async with sem:
                static_text, dynamic_text = await _recall(agent)
                d = await self.reasoner.reason_quantified(
                    agent, states[agent.entity_id], re_engine,
                    recent_events=recent, other_context=others_ctx(agent.entity_id),
                    round_number=round_number, client=client,
                    static_knowledge=static_text, dynamic_memory=dynamic_text,
                    relationship_context=self._rel_context.get(agent.entity_id, {}).get("summary", ""),
                )
                d["actor_id"] = agent.entity_id
                return d

        if self._cancel is not None and self._cancel.is_set():
            return sim_round
        decisions = await asyncio.gather(*[decide(a) for a in ordered])

        # ── 轮前：自动效应（条件触发，逐实体结算）+ 延迟效应到期结算 ──
        ranges = re_engine.ranges()
        auto_deltas = re_engine.evaluate_auto_effects(states)
        for eid, d in auto_deltas.items():
            if eid in states:
                states[eid].apply_deltas(d, round_number, ranges)
        for eid, st in states.items():
            delay_d = st.resolve_delays(round_number)
            if delay_d:
                st.apply_deltas(delay_d, round_number, ranges)

        # 轮初快照(批量应用语义) + 交互解算（收集逐交互归因，供因果链硬档写入）
        name_to_id = {a.name: a.entity_id for a in self.agents}
        deltas, interactions = re_engine.resolve_round(
            states, decisions, name_to_id, self._env, collect_interactions=True)
        inter_by_actor: dict[str, list[dict[str, Any]]] = {}
        for _it in interactions:
            inter_by_actor.setdefault(_it["actor"], []).append(_it)
        for eid, d in deltas.items():
            if eid in states:
                states[eid].apply_deltas(d, round_number, ranges)

        # ── 轮后：调度延迟效应（动作触发的 delay_effects）──
        for dec in decisions:
            actor = dec.get("actor_id")
            if actor not in states:
                continue
            for action, sub_intensity, _target in re_engine._iter_subactions(dec):
                delay_cfg = re_engine.pack.get("delay_effects", {}).get(action)
                if delay_cfg and sub_intensity > 0:
                    dr = int(delay_cfg.get("delay", 1))
                    eff = {k: v * sub_intensity for k, v in delay_cfg.get("effects", {}).items()}
                    states[actor].schedule_delays(round_number, dr, eff)

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
            # W4: 量化轮事件写入 LanceDB 动态表(仅主推演 persist_events=True；优化器隔离不写)
            if self._persist_events and self._preprocessor is not None:
                try:
                    self._preprocessor.add_event_memory(
                        content=content, agent_id=actor,
                        round_number=round_number,
                        event_type=dec["action_type"], priority=0.5)
                except Exception as e:
                    logger.debug("[Simulator] 量化事件写入 LanceDB 失败: %s", e)
            # B+因果链: 量化轮写 Event 节点 + ACTED 边 + TARGETS/CAUSED(确定性数值归因)
            # 仅主推演 persist_events=True；优化器隔离不写。
            if self._persist_events and self.graph is not None:
                try:
                    _ts = datetime.now().isoformat()
                    _eid = f"evt-{uuid.uuid4().hex[:8]}"
                    _inters = inter_by_actor.get(actor, [])
                    _primary_tid = _inters[0]["target"] if _inters else ""
                    self.graph.add_event(_eid, content[:200], dec["action_type"], _ts, actor,
                                         round_number=round_number, target_id=_primary_tid)
                    self.graph.add_acted(actor, _eid, dec["action_type"], _ts)
                    _seen_targets: set[str] = set()
                    for _it in _inters:
                        _tid = _it["target"]
                        if _tid not in _seen_targets:
                            self.graph.add_targets(_eid, _tid)
                            _seen_targets.add(_tid)
                        for _metric, _amount in _it["deltas"].items():
                            self.graph.add_caused(_eid, _tid, _metric, float(_amount))
                except Exception as e:
                    logger.debug("[Simulator] 量化因果写入 Kuzu 失败: %s", e)
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

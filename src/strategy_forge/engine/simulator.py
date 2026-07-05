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
from string import Template
from typing import Any

import numpy as np

from strategy_forge.storage.graph_store import DeductionGraphStore

from ._utils import extract_text
from .models import DeductionAgentProfile, SimulationAction, SimulationRound
from .orchestrator import _PhaseCancelledError
from .preprocessor import DeductionPreprocessor

logger = logging.getLogger(__name__)


_ACTION_PROMPT = """你是一个推演模拟中的智能体。根据你的角色设定和当前世界状态，决定你的下一步行动。

## 你的固定人格（基于原文）
$persona

## 你的背景
$background

## 你的目标
$goals

## 当前轮次
第 $round_number 轮

## 近期模拟动态事件（重要！以下是其他角色刚刚做过的事）
$dynamic_memory

## 你的原著背景参考（仅供参考）
$static_knowledge

## 近期世界缓存
$recent_events

## 输出 JSON — 选择一种行动
```json
{
  "action": "post|reply|interact|observe",
  "target": "目标实体名或留空",
  "content": "行动内容 (30-100字)"
}
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
        algorithm_modules: list | None = None,
        fsm_override_store: dict | None = None,
    ) -> None:
        self.agents = agents
        self.graph = graph
        self._name_to_id: dict[str, str] = {a.name: a.entity_id for a in agents}
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
        self._algorithm_modules: list = algorithm_modules or []
        self._fsm_override_store: dict = fsm_override_store if fsm_override_store is not None else {}
        self._spatial_state = None   # cached SpatialState, updated after each module run
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

    # ── 用户强制 override（按体强制动作，跳过 FSM/LLM）──
    def _pop_override(self, agent: Any) -> dict | None:
        """取出并消费该 agent 的强制动作（按名称或 entity_id 匹配）。remaining 归零即删除。"""
        store = self._fsm_override_store
        if not store:
            return None
        key = None
        for k in (agent.name, agent.entity_id):
            if k in store:
                key = k
                break
        if key is None:
            return None
        ov = store[key]
        try:
            remaining = int(ov.get("remaining", 1))
        except (TypeError, ValueError):
            remaining = 1
        remaining -= 1
        if remaining <= 0:
            store.pop(key, None)
        else:
            ov["remaining"] = remaining
        return {
            "action_type": str(ov.get("action_type", "observe")),
            "intensity": float(ov.get("intensity", 0.6)),
            "target": str(ov.get("target", "") or ""),
            "rationale": f"[用户强制] {ov.get('action_type', 'observe')}"
                         + (f" → {ov.get('target')}" if ov.get("target") else ""),
        }

    def _describe_fsm_action(self, agent: Any, state: str, action_type: str) -> str:
        """FSM 确定性动作的数据差异化描述：突出该体当前最危险的受阈值约束指标。"""
        st = self._states.get(agent.entity_id) if self._quantified else None
        thresholds = self._rule_engine.thresholds() if self._rule_engine is not None else {}
        if st is not None and thresholds:
            worst_metric, worst_ratio, worst_val, worst_thr = None, None, None, None
            for m, thr in thresholds.items():
                try:
                    thr_f = float(thr)
                    val = float(st.get_metric(m))
                except (TypeError, ValueError):
                    continue
                ratio = val / thr_f if thr_f > 0 else val
                if worst_ratio is None or ratio < worst_ratio:
                    worst_metric, worst_ratio, worst_val, worst_thr = m, ratio, val, thr_f
            if worst_metric is not None:
                tag = "告急" if worst_val <= worst_thr * 1.2 else "偏紧"
                return f"{action_type}（{worst_metric}={worst_val:.0f}{tag}，阈值{worst_thr:.0f}｜{state}）"
        return f"{action_type}（{state}）"

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

        for agent in ordered:
            if self._cancel is not None and self._cancel.is_set():
                raise _PhaseCancelledError()
            action = await process_agent(agent)
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
                    agent.name, config.deduction_retrieve_top_k,
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
                    query, config.deduction_retrieve_top_k, min_similarity=config.deduction_similarity_threshold,
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
            messages = [Message(role="user", content=Template(_ACTION_PROMPT).substitute(
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
        alive_ids = [a.entity_id for a in alive_agents]
        if not alive_agents:
            return sim_round

        ordered = list(alive_agents)
        self._rng.shuffle(ordered)

        recent = "\n".join(
            f"- [{e.get('round','?')}] {e.get('agent_name','?')}: {e.get('content','')[:80]}"
            for e in self._event_history[-5:]
        ) or "（无近期事件）"

        # Pre-build O(1) entity-id→index map for spatial lookups
        alive_id_to_idx = {eid: i for i, eid in enumerate(alive_ids)} if alive_ids else {}

        def others_ctx(self_id: str) -> str:
            if len(alive_agents) > 30:
                return _build_summary_ctx(self_id, alive_agents, states, alive_id_to_idx,
                                          self._spatial_state, re_engine)
            lines = []
            idx_self = alive_id_to_idx.get(self_id)
            for a in alive_agents:
                if a.entity_id == self_id:
                    continue
                st = states[a.entity_id]
                line = st.to_prompt_context()
                # ── Trend perception: show multi-round metric deltas ──
                hist = getattr(st, "history", []) or []
                if len(hist) >= 6:  # at least 2 rounds with ~3 metrics each
                    trend_parts = []
                    # Reconstruct per-round metric values from history entries
                    by_round: dict[int, dict[str, float]] = {}
                    for entry in hist:
                        if isinstance(entry, dict):
                            r = entry.get("round", 0)
                            metric = entry.get("metric", "")
                            val = entry.get("new", entry.get("value", 0))
                            if r and metric:
                                by_round.setdefault(r, {})[metric] = float(val)
                    rounds = sorted(by_round.keys())
                    if len(rounds) >= 2:
                        first, last = by_round[rounds[0]], by_round[rounds[-1]]
                        for metric in re_engine.metrics():
                            v0, v1 = first.get(metric, 0), last.get(metric, 0)
                            if v0 > 0 and abs(v1 - v0) > 3.0:
                                symbol = "↑" if v1 > v0 else "↓"
                                trend_parts.append(f"{metric}{symbol}{abs(v1-v0):.0f}")
                    if trend_parts:
                        line += f"  多轮趋势: {', '.join(trend_parts)}"
                if self._spatial_state is not None and idx_self is not None:
                    sp = self._spatial_state
                    idx_other = alive_id_to_idx.get(a.entity_id)
                    if idx_other is not None and idx_self < len(sp.positions) and idx_other < len(sp.positions):
                        dist = float(np.linalg.norm(sp.positions[idx_self] - sp.positions[idx_other]))
                        line += f"  距离: {dist:.0f}m"
                lines.append(line)
            return "\n".join(lines) or "（无其他参与方）"

        def _build_summary_ctx(self_id, alive_agents, states, id_to_idx, spatial_state, re_engine):
            """N>30: bucket entities by distance + show global averages. Keeps prompts O(1)."""
            import numpy as _np
            ml = []
            idx_self = id_to_idx.get(self_id)
            metrics_list = re_engine.metrics()
            buckets = {"close": [], "mid": [], "far": []}
            for a in alive_agents:
                if a.entity_id == self_id: continue
                st = states[a.entity_id]
                dist = 9999.0
                if spatial_state is not None and idx_self is not None:
                    idx_other = id_to_idx.get(a.entity_id)
                    if idx_other is not None and idx_self < len(spatial_state.positions) and idx_other < len(spatial_state.positions):
                        dvec = spatial_state.positions[idx_self] - spatial_state.positions[idx_other]
                        dist = float(_np.linalg.norm(dvec))
                entry = {"name": st.name, "metrics": dict(st.metrics), "dist": dist}
                if dist < 50: buckets["close"].append(entry)
                elif dist < 200: buckets["mid"].append(entry)
                else: buckets["far"].append(entry)
            def _s(e, cnt):
                m = e["metrics"]
                kv = ", ".join(f"{k}={m.get(k,0):.0f}" for k in metrics_list[:cnt])
                return f"{e['name']}({kv}, d={e['dist']:.0f}m)"
            if buckets["close"]:
                buckets["close"].sort(key=lambda e: e["dist"])
                ml.append(f"邻近威胁（<50m, {len(buckets['close'])}个）:")
                for e in buckets["close"][:8]: ml.append(f"  {_s(e, 4)}")
            if buckets["mid"]:
                buckets["mid"].sort(key=lambda e: e["dist"])
                ml.append(f"中等距离（50-200m, {len(buckets['mid'])}个）:")
                for e in buckets["mid"][:5]: ml.append(f"  {_s(e, 2)}")
            fc = len(buckets["far"])
            if fc > 0: ml.append(f"远处（>200m, {fc}个）")
            all_m = [_np.array(list(states[e.entity_id].metrics.values()), dtype=_np.float64)
                      for e in alive_agents if e.entity_id != self_id]
            if all_m:
                arr = _np.stack(all_m)
                avgs = _np.mean(arr, axis=0)
                mins = _np.min(arr, axis=0)
                maxs = _np.max(arr, axis=0)
                parts = []
                for i, m in enumerate(metrics_list):
                    parts.append(f"{m}: avg={avgs[i]:.0f} [{mins[i]:.0f}-{maxs[i]:.0f}]")
                ml.append("全局统计: " + ", ".join(parts))
            return "\n".join(ml) if ml else "（无其他参与方）"

        def env_context() -> str:
            """Build terrain/weather description for the LLM prompt."""
            parts = []
            if self._env:
                weather = self._env.get("weather", "").strip()
                terrain = self._env.get("terrain", "").strip()
                if weather:
                    parts.append(f"天气: {weather}")
                if terrain:
                    parts.append(f"地形: {terrain}")
            if parts:
                return "； ".join(parts)
            return ""

        def spatial_self_ctx(self_id: str) -> str:
            if self._spatial_state is None:
                return ""
            sp = self._spatial_state
            idx = alive_id_to_idx.get(self_id)
            if idx is None or idx >= len(sp.positions):
                return ""
            pos = sp.positions[idx]
            dists: list[tuple[str, float]] = []
            for i, a in enumerate(alive_agents):
                if a.entity_id == self_id or i >= len(sp.positions):
                    continue
                d = float(np.linalg.norm(sp.positions[idx] - sp.positions[i]))
                if d < 200:
                    dists.append((a.name, d))
            dists.sort(key=lambda x: x[1])
            lines = [f"位置: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})"]
            if dists:
                lines.append("邻近实体: " + "; ".join(f"{n}({d:.0f}m)" for n, d in dists[:5]))
            # Collision contact
            in_contact = []
            for i, a in enumerate(alive_agents):
                if a.entity_id == self_id or i >= len(sp.positions):
                    continue
                d = float(np.linalg.norm(sp.positions[idx] - sp.positions[i]))
                min_d = sp.radii[idx] + sp.radii[i] if i < len(sp.radii) else 10
                if d < min_d:
                    in_contact.append(a.name)
            if in_contact:
                lines.append("接触/碰撞中: " + "、".join(in_contact))
            return "\n".join(lines)

        from strategy_forge.core.config import config as _cfg
        sem = asyncio.Semaphore(self._max_concurrent)

        # Clear round-level caches at start of round
        if self._preprocessor is not None and hasattr(self._preprocessor, "clear_round_cache"):
            self._preprocessor.clear_round_cache()

        async def _recall(agent: DeductionAgentProfile) -> tuple[str, str]:
            """量化轮的 LanceDB 语义召回：Path A 原著静态(只读，优化器也启用) + Path B 动态事件。"""
            static_text, dynamic_text = "", ""
            pp = self._preprocessor
            if pp is not None and getattr(pp, "result", None):
                try:
                    frags = await asyncio.to_thread(
                        pp.retrieve_for_entity, agent.name, _cfg.deduction_retrieve_top_k,
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
                        pp.retrieve_dynamic_events, query, _cfg.deduction_retrieve_top_k,
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

        # Pre-compute per-agent contexts once before concurrent execution
        _other_ctxs = {a.entity_id: others_ctx(a.entity_id) for a in alive_agents}
        _spatial_ctxs = {a.entity_id: spatial_self_ctx(a.entity_id) for a in alive_agents}
        _env_ctx = env_context()
        # ── Causal feedback: per-agent last round outcomes ──
        _causal_ctxs = {
            a.entity_id: "\n".join(getattr(self, "_last_round_outcomes", {}).get(a.entity_id, []))
            for a in alive_agents
        }

        async def decide(agent: DeductionAgentProfile) -> dict[str, Any]:
            async with sem:
                static_text, dynamic_text = await _recall(agent)
                rel_ctx = self._rel_context.get(agent.entity_id, {}).get("summary", "")
                causal = _causal_ctxs.get(agent.entity_id, "")
                if causal:
                    rel_ctx = f"上一轮行动效果: {causal}\n{rel_ctx}" if rel_ctx else f"上一轮行动效果: {causal}"
                d = await self.reasoner.reason_quantified(
                    agent, states[agent.entity_id], re_engine,
                    recent_events=recent, other_context=_other_ctxs.get(agent.entity_id, ""),
                    round_number=round_number, client=client,
                    static_knowledge=static_text, dynamic_memory=dynamic_text,
                    relationship_context=rel_ctx,
                    spatial_context=_spatial_ctxs.get(agent.entity_id, ""),
                    env_context=_env_ctx,
                )
                d["actor_id"] = agent.entity_id
                return d

        if self._cancel is not None and self._cancel.is_set():
            return sim_round
        # 逐代理决策（逐个等待，确保取消信号在代理之间能被及时检测）
        # ── FSM 分流：上一轮的 FSM 状态决定本轮哪些代理走 LLM ──
        fsm_states = getattr(self, "_last_fsm_states", None)
        fsm_actions = getattr(self, "_last_fsm_actions", None)
        fsm_command = getattr(self, "_last_fsm_command_states", {"combat"})
        decisions: list[dict[str, Any]] = []

        for i, agent in enumerate(ordered):
            if self._cancel is not None and self._cancel.is_set():
                self._log("simulation", f"取消信号：已处理 {i}/{len(ordered)} 代理后停止")
                return sim_round
            # ── 用户强制 override：最高优先，跳过 FSM 与 LLM ──
            ov = self._pop_override(agent)
            if ov is not None:
                ov["actor_id"] = agent.entity_id
                ov["driver"] = "forced"
                decisions.append(ov)
                self._log("simulation", f"[用户强制] {agent.name} → {ov.get('action_type')}")
                continue
            # Check if FSM should drive this agent
            state = fsm_states[i] if fsm_states is not None and i < len(fsm_states) else None
            if state is not None and state not in fsm_command:
                # FSM deterministic action — skip LLM
                act = None
                if fsm_actions is not None and i < len(fsm_actions) and fsm_actions[i]:
                    act = dict(fsm_actions[i])
                if act is None:
                    act = {"action_type": "observe", "intensity": 0.3, "target": ""}
                # 数据差异化描述：结合当前指标最危险项，避免"[FSM] observe"千篇一律
                act["rationale"] = self._describe_fsm_action(agent, state, act.get("action_type", "observe"))
                act["driver"] = "fsm"
                act["actor_id"] = agent.entity_id
                decisions.append(act)
                continue
            # LLM decision for command-state agents
            raw = await decide(agent)
            if isinstance(raw, BaseException):
                self._log("simulation", f"agent {agent.name} 决策失败: {raw}")
            else:
                decisions.append(raw)
        # raw_results kept below for backward compat
        raw_results = decisions

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
        deltas, interactions = re_engine.resolve_round(
            states, decisions, self._name_to_id, self._env, collect_interactions=True)
        inter_by_actor: dict[str, list[dict[str, Any]]] = {}
        for _it in interactions:
            bucket = inter_by_actor.get(_it["actor"])
            if bucket is None:
                inter_by_actor[_it["actor"]] = [_it]
            else:
                bucket.append(_it)
        # Bulk JIT delta application for large entity counts
        if len(states) >= 20:
            _bulk_apply_deltas(states, deltas, ranges, re_engine.metrics())
        else:
            for eid, d in deltas.items():
                if eid in states:
                    states[eid].apply_deltas(d, round_number, ranges)

        # ── 轮后：调度延迟效应 + 保存因果反馈 ──
        self._last_round_outcomes: dict[str, list[dict]] = {}
        for dec in decisions:
            actor = dec.get("actor_id")
            if actor not in states:
                continue
            # Causal feedback for next round
            my_deltas = deltas.get(actor, {})
            if my_deltas:
                summary = ", ".join(f"{k}{v:+.1f}" for k, v in my_deltas.items())
                target = dec.get("target", "")
                action = dec.get("action_type", "?")
                target_name = target if target else "自身"
                self._last_round_outcomes.setdefault(actor, []).append(
                    f"你的 {action} 对 {target_name} 造成: {summary}" if target else
                    f"你的 {action} 自身效应: {summary}"
                )
            # Delay effect scheduling
            for action, sub_intensity, _target in re_engine._iter_subactions(dec):
                delay_cfg = re_engine.pack.get("delay_effects", {}).get(action)
                if delay_cfg and sub_intensity > 0:
                    dr = int(delay_cfg.get("delay", 1))
                    eff = {k: v * sub_intensity for k, v in delay_cfg.get("effects", {}).items()}
                    states[actor].schedule_delays(round_number, dr, eff)

        # ── Algorithm module chain (ODE + Physics) ──
        if self._algorithm_modules and self._rule_engine is not None:
            from strategy_forge.algorithms.module_utils import (
                apply_context_results,
                build_context,
            )
            entity_ids = [a.entity_id for a in self.agents if a.entity_id in states]
            ctx = build_context(states, self._rule_engine, entity_ids, round_number,
                                prev_spatial=getattr(self, "_spatial_state", None))
            for mod in self._algorithm_modules:
                try:
                    ctx = mod.execute(ctx)
                except Exception as e:
                    self._log("simulation", f"模块 {mod.name} 执行异常: {e}")
            apply_context_results(ctx, states, entity_ids, self._rule_engine)
            # Cache spatial state for next round's decision prompts
            if hasattr(ctx, "spatial"):
                self._spatial_state = ctx.spatial
            # Save FSM state for next round's agent decision split
            if "fsm.agent_states" in ctx.metadata:
                self._last_fsm_states = list(ctx.metadata["fsm.agent_states"])
                self._last_fsm_actions = list(ctx.metadata.get("fsm.agent_actions", []))
                self._last_fsm_command_states = set(
                    ctx.metadata.get("fsm.command_states", ["combat"])
                )

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
                                         round_number=round_number, target_id=_primary_tid,
                                         effect=delta_txt, driver=dec.get("driver", "llm"))
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

        # Build dashboard snapshot for frontend
        sim_round.state_delta["snapshot"] = _build_state_snapshot(
            states, re_engine.thresholds(), self._event_history, round_number, re_engine)

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


def _bulk_apply_deltas(
    states: dict[str, Any],
    deltas: dict[str, dict[str, float]],
    ranges: dict[str, Any],
    metric_names: list[str],
) -> None:
    """Bulk JIT delta application for large entity counts."""
    from strategy_forge.engine._jit_utils import batch_apply_deltas

    entity_ids = list(states.keys())
    if not entity_ids:
        return
    N = len(entity_ids)
    M = len(metric_names)
    metrics_arr = np.zeros((N, M), dtype=np.float64)
    deltas_arr = np.zeros((N, M), dtype=np.float64)
    lo_arr = np.full(M, -1e12, dtype=np.float64)
    hi_arr = np.full(M, 1e12, dtype=np.float64)

    for i, eid in enumerate(entity_ids):
        st = states[eid]
        for m, name in enumerate(metric_names):
            metrics_arr[i, m] = float(st.metrics.get(name, 0.0))
            d = deltas.get(eid, {}).get(name, 0.0)
            deltas_arr[i, m] = float(d) if d is not None else 0.0

    for m, name in enumerate(metric_names):
        rng = ranges.get(name, [0.0, 100.0])
        if rng and len(rng) >= 2:
            lo_arr[m] = float(rng[0])
            hi_arr[m] = float(rng[1])

    batch_apply_deltas(metrics_arr, deltas_arr, lo_arr, hi_arr)

    for i, eid in enumerate(entity_ids):
        st = states[eid]
        for m, name in enumerate(metric_names):
            st.metrics[name] = float(metrics_arr[i, m])


def _build_state_snapshot(states: dict, thresholds: dict, event_history: list,
                          round_num: int, re_engine: Any) -> dict:
    """Build structured snapshot for frontend dashboard panel (no LLM)."""
    metrics_list = re_engine.metrics() if re_engine else []
    # Alerts: metrics within 20% of threshold
    alerts = []
    for st in states.values():
        if not hasattr(st, 'name'):
            continue
        for metric, threshold in thresholds.items():
            val = st.metrics.get(metric, 0)
            if val <= threshold * 1.2:
                severity = "critical" if val <= threshold else "warning"
                alerts.append({
                    "entity": getattr(st, 'name', '?'),
                    "metric": metric, "value": round(val, 1),
                    "threshold": threshold, "severity": severity,
                })
    alerts.sort(key=lambda a: a["value"] - a["threshold"])
    # Group stats by domain
    groups = {}
    for st in states.values():
        domain = getattr(st, "domain", "generic")
        if domain not in groups:
            groups[domain] = {"names": [], "metrics": {m: [] for m in metrics_list}}
        groups[domain]["names"].append(getattr(st, 'name', '?'))
        for m in metrics_list:
            groups[domain]["metrics"][m].append(st.metrics.get(m, 0))
    group_stats = {}
    for domain, data in groups.items():
        group_stats[domain] = {
            "count": len(data["names"]),
            "metrics": {m: round(np.mean(vals), 1) for m, vals in data["metrics"].items() if vals},
        }
    # Recent events
    recent = []
    for e in event_history[-3:]:
        recent.append({
            "agent": e.get("agent_name", "?"),
            "action": e.get("action", ""),
            "content": (e.get("content", "") or "")[:80],
            "round": e.get("round", round_num),
        })
    return {"alerts": alerts[:5], "groups": group_stats, "recent": recent,
            "round": round_num, "entity_count": len(states)}


def _parse_action_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

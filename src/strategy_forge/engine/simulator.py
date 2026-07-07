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

from strategy_forge.core.llm_client import LLMConnectionError
from strategy_forge.storage.graph_store import DeductionGraphStore

from ._utils import extract_text
from .models import DeductionAgentProfile, SimulationAction, SimulationRound
from .orchestrator import _PhaseCancelledError
from .preprocessor import DeductionPreprocessor

logger = logging.getLogger(__name__)

_METRIC_NAME: dict[str, str] = {
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
_mn = _METRIC_NAME.get


def _delta_desc(v: float) -> str:
    """数值 → 定性描述。|v|>15 大幅, |v|>5 默认, |v|<=5 轻微。"""
    mag = abs(v)
    if mag > 15:
        return "大幅"
    if mag > 5:
        return ""
    return "轻微"


def _delta_dir(v: float) -> str:
    return "增长" if v > 0 else "消耗" if v < 0 else "持平"


def _build_causal_feedback(
    actor_id: str, actor_name: str, action: str, target_id: str, target_name: str,
    my_deltas: dict[str, float], target_deltas: dict[str, float],
    auto_deltas: dict[str, float], event_history: list[dict],
    round_number: int, name_to_id: dict[str, str],
) -> str:
    """构建多段落叙事化因果反馈：自身效应 / 目标影响 / 连锁反应 / 后续反应。"""
    parts: list[str] = []
    # 自身效应
    if my_deltas:
        items = []
        for k, v in my_deltas.items():
            label = _mn(k, k)
            desc = _delta_desc(v)
            items.append(f"{label}{desc}{_delta_dir(v)}({v:+.0f})")
        if items:
            parts.append("自身 — " + "，".join(items[:4]))
    # 目标影响
    if target_id and target_id != actor_id and target_deltas:
        items = []
        for k, v in target_deltas.items():
            if v < 0:
                label = _mn(k, k)
                desc = _delta_desc(v)
                items.append(f"{label}{desc}{_delta_dir(v)}({v:+.0f})")
        if items:
            parts.append(f"对{target_name} — " + "，".join(items[:3]))
    # 连锁反应（auto effects）
    if auto_deltas:
        items = []
        for k, v in auto_deltas.items():
            label = _mn(k, k)
            items.append(f"{label}{_delta_dir(v)}({v:+.0f})")
        if items:
            parts.append("连锁反应 — " + "，".join(items[:3]))
    # 后续反应：从同一轮 event_history 中提取他人对 actor 或 target 的回应
    reactions = _extract_reactions(actor_name, target_name, event_history, round_number, name_to_id)
    if reactions:
        parts.append("后续反应 — " + reactions)
    if not parts:
        return f"你的 {action} 已执行（本轮无显著数值变化）"
    return "## 上轮回顾\n" + "\n".join(f"  • {p}" for p in parts)


def _extract_reactions(
    actor_name: str, target_name: str, event_history: list[dict],
    round_number: int, name_to_id: dict[str, str],
) -> str:
    """从当前轮事件历史中提取他人对 acter/target 的回应。"""
    reacting: list[str] = []
    target_events = [e for e in event_history
                     if e.get("round") == round_number
                     and e.get("agent_name", "") not in (actor_name, "")]
    for e in target_events[-6:]:
        name = e.get("agent_name", "?")
        content = (e.get("content", "") or "")[:50]
        if name == target_name:
            reacting.append(f"{name}{content[:40]}")
        elif target_name in content:
            reacting.append(f"{name}回应{target_name}: {content[:35]}")
    if not reacting:
        return ""
    return "；".join(reacting[:4])


# ── 信息传播：信任度驱动延迟/失真 ──

def _compute_delay(trust: float, max_delay: int = 4) -> int:
    """trust ∈ [-5, +5] → delay ∈ [max_delay, 0]（线性）。"""
    if trust >= 4.0:
        return 0
    normalized = max(0.0, (4.0 - trust) / 9.0)
    return max(0, round(normalized * max_delay))


def _compute_distortion(trust: float) -> float:
    """trust ∈ [-5, +5] → distortion ∈ [0.0, 0.30]。"""
    if trust >= 4.0:
        return 0.0
    normalized = max(0.0, (4.0 - trust) / 9.0)
    return normalized * 0.30


def _distort_event_content(content_raw: str, distortion: float) -> str:
    """对事件内容施加数值模糊：低失真保留结构改区间、高失真用定性描述。"""
    if distortion < 0.05 or not content_raw:
        return content_raw
    import re as _re
    parts = _re.findall(r'((?:[\u4e00-\u9fff]|\w)+(?:[+-]\d+(?:\.\d+)?))', content_raw)
    if not parts or distortion >= 0.25:
        # 高失真：去掉所有精确数值，用定性词替换
        return _re.sub(r'[+-]?\d+(?:\.\d+)?', '?', content_raw)
    result = content_raw
    for tok in parts:
        m = _re.match(r'(.*?)([+-]\d+(?:\.\d+)?)', tok)
        if m:
            prefix = m.group(1)
            val = float(m.group(2))
            spread = abs(val) * distortion
            lo, hi = round(val - spread), round(val + spread)
            if lo == hi:
                replacement = f"{prefix}{val:+.0f}"
            else:
                replacement = f"{prefix}约{lo}~{hi}"
            result = result.replace(tok, replacement, 1)
    return result


class ConnectionFailureError(Exception):
    """连接故障导致推演中断（含原文，供界面日志展示）。"""
    pass


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

        # ── 前瞻规划：Rollout 模式 ──
        self._enable_rollout: bool = False
        self._baseline_decisions: dict[str, dict[str, Any]] = {}

        # ── 信息传播：每 agent 的知识队列 ──
        self._agent_knowledge: dict[str, list[dict[str, Any]]] = {}
        # ── 谍报：每 agent 对特定目标的信息优势 ──
        self._intel_bonuses: dict[str, dict[str, float]] = {}  # {source_id: {target_name: bonus}}
        # ── 人格动态化：每 agent 的反思轮次追踪 ──
        self._last_reflection_round: dict[str, int] = {}
        self._personality_log: list[dict[str, Any]] = []  # [{round, agent, old_extra, new_extra}]

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

    def _augment_recall_query(self, base: str, entity_id: str) -> str:
        """④ 关系邻居增强召回：把 Kuzu 盟友/对手名拼进动态事件召回 query，聚焦相关事件。

        默认关（FORGE_RECALL_REL_BOOST=0）时直接返回 base，行为与现状逐字一致。
        """
        from strategy_forge.core.config import config
        if not config.deduction_recall_rel_boost:
            return base
        rel = self._rel_context.get(entity_id, {}) or {}
        names: list[str] = []
        cap = max(0, config.deduction_recall_rel_max)
        for n in (list(rel.get("allies", [])) + list(rel.get("opponents", []))):
            if n and n not in names:
                names.append(n)
            if len(names) >= cap:
                break
        return (base + " " + " ".join(names)).strip() if names else base

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

        if self._cancel is not None and self._cancel.is_set():
            raise _PhaseCancelledError()

        sem = asyncio.Semaphore(self._max_concurrent)

        async def process_agent(agent: DeductionAgentProfile) -> SimulationAction | None:
            async with sem:
                # 取信号量后再查一次取消，便于并发批内尽早短路
                if self._cancel is not None and self._cancel.is_set():
                    return None
                from strategy_forge.core.config import config
                from strategy_forge.core.llm_client import LLMConnectionError
                fails = 0
                max_passes = max(0, config.deduction_llm_retry_passes)
                while True:
                    try:
                        return await self._agent_decide(client, agent, round_number)
                    except LLMConnectionError as e:
                        fails += 1
                        if fails > max_passes:
                            raise
                        delay = min(60.0, 5.0 * (2 ** (fails - 1)))
                        self._log("simulation", f"{agent.name} LLM 连接失败({fails}/{max_passes+1})，{delay:.0f}s 后重试… | {e.endpoint}: {e.cause}")
                        await asyncio.sleep(delay)

        # 并发决策（上限 = FORGE_MAX_CONCURRENT），随后按 ordered 原序回填以保持确定性
        results = await asyncio.gather(
            *(process_agent(agent) for agent in ordered), return_exceptions=True)
        conn_fails = sum(1 for r in results if isinstance(r, LLMConnectionError))
        if conn_fails > 0:
            from strategy_forge.core.config import config
            ratio = conn_fails / max(1, len(ordered))
            if ratio >= config.deduction_sim_fail_ratio:
                first = next((r for r in results if isinstance(r, LLMConnectionError)), None)
                raise ConnectionFailureError(str(first) if first else f"连接故障：{conn_fails}/{len(ordered)} agent 无法连接 LLM")
        for agent, action in zip(ordered, results, strict=False):
            if isinstance(action, BaseException):
                self._log("simulation", f"agent {agent.name} 决策失败: {action}")
                continue
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
        from strategy_forge.core.config import config
        # ── 近期事件 ──
        recent = self._event_history[-max(1, config.deduction_sim_recent_events):]
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
                query = self._augment_recall_query(query, agent.entity_id)
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

    REFLECT_INTERVAL = 5      # 每5轮触发一次人格反思
    REFLECT_DELTA_THRESHOLD = 25.0  # 单指标累计变化超过此阈值也触发

    # ── 前瞻规划：Rollout 反应规则 ──
    _REACTION_RULES: list[tuple[str, str, str]] = [
        ("strength", "<30", "defend"),
        ("strength", "<50", "defensive_buildup"),
        ("morale", "<20", "retreat"),
        ("supply", "<15", "invest"),
        ("cash_flow", "<15", "defensive_buildup"),
        ("morale", ">90", "attack"),
        ("support_rate", "<20", "campaign"),
    ]

    async def _rollout_candidates(
        self, agent: Any, candidates: list[dict[str, Any]],
        rule_engine: Any, current_states: dict[str, Any],
        round_number: int, lookahead: int = 3,
    ) -> list[dict[str, Any]]:
        """对每个候选动作做 2-3 轮轻量 rollout，返回附 future_score 的候选列表。

        三层误差消除：
          第一层：其他 agent 使用本轮真实 LLM 决策（_baseline_decisions）
          第二层：检测被打击方的反应（_REACTION_RULES）
        """
        import copy
        baseline = getattr(self, "_baseline_decisions", {})

        for cand in candidates:
            cloned_states = {eid: copy.deepcopy(st) for eid, st in current_states.items()}
            current_actions = dict(baseline)
            total_score = 0.0

            for r in range(lookahead):
                decisions = []
                for a in self.agents:
                    eid = a.entity_id
                    if eid not in cloned_states:
                        continue
                    if a.entity_id == agent.entity_id:
                        # 候选方：第一轮执行候选动作，后续观察
                        decisions.append(cand if r == 0 else {
                            "actor_id": eid, "action_type": "observe",
                            "intensity": 0.3, "target": "",
                        })
                    else:
                        act = current_actions.get(eid) or {
                            "actor_id": eid, "action_type": "observe",
                            "intensity": 0.3, "target": "",
                        }
                        decisions.append(act)

                # 应用效果
                try:
                    deltas, _interactions = rule_engine.resolve_round(
                        cloned_states, decisions, self._name_to_id, self._env,
                        collect_interactions=False)
                except Exception:
                    break

                for eid, d in deltas.items():
                    if eid in cloned_states:
                        cloned_states[eid].apply_deltas(d, round_number + r,
                                                         rule_engine.ranges())

                # 第二层：检测反应
                for eid, d in deltas.items():
                    if eid == agent.entity_id or eid not in cloned_states:
                        continue
                    st = cloned_states[eid]
                    for metric, cond, new_action in self._REACTION_RULES:
                        if metric in st.metrics:
                            val = st.metrics[metric]
                            cond_ok = False
                            if cond.startswith("<"):
                                cond_ok = val < float(cond[1:])
                            elif cond.startswith(">"):
                                cond_ok = val > float(cond[1:])
                            if cond_ok:
                                current_actions[eid] = {
                                    "actor_id": eid, "action_type": new_action,
                                    "intensity": 0.6, "target": "",
                                }
                                break

                # 累积评分：考察 agent 自身的指标健康度
                if agent.entity_id in cloned_states:
                    st = cloned_states[agent.entity_id]
                    for m, v in st.metrics.items():
                        total_score += v / 100.0  # 简单加权

            cand["_future_score"] = round(total_score / max(lookahead, 1), 2)
            cand["_rollout_lookahead"] = lookahead

        return candidates

    async def _reflect_and_adapt(self, agent: Any, round_number: int,
                                   client: Any) -> str | None:
        """人格动态化：根据近期经历微调 agent 的行为准则。

        仅修改 system_prompt_extra，不覆盖原始 persona/background。
        返回新 system_prompt_extra 字符串，或 None（无变化）。
        """
        from strategy_forge.core.llm_client import Message
        state = self._states.get(agent.entity_id)
        if state is None:
            return None
        history = getattr(state, "history", []) or []
        recent_history = history[-20:]  # 最近20条变化记录
        if not recent_history:
            return None

        # 计算各指标累计变化
        delta_summary: list[str] = []
        deltas_by_metric: dict[str, float] = {}
        for h in recent_history:
            m = h.get("metric", "")
            d = h.get("delta", 0)
            if m:
                deltas_by_metric[m] = deltas_by_metric.get(m, 0) + float(d)
        for m, d in deltas_by_metric.items():
            label = _METRIC_NAME.get(m, m)
            direction = "↑" if d > 0 else "↓"
            delta_summary.append(f"{label}{direction}{abs(d):.0f}")

        # 因果反馈摘要
        causal = getattr(self, "_last_round_outcomes", {}).get(agent.entity_id, "")
        causal_short = (causal[:200] + "...") if len(causal) > 200 else causal

        # 当前状态快照
        metrics = getattr(state, "metrics", {})
        status_summary: list[str] = []
        for m, v in metrics.items():
            if v < 30:
                status_summary.append(f"{_METRIC_NAME.get(m,m)}告急({v:.0f})")

        prompt = (
            f"你是 {agent.name} 的潜意识。回顾你近期的经历，判断你的性格是否需要微调。\n\n"
            f"## 你的核心人格（不可改动）\n{agent.persona or '（无）'}\n\n"
            f"## 你现有的行为准则\n{agent.system_prompt_extra or '（无，完全依据核心人格）'}\n\n"
            f"## 近期指标变化\n{', '.join(delta_summary) if delta_summary else '无显著变化'}\n\n"
            f"## 风险信号\n{'; '.join(status_summary) if status_summary else '无告急指标'}\n\n"
            f"## 近期行动复盘\n{causal_short if causal_short else '无'}\n\n"
            f"## 任务\n"
            f"根据以上经历，判断是否需要添加一条新的行为准则（或修正旧准则），"
            f"使你的行为更符合当前的处境。\n"
            f"- 输出格式：一行简短中文准则（20字以内），直接陈述。\n"
            f"- 如果当前人格已足够应对，输出\"无需调整\"。\n"
            f"- 仅添加/修正，不删除原有准则。\n"
            f"- 示例：\"长期补给不足时应优先休整\" \"连胜后保持谨慎避免冒进\" \"与盟友保持紧密协作\"\n"
            f"\n只输出准则本身或\"无需调整\"，不要解释。"
        )

        try:
            resp = await client.chat(
                [Message(role="user", content=prompt)],
                system="你是潜意识分析师，输出简短行为准则或'无需调整'。",
                temperature=0.3,
                max_tokens=80,
            )
            text = extract_text(resp).strip()
            if not text or "无需调整" in text or len(text) < 2:
                return None
            # 更新 agent 的行为准则
            old_extra = agent.system_prompt_extra
            if old_extra and text not in old_extra:
                agent.system_prompt_extra = f"{old_extra}；{text}"
            elif not old_extra:
                agent.system_prompt_extra = text
            else:
                return None  # 重复准则，不更新
            # 记录日志
            self._personality_log.append({
                "round": round_number, "agent": agent.name,
                "old_extra": old_extra, "new_extra": agent.system_prompt_extra,
            })
            self._log("simulation",
                       f"[人格演化] {agent.name} 新增准则: {text} (R{round_number})")
            return agent.system_prompt_extra
        except Exception as e:
            logger.debug("[Simulator] _reflect_and_adapt failed for %s: %s",
                         agent.name, e)
            return None

    def _dispatch_events(self, round_number: int) -> None:
        """将本轮 _event_history 中新事件按信任度分发至各 agent 知识队列。"""
        from uuid import uuid4
        alive_ids = [a.entity_id for a in self.agents if a.entity_id in self._states]
        name_to_id = self._name_to_id
        for evt in self._event_history:
            if evt.get("round") != round_number:
                continue
            actor_id = evt.get("agent", "")
            actor_name = evt.get("agent_name", "")
            content = evt.get("content", "")
            if not actor_id or not content:
                continue
            for a_id in alive_ids:
                if a_id == actor_id:
                    continue
                # Trust lookup: matrix is indexed by [entity_id][name]; seed_trust stores
                # by source entity_id → target name. We need observer's entity_id (a_id)
                # looking at actor's name.
                trust = self.reasoner.get_trust(a_id, actor_name)
                # Intel bonus: if observer has gathered intel on actor, reduce delay + distortion
                intel_bonus = (self._intel_bonuses.get(a_id, {}).get(actor_name, 0.0)
                               + self._intel_bonuses.get(a_id, {}).get(actor_id, 0.0))
                delay = max(0, _compute_delay(trust + intel_bonus * 2.0) - int(intel_bonus))
                distortion = _compute_distortion(trust + intel_bonus * 2.0)
                delivered_content = _distort_event_content(content, distortion)
                self._agent_knowledge.setdefault(a_id, []).append({
                    "event_id": str(uuid4()),
                    "round_occurred": round_number,
                    "deliver_round": round_number + delay,
                    "content_raw": content,
                    "content_delivered": delivered_content,
                    "actor": actor_name,
                    "target": next((k for k, v in name_to_id.items() if v == actor_id), ""),
                    "importance": 0.5,
                    "_base_distortion": distortion,  # for information decay
                })

    def _deliver_ripe_knowledge(self, agent_id: str, current_round: int) -> list[dict[str, Any]]:
        """交付该 agent 的已熟事件（deliver_round <= 当前轮），从队列中移除。"""
        ripe = []
        remaining = []
        for k in self._agent_knowledge.get(agent_id, []):
            # Apply information decay: events older than current_round lose precision
            age = current_round - k.get("round_occurred", current_round)
            if age > 1:
                base_dist = k.get("_base_distortion", 0.0)
                extra = 0.05 * (age - 1)
                total_dist = min(0.40, base_dist + extra)
                k["content_delivered"] = _distort_event_content(
                    k.get("content_raw", ""), total_dist)
            if k["deliver_round"] <= current_round:
                ripe.append(k)
            else:
                remaining.append(k)
        if remaining != self._agent_knowledge.get(agent_id, []):
            self._agent_knowledge[agent_id] = remaining
        return ripe

    def _update_reputation_after_round(self, decisions: list[dict[str, Any]],
                                        name_to_id: dict[str, str]) -> None:
        """根据本轮交互自动更新 agent 间的信任度。"""
        for dec in decisions:
            actor = dec.get("actor_id", "")
            action = dec.get("action_type", "")
            target_id = dec.get("target", "")
            intensity = float(dec.get("intensity", 0.5))
            if not actor or not target_id:
                continue
            target_name = next((k for k, v in name_to_id.items() if v == target_id), "")
            if not target_name:
                continue
            delta = 0.0
            if action in self.reasoner._TRUST_HOSTILE_ACTIONS:
                delta = -2.5 * intensity
            elif action in self.reasoner._TRUST_FRIENDLY_ACTIONS:
                delta = +1.5 * intensity
            if abs(delta) > 0.01:
                self.reasoner.adjust_trust(actor, target_name, delta)
                self.reasoner.adjust_trust(target_name, actor, delta * 0.6)

    async def _run_round_quantified(self, round_number: int) -> SimulationRound:
        from datetime import datetime

        from strategy_forge.core.config import config as _cfg
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

        # Pre-build O(1) entity-id→index map for spatial lookups
        alive_id_to_idx = {eid: i for i, eid in enumerate(alive_ids)} if alive_ids else {}

        def others_ctx(self_id: str) -> str:
            # B1: 只渲染 Top-K 最相关他方(盟友/对手>最近>最危急)，其余合并为全局摘要，
            # 把每 agent prompt 的他方块从 O(N) 降到 O(K)，消除 O(N^2) 与逐轮膨胀。
            from strategy_forge.core.config import config as _c
            topk = max(1, int(getattr(_c, "deduction_sim_others_topk", 10)))
            metrics_list = re_engine.metrics()
            idx_self = alive_id_to_idx.get(self_id)
            sp = self._spatial_state
            rel = getattr(self, "_rel_context", {}).get(self_id, {}) or {}
            important = set(rel.get("allies", []) or []) | set(rel.get("opponents", []) or [])

            def _dist(a) -> float | None:
                if sp is not None and idx_self is not None:
                    io = alive_id_to_idx.get(a.entity_id)
                    if io is not None and idx_self < len(sp.positions) and io < len(sp.positions):
                        return float(np.linalg.norm(sp.positions[idx_self] - sp.positions[io]))
                return None

            others = [a for a in alive_agents if a.entity_id != self_id]
            if not others:
                return "（无其他参与方）"

            def _salience(a):
                st = states[a.entity_id]
                rel_pri = 0 if st.name in important else 1
                d = _dist(a)
                mtot = sum(st.metrics.values()) if st.metrics else 0.0
                return (rel_pri, d if d is not None else 1e9, mtot)

            ranked = sorted(others, key=_salience)
            shown, rest = ranked[:topk], ranked[topk:]

            def _detail(a) -> str:
                st = states[a.entity_id]
                line = st.to_prompt_context()
                hist = getattr(st, "history", []) or []
                if len(hist) >= 6:
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
                        trend_parts = []
                        for metric in metrics_list:
                            v0, v1 = first.get(metric, 0), last.get(metric, 0)
                            if v0 > 0 and abs(v1 - v0) > 3.0:
                                trend_parts.append(f"{metric}{'↑' if v1 > v0 else '↓'}{abs(v1-v0):.0f}")
                        if trend_parts:
                            line += f"  多轮趋势: {', '.join(trend_parts)}"
                d = _dist(a)
                if d is not None:
                    line += f"  距离: {d:.0f}m"
                return line

            lines = [_detail(a) for a in shown]
            if rest:
                arr = np.stack([
                    np.array([states[a.entity_id].metrics.get(m, 0.0) for m in metrics_list],
                             dtype=np.float64) for a in rest]) if metrics_list else None
                if arr is not None and arr.size:
                    avgs, mins, maxs = arr.mean(0), arr.min(0), arr.max(0)
                    stat = ", ".join(f"{m}: avg={avgs[i]:.0f} [{mins[i]:.0f}-{maxs[i]:.0f}]"
                                     for i, m in enumerate(metrics_list))
                    lines.append(f"其余 {len(rest)} 方（全局）: {stat}")
                else:
                    lines.append(f"其余 {len(rest)} 方")
            return "\n".join(lines) or "（无其他参与方）"

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

        sem = asyncio.Semaphore(self._max_concurrent)

        # Clear round-level caches at start of round
        if self._preprocessor is not None and hasattr(self._preprocessor, "clear_round_cache"):
            self._preprocessor.clear_round_cache()

        async def _recall(agent: DeductionAgentProfile) -> tuple[str, str]:
            """量化轮的 LanceDB 语义召回：Path A 原著静态(只读，优化器也启用) + Path B 动态事件。"""
            static_text, dynamic_text = "", ""
            # B2: 模拟召回片段数与字符预算独立于 graph/agents 的检索深度
            _rk = max(1, _cfg.deduction_sim_recall_topk)
            _rc = max(200, _cfg.deduction_sim_recall_chars)
            pp = self._preprocessor
            if pp is not None and getattr(pp, "result", None):
                try:
                    frags = await asyncio.to_thread(
                        pp.retrieve_for_entity, agent.name, _rk,
                        {agent.name} if agent.name else None)
                    if frags:
                        static_text = "\n---\n".join(f[:300] for f in frags[:_rk])[:_rc]
                except Exception as e:
                    logger.debug("[Simulator] 量化静态召回失败 %s: %s", agent.name, e)
            if self._persist_events and pp is not None:
                try:
                    aliases: set[str] = set()
                    if pp.result:
                        aliases = set(pp.result.high_freq_entities.get(agent.name, set()))
                        aliases.update(pp.result.low_freq_entities.get(agent.name, set()))
                    query = (agent.name + " " + " ".join(aliases - {agent.name})).strip()
                    query = self._augment_recall_query(query, agent.entity_id)
                    frags = await asyncio.to_thread(
                        pp.retrieve_dynamic_events, query, _rk,
                        _cfg.deduction_similarity_threshold)
                    if frags:
                        dynamic_text = "\n---\n".join(frags[:_rk])[:_rc]
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
        # ── 增强因果反馈：per-agent 上次行动复盘 ──
        _causal_ctxs = {
            a.entity_id: getattr(self, "_last_round_outcomes", {}).get(a.entity_id, "")
            for a in alive_agents
        }
        # ── 信息传播：per-agent 近期事件（信任度驱动延迟/失真）──
        _recent_ctxs: dict[str, str] = {}
        for a in alive_agents:
            ripe = self._deliver_ripe_knowledge(a.entity_id, round_number)
            items: list[str] = []
            for k in ripe[-8:]:
                items.append(f"• [{k['round_occurred']}] {k['content_delivered']}")
            # 补充 agent 自身相关的事件（来自 _event_history）
            own_events = [e for e in self._event_history[-5:]
                          if e.get("agent") == a.entity_id or a.name in e.get("content", "")]
            for e in own_events:
                text = e.get("content", "")[:80]
                items.append(f"• [R{e.get('round','?')}] {text}")
            _recent_ctxs[a.entity_id] = "\n".join(items[-8:]) or "（无近期事件）"

        async def decide(agent: DeductionAgentProfile) -> dict[str, Any] | None:
            from strategy_forge.core.config import config as _cr
            from strategy_forge.core.llm_client import LLMConnectionError
            fails = 0
            max_passes = max(0, _cr.deduction_llm_retry_passes)
            last_err = None
            while True:
                try:
                    async with sem:
                        if self._cancel is not None and self._cancel.is_set():
                            return None
                        static_text, dynamic_text = await _recall(agent)
                        rel_ctx = self._rel_context.get(agent.entity_id, {}).get("summary", "")
                        causal = _causal_ctxs.get(agent.entity_id, "")
                        agent_recent = _recent_ctxs.get(agent.entity_id, "（无近期事件）")
                        d = await self.reasoner.reason_quantified(
                            agent, states[agent.entity_id], re_engine,
                            recent_events=agent_recent, other_context=_other_ctxs.get(agent.entity_id, ""),
                            round_number=round_number, client=client,
                            static_knowledge=static_text, dynamic_memory=dynamic_text,
                            relationship_context=rel_ctx, causal_feedback=causal,
                            spatial_context=_spatial_ctxs.get(agent.entity_id, ""),
                            env_context=_env_ctx,
                            multi_candidate=getattr(self, "_enable_rollout", False),
                        )
                        d["actor_id"] = agent.entity_id

                        # ── 前瞻规划：如果设定了 enable_rollout，做多候选评分 ──
                        if getattr(self, "_enable_rollout", False):
                            try:
                                candidates_raw = d.get("_candidates", [])
                                if candidates_raw and len(candidates_raw) > 1:
                                    scored = await self._rollout_candidates(
                                        agent, candidates_raw, re_engine,
                                        states, round_number, lookahead=3)
                                    if scored:
                                        best = max(scored, key=lambda c: c.get("_future_score", 0))
                                        best["actor_id"] = agent.entity_id
                                        best["_original"] = d
                                        best["_rollout_score"] = best.get("_future_score", 0)
                                        best["driver"] = "llm_rollout"
                                        return best
                            except Exception:
                                pass  # rollout 失败 → 安全回退到 LLM 直接决策

                        return d
                except LLMConnectionError as e:
                    fails += 1
                    if fails > max_passes:
                        raise
                    delay = min(60.0, 5.0 * (2 ** (fails - 1)))
                    self._log("simulation", f"{agent.name} LLM 连接失败({fails}/{max_passes+1})，{delay:.0f}s 后重试… | {e.endpoint}: {e.cause}")
                    await asyncio.sleep(delay)

        if self._cancel is not None and self._cancel.is_set():
            return sim_round
        # ── FSM 分流：上一轮的 FSM 状态决定本轮哪些代理走 LLM ──
        fsm_states = getattr(self, "_last_fsm_states", None)
        fsm_actions = getattr(self, "_last_fsm_actions", None)
        fsm_command = getattr(self, "_last_fsm_command_states", {"combat"})

        # 第一遍（顺序）：override / FSM 走确定性动作（纯 Python、含状态消费），
        # command 态标记为 None 待第二遍并发 LLM 决策；plan 与 ordered 索引对齐以保序。
        plan: list[dict[str, Any] | None] = []
        for i, agent in enumerate(ordered):
            # ── 用户强制 override：最高优先，跳过 FSM 与 LLM ──
            ov = self._pop_override(agent)
            if ov is not None:
                ov["actor_id"] = agent.entity_id
                ov["driver"] = "forced"
                plan.append(ov)
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
                plan.append(act)
                continue
            plan.append(None)  # command 态 → 待并发 LLM 决策

        # 第二遍：command 态 agent 并发 LLM 决策（上限 = FORGE_MAX_CONCURRENT），按索引回填保序
        llm_idx = [i for i, p in enumerate(plan) if p is None]
        if llm_idx and not (self._cancel is not None and self._cancel.is_set()):
            from strategy_forge.core.llm_client import LLMConnectionError
            llm_results = await asyncio.gather(
                *(decide(ordered[i]) for i in llm_idx), return_exceptions=True)
            conn_fails = sum(1 for r in llm_results if isinstance(r, LLMConnectionError))
            if conn_fails > 0:
                ratio = conn_fails / max(1, len(llm_idx))
                if ratio >= (_cfg.deduction_sim_fail_ratio if '_cfg' in dir() else 0.75):
                    first = next((r for r in llm_results if isinstance(r, LLMConnectionError)), None)
                    raise ConnectionFailureError(str(first) if first else f"连接故障：{conn_fails}/{len(llm_idx)} agent 无法连接 LLM")
            for i, raw in zip(llm_idx, llm_results, strict=False):
                if isinstance(raw, BaseException):
                    self._log("simulation", f"agent {ordered[i].name} 决策失败: {raw}")
                else:
                    plan[i] = raw

        # 按 ordered 原序装配 decisions（跳过 override/FSM 之外未成功的项）
        decisions: list[dict[str, Any]] = [p for p in plan if p is not None]
        # raw_results kept below for backward compat
        raw_results = decisions

        # ── 前瞻规划：保存本轮真实 LLM 决策为下轮的 Rollout 基线 ──
        if self._enable_rollout:
            self._baseline_decisions = {}
            for dec in decisions:
                self._baseline_decisions[dec.get("actor_id", "")] = {
                    "actor_id": dec.get("actor_id", ""),
                    "action_type": dec.get("action_type", "observe"),
                    "target": dec.get("target", ""),
                    "intensity": dec.get("intensity", 0.5),
                }

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

        # ── 轮后：调度延迟效应 + 增强因果反馈 ──
        self._last_round_outcomes: dict[str, str] = {}
        for dec in decisions:
            actor = dec.get("actor_id")
            if actor not in states:
                continue
            my_deltas = deltas.get(actor, {})
            target_id = dec.get("target", "")
            target_deltas = deltas.get(target_id, {}) if target_id and target_id in states else {}
            action = dec.get("action_type", "?")
            target_name = target_id if target_id else "自身"
            agent = next((a for a in self.agents if a.entity_id == actor), None)
            nm = agent.name if agent else actor[:8]
            auto_d = auto_deltas.get(actor, {})
            # 增强因果反馈（多段落叙事）
            feedback = _build_causal_feedback(
                actor_id=actor, actor_name=nm, action=action,
                target_id=target_id or "", target_name=target_name,
                my_deltas=my_deltas, target_deltas=target_deltas,
                auto_deltas=auto_d, event_history=self._event_history,
                round_number=round_number, name_to_id=self._name_to_id,
            )
            self._last_round_outcomes[actor] = feedback
            # Delay effect scheduling
            for action, sub_intensity, _target in re_engine._iter_subactions(dec):
                delay_cfg = re_engine.pack.get("delay_effects", {}).get(action)
                if delay_cfg and sub_intensity > 0:
                    dr = int(delay_cfg.get("delay", 1))
                    eff = {k: v * sub_intensity for k, v in delay_cfg.get("effects", {}).items()}
                    states[actor].schedule_delays(round_number, dr, eff)

        # ── 声誉系统：根据本轮交互自动更新信任度 ──
        self._update_reputation_after_round(decisions, self._name_to_id)

        # ── 谍报处理：检测 _intel_exposed 标记 → 授予信息优势 ──
        for dec in decisions:
            actor = dec.get("actor_id", "")
            tgts = dec.get("target", "")
            tgts_list = tgts.split(",") if isinstance(tgts, str) and "," in tgts else [tgts]
            for tgt in tgts_list:
                tgt = tgt.strip()
                if not tgt or tgt not in self._name_to_id:
                    continue
                # Check if this interaction triggered _intel_exposed
                inter = next((it for it in inter_by_actor.get(actor, [])
                              if it.get("target") == tgt), None)
                if inter and any(
                    "_intel_exposed" in d or "intel" in k.lower()
                    for k, d in inter.get("deltas", {}).items()
                    if isinstance(d, dict)
                ):
                    # Also check via keys
                    pass
                if inter:
                    for k in inter.get("deltas", {}):
                        if "_intel_exposed" in str(k).lower() or "intel" in str(k).lower():
                            bonus = self._intel_bonuses.setdefault(actor, {}).get(tgt, 0.0)
                            self._intel_bonuses.setdefault(actor, {})[tgt] = min(5.0, bonus + 2.0)
                            self._log("simulation", f"[谍报] {actor} 对 {tgt} 获得信息优势 (+2.0, 总和={bonus+2.0:.1f})")
                            break

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

        # ── 信息传播：将本轮事件按信任度分发至各 agent 知识队列 ──
        self._dispatch_events(round_number)

        # ── 人格动态化：每 REFLECT_INTERVAL 轮或重大指标变化时触发反思 ──
        for agent in self.agents:
            if agent.entity_id not in states:
                continue
            last_rf = self._last_reflection_round.get(agent.entity_id, 0)
            should_reflect = (round_number - last_rf >= self.REFLECT_INTERVAL)
            if not should_reflect:
                # 检查是否有指标累计变化超过阈值
                st = states[agent.entity_id]
                history = getattr(st, "history", []) or []
                recent = [h for h in history if h.get("round", 0) > last_rf]
                cumsum: dict[str, float] = {}
                for h in recent:
                    cumsum[h.get("metric", "")] = cumsum.get(h.get("metric", ""), 0) + abs(float(h.get("delta", 0)))
                if any(v >= self.REFLECT_DELTA_THRESHOLD for v in cumsum.values()):
                    should_reflect = True
            if should_reflect:
                await self._reflect_and_adapt(agent, round_number, client)
                self._last_reflection_round[agent.entity_id] = round_number

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
            f"将第 {round_number} 轮量化推演结果改写为一段生动简洁的战局叙事（200 字以内）。\n\n"
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
            "round": round_num, "entity_count": len(states),
            "_thresholds": thresholds,
            "entities": [{k: v for k, v in {
                "name": getattr(st, 'name', '?'),
                "metrics": {m: round(st.metrics.get(m, 0), 1) for m in metrics_list},
                "alive": re_engine.is_alive(st) if re_engine else True,
            }.items() if v is not None and (k != "metrics" or isinstance(v, dict) and len(v) > 0)}
            for st in states.values() if hasattr(st, 'name')]}


def _parse_action_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

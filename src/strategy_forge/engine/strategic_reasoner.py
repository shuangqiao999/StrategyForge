"""Strategic Reasoner — multi-candidate generation + heuristic scoring + trust matrix.

Provides deep strategic reasoning for simulation agents, replacing inline prompt assembly.
Supports user intervention awareness via LanceDB priority events.
"""
from __future__ import annotations

import asyncio
import heapq
import json
import logging
import re
from collections import defaultdict
from string import Template
from typing import Any

from ._utils import extract_text
from .models import DeductionAgentProfile
from .preprocessor import DeductionPreprocessor

logger = logging.getLogger(__name__)

_POSITIVE_KW = frozenset({"support", "help", "cooperate", "praise", "agree"})
_NEGATIVE_KW = frozenset({"oppose", "attack", "betray", "insult", "threaten", "block"})


_CANDIDATE_PROMPT = """You are a strategic advisor. Generate $candidate_count distinct action strategies for $agent_name.

## Immutable Goals (highest priority — persist throughout entire simulation)
$immutable_goals

## Override Directive (highest priority — must influence every candidate)
$user_intervention

## Agent Profile
Persona: $persona
Background: $background
Goals: $goals

## Current World State
Round: $round_number
Recent events: $recent_events

## Trust Relationship Summary
$trust_summary

## 关系网络（来自知识图谱：盟友 / 对手）
$relationship_context

## Output — pure JSON array
[
  {
    "action": "post|reply|interact|observe",
    "target": "target entity name or empty",
    "content": "action description (30-100 chars)",
    "rationale": "why this action (20-60 chars)",
    "risk_level": "low|medium|high"
  }
]

Output ONLY the JSON array. No markdown, no explanations."""


class StrategicReasoner:
    """Multi-candidate strategic reasoning engine.

    For each agent decision:
      1. Generate N candidate strategies via LLM
      2. Score candidates heuristically (trust matrix, risk, goal alignment)
      3. Select best candidate or fall back to LLM tiebreak
    """

    def __init__(self, candidate_count: int = 3, preprocessor: DeductionPreprocessor | None = None, chat_fn: Any = None, immutable_goals: list[str] | None = None, temperature: float = 0.7, enable_multi_action: bool = False, max_actions: int = 3):
        self.candidate_count = candidate_count
        self._preprocessor = preprocessor
        self._chat_fn = chat_fn
        self._immutable_goals: list[str] = list(immutable_goals or [])
        self._temperature = temperature
        self._enable_multi_action = enable_multi_action
        self._max_actions = max(1, int(max_actions))
        self._trust_matrix: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._action_catalog_cache: dict[str, str] = {}
        self._intervention_cache: str | None = None
        self._intervention_round: int = -1

    def _cached_intervention(self, round_number: int) -> str:
        """Fetch latest intervention, cached per round."""
        if self._intervention_round == round_number and self._intervention_cache is not None:
            return self._intervention_cache
        self._intervention_round = round_number
        if self._preprocessor is not None:
            try:
                iv = self._preprocessor.retrieve_latest_intervention()
                if iv:
                    self._intervention_cache = iv.get("content", "")
                    return self._intervention_cache
            except Exception:
                pass
        self._intervention_cache = ""
        return ""

    def _cached_action_catalog(self, rule_engine: Any) -> str:
        domain = getattr(rule_engine, "domain", "default")
        key = f"{domain}_{self._enable_multi_action}_{self._max_actions}"
        if key not in self._action_catalog_cache:
            self._action_catalog_cache[key] = rule_engine.action_catalog()
        return self._action_catalog_cache[key]

    def record_interaction(
        self, source: str, target: str, action_type: str, content: str,
    ) -> None:
        """Update trust matrix based on interaction sentiment."""
        delta = 0.0
        text_lower = content.lower()
        if action_type == "reply" or action_type == "interact":
            if any(w in text_lower for w in _POSITIVE_KW):
                delta = 0.3
            elif any(w in text_lower for w in _NEGATIVE_KW):
                delta = -0.5
        elif action_type == "post":
            if any(w in text_lower for w in _POSITIVE_KW):
                delta = 0.1
            elif any(w in text_lower for w in _NEGATIVE_KW):
                delta = -0.2
        if delta != 0.0:
            current = self._trust_matrix[source][target]
            self._trust_matrix[source][target] = max(-5.0, min(5.0, current + delta))

    def get_trust(self, source: str, target: str) -> float:
        return self._trust_matrix.get(source, {}).get(target, 0.0)

    def seed_trust(self, source_id: str, allies: list[str], opponents: list[str],
                   weight: float = 2.0) -> None:
        """用图谱既有关系初始化信任矩阵（键用对方名称，匹配运行时 get_trust 查找约定）。

        仅影响定性 reason() 的启发式打分与信任摘要；量化路径不读信任矩阵，
        关系信息在量化模式经 relationship_context 注入 Prompt。
        """
        w = max(-5.0, min(5.0, weight))
        for name in allies or []:
            if name:
                self._trust_matrix[source_id][name] = w
        for name in opponents or []:
            if name:
                self._trust_matrix[source_id][name] = -w

    def _trust_summary_for(self, agent_id: str) -> str:
        relations = self._trust_matrix.get(agent_id, {})
        if not relations:
            return "No prior trust history"
        top = heapq.nlargest(5, relations.items(), key=lambda x: abs(x[1]))
        lines = []
        for other, score in top:
            label = "trusts" if score > 0 else "distrusts" if score < 0 else "neutral to"
            lines.append(f"  {label} {other[:12]} (score={score:+.1f})")
        return "\n".join(lines) if lines else "No significant trust relations"

    async def reason(
        self, agent: DeductionAgentProfile, world_state: dict, round_number: int,
        client: Any = None,
    ) -> dict[str, Any]:
        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
        from strategy_forge.core.llm_client import Message

        # 1. Check for user intervention (cached per round)
        user_cmd = "No external directive — act autonomously based on your profile."
        intervention_text = self._cached_intervention(round_number)
        if intervention_text:
            user_cmd = intervention_text

        # 2. Build trust summary
        trust = self._trust_summary_for(agent.entity_id)

        # 3. Generate candidates via LLM
        recent = world_state.get("recent_events", "None")
        system = "You are a JSON-only strategic advisor. Output ONLY a valid JSON array."
        llm = client if client is not None else LLMClient()
        messages = [Message(role="user", content=Template(_CANDIDATE_PROMPT).substitute(
            candidate_count=self.candidate_count,
            agent_name=agent.name,
            immutable_goals="\n".join(f"- {g}" for g in self._immutable_goals) if self._immutable_goals else "No immutable goals — act freely based on your profile.",
            user_intervention=user_cmd,
            persona=agent.persona,
            background=agent.background,
            goals=", ".join(agent.goals) if agent.goals else "act naturally",
            round_number=round_number,
            recent_events=str(recent)[:500],
            trust_summary=trust,
            relationship_context=world_state.get("relationship_context", "") or "（无已知关系）",
        ))]

        candidates: list[dict[str, Any]] = []
        try:
            if self._chat_fn is not None:
                content = await asyncio.to_thread(self._chat_fn, messages, system, self._temperature)
            else:
                response = await llm.chat(messages, system=system, temperature=self._temperature)
                content = extract_text(response)
            candidates = _parse_candidates(content)
        except Exception as e:
            logger.warning("[Reasoner] LLM candidate generation failed: %s", e)

        # 4. Fallback if no candidates
        if not candidates:
            return {
                "selected": {"action": "observe", "target": "", "content": f"{agent.name}观察着周围环境", "rationale": "fallback"},
                "candidates": [],
                "trust_used": False,
            }

        # 5. Heuristic scoring
        for c in candidates:
            score = 0.0
            # Risk penalty
            risk = c.get("risk_level", "medium")
            if risk == "high":
                score -= 0.3
            elif risk == "low":
                score += 0.1
            # User intervention bonus: match actual intervention keywords
            if intervention_text:
                keywords = [w for w in intervention_text[:40].split() if len(w) >= 2]
                if any(kw in c.get("content", "") or kw in c.get("rationale", "")
                       for kw in keywords):
                    score += 0.5
            # Goal alignment bonus
            if agent.goals and any(g[:4] in c.get("content", "") or g[:4] in c.get("rationale", "")
                                  for g in agent.goals):
                score += 0.2
            # Trust awareness: prefer interacting with trusted agents
            target = c.get("target", "")
            if target and self.get_trust(agent.entity_id, target) > 1.0:
                score += 0.2
            elif target and self.get_trust(agent.entity_id, target) < -2.0:
                score -= 0.3
            c["_score"] = score

        candidates.sort(key=lambda c: c.get("_score", 0), reverse=True)
        selected = candidates[0]

        return {
            "selected": selected,
            "candidates": candidates,
            "trust_used": any(abs(v) > 0.5 for v in self._trust_matrix.get(agent.entity_id, {}).values()),
        }

    async def reason_quantified(
        self, agent: DeductionAgentProfile, state: Any, rule_engine: Any,
        recent_events: str = "", other_context: str = "", round_number: int = 0,
        client: Any = None, static_knowledge: str = "", dynamic_memory: str = "",
        relationship_context: str = "",
        spatial_context: str = "",
        env_context: str = "",
    ) -> dict[str, Any]:
        """量化模式决策。

        - 单动作（默认）：输出 {action_type, target, intensity, rationale}，与 v2.0 一致。
        - 多动作分配（self._enable_multi_action）：输出 {budget, actions:[{action_type, weight, target}], rationale}，
          解析后归一化并裁剪到 max_actions；统一返回时附带 action_type/target/intensity（取主导动作），
          以兼容下游 SimulationAction 构造与叙事。
        """
        from ._utils import extract_text
        from .graph_builder import try_extract_json
        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
        from strategy_forge.core.llm_client import Message

        actions = rule_engine.actions()
        user_cmd = self._cached_intervention(round_number)

        goals = ", ".join(agent.goals) if agent.goals else "依据人格自主行动"
        imm = "；".join(self._immutable_goals) if self._immutable_goals else "无"
        select_hint = ("从可选行动中分配资源给一个或多个动作"
                       if self._enable_multi_action else "从可选行动中选择一个并给出投入力度")
        diversity_hint = "。避免连续3轮以上重复相同策略，结合当前局势变化探索多元行动" if round_number > 3 else ""
        header = (
            f"你是「{agent.name}」，正处于一场量化推演的第 {round_number} 轮。"
            f"请基于你的人格、目标与当前数值状态，{select_hint}{diversity_hint}。\n\n"
            f"## 你的人格\n{agent.persona or '（无）'}\n"
            f"## 你的目标\n{goals}\n"
            f"## 不可变战略指令（最高优先级）\n{imm}\n"
            + (f"## 外部干预指令（最高优先级）\n{user_cmd}\n" if user_cmd else "")
            + f"## 你的当前状态\n{state.to_prompt_context()}\n"
            f"## 其他参与方状态\n{other_context or '（暂无）'}\n"
            + (f"## 关系网络（盟友/对手）\n{relationship_context}\n" if relationship_context else "")
            + (f"## 原著背景（语义召回）\n{static_knowledge}\n" if static_knowledge else "")
            + (f"## 历史记忆（语义召回）\n{dynamic_memory}\n" if dynamic_memory else "")
            + f"## 近期局势\n{recent_events or '（无）'}\n"
            + (f"## 空间环境\n{spatial_context}\n" if spatial_context else "")
            + (f"## 地形与天气\n{env_context}\n" if env_context else "")
            + f"\n## 可选行动\n{self._cached_action_catalog(rule_engine)}\n\n"
        )
        if self._enable_multi_action:
            output_spec = (
                '## 输出 JSON（仅 JSON，无解释）\n'
                '{"budget": 0.0到1.0, "actions": ['
                '{"action_type": "上面之一", "weight": 0.0到1.0, '
                '"target": "目标方名称(进攻/竞争/外交时填，否则留空)"}], '
                '"rationale": "20-50字理由"}\n'
                f"- 最多 {self._max_actions} 个动作，可同时分配资源（如同时进攻与防守，或对不同对手施压）\n"
                "- budget：本轮总投入力度，0.1=保守，0.5=常规，1.0=倾尽全力\n"
                "- weight：各动作占总投入的比例，所有 weight 之和应约等于 1.0"
            )
        else:
            output_spec = (
                '## 输出 JSON（仅 JSON，无解释）\n'
                '{"action_type": "上面之一", "target": "目标方名称(进攻/竞争/外交时填，否则留空)", '
                '"intensity": 0.0到1.0, "rationale": "20-50字理由"}\n'
                "- intensity：投入力度，0.1=试探，0.5=常规，1.0=倾尽全力"
            )
        prompt = header + output_spec
        system = "你是量化推演中的战略决策者，只输出 JSON。"
        llm = client if client is not None else LLMClient()
        try:
            if self._chat_fn is not None:
                content = await asyncio.to_thread(
                    self._chat_fn, [Message(role="user", content=prompt)], system, self._temperature)
            else:
                resp = await llm.chat([Message(role="user", content=prompt)],
                                      system=system, temperature=self._temperature)
                content = extract_text(resp)
            data = try_extract_json(content)
            if not isinstance(data, dict):
                data = {}
        except Exception as e:
            logger.warning("[Reasoner] 量化决策失败，回退 observe: %s", e)
            data = {}

        rationale = str(data.get("rationale", ""))[:120]

        if self._enable_multi_action:
            try:
                budget = max(0.0, min(1.0, float(data.get("budget", data.get("intensity", 0.5)))))
            except (TypeError, ValueError):
                budget = 0.5
            subs: list[dict[str, Any]] = []
            raw_actions = data.get("actions")
            if isinstance(raw_actions, list):
                for a in raw_actions:
                    if not isinstance(a, dict):
                        continue
                    act = str(a.get("action_type", "")).strip()
                    if act not in actions:
                        continue
                    try:
                        w = float(a.get("weight", 0.0))
                    except (TypeError, ValueError):
                        w = 0.0
                    if w <= 0:
                        continue
                    subs.append({"action_type": act, "weight": w,
                                 "target": str(a.get("target", "") or "").strip()})
            if not subs:
                return {"action_type": "observe", "target": "", "intensity": budget,
                        "budget": budget,
                        "actions": [{"action_type": "observe", "weight": 1.0, "target": ""}],
                        "rationale": rationale}
            subs.sort(key=lambda s: s["weight"], reverse=True)
            subs = subs[: self._max_actions]
            total = sum(s["weight"] for s in subs) or 1.0
            for s in subs:
                s["weight"] = round(s["weight"] / total, 4)
            primary = subs[0]
            return {
                "action_type": primary["action_type"],
                "target": primary["target"],
                "intensity": budget,
                "budget": budget,
                "actions": subs,
                "rationale": rationale,
            }

        action = str(data.get("action_type", "observe"))
        if action not in actions:
            action = "observe"
        try:
            intensity = max(0.0, min(1.0, float(data.get("intensity", 0.5))))
        except (TypeError, ValueError):
            intensity = 0.5
        return {
            "action_type": action,
            "target": str(data.get("target", "") or "").strip(),
            "intensity": intensity,
            "rationale": rationale,
        }


def _parse_candidates(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', raw)
    cleaned = re.sub(r'\n?```', '', cleaned).strip()
    for pat in (r'\[[\s\S]*\]', r'\{[\s\S]*\}'):
        m = re.search(pat, cleaned)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [c for c in data if isinstance(c, dict) and "action" in c]
                if isinstance(data, dict) and "action" in data:
                    return [data]
            except (json.JSONDecodeError, ValueError):
                continue
    return []

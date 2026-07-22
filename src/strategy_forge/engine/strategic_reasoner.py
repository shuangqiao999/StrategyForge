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


_CANDIDATE_PROMPT = """你是一个战略顾问。为 $agent_name 生成 $candidate_count 个不同的行动策略。

## 不可变目标（最高优先级，贯穿整个模拟）
$immutable_goals

## 外部指令（最高优先级，必须影响每个候选）
$user_intervention

## 智能体档案
人格: $persona
背景: $background
目标: $goals

## 你最近几轮已执行的行动（本轮禁止生成与这些雷同的行动——必须推进新的实质性进展，如升级冲突、兑现承诺、改变结盟、迫使摊牌）
$recent_own_actions

## 当前世界状态
轮次: $round_number
近期事件: $recent_events

## 信任关系摘要
$trust_summary

## 关系网络（来自知识图谱：盟友 / 对手）
$relationship_context

## 现实性约束（硬性规则）
- 行动手段必须限于你当前身份在现实中可用的手段（职权、人脉、资金、信息），禁止超现实桥段：黑客奇迹、凭空巨额资金、一夜掌控他人系统等。
- 行动只能表达"你做了什么"，不能宣称单方面完成需要多方配合的结果（如接管、罢免、收购需经程序，只能"推动/发起"）。
- 已死亡或已退场的人物不得作为行动者或对话对象出现。
- observe 仅在没有明确威胁且局势不明时使用——如核心目标未达成，应选择低风险主动行动（collaborate/respond）而非被动观察。

## 行动示例
正确（具体、有因果逻辑、与人格相关）：
  {"action": "compete", "target": "OpenAI", "content": "DeepSeek宣布开源MoE训练框架，同步将商业API价格下调60%——以开源+降价双重施压对手企业客户迁移", "rationale": "利用对手API稳定性争议，用价格和技术优势抢占市场", "risk_level": "medium"}
错误（模糊、模板化）：
  {"action": "compete", "target": "对手", "content": "继续加大研发投入，推出新产品，提升核心竞争力", "rationale": "为了发展", "risk_level": "low"}

## 输出 — 纯 JSON 数组
[
  {
    "action": "initiate|respond|collaborate|compete|observe",
    "target": "目标实体名或留空",
    "content": "行动描述 (30-100字)",
    "rationale": "行动理由 (20-60字)",
    "risk_level": "low|medium|high"
  }
]

只输出 JSON 数组，不要 markdown，不要解释。"""


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
        # 叙事模式行动去重：每 agent 最近 5 轮已执行行动摘要
        self._recent_actions: dict[str, list[str]] = defaultdict(list)

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

    def adjust_trust(self, source_id: str, target_name: str, delta: float) -> float:
        """根据交互结果动态调整信任度。返回新信任值。

        delta>0 表示正向互动（合作/援助），delta<0 表示负向（攻击/背叛）。
        信任度始终钳制在 [-5.0, +5.0] 区间。
        """
        if not target_name:
            return 0.0
        old = self._trust_matrix[source_id].get(target_name, 0.0)
        new = max(-5.0, min(5.0, old + delta))
        self._trust_matrix[source_id][target_name] = new
        return new

    # ── 动作→声誉映射：用于自动信任更新 ──
    _TRUST_HOSTILE_ACTIONS = frozenset({
        "attack", "siege", "price_war", "embargo", "export_control",
        "talent_war", "poach_talent", "compete", "attack_opponent",
        "electronic_warfare", "military_offensive", "trade_warfare",
        "propaganda", "leak_info", "framing_battle",
    })
    _TRUST_FRIENDLY_ACTIONS = frozenset({
        "diplomacy", "partner", "invest", "invest_rnd", "welfare",
        "diplomatic_engagement", "fact_check", "conserve", "restoration",
    })

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """字符2-gram重合率，用于行动去重打分。返回 [0,1]。"""
        ga = {a[i:i + 2] for i in range(len(a) - 1)}
        gb = {b[i:i + 2] for i in range(len(b) - 1)}
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / min(len(ga), len(gb))

    @staticmethod
    def _normalize_action(action: str) -> str:
        """枚举校验：清除 'initiate|respond|...' 字面量泄漏等脏 action 值。"""
        act = str(action or "").strip()
        if not act:
            return "observe"
        if "|" in act:
            first = act.split("|")[0].strip()
            return first if first else "observe"
        return act

    @staticmethod
    def _persona_with_evolution(agent: Any) -> str:
        """人格 = 原始人格（不可变） + 推演中演化出的行为准则（信念增量层）。"""
        base = agent.persona or "（无）"
        extra = getattr(agent, "system_prompt_extra", "") or ""
        if extra:
            return f"{base}\n【行为准则·由推演经历塑造，影响你的决策】{extra}"
        return base

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

        # 2.5 全局目标块：附加"推动收敛"指令，避免 agent 无方向即兴表演
        if self._immutable_goals:
            goals_block = "\n".join(f"- {g}" for g in self._immutable_goals) + (
                "\n注意：上述目标是本次推演要回答的核心问题。每个候选策略都应实质性推动局势向"
                "\"该问题可被明确判定\"的方向演进——扩大或削弱关键方的控制力、迫使摊牌、打破僵局。"
                "避免与核心问题无关的原地观望或重复性表态。")
        else:
            goals_block = "No immutable goals — act freely based on your profile."

        # 3. Generate candidates via LLM
        recent = world_state.get("recent_events", "None")
        recent_own = self._recent_actions.get(agent.entity_id, [])
        recent_own_text = ("\n".join(f"- {a}" for a in recent_own)
                           if recent_own else "（无——这是你的首轮行动）")
        system = "你是战略顾问，只输出 JSON 数组。"
        llm = client if client is not None else LLMClient()
        messages = [Message(role="user", content=Template(_CANDIDATE_PROMPT).substitute(
            candidate_count=self.candidate_count,
            agent_name=agent.name,
            immutable_goals=goals_block,
            user_intervention=user_cmd,
            persona=self._persona_with_evolution(agent),
            background=agent.background,
            goals=", ".join(agent.goals) if agent.goals else "act naturally",
            recent_own_actions=recent_own_text,
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
            # Global immutable-goal alignment bonus: reward candidates whose
            # rationale engages with the core deduction question's keywords
            if self._immutable_goals:
                imm_kws = {w for g in self._immutable_goals
                           for w in re.findall(r"[\u4e00-\u9fff]{2,4}|[A-Za-z]{3,}", g)}
                text = c.get("content", "") + c.get("rationale", "")
                hits = sum(1 for kw in imm_kws if kw in text)
                if hits >= 2:
                    score += 0.4
                elif hits == 1:
                    score += 0.2
            # Penalize passive observation when a core question awaits resolution
            if self._immutable_goals and c.get("action") == "observe":
                score -= 0.2
            # Trust awareness: prefer interacting with trusted agents
            target = c.get("target", "")
            if target and self.get_trust(agent.entity_id, target) > 1.0:
                score += 0.2
            elif target and self.get_trust(agent.entity_id, target) < -2.0:
                score -= 0.3
            # Repetition penalty: discourage repeating recent own actions
            if recent_own:
                max_sim = max(self._text_overlap(c.get("content", ""), prev)
                              for prev in recent_own[-3:])
                if max_sim >= 0.5:
                    score -= 0.4
                elif max_sim >= 0.3:
                    score -= 0.2
            c["_score"] = score

        candidates.sort(key=lambda c: c.get("_score", 0), reverse=True)
        selected = candidates[0]
        selected["action"] = self._normalize_action(selected.get("action", "observe"))

        # 记录本轮选中行动，供后续轮次去重（每 agent 保留最近 5 条）
        hist = self._recent_actions[agent.entity_id]
        hist.append(str(selected.get("content", ""))[:80])
        if len(hist) > 5:
            del hist[:-5]

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
        causal_feedback: str = "",
        multi_candidate: bool = False,
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
        diversity_hint = ("。避免连续3轮以上重复相同策略，结合当前局势变化探索多元行动"
                          if round_number > 3 else "")
        if round_number > 5:
            diversity_hint += "。注意：你已连续多轮执行相同类型的行动，这可能导致策略僵化和对手预判。请认真考虑在当前轮选择一个不同类型的行动。"
        # B4: 云端 prompt 缓存友好排序 —— 同一轮所有 agent 相同的"共享前缀"放最前
        # (不可变目标/可选动作/地形/干预/近期局势)，agent 私有部分居中，输出规范置末。
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
            if multi_candidate:
                output_spec += (
                    '\n\n## 多候选模式（输出3个不同的候选策略）\n'
                    '{"candidates": ['
                    '{"action_type": "...", "target": "...", "intensity": 0.0~1.0, "rationale": "理由1"},'
                    '{"action_type": "...", "target": "...", "intensity": 0.0~1.0, "rationale": "理由2"},'
                    '{"action_type": "...", "target": "...", "intensity": 0.0~1.0, "rationale": "理由3"}'
                    ']}\n'
                    "- 3个候选应包含不同策略方向（如进攻/防守/外交），不是同一方向的微调"
                )

        # ── 共享前缀（同轮所有 agent 一致，利于云端前缀缓存）──
        prefix_parts: list[str] = []
        if imm != "无":
            prefix_parts.append(f"## 核心战略问题（你必须回答，贯穿所有轮次）\n{imm}\n")
        prefix_parts.append(f"## 可选行动\n{self._cached_action_catalog(rule_engine)}\n")
        # 合作目标约束：partner 必须从关系网络中的已知合作方或同行业实体中选择
        if relationship_context and "（无" not in relationship_context:
            prefix_parts.append(f"## 合作目标约束\npartner 动作的目标必须从你的关系网络中选择已知合作方，"
                                f"或同行业/上游供应链实体。禁止选择直接竞对、跨行业无关方。"
                                f"当前已知关系：\n{relationship_context}\n")
        if env_context:
            prefix_parts.append(f"## 地形与天气\n{env_context}\n")
        if user_cmd:
            prefix_parts.append(f"## 外部干预指令（最高优先级）\n{user_cmd}\n")
        prefix_parts.append(f"## 近期局势\n{recent_events or '（无）'}\n")
        shared_prefix = "\n".join(prefix_parts) + "\n"

        # ── agent 私有部分 ──
        agent_parts = [
            f"你是「{agent.name}」，正处于一场量化推演的第 {round_number} 轮。"
            f"请基于战略问题、你的人格、目标与当前数值状态，{select_hint}{diversity_hint}。\n",
        ]
        if causal_feedback:
            agent_parts.append(f"{causal_feedback}\n")
        agent_parts.extend([
            f"## 你的人格\n{self._persona_with_evolution(agent)}\n",
            f"## 你的目标\n{goals}\n",
            f"## 你的当前状态\n{state.to_prompt_context()}\n",
            f"## 其他参与方状态\n{other_context or '（暂无）'}\n",
        ])
        if relationship_context:
            agent_parts.append(f"## 关系网络（盟友/对手）\n{relationship_context}\n")
        if static_knowledge:
            agent_parts.append(f"## 原著背景（语义召回）\n{static_knowledge}\n")
        if dynamic_memory:
            agent_parts.append(f"## 历史记忆（语义召回）\n{dynamic_memory}\n")
        if spatial_context:
            agent_parts.append(f"## 空间环境\n{spatial_context}\n")

        prompt = shared_prefix + "".join(agent_parts) + "\n" + output_spec
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
        result: dict[str, Any] = {
            "action_type": action,
            "target": str(data.get("target", "") or "").strip(),
            "intensity": intensity,
            "rationale": str(rationale)[:120],
        }
        # ── 多候选模式：优先使用 candidates ──
        cands_raw = data.get("candidates")
        if multi_candidate and isinstance(cands_raw, list) and len(cands_raw) > 0:
            cands: list[dict[str, Any]] = []
            for c in cands_raw:
                if not isinstance(c, dict):
                    continue
                act = str(c.get("action_type", "")).strip()
                if act not in actions:
                    continue
                try:
                    ci = max(0.0, min(1.0, float(c.get("intensity", 0.5))))
                except (TypeError, ValueError):
                    ci = 0.5
                cands.append({
                    "action_type": act,
                    "target": str(c.get("target", "") or "").strip(),
                    "intensity": ci,
                    "rationale": str(c.get("rationale", ""))[:120],
                })
            if cands:
                first = cands[0]
                if first["action_type"] == "observe" and len(cands) > 1:
                    first = cands[1]
                result = {
                    "action_type": first["action_type"],
                    "target": first.get("target", ""),
                    "intensity": first["intensity"],
                    "rationale": first.get("rationale", str(rationale)[:120]),
                    "_candidates": cands,
                }
        return result


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

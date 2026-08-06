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
    reactions = _extract_reactions(actor_name, target_name, event_history, round_number, name_to_id, actor_id, actor_name)
    if reactions:
        parts.append("后续反应 — " + reactions)
    if not parts:
        return f"你的 {action} 已执行（本轮无显著数值变化）"
    return "## 上轮回顾\n" + "\n".join(f"  • {p}" for p in parts)


def _extract_reactions(
    actor_name: str, target_name: str, event_history: list[dict],
    round_number: int, name_to_id: dict[str, str],
    observer_id: str = "", observer_name: str = "",
) -> str:
    """从当前轮事件历史中提取他人对 acter/target 的回应（消上帝视角L2：可见性过滤）。"""
    reacting: list[str] = []
    target_events = [e for e in event_history
                     if e.get("round") == round_number
                     and e.get("agent_name", "") not in (actor_name, "")
                     and _is_event_visible_to(observer_id, observer_name, e)]
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


# 受限可见性：这些级别的事件仅对参与者+发起者可见，不会全局广播
_RESTRICTED_VIS = frozenset({"private", "alliance"})

# 谍报暴露标记：rules.json 中 target_effects 使用此 key 表示情报获取成功
_INTEL_EXPOSED_KEY = "_intel_exposed"


def _is_event_visible_to(entity_id: str, entity_name: str, evt: dict) -> bool:
    """事件可见性判定：public 全图可见，private/alliance 仅参与者/发起者可见。

    供 _agent_decide 可见历史与 _dispatch_events 知识队列分发两处复用，
    保证私密事件不会经任何途径泄漏给非参与者。
    alliance 级别与 private 相同过滤逻辑——发起方创建事件时
    将盟友名写入 participants 字段，即可实现盟友间情报共享。
    """
    vis = (evt.get("visibility", "") or "public").strip()
    if vis not in _RESTRICTED_VIS:
        return True
    parts = evt.get("participants", "") or ""
    if entity_name in parts or entity_id in parts:
        return True
    return evt.get("agent") == entity_id


def _state_snapshot_sig(states: dict, alive_ids: list) -> int:
    """计算量化状态快照的签名，用于 _other_ctxs 跨轮缓存失效判断。

    仅对活体实体取 metrics（有序值元组）与 history 长度/末条哈希，成本 O(alive)。
    """
    parts = []
    for eid in alive_ids:
        st = states.get(eid)
        if st is None:
            continue
        parts.append((eid, tuple(st.metrics.get(k, 0.0)
                                 for k in sorted(st.metrics or {}))))
        hist = getattr(st, "history", []) or []
        if hist:
            last = hist[-1]
            parts.append((eid, len(hist), str(last)[:120]))
    return hash(tuple(parts))


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

## 行动规则
- 行动必须与你的人格一致，禁止做出与人格矛盾的决策
- 不得重复近期动态事件中你已做过的相同行动（如果有）
- observe 仅在没有明确威胁且局势不明时才使用——如核心目标未达成，应选择低风险主动行动而非观察
- 行动内容必须是该角色在现实中可能采取的具体措施（30-100字）
- 秘密行动（间谍/卧底/密谈/潜入/暗中交易等）必须在 metadata.visibility 标记为 "private"
- 禁止在行动描述中透露其他角色不应知晓的私密信息（如他方的秘密计划/内部数据/私下布署）

## 正确示例
{"action": "compete", "target": "竞品X", "content": "A公司宣布旗舰产品全系降价8%，同时开放核心设施给第三方合作伙伴，以价格优势和生态扩张挤压对手利润空间——这一举动与其'成本控制优先'的策略一脉相承。"}
{"action": "collaborate", "target": "合作伙伴Y", "content": "B公司利用盟友的成熟渠道网络，以'轻资产协同'模式迅速建立海外售后体系，规避区域性贸易壁垒的直接冲击。"}

## 错误示例（禁止）
{"action": "observe", "target": "", "content": "观察市场变化。"}  ← 核心目标未达成时使用observe
{"action": "compete", "target": "某国", "content": "公司继续加大研发投入，提升竞争力。"}  ← 模糊、模板化、无具体行动

## 输出 JSON — 选择一种行动
{
  "action": "initiate|respond|collaborate|compete|observe",
  "target": "目标实体名或留空",
  "content": "行动内容 (30-100字)"
}

只返回 JSON，不要解释。"""


# 关系→敌友的静态关键字表已迁移至 relation_polarity 模块（Layer B 兜底），
# 结构化映射（Layer A）由 ontology/适配器提供并经 relation_polarity 参数注入。
from strategy_forge.engine.relation_polarity import infer_polarity, merge_polarity_map


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
        domain: str = "",
        relation_polarity: dict | None = None,
        injected_events_store: dict | None = None,
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
        # 融合架构·盲点4：外部注入事件队列（按引用，供通道①每轮消费）
        self._injected_events_store: dict = injected_events_store if injected_events_store is not None else {}
        # 叙事模式环境变量（仅叙事模式使用）
        self._narrative_env: dict[str, float] = self._init_narrative_env(domain)
        self._domain = domain
        self._reflection_thresholds = self._load_reflection_config(domain)
        self._merged_triggers = self._merge_domain_triggers(domain)
        self._severity_map = self._load_severity_map(domain)
        self._spatial_state = None   # cached SpatialState, updated after each module run
        # 关系→敌友极性：Layer A 结构化映射（ontology+适配器）；None 时仅用 Layer B 关键字兜底
        self._relation_polarity: dict[str, str] = merge_polarity_map(relation_polarity)
        # Layer C: 对 A/B 均 neutral 的高频交互边做 LLM 精判的缓存（relation → polarity）
        self._relation_llm_overrides: dict[str, str] = {}
        from strategy_forge.core.config import config
        from strategy_forge.core.providers import registry as _reg

        self._max_concurrent = (
            max_concurrent if max_concurrent is not None
            else _reg.max_concurrent
        )

        # ── 前瞻规划：Rollout 模式 ──
        self._enable_rollout: bool = False
        self._baseline_decisions: dict[str, dict[str, Any]] = {}

        # ── 信息传播：每 agent 的知识队列 ──
        self._agent_knowledge: dict[str, list[dict[str, Any]]] = {}
        # ── 谍报：每 agent 对特定目标的信息优势 ──
        self._intel_bonuses: dict[str, dict[str, float]] = {}  # {source_id: {target_name: bonus}}
        # ── 人格动态化：每 agent 的反思轮次追踪 ──
        self._personality_log: list[dict[str, Any]] = []  # [{round, agent, old_extra, new_extra}]

        from .strategic_reasoner import StrategicReasoner
        self.reasoner = StrategicReasoner(
            candidate_count=_reg.candidate_count,
            preprocessor=preprocessor,
            chat_fn=chat_fn,
            immutable_goals=self._immutable_goals,
            temperature=temperature,
            enable_multi_action=self._enable_multi_action,
            max_actions=self._max_actions,
        )

        # A. 关系反哺：开局一次性从 Kuzu 预取盟友/对手并播种信任(关系在一次推演内静态)
        self._rel_context: dict[str, dict] = {}
        # P1#7: 量化 other_ctxs 跨轮缓存
        self._other_ctxs_cache: dict = {}
        self._other_ctxs_sig = None
        self._build_relationship_context()
        seeded = sum(1 for v in self._rel_context.values() if v["summary"])

        # ── B. 补全：无图谱关系的 agent 用 polarization 自动划分敌友 ──
        self._seed_polarization_relations(seeded)

    def _init_narrative_env(self, domain: str = "") -> dict[str, float]:
        """根据域类型初始化不同的叙事环境变量。"""
        business_domains = {"business", "business_narrative"}
        if domain in business_domains:
            return {
                "市场竞争烈度": 60.0,  # 0=垄断 50=充分竞争 100=恶性价格战
                "资本流动性": 50.0,    # 0=枯竭 50=正常 100=过热
                "技术壁垒高度": 40.0,  # 0=无壁垒 50=中等壁垒 100=极高壁垒
                "监管压力": 30.0,      # 0=宽松 50=适度 100=高压
                "消费者信心": 55.0,    # 0=恐慌 50=中性 100=狂热
            }
        return {
            "舆论风向": 50.0,    # 0=批判 50=中性 100=支持
            "抗议规模": 0.0,     # 0=无 50=局部 100=全城
            "媒体关注": 20.0,    # 0=无人 50=全国 100=全球
            "国际压力": 10.0,    # 0=无视 50=关注 100=干预
            "社会分裂": 30.0,    # 0=团结 50=分歧 100=对立
        }

    def _load_reflection_config(self, domain: str) -> dict:
        """从 methodology.yaml 加载域专属反思配置。"""
        try:
            from pathlib import Path
            import yaml
            path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "methodology.yaml"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                thresholds = data.get("_reflection_thresholds", {}) or {}
                domain_cfg = thresholds.get(domain, {})
                if domain_cfg:
                    return dict(domain_cfg)
        except Exception:
            pass
        return {}

    def _merge_domain_triggers(self, domain: str) -> dict[str, list[str]]:
        """合并通用触发词与域专属触发词。"""
        merged = dict(self._TRIGGER_RULES)
        try:
            from pathlib import Path
            import yaml
            path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "methodology.yaml"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                domain_triggers = data.get("_domain_triggers", {}) or {}
                extra = domain_triggers.get(domain, {})
                if isinstance(extra, dict):
                    for cat, kws in extra.items():
                        if cat not in merged:
                            merged[cat] = []
                        merged[cat] = list(dict.fromkeys(merged[cat] + list(kws)))
        except Exception:
            pass
        return merged

    def _load_severity_map(self, domain: str) -> dict[str, set[str]]:
        """加载域专属严重度映射，一次读取缓存（避免热路径反复解析 YAML）。"""
        sev_map = {}
        try:
            from pathlib import Path
            import yaml
            path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "methodology.yaml"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                all_map = data.get("_severity_mapping", {}) or {}
                domain_sev = all_map.get(domain, all_map.get("default", {}))
                sev_map["heavy"] = set(domain_sev.get("heavy", []))
                sev_map["medium"] = set(domain_sev.get("medium", []))
                sev_map["light"] = set(domain_sev.get("light", []))
        except Exception:
            pass
        if not sev_map:
            sev_map = {
                "heavy": {"遭攻击", "战败", "开战", "遭背叛", "关系恶化", "被胁迫", "内乱"},
                "medium": {"遭制裁", "资源危机", "情报泄露", "重大失败", "声誉危机"},
                "light": {"意外转折", "联盟变动", "意外成功"},
            }
        return sev_map

    def _get_reflection_threshold(self, key: str, default: int | float) -> int | float:
        """从 methodology 读取域专属反思阈值，无配置回退默认。"""
        return self._reflection_thresholds.get(key, default)

    def _max_persona_rules(self) -> int:
        return int(self._get_reflection_threshold("max_persona_rules", 3))

    def _classify_relation(self, relation: str) -> str:
        # Layer A: 结构化映射（ontology + 适配器覆盖，确定性）
        if relation in self._relation_polarity:
            return self._relation_polarity[relation]
        # Layer C: 单边 LLM 精判缓存（按 agent.entity_id 的邻居名索引）
        if self._relation_llm_overrides and relation in self._relation_llm_overrides:
            return self._relation_llm_overrides[relation]
        # Layer B: 关键字兜底
        return infer_polarity(relation)

    def _build_relationship_context(self) -> None:
        """开局一次性从 Kuzu 预取各 agent 的盟友/对手(关系静态)，缓存并播种信任矩阵。

        顺序执行(非并发)，规避 Kuzu 单连接线程安全问题；运行中只读缓存，
        不在并发 decide() 里查图。量化经 relationship_context 注入 Prompt，
        定性额外经 seed_trust 影响打分/信任摘要。
        """
        if self.graph is None:
            self._log("simulation", "关系反哺跳过: self.graph is None (图数据库未初始化)")
            return
        if not self.agents:
            self._log("simulation", "关系反哺跳过: self.agents 为空")
            return
        self._log("simulation", f"关系反哺开始: {len(self.agents)} 个智能体, 图状态={self.graph is not None}")
        total_neighbors = 0
        # Layer C: 收集 A/B 均 neutral 的边，供开局异步精判（按关系名聚合，保留样本）
        layer_c_pending: dict[str, list[tuple[str, str]]] = {}
        for a in self.agents:
            allies: list[str] = []
            foes: list[str] = []
            try:
                data = self.graph.get_entity_neighbors(a.entity_id, max_depth=1)
            except Exception as e:
                logger.debug("[Simulator] 关系预取失败 %s (%s): %s", a.name, a.entity_id, e)
                continue
            nebs = data.get("neighbors", [])
            total_neighbors += len(nebs)
            for nb in nebs:
                nm = nb.get("name", "")
                if not nm or nm == a.name:
                    continue
                rel = nb.get("relation", "")
                kind = self._classify_relation(rel)
                if kind == "neutral":
                    # 记录样本用于 Layer C；仅当关系名不在结构化映射且关键字也为 neutral 时
                    if rel and rel not in self._relation_polarity:
                        samples = layer_c_pending.setdefault(rel, [])
                        if len(samples) < 4:
                            samples.append((a.name, nm))
                elif kind == "ally" and nm not in allies:
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
            if nebs:
                rel_types = list({nb.get("relation","?") for nb in nebs})
                self._log("simulation", f"  {a.name}: {len(nebs)} 邻居, 关系类型={rel_types}, 盟友={len(allies)}, 对手={len(foes)}")
        self._log("simulation", f"关系反哺概况: 全图 {total_neighbors} 条邻居边")
        seeded = sum(1 for v in self._rel_context.values() if v["summary"])
        if seeded:
            self._log("simulation", f"关系反哺：{seeded} 个智能体注入图谱盟友/对手并播种信任")
        # Layer C 待精判集合（relation → 样本对），供 _run_layer_c_judgment 消费
        self._layer_c_pending = layer_c_pending

    async def _run_layer_c_judgment(self) -> None:
        """Layer C: 对 A/B 均 neutral 的高频交互边做 LLM 批量精判，并重建关系上下文。

        只执行一次（由 run_round 首次调用），失败静默回退（保持 A/B 结果）。
        """
        pending = getattr(self, "_layer_c_pending", {}) or {}
        if not pending or getattr(self, "_layer_c_done", False):
            return
        self._layer_c_done = True
        if self.graph is None:
            return
        # 过滤：精判所有出现过的 neutral 关系（>=1 样本，放宽阈值），
        # 但设总候选数上限防一次性大量 LLM 调用（缺口5 修复）
        candidates = {rel: samples for rel, samples in pending.items() if samples}
        if len(candidates) > 12:
            # 按样本数排序，优先精判出现更频繁的关系
            candidates = dict(sorted(candidates.items(),
                                     key=lambda kv: len(kv[1]), reverse=True)[:12])
        if not candidates:
            return
        try:
            from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
            from strategy_forge.core.llm_client import Message
            from strategy_forge.core.config import config
            client = LLMClient()
            judged: dict[str, str] = {}
            for rel, samples in candidates.items():
                sample_lines = "\n".join(
                    f"- {a} → {b} 关系「{rel}」" for a, b in samples)
                prompt = (
                    "你是战略关系语义分析专家。判断以下实体间关系属于哪种利益极性。\n"
                    "只输出 JSON：{\"polarity\": \"foe|ally|neutral\"}\n\n"
                    f"{sample_lines}\n\n"
                    "判定方法：\n"
                    "- foe：零和/利益直接冲突/此消彼长\n"
                    "- ally：共赢/协同/利益一致\n"
                    "- neutral：无明确利益倾向\n"
                    "若从样本无法判断，输出 neutral。"
                )
                resp = await client.chat_json(
                    [Message(role="user", content=prompt)],
                    system="你是战略关系语义分析专家，只输出 JSON。",
                    schema_name="relation_polarity", temperature=0,
                    max_tokens=min(config.deduction_intel_max_tokens, 200),
                )
                try:
                    import json as _json
                    from strategy_forge.engine._utils import extract_text as _et
                    data = _json.loads(_et(resp))
                    pol = str(data.get("polarity", "neutral")).strip().lower()
                    if pol in ("foe", "ally"):
                        judged[rel] = pol
                except Exception:
                    continue
            if judged:
                self._relation_llm_overrides.update(judged)
                self._log("simulation",
                          f"关系精判(Layer C): {len(judged)} 种关系升级 — "
                          + ", ".join(f"{k}={v}" for k, v in judged.items()))
                # 基于精判结果重建关系上下文与信任矩阵
                self._rel_context = {}
                self._build_relationship_context()
        except Exception as e:
            logger.warning("[Simulator] Layer C 关系精判失败，使用 A/B 结果: %s", e)

    def _seed_polarization_relations(self, graph_seeded: int) -> None:
        """对无图谱关系的 agent，按 polarization 指标自动划分敌友。
        
        消上帝视角L3：仅扫描已知邻居（共享事件 / 图谱邻居），不遍历全体 agent 的 polarization。
        """
        # 构建已知邻居集合：图谱邻居 + 共享事件的 agent
        known_neighbors: dict[str, set[str]] = {}
        for a in self.agents:
            kn = set()
            # 图谱邻居
            rel = self._rel_context.get(a.entity_id, {})
            kn.update(rel.get("allies", []))
            kn.update(rel.get("opponents", []))
            # 共享事件的 agent（最近20条事件）
            for evt in self._event_history[-20:]:
                if evt.get("agent") == a.entity_id:
                    tgt = evt.get("target", "")
                    if tgt and tgt in {o.entity_id for o in self.agents}:
                        kn.add(tgt)
                parts = evt.get("participants", "") or ""
                for p in parts.split(","):
                    p = p.strip()
                    if p and p in {o.entity_id for o in self.agents}:
                        kn.add(p)
            known_neighbors[a.entity_id] = kn

        for a in self.agents:
            if self._rel_context.get(a.entity_id, {}).get("summary"):
                continue
            state = self._states.get(a.entity_id)
            if state is None:
                continue
            polar = state.metrics.get("polarization", 0)
            allies: list[str] = []
            foes: list[str] = []
            kn = known_neighbors.get(a.entity_id, set())
            for other in self.agents:
                if other.entity_id == a.entity_id:
                    continue
                if other.entity_id not in kn:
                    continue  # 仅扫描已知邻居
                other_st = self._states.get(other.entity_id)
                if other_st is None:
                    continue
                other_polar = other_st.metrics.get("polarization", 0)
                if abs(polar) < 0.5 and abs(other_polar) < 0.5:
                    continue
                if (polar > 0 and other_polar > 0) or (polar < 0 and other_polar < 0):
                    allies.append(other.name)
                elif (polar * other_polar) < 0:
                    foes.append(other.name)
            if allies or foes:
                parts = []
                if allies:
                    parts.append("盟友: " + "、".join(allies[:6]))
                if foes:
                    parts.append("对手: " + "、".join(foes[:6]))
                self._rel_context[a.entity_id] = {
                    "allies": allies, "opponents": foes,
                    "summary": "；".join(parts)}
                self.reasoner.seed_trust(a.entity_id, allies, foes)
        post_seeded = sum(1 for v in self._rel_context.values() if v["summary"])
        if post_seeded > graph_seeded:
            self._log("simulation",
                       f"极化补全：{post_seeded - graph_seeded} 个智能体通过 polarization 自动划分敌友")

    def _augment_recall_query(self, base: str, entity_id: str) -> str:
        """④ 关系邻居增强召回：把 Kuzu 盟友/对手名拼进动态事件召回 query，聚焦相关事件。

        默认关（FORGE_RECALL_REL_BOOST=0）时直接返回 base，行为与现状逐字一致。
        """
        from strategy_forge.core.config import config
        from strategy_forge.core.providers import registry as _reg
        if not _reg.recall_rel_boost:
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

    def get_state(self) -> dict[str, Any]:
        """导出暂态快照供暂停时持久化。"""
        return {
            "_event_history": list(self._event_history[-100:]),
            "_narrative_env": dict(getattr(self, "_narrative_env", {})),
            "_agent_knowledge": {k: [dict(e) for e in v[-200:]]
                                 for k, v in getattr(self, "_agent_knowledge", {}).items()},
            "_intel_bonuses": {k: dict(v) for k, v in
                               getattr(self, "_intel_bonuses", {}).items()},
            "_personality_log": list(getattr(self, "_personality_log", [])),
            "_character_journal": {k: list(v) for k, v in
                                  getattr(self, "_character_journal", {}).items()},
            "_reflection_baselines": {k: dict(v) for k, v in
                                      getattr(self, "_reflection_baselines", {}).items()},
            "_env_snapshots": {k: [dict(s) for s in v]
                               for k, v in getattr(self, "_env_snapshots", {}).items()},
            "_last_reflection_round_n": dict(getattr(self, "_last_reflection_round_n", {})),
            "_last_round_outcomes": dict(getattr(self, "_last_round_outcomes", {})),
            "_prev_rel_map": {k: {"allies": list(v.get("allies", [])),
                                  "opponents": list(v.get("opponents", []))}
                              for k, v in getattr(self, "_prev_rel_map", {}).items()},
            # 融合架构：通道② 已触发的 (实体id, 事件名) 去重集合（list of list，JSON 可序列化）
            "_event_trigger_fired": [
                list(x) if isinstance(x, (tuple, list)) else [x]
                for x in (getattr(self, "_event_trigger_fired", set()) or set())
            ],
            # 融合架构：已分发事件去重集合（缺陷6：纳入快照 + 有界）
            "_dispatched_eids": list(getattr(self, "_dispatched_eids", set()) or set()),
            # 反思机制动态塑造的行为准则：暂停恢复后不丢失（P0#2）
            "_agent_extra_rules": {
                a.entity_id: (a.system_prompt_extra or "")
                for a in getattr(self, "agents", [])
            },
        }

    def restore_state(self, saved: dict[str, Any]) -> None:
        """从暂停快照恢复暂态。"""
        self._event_history = saved.get("_event_history", [])
        self._narrative_env = saved.get("_narrative_env",
                                         getattr(self, "_narrative_env", {}))
        self._agent_knowledge = saved.get("_agent_knowledge", {})
        self._intel_bonuses = saved.get("_intel_bonuses", {})
        self._personality_log = saved.get("_personality_log", [])
        self._character_journal = saved.get("_character_journal", {})
        self._reflection_baselines = saved.get("_reflection_baselines", {})
        self._env_snapshots = saved.get("_env_snapshots", {})
        self._last_reflection_round_n = saved.get("_last_reflection_round_n", {})
        self._last_round_outcomes = saved.get("_last_round_outcomes", {})
        self._prev_rel_map = saved.get("_prev_rel_map", {})
        # 融合架构：恢复通道②去重集合（兼容 JSON 往返：元素可能是 tuple 或 list）
        _fired = saved.get("_event_trigger_fired", [])
        self._event_trigger_fired = set()
        for _x in (_fired if isinstance(_fired, list) else []):
            if isinstance(_x, (tuple, list)) and len(_x) == 2:
                self._event_trigger_fired.add((str(_x[0]), str(_x[1])))
            elif _x is not None:
                self._event_trigger_fired.add(str(_x))
        # 融合架构：恢复已分发事件去重集合（缺陷6）
        _dsp = saved.get("_dispatched_eids", [])
        self._dispatched_eids = set(_dsp) if isinstance(_dsp, list) else set()
        # 恢复反思行为准则到 agent 对象（P0#2）
        extra_rules = saved.get("_agent_extra_rules", {}) or {}
        if extra_rules and getattr(self, "agents", None):
            for a in self.agents:
                extra = extra_rules.get(a.entity_id, "")
                if extra:
                    a.system_prompt_extra = extra

    # ── 用户强制 override ──
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

    async def _shared_dual_recall(
        self, agent: DeductionAgentProfile,
        recall_top_k: int | None = None,
        recall_chars: int | None = None,
    ) -> tuple[str, str]:
        """共享双路 LanceDB 召回：Path A 静态原著 + Path B 动态事件。
        叙事和量化模式统一调用本方法，消除 ~40 行重复代码。
        """
        from strategy_forge.core.providers import registry as _reg
        from strategy_forge.core.config import config as _cfg
        rk = recall_top_k or _reg.retrieve_top_k
        rc = recall_chars or 300
        pp = self._preprocessor
        static_text, dynamic_text = "", ""
        if pp is not None and getattr(pp, "result", None):
            try:
                frags = await asyncio.to_thread(
                    pp.retrieve_for_entity, agent.name, rk,
                    {agent.name} if agent.name else None)
                if frags:
                    static_text = "\n---\n".join(f[:300] for f in frags[:rk])[:rc]
            except Exception as e:
                logger.warning("[Simulator] 静态召回失败 %s: %s", agent.name, e)
        if self._persist_events and pp is not None:
            try:
                aliases: set[str] = set()
                if getattr(pp, "result", None):
                    aliases = set(pp.result.high_freq_entities.get(agent.name, set()))
                    aliases.update(pp.result.low_freq_entities.get(agent.name, set()))
                query = (agent.name + " " + " ".join(aliases - {agent.name})).strip()
                query = self._augment_recall_query(query, agent.entity_id)
                frags = await asyncio.to_thread(
                    pp.retrieve_dynamic_events, query, rk,
                    _cfg.deduction_similarity_threshold, agent.name)
                if frags:
                    dynamic_text = "\n---\n".join(frags[:rk])[:rc]
            except Exception as e:
                logger.warning("[Simulator] 动态召回失败 %s: %s", agent.name, e)
        elif not self._persist_events:
            mem = [e for e in self._event_history[-20:]
                   if (agent.name in e.get("content", "") or e.get("agent") == agent.entity_id)
                   and _is_event_visible_to(agent.entity_id, agent.name, e)]
            if mem:
                dynamic_text = "\n".join(f"- {e.get('content', '')[:80]}" for e in mem[-3:])
        return static_text or "无特定背景", dynamic_text or "无近期模拟事件"

    # ── 事件驱动反思：触发规则（纯关键词匹配，0 LLM 成本）──
    _TRIGGER_RULES: dict[str, list[str]] = {
        # 军事冲突
        "遭攻击": ["遭到攻击", "被袭击", "被入侵", "被轰炸", "受到打击", "遭到空袭",
                   "被伏击", "被围困", "防线被突破", "阵地失守"],
        # 战败
        "战败": ["战败", "溃败", "大败", "撤退", "失守", "全军覆没",
                 "损失惨重", "伤亡巨大", "被歼灭", "败退"],
        # 主动战争
        "开战": ["开战", "宣战", "发动攻击", "入侵", "进攻", "突袭",
                 "空袭", "打击", "轰炸", "出兵"],
        # 外交背叛
        "遭背叛": ["背叛", "被出卖", "被欺骗", "被利用", "背信弃义",
                   "违约", "撕毁协议", "毁约", "背弃承诺", "倒戈",
                   "投敌", "叛变", "内奸", "间谍", "两面三刀"],
        # 断交/关系恶化
        "关系恶化": ["断交", "断绝关系", "撕毁条约", "退出联盟", "关系破裂",
                     "分道扬镳", "决裂", "反目成仇", "盟友倒向敌方"],
        # 战略重大失败
        "重大失败": ["失败", "未能", "功亏一篑", "计划落空", "无功而返",
                     "重大失误", "战略失误", "判断错误", "失策", "功败垂成",
                     "未达预期", "空手而归", "无法达成", "受阻"],
        # 意外（负面）
        "意外转折": ["意外", "突然", "出乎意料", "措手不及", "始料未及",
                     "突发", "变故", "晴天霹雳", "逆转", "翻盘",
                     "意外失败", "出人意料"],
        # 被制裁/封锁
        "遭制裁": ["被制裁", "被封锁", "被禁运", "被冻结资产", "被列入黑名单",
                   "被孤立", "被排挤", "被切断", "被禁", "受限",
                   "被遏制", "被围堵"],
        # 资源危机
        "资源危机": ["资源枯竭", "粮草不足", "资金断裂", "供应中断", "物资匮乏",
                     "能源危机", "断供", "储备耗尽", "入不敷出", "弹尽粮绝"],
        # 内部动荡
        "内乱": ["政变", "内乱", "叛乱", "分裂", "内讧", "哗变",
                 "抗议", "暴动", "暗杀", "刺杀", "内部分裂",
                 "权力斗争", "派系冲突", "民怨沸腾"],
        # 情报相关
        "情报泄露": ["情报泄露", "秘密暴露", "被发现", "暴露", "泄密",
                     "走漏风声", "被识破", "被揭穿", "阴谋败露"],
        # 联盟变化
        "联盟变动": ["结盟", "结为新盟友", "缔结同盟", "联手", "联合",
                     "瓦解联盟", "盟友倒戈", "投靠", "投诚", "归附"],
        # 意外成功（正面触发反思）
        "意外成功": ["意外成功", "大获全胜", "超出预期", "意外收获", "重大突破",
                     "不战而胜", "意外得利", "趁火打劫得手"],
        # 被威胁/勒索
        "被胁迫": ["被威胁", "被勒索", "被要挟", "被逼无奈", "被迫让步",
                   "身不由己", "受制于人", "被裹挟"],
        # 舆论/声誉
        "声誉危机": ["舆论压力", "声誉受损", "形象崩塌", "被谴责", "被孤立",
                     "信任危机", "信用崩塌", "名誉扫地", "千夫所指"],
    }

    def _has_trigger_event(self, action) -> tuple[str, str, int] | None:
        """检测 action 是否包含触发反思的关键事件。返回 (类别, 匹配词, 严重度1-4) 或 None。"""
        content = getattr(action, "content", "") or ""
        action_type = getattr(action, "action_type", "") or ""
        text = f"{action_type} {content}"
        for category, keywords in self._merged_triggers.items():
            for kw in keywords:
                if kw in text:
                    severity = self._get_severity(category)
                    return category, kw, severity
        return None

    def _get_severity(self, category: str) -> int:
        """D3: 事件权重分级。1=轻度 2=中度 3=重度 4=灾难。
        优先使用 __init__ 缓存的 domain-specific severity mapping，无匹配回退通用默认。
        """
        sm = getattr(self, "_severity_map", None)
        if sm:
            if category in sm.get("heavy", set()):
                return 3
            if category in sm.get("medium", set()):
                return 2
            if category in sm.get("light", set()):
                return 1
        # 通用默认回退
        _heavy = {"遭攻击", "战败", "开战", "遭背叛", "关系恶化", "被胁迫", "内乱"}
        _medium = {"遭制裁", "资源危机", "情报泄露", "重大失败", "声誉危机"}
        _light = {"意外转折", "联盟变动", "意外成功"}
        if category in _heavy:
            return 3
        if category in _medium:
            return 2
        if category in _light:
            return 1
        return 1

    # ── D4: 二阶复盘数据结构 ──
    _pending_corrections: dict = {}      # agent_id → list[{round, raw_rule, trigger}]
    _event_category_log: dict = {}       # agent_id → {category: [rounds]}
    _rule_history: dict = {}             # agent_id → [{"round":R, "rule":str, "status":"active"|"retired"}]
    _strategy_depth: dict = {}           # agent_id → 0=正常 1=谨慎 2=危机 3=重构

    # ── D1-D4 多模式 Prompt 模板 ──
    _PROMPT_EMOTIONAL = """你是 {name} 的本能反应层。刚刚经历了：{trigger_summary}
这是人类的即时情绪反应——允许短期偏激，允许防卫过当，允许非理性。

## 你的核心人格（不可撼动）
{persona}

## 触发事件内容
{action_content}

## 近期经历
{recent_events}

## 输出
一行代表人遭遇此类事件时本能情绪反应的准则（≤15字）。
示例："被背叛后复仇心压倒一切" "遭受攻击后全面戒备"
只输出准则本身。"""

    _PROMPT_STRATEGIC = """你是 {name} 的理性分析层。基于长期数据趋势，做全局战略判断。

## 核心人格
{persona}

## 当前宏观环境（0-100）
{env_stats}

## 近期变化趋势
{delta_summary}

## 多轮环境趋势
{trend_data}

## 现有准则
{current_rules}

## 输出
一行基于长期趋势的战略修正准则（≤20字），保持理性和远见。
示例："资源持续下滑应收缩防线保核心区"
只输出准则本身，如果当前战略正确输出"无需调整"。"""

    _PROMPT_DEEP = """你是 {name} 的深度战略重构层。{severity_label} 事件触发了你的全盘重思。

## 核心人格（不可撼动）
{persona}

## 触发事件
{trigger_summary}

## 近期全部经历
{recent_events}

## 现有准则（将被替换）
{current_rules}

## 历史同类经历
{history_patterns}

## 任务
这次打击极其严重——你需要生成一条根本性的新战略准则（≤25字），用于完全替代旧准则。
这条准则必须是经历过重大打击后产生的深层次策略转变。
示例："永不以信任作为博弈筹码" "在确保绝对实力前停止一切冒险扩张"
只输出准则本身。"""

    _PROMPT_RECONSTRUCT = """你是 {name} 的价值观重构层。毁灭级事件彻底瓦解了你的旧世界观。

## 仅存的核心人格
{persona}

## 毁灭级事件
{trigger_summary}

## 近期全部经历
{recent_events}

## 历史伤痛
{history_patterns}

## 旧准则（全部作废）
{current_rules}

## 任务
输出一条全新的、从根本上重构你行为方式的准则（≤30字）。
这不应该是对旧准则的修补——而是彻底的范式转变。
示例："从多边协作彻底转向自给自足的孤立主义" "放弃一切外交幻想，武力是唯一语言"
只输出准则本身。"""

    _PROMPT_INTERNAL = """你是 {name} 的自我审视层。当前一切平稳——但你需要主动检查。

## 核心人格
{persona}

## 现有准则
{current_rules}

## 近期行动
{recent_events}

## 历史遭遇模式
{history_patterns}

## 任务
查看过去的重复遭遇，判断是否有战略盲点需要一条新准则来覆盖。
如果现有准则已覆盖所有模式 → "无需调整"。如果发现脆弱点 → 输出新准则（≤20字）。
示例："警惕信任被反复利用" "在对方示弱时核查动机"
只输出准则或"无需调整"。"""

    _PROMPT_RETROSPECT = """你是 {name} 的纠错层。定期回溯你的全部行为准则，结合历史遭遇和近期事件清理过时项。

## 历史遭遇模式
{history_patterns}

## 近期自身事件
{recent_events}

## 现有准则
{current_rules}

## 判断标准
- 局势已变→准则过时→删除
- 过于极端→需要缓和→标记修正
- 曾因冲动产生→冷静后不适用→删除
- 历史模式显示同类问题未再出现→该应对准则可删除
- 近期事件表明某准则还在发挥作用→保留

## 输出
每行一个操作：
  删除：<准则原文>
  修正：<旧准则> → <新准则>
无变化输出"无需调整"。"""

    def _should_reflect(self, agent_id: str, round_number: int,
                         state: Any = None, rule_engine: Any = None) -> str | None:
        """共享反思闸门：环境漂移 + 关系变化 + 长期无反思保护。
        叙事和量化模式统一调用，返回触发原因或 None。
        阈值从 methodology.yaml 读取，无配置回退硬编码默认。
        """
        baseline = self._reflection_baselines.get(agent_id, dict(self._narrative_env))
        last_r = self._last_reflection_round_n.get(agent_id, 0)
        drift_single = self._get_reflection_threshold("env_drift_single", 5)
        drift_cumul = self._get_reflection_threshold("env_drift_cumulative", 12)
        no_reflect_rounds = self._get_reflection_threshold("no_reflect_rounds", 4)
        alarm_mult = self._get_reflection_threshold("metric_alarm_mult", 1.3)
        self_interval = self._get_reflection_threshold("self_reflect_interval", 5)
        pattern_cnt = self._get_reflection_threshold("pattern_warn_count", 8)
        # 条件1：环境累积剧变
        total_drift = 0.0
        for k in self._narrative_env:
            delta = self._narrative_env[k] - baseline.get(k, self._narrative_env[k])
            total_drift += abs(delta)
            if abs(delta) > drift_single:
                return f"环境剧变({k}{delta:+.0f})"
        if total_drift > drift_cumul:
            return f"环境累计漂移({total_drift:.0f})"
        # 条件2：关系网络变化（第1轮豁免——初始化过程中无真正"变化"可言）
        if round_number > 1:
            prev_rels = getattr(self, "_prev_rel_map", {})
            curr_rels = self._rel_context.get(agent_id, {})
            prev_allies = set(prev_rels.get(agent_id, {}).get("allies", []))
            curr_allies = set(curr_rels.get("allies", []))
            prev_opps = set(prev_rels.get(agent_id, {}).get("opponents", []))
            curr_opps = set(curr_rels.get("opponents", []))
            if prev_allies != curr_allies or prev_opps != curr_opps:
                return "关系网络变化"
        # 条件3：长期无反思保护
        if (round_number - last_r) > no_reflect_rounds:
            return "长期无反思保护"
        # 量化模式补充：指标告急触发反思
        if state is not None and rule_engine is not None:
            thr_map = getattr(rule_engine, "thresholds", lambda: {})()
            for m, v in getattr(state, "metrics", {}).items():
                thr = thr_map.get(m, 0)
                if thr > 0 and v <= thr * alarm_mult:
                    return f"指标告急({m}={v:.0f},阈={thr:.0f})"
        # 条件5: 内源主动自省
        if round_number > 0 and round_number % self_interval == 0:
            return "内源主动自省"
        # 条件6: D5 模式累积升级
        cat_log = self._event_category_log.get(agent_id, {})
        for cat, entries in cat_log.items():
            if len(entries) >= pattern_cnt:
                return f"模式预警({cat}×{len(entries)})"
        return None

    def _append_event(self, event: dict) -> None:
        """共享事件历史追加 + 截断。"""
        self._event_history.append(event)
        if len(self._event_history) > 200:
            self._event_history = self._event_history[-200:]

    def _apply_event_impacts(self, round_number: int) -> int:
        """融合架构·通道①：本轮高优先级事件按规则包 event_impact 冲击相关实体指标。

        仅量化模式（_rule_engine 非空）生效。事件只对发起者与目标实体生效，
        一次性注入指标增量，使"制裁/并购"等事件确定性地改变局势而非仅作文字上下文。

        关键（缺陷2 修复）：只处理【系统事件 / 外部注入事件】，跳过 agent 主动行动——
        agent 主动行动的效果已由 rule_engine.compute_deltas 结算，再走事件冲击会造成双重扣减。
        返回实际应用的事件数。
        """
        if not self._quantified or self._rule_engine is None:
            return 0
        impact_map = self._rule_engine.event_impact_map()
        # 规则包内的 agent 可执行动作（其效果已由行动结算处理，事件冲击需跳过）
        action_set = set(self._rule_engine.actions())
        applied = 0
        ranges = self._rule_engine.ranges()

        # 融合架构·盲点4：消费外部注入的系统事件（注入为当前轮系统事件，随后走通道①）
        pending = self._injected_events_store.get("pending", []) if self._injected_events_store else []
        if pending:
            # name→id 解析（修复缺陷4：target_id 传实体名时也能命中）
            _name_to_id = {a.name: a.entity_id for a in getattr(self, "agents", [])}
            for _ev in pending:
                _et = _ev.get("event_type", "")
                _imp = impact_map.get(_et)
                # 自定义 impact 优先（修复缺陷5）：事件自带 impact 且未匹配规则包时使用
                if not _imp:
                    _imp = _ev.get("impact") if isinstance(_ev.get("impact"), dict) else None
                _tid = str(_ev.get("target_id", "") or "").strip()
                if _tid and _tid not in self._states and _tid in _name_to_id:
                    _tid = _name_to_id[_tid]
                self._append_event({
                    "agent": "system",
                    "agent_name": "系统",
                    "action": "system_injected",
                    "content": _ev.get("content", _et or "外部事件"),
                    "round": round_number,
                    "event_type": _et,
                    "target_id": _tid,
                    "is_system_event": True,
                    "_injected_impact": _imp,
                })
            self._injected_events_store["pending"] = []

        for evt in self._event_history:
            if evt.get("round") != round_number:
                continue
            et = evt.get("event_type", "") or evt.get("action", "")
            # 注入事件可携带自定义 impact（缺陷5）；否则查规则包映射
            imp = evt.get("_injected_impact") or impact_map.get(et)
            if not imp or not isinstance(imp, dict):
                continue
            # 跳过 agent 主动行动：避免"行动结算 + 事件冲击"双重扣减
            if not evt.get("is_system_event") and et in action_set:
                continue
            targets = set()
            if evt.get("is_system_event") and not evt.get("target_id"):
                # 无目标指定 → 系统级事件作用于全部存活实体（修复缺陷4）
                targets.update(eid for eid in self._states)
            else:
                if evt.get("agent") and evt.get("agent") in self._states:
                    targets.add(evt["agent"])
                if evt.get("target_id") and evt["target_id"] in self._states:
                    targets.add(evt["target_id"])
            for eid in targets:
                st = self._states.get(eid)
                if st is None:
                    continue
                # 修复缺陷7：不结算已出局实体（避免污染死后历史）
                if self._rule_engine is not None and not self._rule_engine.is_alive(st):
                    continue
                st.apply_deltas({k: float(v) for k, v in imp.items()},
                                round_number, ranges)
            applied += 1
        return applied

    def _trigger_events_from_metrics(self, round_number: int) -> int:
        """融合架构·通道②：指标越界自动生成系统事件，进入事件流反向影响决策。

        仅量化模式生效。按规则包 event_triggers 对存活实体检查阈值，
        越界则追加系统事件（is_system_event=True），事件自带冲击映射。
        once 触发在单次推演内去重。
        """
        if not self._quantified or self._rule_engine is None:
            return 0
        triggers = self._rule_engine.event_triggers()
        if not triggers:
            return 0
        if not hasattr(self, "_event_trigger_fired"):
            self._event_trigger_fired: set = set()
        fired = self._event_trigger_fired
        generated = 0
        # 不按全局 is_alive 过滤：指标越界触发的事件往往正是"濒死预警/崩溃"本身，
        # 触发阈值可能高于死亡阈值（如 supply_chain<30 预警），过早跳过会漏掉关键转折。
        for eid, st in self._states.items():
            for trig in self._rule_engine.check_event_triggers(st, fired):
                name = trig.get("event", "")
                if not name:
                    continue
                if trig.get("once"):
                    fired.add((st.id, name))
                content = trig.get("content") or name
                self._append_event({
                    "agent": st.id, "agent_name": st.name,
                    "action": "system_trigger",
                    "content": content,
                    "round": round_number,
                    "event_type": name,
                    "target_id": "",
                    "is_system_event": True,
                    "impact_map": trig.get("impact") or {},
                })
                # 事件自带冲击映射（可选）立即结算
                imp = trig.get("impact")
                if isinstance(imp, dict):
                    st.apply_deltas({k: float(v) for k, v in imp.items()},
                                    round_number, self._rule_engine.ranges())
                # 系统事件持久化到 LanceDB（缺陷1 修复）：使其可被后续轮次语义召回，
                # 而非仅停留在内存 _event_history 的最近几条。
                if self._persist_events and self._preprocessor is not None:
                    try:
                        self._preprocessor.add_event_memory(
                            content=content, agent_id=st.id,
                            round_number=round_number,
                            event_type=f"system_{name}", priority=0.9)
                    except Exception as _e:
                        logger.debug("[Simulator] 系统事件写入 LanceDB 失败: %s", _e)
                # 系统事件写入 Kuzu Event 节点 + ACTED 边（缺陷2 修复）：
                # 使系统事件在报告时间线/因果视图与暂停恢复中可见，而非仅存内存/向量库。
                if self._persist_events and getattr(self, "graph", None) is not None:
                    try:
                        from datetime import datetime as _dt
                        _eid = f"evt-{uuid.uuid4().hex[:8]}"
                        _ts = _dt.now().isoformat()
                        self.graph.add_event(_eid, content[:200], f"system_{name}",
                                             _ts, st.id, round_number=round_number,
                                             effect=", ".join(f"{k}{v:+.0f}" for k, v in (imp or {}).items()),
                                             driver="system")
                        self.graph.add_acted(st.id, _eid, f"system_{name}", _ts)
                    except Exception as _e:
                        logger.debug("[Simulator] 系统事件写入 Kuzu 失败: %s", _e)
                generated += 1
        return generated

    async def run_round(self, round_number: int) -> SimulationRound:
        # Layer C: 首次开局精判 neutral 关系（幂等，仅一次）。
        # 置于量化/叙事分支之前，保证两种模式都执行（Bug1 修复）。
        await self._run_layer_c_judgment()

        if self._quantified:
            return await self._run_round_quantified(round_number)

        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient

        sim_round = SimulationRound(round_number=round_number)
        client = LLMClient()

        ordered = list(self.agents)
        self._rng.shuffle(ordered)

        if self._cancel is not None and self._cancel.is_set():
            raise _PhaseCancelledError()

        async def process_agent(agent: DeductionAgentProfile) -> SimulationAction | None:
            # 取信号量后再查一次取消，便于并发批内尽早短路
            if self._cancel is not None and self._cancel.is_set():
                return None
            from strategy_forge.core.config import config
            from strategy_forge.core.providers import registry as _reg
            from strategy_forge.core.llm_client import LLMConnectionError
            fails = 0
            max_passes = max(0, _reg.retry_passes)
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
            from strategy_forge.core.providers import registry as _reg
            ratio = conn_fails / max(1, len(ordered))
            if ratio >= _reg.sim_fail_threshold:
                first = next((r for r in results if isinstance(r, LLMConnectionError)), None)
                raise ConnectionFailureError(str(first) if first else f"连接故障：{conn_fails}/{len(ordered)} agent 无法连接 LLM")
        for agent, action in zip(ordered, results, strict=False):
            if isinstance(action, BaseException):
                self._log("simulation", f"agent {agent.name} 决策失败: {action}")
                continue
            if action is not None:
                sim_round.actions.append(action)
                _actor_name = getattr(
                    next((a for a in self.agents if a.entity_id == action.agent_id), None),
                    "name", action.agent_id[:8])
                from .narrative_actions import is_secret_action
                _secret = is_secret_action(action.action_type, action.content)
                _participants = "|".join(filter(None, [
                    _actor_name, action.agent_id, str(action.target_id or "")]))
                self._append_event({
                    "agent": action.agent_id,
                    "agent_name": _actor_name,
                    "action": action.action_type,
                    "content": action.content,
                    "round": round_number,
                    "timestamp": action.timestamp,
                    "visibility": "private" if _secret else "public",
                    "participants": _participants,
                })

        # Write round events to Kuzu graph + LanceDB dynamic event table
        # 蒙特卡洛隔离模式 (persist_events=False): 不落盘、不写向量库，仅保留内存事件历史，
        # 保证 M×N 次模拟相互隔离、可并发，且不污染主会话数据。
        if self._persist_events:
            for action in sim_round.actions:
                event_id = f"evt-{uuid.uuid4().hex[:8]}"
                _vis = (getattr(action, "metadata", {}) or {}).get("visibility", "public")
                self.graph.add_event(
                    event_id, action.content[:200], action.action_type,
                    action.timestamp, action.agent_id,
                    visibility=_vis,
                )
                self.graph.add_acted(action.agent_id, event_id, action.action_type, action.timestamp)

                # ★ 动态事件写入 LanceDB (下一轮决策即可语义召回)
                if self._preprocessor is not None:
                    try:
                        from .narrative_actions import is_secret_action as _isa
                        _sec = _isa(action.action_type, action.content)
                        _an = getattr(
                            next((a for a in self.agents if a.entity_id == action.agent_id), None),
                            "name", action.agent_id[:8])
                        self._preprocessor.add_event_memory(
                            content=action.content,
                            agent_id=action.agent_id,
                            round_number=round_number,
                            event_type=action.action_type,
                            visibility="private" if _sec else "public",
                            participants="|".join(filter(None, [
                                _an, action.agent_id, str(action.target_id or "")])),
                        )
                    except Exception as e:
                        logger.warning("[Simulator] Event memory write failed for %s: %s",
                                     action.agent_id, e)

        # ── 事件驱动反思：初始化基础设施 + 检测关键事件 ──
        if not hasattr(self, "_reflection_baselines"):
            self._reflection_baselines: dict[str, dict[str, float]] = {}
            self._last_reflection_round_n: dict[str, int] = {}
            import random as _random
            for agent in self.agents:
                self._last_reflection_round_n[agent.entity_id] = _random.randint(0, 2)
        if not hasattr(self, "_pending_corrections"):
            self._pending_corrections = {}
        if not hasattr(self, "_event_category_log"):
            self._event_category_log = {}

        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
        _erc = LLMClient()
        for action in sim_round.actions:
            trigger = self._has_trigger_event(action)
            if trigger:
                category, keyword, severity = trigger
                agent = next((a for a in self.agents if a.entity_id == action.agent_id), None)
                if agent:
                    eid = agent.entity_id
                    last_r = self._last_reflection_round_n.get(eid, 0)
                    if round_number - last_r < 1:
                        continue
                    content = getattr(action, "content", "")[:200]
                    # D1: 情绪应激反思
                    rule_added = await self._reflect_narrative(
                        agent, round_number, _erc, mode="emotional",
                        trigger_category=category, trigger_keyword=keyword,
                        action_content=content, severity=severity)
                    if rule_added:
                        # D4: 存入快照供下轮冷静纠错
                        self._pending_corrections.setdefault(eid, []).append({
                            "round": round_number, "rounds_ago": 1,
                            "raw_rule": agent.system_prompt_extra.split("；")[-1]
                            if agent.system_prompt_extra else "",
                            "trigger": f"{category}({keyword})",
                        })
                        self._reflection_baselines[eid] = dict(self._narrative_env)
                        self._last_reflection_round_n[eid] = round_number
                    # D5: 记录到历史分类日志
                    self._log_category_event(eid, category, round_number,
                                              action.content)
                    self._log("simulation",
                        f"[事件反思-D1] {agent.name}: {category}({keyword}) severity={severity} → "
                        f"{'新增准则' if rule_added else '无需调整'} (R{round_number})")
                    # D3: L3+ 重度事件 → 深度/价值观重构反思
                    if severity >= 3 and rule_added:
                        deep_mode = "reconstruct" if severity >= 4 else "deep"
                        depth_label = "重构" if severity >= 4 else "深度"
                        await self._reflect_narrative(
                            agent, round_number, _erc, mode=deep_mode,
                            trigger_category=category, trigger_keyword=keyword,
                            action_content=content, severity=severity)
                        self._log("simulation",
                            f"[事件反思-D3{depth_label}] {agent.name}: severity={severity} (R{round_number})")

        # ── 叙事模式环境评估（每轮最多 3 个 Agent 抽样）──
        await self._assess_env_impact(sim_round, round_number)

        # ── 环境自然衰减──
        for key in self._narrative_env:
            self._narrative_env[key] = max(0.0, min(100.0,
                round(self._narrative_env[key] * 0.95, 1)))

        # ── 共享反思闸门（叙事模式调用 _reflect_narrative）──
        if not hasattr(self, "_reflection_baselines"):
            self._reflection_baselines: dict[str, dict[str, float]] = {}
            self._last_reflection_round_n: dict[str, int] = {}
            import random as _random
            for agent in self.agents:
                self._last_reflection_round_n[agent.entity_id] = _random.randint(0, 2)

        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
        from strategy_forge.core.providers import registry as _reg
        _rc = LLMClient()
        _max_conc = max(1, _reg.max_concurrent)
        _sem = asyncio.Semaphore(_max_conc)

        # ── D4-phase2: 并行纠错 ──
        async def _correct_one(eid: str, remaining: list, agent, snapshots) -> None:
            async with _sem:
                for snap in snapshots:
                    if round_number - snap["round"] >= 1:
                        corrected = await self._reflect_correct(agent, snap, round_number, _rc)
                        if corrected and agent.system_prompt_extra:
                            old_rules = agent.system_prompt_extra.split("；")
                            if snap["raw_rule"] in old_rules:
                                new_rules = [r for r in old_rules if r != snap["raw_rule"]]
                                new_rules.append(corrected)
                                agent.system_prompt_extra = "；".join(new_rules)
                            self._log("simulation",
                                f"[D4纠错] {agent.name}: {snap['trigger']} 冲动准则已修正 (R{round_number})")
                    else:
                        remaining.append(snap)
                if remaining:
                    self._pending_corrections[eid] = remaining
                else:
                    self._pending_corrections.pop(eid, None)

        d4_tasks = []
        for eid, snapshots in list(self._pending_corrections.items()):
            agent = next((a for a in self.agents if a.entity_id == eid), None)
            if agent:
                remaining: list = []
                d4_tasks.append(_correct_one(eid, remaining, agent, snapshots))
        if d4_tasks:
            await asyncio.gather(*d4_tasks)

        # ── 主要反思循环：并行 ──
        async def _reflect_one(agent) -> None:
            async with _sem:
                eid = agent.entity_id
                reason = self._should_reflect(eid, round_number)
                if reason:
                    if "内源" in reason:
                        rule_added = await self._reflect_narrative(agent, round_number, _rc, mode="internal")
                        tag = "[内源自省]"
                    elif "环境" in reason or "累计" in reason or "模式" in reason:
                        rule_added = await self._reflect_narrative(agent, round_number, _rc, mode="strategic")
                        tag = "[D2数值复盘]" if "模式" not in reason else "[模式预警]"
                    else:
                        rule_added = await self._reflect_narrative(agent, round_number, _rc)
                        tag = "[反思]"
                    if rule_added:
                        self._reflection_baselines[eid] = dict(self._narrative_env)
                        self._last_reflection_round_n[eid] = round_number
                    self._log("simulation",
                        f"{tag} {agent.name}: {reason} → "
                        f"{'新增准则' if rule_added else '无需调整'} (R{round_number})")

        reflect_tasks = [_reflect_one(a) for a in self.agents]
        await asyncio.gather(*reflect_tasks)

        # D2 趋势：记录本轮环境快照
        for a in self.agents:
            self._record_env_snapshot(a.entity_id)

        # ── D6.2: 并行回溯悔悟 ──
        if round_number > 0 and round_number % 5 == 0:
            async def _retrospect_one(agent) -> None:
                async with _sem:
                    await self._reflect_retrospect(agent, round_number, _rc)
            retro_tasks = [_retrospect_one(a) for a in self.agents]
            await asyncio.gather(*retro_tasks)

        # 保存本轮关系网络快照供下轮对比
        if not hasattr(self, "_prev_rel_map"):
            self._prev_rel_map: dict = {}
        for eid, ctx in self._rel_context.items():
            self._prev_rel_map[eid] = {
                "allies": list(ctx.get("allies", [])),
                "opponents": list(ctx.get("opponents", [])),
            }

        # ── 叙事模式态势快照（供前端 dashboard tab 使用）──
        sim_round.state_delta["snapshot"] = {
            "round": round_number,
            "entity_count": len(self.agents),
            "_thresholds": {},  # 空阈值避免前端判定全部"已淘汰"
            "entities": [
                {"name": a.name, "alive": True}
                for a in self.agents
            ],
            "recent": [
                {"agent": e.get("agent_name", ""), "action": e.get("action", ""),
                 "content": e.get("content", "")[:80], "round": e.get("round", 0)}
                for e in self._event_history[-8:]
                if (e.get("visibility", "") or "public") not in _RESTRICTED_VIS
            ],
        }

        return sim_round

    async def _reflect_narrative(
        self, agent: DeductionAgentProfile, round_number: int, client: Any,
        mode: str = "default",
        trigger_category: str = "", trigger_keyword: str = "",
        action_content: str = "", severity: int = 1,
    ) -> bool:
        """叙事模式人格反思。mode: default/emotional/strategic/deep/internal。
        返回 True=新增规则，False=无需调整。"""
        from strategy_forge.core.llm_client import Message
        from ._utils import extract_text

        my_events = [
            e for e in self._event_history[-30:]
            if e.get("agent") == agent.entity_id or e.get("agent_name") == agent.name
        ]
        if not my_events:
            # observe 型 agent 无事件记录 → 基于环境状态构建反思上下文
            env_state = "\n".join(
                f"- {k}: {v:.0f}" for k, v in self._narrative_env.items())
            events_text = (
                f"（你在本轮未采取行动，处于观察状态）\n"
                f"当前环境状态：\n{env_state}"
            )
        else:
            events_text = "\n".join(
                f"- [R{e.get('round','?')}] {e.get('content','')[:100]}"
                for e in my_events[-15:]
            )
        # ── 模版路由：根据不同反思维度选择 prompt 和参数 ──
        history_patterns = self._get_history_patterns(agent.entity_id)
        current_rules = agent.system_prompt_extra or "（无）"
        env_stats = "\n".join(f"- {k}: {v:.0f}" for k, v in self._narrative_env.items())
        persona = agent.persona or "（无）"

        if mode == "emotional":
            prompt = self._PROMPT_EMOTIONAL.format(
                name=agent.name, persona=persona,
                trigger_summary=f"{trigger_category}({trigger_keyword})",
                action_content=action_content[:200],
                recent_events=events_text,
            )
            _mt = 80
        elif mode == "strategic":
            baseline = self._reflection_baselines.get(agent.entity_id, dict(self._narrative_env))
            deltas = []
            for k in self._narrative_env:
                d = self._narrative_env[k] - baseline.get(k, self._narrative_env[k])
                deltas.append(f"- {k}: {d:+.0f}")
            # D2 增强：最近3轮环境趋势
            trend_text = self._build_env_trend(agent.entity_id)
            prompt = self._PROMPT_STRATEGIC.format(
                name=agent.name, persona=persona,
                env_stats=env_stats, delta_summary="\n".join(deltas),
                current_rules=current_rules,
                trend_data=trend_text,
            )
            _mt = 120
        elif mode in ("deep", "reconstruct"):
            sev_label = {3: "重度—格局级", 4: "灾难—价值观重构级"}.get(severity, "重度")
            template = self._PROMPT_RECONSTRUCT if mode == "reconstruct" else self._PROMPT_DEEP
            prompt = template.format(
                name=agent.name, persona=persona,
                severity_label=sev_label,
                trigger_summary=f"{trigger_category}({trigger_keyword})",
                current_rules=current_rules,
                history_patterns=history_patterns or "首次遭遇此类事件",
                recent_events=events_text,
            )
            _mt = 250 if severity == 3 else 400
        elif mode == "internal":
            prompt = self._PROMPT_INTERNAL.format(
                name=agent.name, persona=persona,
                current_rules=current_rules,
                recent_events=events_text,
                history_patterns=history_patterns or "首次自省",
            )
            _mt = 120
        else:
            prompt = (
            f"你是 {agent.name} 的潜意识。回顾你近期的行动经历，"
            f"判断你的性格是否需要微调。\n\n"
            f"## 你的核心人格（不可改动）\n{agent.persona or '（无）'}\n\n"
            f"## 你现有的行为准则\n{agent.system_prompt_extra or '（无，完全依据核心人格）'}\n\n"
            f"## 近期行动经历\n{events_text}\n\n"
            f"## 任务\n"
            f"根据以上经历，判断是否需要添加一条新的行为准则（或修正旧准则），"
            f"使你的行为更符合当前的处境。\n"
            f"【重要】你的人格核心不可动摇，新准则只能是对核心人格的策略性微调，"
            f"禁止产生与核心人格根本矛盾的方向性反转。\n"
            f"【重要】新准则必须由上方「近期行动经历」中的某条具体经历直接引出，"
            f"禁止脱离经历凭空生成；准则应符合现实中该处境下真实的人会有的心理变化。\n"
            f"- 输出格式：一行简短中文准则（20字以内），直接陈述。\n"
            f"- 如果当前人格已足够应对，输出\"无需调整\"。\n"
            f"- 仅添加/修正，不删除原有准则。最多保留3条准则，超限时替换最旧的一条。\n"
            f"- 示例：\"遭受背叛后更谨慎选择盟友\" \"危急时刻敢于孤注一掷\"\n"
            f"- 何时输出\"无需调整\"：近期经历与人格一致、现有准则已覆盖行为模式\n"
            f"- 如果本轮经历了重要的人际承诺、债务或背叛，另起一行输出：\n"
            f"  记忆：向[某角色]承诺/欠/被[具体事件]\n"
            f"  没有重要人际事件则省略此行。\n"
            f"\n只输出准则本身或\"无需调整\"，不要解释。"
        )
            _mt = 80
        try:
            # P2#14: 结构化反思输出——统一约束为 JSON，替代自由文本字符串匹配。
            # prompt 末尾追加 JSON 输出说明（不改动各 mode 模板主体）。
            json_instruct = (
                "\n\n## 输出（仅 JSON，无其他文字）\n"
                '{"new_rule": "≤20字的新行为准则；无需调整时留空字符串", '
                '"changed": true/false, "memory": "重要人际承诺/债务/背叛，无则空字符串"}'
            )
            resp = await client.chat_json(
                [Message(role="user", content=prompt + json_instruct)],
                system="你是潜意识分析师。输出结构化 JSON，new_rule 为 ≤20字中文行为准则；无需调整时 changed=false 且 new_rule 为空。",
                schema_name="reflection_result",
                temperature=0.3 if mode != "emotional" else 0.5,
                max_tokens=_mt,
            )
            text = extract_text(resp).strip()
            if not text:
                return False
            try:
                import json as _json
                data = _json.loads(text)
                changed = bool(data.get("changed", False))
                rule_text = str(data.get("new_rule", "") or "").strip()
                mem_text = str(data.get("memory", "") or "").strip()[:40]
            except (ValueError, TypeError):
                # 兼容：偶发非 JSON 回退自由文本解析
                data = {}
                changed = text not in ("", "无需调整")
                rule_text = "" if "无需调整" in text else text
                mem_text = ""
            # 提取私人记忆行（结构化 memory 字段），与人格准则分离处理
            if mem_text:
                if not hasattr(self, "_character_journal"):
                    self._character_journal: dict[str, list[str]] = {}
                self._character_journal.setdefault(agent.entity_id, []).append(
                    f"R{round_number}: {mem_text}")
                if len(self._character_journal[agent.entity_id]) > 5:
                    self._character_journal[agent.entity_id] = \
                        self._character_journal[agent.entity_id][-5:]
            if not changed or not rule_text or "无需调整" in rule_text or len(rule_text) < 2:
                return False
            text = rule_text
            old_extra = agent.system_prompt_extra
            # 去重：新准则与已有准则相似度 > 0.7 则跳过，避免反复生成近似内容
            if old_extra:
                existing_rules = old_extra.split("；")
                for old in existing_rules:
                    if old and text:
                        common = len(set(old) & set(text))
                        denom = max(len(set(old)), len(set(text)), 1)
                        if common / denom > 0.7:
                            return False
            if old_extra and text not in old_extra:
                parts = old_extra.split("；")
                max_rules = self._max_persona_rules()
                if len(parts) >= max_rules:
                    parts = parts[1:]  # 丢弃最旧
                    agent.system_prompt_extra = "；".join(parts + [text])
                else:
                    agent.system_prompt_extra = f"{old_extra}；{text}"
            elif not old_extra:
                agent.system_prompt_extra = text
            else:
                return False
            self._personality_log.append({
                "round": round_number, "agent": agent.name,
                "old_extra": old_extra, "new_extra": agent.system_prompt_extra,
            })
            self._log("simulation",
                       f"[叙事人格演化] {agent.name} 新增准则: {text} (R{round_number})")
            return True
        except Exception as e:
            logger.debug("[Simulator] 叙事反思失败: %s", e)
            return False

    def _build_env_trend(self, agent_id: str) -> str:
        """D2 趋势：最近3轮环境快照 → 分段变化方向。"""
        snaps = getattr(self, "_env_snapshots", {}).get(agent_id, [])
        if len(snaps) < 2:
            return "（数据不足，无法生成趋势）"
        lines = []
        for i in range(1, len(snaps)):
            prev = snaps[i - 1]
            curr = snaps[i]
            round_tag = f"R{i}→R{i+1}"
            changes = []
            for k in curr:
                d = curr[k] - prev.get(k, curr[k])
                if abs(d) >= 2:
                    direction = "+" if d > 0 else ""
                    changes.append(f"{k}{direction}{d:.0f}")
            if changes:
                lines.append(f"{round_tag}: {', '.join(changes)}")
        return " ｜ ".join(lines) if lines else "（无显著变化）"

    def _record_env_snapshot(self, agent_id: str) -> None:
        """记录当前环境快照至趋势历史（保留最近5轮）。"""
        if not hasattr(self, "_env_snapshots"):
            self._env_snapshots = {}
        snap = dict(self._narrative_env)
        hist = self._env_snapshots.setdefault(agent_id, [])
        hist.append(snap)
        if len(hist) > 5:
            self._env_snapshots[agent_id] = hist[-5:]

    def _get_history_patterns(self, agent_id: str) -> str:
        """D5: 聚合历史同类事件的内容摘要，供反思层识别策略模式。
        兼容旧 int(round_number) 格式和新 {round,content} dict 格式。
        """
        log = self._event_category_log.get(agent_id, {})
        if not log:
            return ""
        patterns = []
        for cat, entries in log.items():
            if len(entries) < 2:
                continue
            recent = entries[-5:]
            samples = []
            for e in recent:
                if isinstance(e, dict):
                    c = (e.get("content") or "").strip()
                    samples.append(f"R{e.get('round','?')}:{c[:50]}" if c else f"R{e.get('round','?')}")
                else:
                    # 旧格式兼容：int(round_number)
                    samples.append(f"R{e}")
            patterns.append(f"- {cat}（共{len(entries)}次）：{' | '.join(samples)}")
        return "\n".join(patterns) if patterns else ""

    def _log_category_event(self, agent_id: str, category: str, round_number: int,
                            event_content: str = "") -> None:
        """D5: 记录触发事件类别、轮号及内容摘要，供历史模式识别。"""
        raw = (event_content or "").strip()
        if not raw:
            raw = "无详细记录"
        # 智能截断：在句号/问号/感叹号处自然断句
        if len(raw) > 100:
            raw = raw[:100]
        for sep in ("。", "！", "？"):
            idx = raw.rfind(sep, 0, 80)
            if idx > 40:
                raw = raw[:idx + 1]
                break
        entry = {"round": round_number, "content": raw[:80]}
        agent_log = self._event_category_log.setdefault(agent_id, {})
        agent_log.setdefault(category, []).append(entry)
        if len(agent_log[category]) > 20:
            agent_log[category] = agent_log[category][-20:]

    async def _reflect_correct(
        self, agent: DeductionAgentProfile, snapshot: dict,
        round_number: int, client: Any,
    ) -> str | None:
        """D4-phase2: 二阶冷静纠错。对比冲动快照，生成修正。返回修正后准则或 None。"""
        from strategy_forge.core.llm_client import Message
        from ._utils import extract_text
        my_events = [
            e for e in self._event_history[-20:]
            if e.get("agent") == agent.entity_id or e.get("agent_name") == agent.name
        ]
        events_since = "\n".join(
            f"- [R{e.get('round','?')}] {e.get('content','')[:80]}"
            for e in my_events[-8:]
        ) or "无后续事件"
        prompt = (
            f"你是 {agent.name} 的理性纠错层。{snapshot['rounds_ago']} 轮前，你因"
            f"「{snapshot['trigger']}」产生了冲动反应：\n\n"
            f"## 当时的冲动准则\n\"{snapshot['raw_rule']}\"\n\n"
            f"## 此后发生的事\n{events_since}\n\n"
            f"## 任务\n这条冲动准则在冷静后是否需要修正？如果需要，输出修正后准则（≤20字）。"
            f"如果当时判断正确无需修正，输出\"无需修正\"。\n只输出准则或\"无需修正\"。"
        )
        try:
            resp = await client.chat(
                [Message(role="user", content=prompt)],
                system="你是冷静的纠错分析师，输出修正准则或'无需修正'。",
                temperature=0.1, max_tokens=80,
            )
            text = extract_text(resp).strip()
            if not text or "无需修正" in text or len(text) < 2:
                return None
            return text
        except Exception as e:
            logger.debug("[Simulator] D4纠错失败: %s", e)
            return None

    async def _reflect_retrospect(
        self, agent: DeductionAgentProfile, round_number: int, client: Any,
    ) -> int:
        """D6.2: 回溯悔悟。检查当前全部准则，删除过时项。返回被删除数。"""
        current = agent.system_prompt_extra
        if not current:
            return 0
        from strategy_forge.core.llm_client import Message
        from ._utils import extract_text
        history_patterns = self._get_history_patterns(agent.entity_id)
        # 最近自身事件上下文
        my_events = [
            e for e in self._event_history[-15:]
            if e.get("agent") == agent.entity_id or e.get("agent_name") == agent.name
        ]
        recent_context = "\n".join(
            f"- [R{e.get('round','?')}] {e.get('content','')[:60]}"
            for e in my_events[-5:]
        ) or "无近期事件"
        prompt = self._PROMPT_RETROSPECT.format(
            name=agent.name, current_rules=current,
            history_patterns=history_patterns or "无历史模式记录",
            recent_events=recent_context,
        )
        try:
            resp = await client.chat(
                [Message(role="user", content=prompt)],
                system="你是准则审计师。逐条判断现有准则是否过时。",
                temperature=0.1, max_tokens=200,
            )
            text = extract_text(resp).strip()
            if not text or "无需调整" in text or len(text) < 2:
                return 0
            deleted = 0
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("删除：") or line.startswith("删除:"):
                    old_rule = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                    if old_rule and old_rule in current:
                        current = current.replace("；" + old_rule, "")
                        current = current.replace(old_rule + "；", "")
                        current = current.replace(old_rule, "")
                        deleted += 1
                elif line.startswith("修正：") or line.startswith("修正:"):
                    mapping = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                    if "→" in mapping:
                        old_str, new_str = mapping.split("→", 1)
                        old_str, new_str = old_str.strip(), new_str.strip()
                        if old_str in current and new_str:
                            current = current.replace(old_str, new_str)
                            deleted += 1
            # Clean up double semicolons
            while "；；" in current:
                current = "；".join(p for p in current.split("；") if p)
            current = current.strip("；")
            agent.system_prompt_extra = current or None
            if deleted:
                self._log("simulation",
                    f"[回溯悔悟] {agent.name}: 清理了 {deleted} 条过时准则 (R{round_number})")
            return deleted
        except Exception as e:
            logger.debug("[Simulator] D6.2回溯失败: %s", e)
            return 0

    async def _assess_env_impact(self, sim_round: SimulationRound, round_number: int) -> None:
        """叙事模式环境评估：随机抽 3 个 Agent 用 LLM 评估其动作对环境的影响。"""
        if not sim_round.actions:
            return
        import random as _random
        sample = sim_round.actions[:]
        _random.shuffle(sample)
        _max = max(3, min(len(sample), len(self.agents) // 2))
        sample = sample[:_max]

        env_state = "\n".join(
            f"- {k}: {v:.0f}" for k, v in self._narrative_env.items()
        )
        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient, Message
        from ._utils import extract_json
        client = LLMClient()
        import re as _re

        total_deltas: dict[str, float] = {k: 0.0 for k in self._narrative_env}
        for action in sample:
            # 消上帝视角L8：私有/秘密事件不应影响公众可观察的环境变量
            action_events = [e for e in self._event_history[-50:]
                            if e.get("agent") == action.agent_id
                            and e.get("round") == round_number
                            and e.get("content", "")[:30] in (action.content or "")[:30]]
            if action_events and any(
                (e.get("visibility", "") or "public") in _RESTRICTED_VIS for e in action_events
            ):
                continue
            agent_name = next((a.name for a in self.agents if a.entity_id == action.agent_id), action.agent_id[:8])
            prompt = (
                f"你是环境观察者。角色「{agent_name}」执行了「{action.action_type}」：{action.content[:80]}\n\n"
                f"当前环境：\n{env_state}\n\n"
                f"评估该动作对以下 5 个环境变量的影响（每个 -10 到 +10）：\n"
                f'{{"舆论风向": 0, "抗议规模": 0, "媒体关注": 0, "国际压力": 0, "社会分裂": 0}}\n'
                f"参考示例：\n"
                f"角色「X公司」发起 price_war：将全线产品降价30% → {{\"舆论风向\": +5, \"抗议规模\": 0, \"媒体关注\": +8, \"国际压力\": 0, \"社会分裂\": 0}}\n"
                f"角色「Y政客」公开指责对手通敌并展示证据 → {{\"舆论风向\": -10, \"抗议规模\": +5, \"媒体关注\": +10, \"国际压力\": +3, \"社会分裂\": +7}}\n"
                f"只输出 JSON。"
            )
            try:
                resp = await client.chat_json(
                    [Message(role="user", content=prompt)],
                    system="你是环境观察者，评估单一动作的环境影响。只输出 JSON。",
                    schema_name="env_impact", temperature=0.2,
                    max_tokens=100,
                )
                data = extract_json(str(resp))
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k in total_deltas:
                            delta = max(-10.0, min(10.0, float(v)))
                            total_deltas[k] += delta
            except Exception as e:
                logger.debug("[Simulator] 环境评估失败: %s", e)

        # 限幅：单轮单变量总变化不超过 ±15
        for k in total_deltas:
            clamped = max(-15.0, min(15.0, total_deltas[k] / max(1, len(sample))))
            self._narrative_env[k] = max(0.0, min(100.0,
                round(self._narrative_env[k] + clamped, 1)))

    async def _agent_decide(
        self, client: Any, agent: DeductionAgentProfile, round_number: int
    ) -> SimulationAction | None:
        from strategy_forge.core.config import config
        # ── 近期事件（可见性过滤：私密事件仅参与者可见）──
        visible_history = [e for e in self._event_history
                           if _is_event_visible_to(agent.entity_id, agent.name, e)]
        recent = visible_history[-max(1, config.deduction_sim_recent_events):]
        recent_text = "\n".join(
            f"- [{e.get('round', '?')}] {e.get('agent_name', e.get('agent', '?'))}: "
            f"{e.get('content', '')[:80]}"
            for e in recent
        ) or "无近期事件"
        recent_text = ("（以下是各方公开可见的行为记录，标注了行为主体——"
                       "注意区分他人行为与你自己的行动，不要把他人做过的事当成自己做过）\n"
                       + recent_text) if recent else recent_text

        # ── 三幕节拍指令 + 世界时钟（仅叙事模式）──
        stage_text = ""
        if not self._quantified and self.total_rounds > 0:
            progress = round_number / max(1, self.total_rounds)
            if self.total_rounds == 1:
                stage_hint = ("唯一轮次：请做出关键决策以推动局势发展")
            elif progress <= 0.3:
                stage_hint = ("当前为铺垫幕：自由布局，建立关系、收集信息、埋设伏笔均可，"
                              "但每轮行动都应产生新信息或新关系，不要空转。")
            elif progress <= 0.8:
                stage_hint = ("当前为对抗幕：冲突必须升级。你的行动应针对既有对手或矛盾"
                              "采取实质性动作（施压、反制、结盟、揭露），"
                              "禁止停留在观察和重复性会面。")
            else:
                stage_hint = ("当前为收束幕：兑现你此前埋下的线索和承诺，迫使关键矛盾摊牌，"
                              "禁止开启全新的支线。你的行动应直接影响最终格局。")
            days = round_number * 5
            stage_text = (f"## 推演节拍\n{stage_hint}\n"
                          f"推演内时间：约第{days}天（1轮≈5天）。保持时间逻辑一致——"
                          f"一次性事件（葬礼、发布会、签约）不应跨多轮持续存在。\n\n")

        # ── 叙事环境上下文（注入到决策 prompt 中）──
        from .narrative_actions import get_narrative_actions
        env_lines = [f"- {k}: {v:.0f}" for k, v in self._narrative_env.items()]
        env_text = "当前社会环境：\n" + "\n".join(env_lines) if not self._quantified else ""
        bt = getattr(agent, "base_type", "Agent") or "Agent"
        action_list = get_narrative_actions(agent.entity_type, base_type=bt) if not self._quantified else []
        # 附加 base_type 能力提示
        if bt != "Agent" and not self._quantified:
            action_catalog_text = f"\n## 你的角色类型\n你是 {bt} 类型实体，决策范围受限于你的身份和资源。\n"
        else:
            action_catalog_text = ""
        action_catalog_text += ("\n## 你可用的动作（按你的身份）\n" + "\n".join(f"- {a}" for a in action_list)
                                if action_list and not self._quantified else "")

        # ── 共享双路 LanceDB 召回 ──
        static_text, dynamic_text = await self._shared_dual_recall(agent)

        # ── Strategic Reasoning (primary path) ──
        context_text = recent_text
        # 注入角色私人记忆（人际承诺/债务/背叛）
        journal = getattr(self, "_character_journal", {}).get(agent.entity_id, [])
        if journal:
            context_text = "## 你的私人记忆\n" + "\n".join(f"- {j}" for j in journal) + "\n\n" + context_text
        if stage_text:
            context_text = stage_text + context_text
        if env_text:
            context_text = env_text + "\n\n" + context_text
        if action_catalog_text:
            context_text = context_text + action_catalog_text
        world = {"recent_events": context_text, "static_knowledge": static_text,
                  "dynamic_memory": dynamic_text,
                  "relationship_context": self._rel_context.get(agent.entity_id, {}).get("summary", "")}

        # ── D6.3: 对手镜像博弈视角 ──
        if "relationship_context" in world:
            rel_ctx = self._rel_context.get(agent.entity_id, {})
            opponents = rel_ctx.get("opponents", [])
            allies = rel_ctx.get("allies", [])
            key_targets = opponents[:3] + allies[:1]  # 前3对手 + 1盟友
            if key_targets:
                mirror_parts = ["## 对手视角预判（站在对方立场思考）"]
                for target_id in key_targets:
                    tgt = next((a for a in self.agents if a.entity_id == target_id), None)
                    if not tgt:
                        continue
                    # 仅使用当前 agent 可见的目标事件（消上帝视角L1+L5）
                    tgt_events = [e for e in self._event_history[-10:]
                                  if e.get("agent") == target_id
                                  and _is_event_visible_to(agent.entity_id, agent.name, e)]
                    tgt_recent = "; ".join(e.get("content", "")[:50] for e in tgt_events[-3:]) or "无记录"
                    mirror_parts.append(
                        f"### {tgt.name}\n"
                        f"近三轮可见行动: {tgt_recent}\n"
                        f"→ 站在 {tgt.name} 的角度预判：如果 TA 推测到你的意图，TA 会如何反制？"
                    )
                world["recent_events"] = context_text + "\n\n" + "\n".join(mirror_parts)
        decision = None
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
                persona=(f"{agent.persona}\n【行为准则·由推演经历塑造】{agent.system_prompt_extra}"
                         if agent.system_prompt_extra else agent.persona),
                background=agent.background,
                goals=", ".join(agent.goals) if agent.goals else "参与互动",
                round_number=round_number, recent_events=context_text,
                static_knowledge=static_text, dynamic_memory=dynamic_text,
            ))]
            try:
                if self._chat_fn is not None:
                    response = await asyncio.to_thread(self._chat_fn, messages, system, 0.7)
                    content = response
                else:
                    response = await client.chat_json(messages, system=system, schema_name="action_fallback", temperature=0.6, max_tokens=1500)
                    content = extract_text(response)
                action_data = _parse_action_json(content)
            except Exception as e2:
                logger.warning("[Deduction] Agent %s decision failed: %s", agent.name, e2)
                return None

        from datetime import datetime
        _vis = (action_data.get("visibility") or "").strip() or "public"
        # 存储落选候选方案（供反思日志和报告复盘使用）
        meta = {"visibility": _vis}
        if decision and decision.get("candidates"):
            all_cands = decision["candidates"]
            rejected = []
            for c in all_cands:
                if c.get("content") != action_data.get("content"):
                    rejected.append({
                        "action": c.get("action", ""),
                        "target": c.get("target", ""),
                        "content": c.get("content", "")[:120],
                        "rationale": c.get("rationale", "")[:80],
                        "risk_level": c.get("risk_level", ""),
                        "score": c.get("_score", 0),
                        "blind_spots": c.get("blind_spots", ""),
                    })
            if rejected:
                meta["_rejected_candidates"] = rejected
        return SimulationAction(
            agent_id=agent.entity_id,
            action_type=action_data.get("action", "observe"),
            target_id=action_data.get("target", ""),
            content=action_data.get("content", f"{agent.name}观察着周围环境"),
            timestamp=datetime.now().isoformat(),
            metadata=meta,
        )

    # ── 量化模式：决策 → 快照交互解算 → 批量应用 → 阈值淘汰 → 可选解读 ──

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
                                   client: Any, mode: str = None) -> str | None:
        """人格动态化：根据近期经历微调 agent 的行为准则（量化模式）。
        mode: None=默认, "strategic"=D2长期理性, "internal"=D6.1内源自省。

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

        # D5 历史模式
        history_patterns = self._get_history_patterns(agent.entity_id)
        patterns_text = history_patterns or "无重复模式记录"

        # D2 趋势增强：最近指标变化的方向是加速还是减缓
        trend_parts = []
        if len(recent_history) >= 6:
            early = sum(float(h.get("delta", 0)) for h in recent_history[-10:-5])
            late = sum(float(h.get("delta", 0)) for h in recent_history[-5:])
            for m in set(h.get("metric","") for h in recent_history if h.get("metric")):
                e = sum(float(h.get("delta", 0)) for h in recent_history[-10:-5] if h.get("metric") == m)
                l = sum(float(h.get("delta", 0)) for h in recent_history[-5:] if h.get("metric") == m)
                if abs(e) < 0.5 and abs(l) < 0.5:
                    continue
                label = _METRIC_NAME.get(m, m)
                if e < -1 and l < -1:
                    trend_parts.append(f"{label}持续恶化")
                elif e > 1 and l > 1:
                    trend_parts.append(f"{label}持续改善")
                elif e < -1 and l > 0:
                    trend_parts.append(f"{label}触底反弹")
                elif e > 1 and l < 0:
                    trend_parts.append(f"{label}由升转跌")
        trend_line = "；".join(trend_parts) if trend_parts else "无显著趋势"

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
            f"## 趋势判断\n{trend_line}\n\n"
            f"## 历史遭遇模式\n{patterns_text}\n\n"
            f"## 风险信号\n{'; '.join(status_summary) if status_summary else '无告急指标'}\n\n"
            f"## 近期行动复盘\n{causal_short if causal_short else '无'}\n\n"
            f"## 任务\n"
            f"根据以上经历，判断是否需要添加一条新的行为准则（或修正旧准则），"
            f"使你的行为更符合当前的处境。\n"
            f"- 输出格式：一行简短中文准则（20字以内），直接陈述。\n"
            f"- 如果当前人格已足够应对，输出\"无需调整\"。\n"
            f"- 仅添加/修正，不删除原有准则。\n"
            f"- 示例：\"资源持续消耗时应优先补充而非扩张\" \"连续成功后应警惕过度自信\" \"核心关系需定期维护\"\n"
            f"- 何时输出\"无需调整\"：指标稳定、现有准则已覆盖所有风险信号、近期无方向性变化\n"
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
            old_extra = agent.system_prompt_extra
            # 去重：新准则与已有准则相似度 > 0.7 则跳过
            if old_extra:
                existing_rules = old_extra.split("；")
                for old in existing_rules:
                    if old and text:
                        common = len(set(old) & set(text))
                        denom = max(len(set(old)), len(set(text)), 1)
                        if common / denom > 0.7:
                            return None
            if old_extra and text not in old_extra:
                parts = old_extra.split("；")
                max_rules = self._max_persona_rules()
                if len(parts) >= max_rules:
                    parts = parts[1:]
                agent.system_prompt_extra = "；".join(parts + [text])
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
        """将本轮 _event_history 中新事件按信任度分发至各 agent 知识队列。

        缺陷3 修复：事件带唯一 _eid，已分发的事件记录在 _dispatched_eids 中，
        避免通道②二次分发时重复注入知识队列。
        """
        from uuid import uuid4
        alive_ids = [a.entity_id for a in self.agents if a.entity_id in self._states]
        name_to_id = self._name_to_id
        if not hasattr(self, "_dispatched_eids"):
            self._dispatched_eids: set = set()
        dispatched = self._dispatched_eids
        for evt in self._event_history:
            if evt.get("round") != round_number:
                continue
            actor_id = evt.get("agent", "")
            actor_name = evt.get("agent_name", "")
            content = evt.get("content", "")
            if not actor_id or not content:
                continue
            _eid = evt.get("_eid")
            if _eid is None:
                _eid = evt["_eid"] = f"eid-{uuid4().hex[:8]}"
            if _eid in dispatched:
                continue
            for a_id in alive_ids:
                if a_id == actor_id:
                    continue
                # 可见性过滤：私密事件仅分发给参与者/发起者，避免信息不对称泄漏
                a_name = name_to_id and next(
                    (k for k, v in name_to_id.items() if v == a_id), actor_name)
                if not _is_event_visible_to(a_id, a_name, evt):
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
                # 队列上限保护
                queue = self._agent_knowledge[a_id]
                if len(queue) > 500:
                    self._agent_knowledge[a_id] = queue[-500:]
            # 事件分发完成，标记去重（缺陷3 修复）
            dispatched.add(_eid)
        # 缺陷6：有界化去重集合——只保留最近 ~400 条，防止长期推演内存无界膨胀
        if len(dispatched) > 400:
            for _old in list(dispatched)[: len(dispatched) - 400]:
                dispatched.discard(_old)

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
            # 缺陷8：全灭后无实体可结算，清空待处理注入事件，避免滞留内存
            if getattr(self, "_injected_events_store", None):
                self._injected_events_store["pending"] = []
            return sim_round

        # 融合架构·通道①：本轮高优先级事件冲击相关实体指标（在决策前结算）
        _n_impacts = self._apply_event_impacts(round_number)
        if _n_impacts:
            self._log("simulation", f"事件冲击(融合): {_n_impacts} 个事件影响实体指标")

        ordered = list(alive_agents)
        self._rng.shuffle(ordered)

        # Pre-build O(1) entity-id→index map for spatial lookups
        alive_id_to_idx = {eid: i for i, eid in enumerate(alive_ids)} if alive_ids else {}

        def others_ctx(self_id: str) -> str:
            # B1: 只渲染 Top-K 最相关他方(盟友/对手>最近>最危急)，其余合并为全局摘要，
            # 把每 agent  prompt 的他方块从 O(N) 降到 O(K)，消除 O(N^2) 与逐轮膨胀。
            # 消上帝视角F1: 详细指标仅限"已知邻居"（盟友/对手 + 共享事件），其余仅汇总。
            from strategy_forge.core.config import config as _c
            topk = max(1, int(getattr(_c, "deduction_sim_others_topk", 10)))
            metrics_list = re_engine.metrics()
            idx_self = alive_id_to_idx.get(self_id)
            sp = self._spatial_state
            rel = getattr(self, "_rel_context", {}).get(self_id, {}) or {}
            important = set(rel.get("allies", []) or []) | set(rel.get("opponents", []) or [])

            # 构建已知邻居集合：关系上下文 + 共享事件
            known_set: set[str] = set(important)
            for evt in self._event_history[-30:]:
                if evt.get("agent") == self_id:
                    tgt = evt.get("target", "")
                    if tgt:
                        known_set.add(tgt)
                parts_s = evt.get("participants", "") or ""
                if self_id in parts_s:
                    for p in parts_s.split(","):
                        p = p.strip()
                        if p:
                            known_set.add(p)

            def _dist(a) -> float | None:
                if sp is not None and idx_self is not None:
                    io = alive_id_to_idx.get(a.entity_id)
                    if io is not None and idx_self < len(sp.positions) and io < len(sp.positions):
                        return float(np.linalg.norm(sp.positions[idx_self] - sp.positions[io]))
                return None

            others = [a for a in alive_agents if a.entity_id != self_id]
            if not others:
                return "（无其他参与方）"

            def _detail(a) -> str:
                st = states[a.entity_id]
                if a.entity_id not in known_set:
                    return f"{st.name}: 信息不足（未直接接触）"
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

            def _salience(a):
                st = states[a.entity_id]
                rel_pri = 0 if a.entity_id in known_set else 2
                d = _dist(a)
                mtot = sum(st.metrics.values()) if st.metrics else 0.0
                return (rel_pri, d if d is not None else 1e9, mtot)

            ranked = sorted(others, key=_salience)
            shown, rest = ranked[:topk], ranked[topk:]

            lines = [_detail(a) for a in shown]
            if rest:
                lines.append(f"其余 {len(rest)} 方（未直接接触，态势未知）")
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

        # Clear round-level caches at start of round
        if self._preprocessor is not None and hasattr(self._preprocessor, "clear_round_cache"):
            self._preprocessor.clear_round_cache()

        async def _recall(agent: DeductionAgentProfile) -> tuple[str, str]:
            _rk = max(1, _cfg.deduction_sim_recall_topk)
            _rc = max(200, _cfg.deduction_sim_recall_chars)
            return await self._shared_dual_recall(agent, _rk, _rc)

        # P1#7: _other_ctxs 跨轮缓存——仅当各实体 metrics/history 快照变化时重建，
        # 避免数千实体下每轮全量 O(N) 字符串构建。
        sig = hash((_state_snapshot_sig(states, alive_ids),))
        if getattr(self, "_other_ctxs_cache", None) is None:
            self._other_ctxs_cache = {}
        if self._other_ctxs_sig == sig and self._other_ctxs_cache:
            _other_ctxs = self._other_ctxs_cache
        else:
            _other_ctxs = {a.entity_id: others_ctx(a.entity_id) for a in alive_agents}
            self._other_ctxs_cache = _other_ctxs
            self._other_ctxs_sig = sig
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
            # 补充 agent 自身相关的事件 + 全部系统事件（融合架构：系统事件全局可见）。
            # 噪音防御（措施1）：系统事件每轮每 agent 限幅，防止多实体同时触发时淹没自身事件。
            # 同时按 (round, event_type) 去重，避免同一系统事件重复展示。
            sys_seen = 0
            _sys_limit = 2  # 每轮每个 agent 最多注入 2 条系统事件
            own_events = [e for e in self._event_history[-8:]
                          if e.get("is_system_event")
                          or (_is_event_visible_to(a.entity_id, a.name, e) and
                              (e.get("agent") == a.entity_id
                               or a.name in e.get("content", "")))]
            _shown_sys = set()
            for e in own_events:
                if e.get("is_system_event"):
                    _sig = (e.get("round"), e.get("event_type"))
                    if _sig in _shown_sys:
                        continue
                    if sys_seen >= _sys_limit:
                        continue
                    _shown_sys.add(_sig)
                    sys_seen += 1
                text = e.get("content", "")[:80]
                prefix = "【系统事件】" if e.get("is_system_event") else ""
                items.append(f"• [R{e.get('round','?')}] {prefix}{text}")
            _recent_ctxs[a.entity_id] = "\n".join(items[-8:]) or "（无近期事件）"

        async def decide(agent: DeductionAgentProfile) -> dict[str, Any] | None:
            from strategy_forge.core.config import config as _cr
            from strategy_forge.core.llm_client import LLMConnectionError
            fails = 0
            max_passes = max(0, _cr.deduction_llm_retry_passes)
            last_err = None
            while True:
                try:
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
        fsm_state_map = getattr(self, "_last_fsm_states_map", None) or {}
        fsm_action_map = getattr(self, "_last_fsm_actions_map", None) or {}
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
            # Check if FSM should drive this agent (entity_id based lookup, not index)
            state = fsm_state_map.get(agent.entity_id) if fsm_state_map else None
            if state is not None and state not in fsm_command:
                # FSM deterministic action — skip LLM
                act = fsm_action_map.get(agent.entity_id) if fsm_action_map else None
                if act is not None:
                    act = dict(act)
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
                    _INTEL_EXPOSED_KEY in d or _INTEL_EXPOSED_KEY in k.lower()
                    for k, d in inter.get("deltas", {}).items()
                    if isinstance(d, dict)
                ):
                    # Also check via keys
                    pass
                if inter:
                    for k in inter.get("deltas", {}):
                        if _INTEL_EXPOSED_KEY in str(k).lower():
                            bonus = self._intel_bonuses.setdefault(actor, {}).get(tgt, 0.0)
                            self._intel_bonuses.setdefault(actor, {})[tgt] = min(5.0, bonus + 2.0)
                            self._log("simulation", f"[谍报] {actor} 对 {tgt} 获得信息优势 (+2.0, 总和={bonus+2.0:.1f})")
                            break

        # ── Algorithm module chain (ODE + Physics) ──
        # 模块顺序已由 build_pipeline() 内置排序（IS_FINALIZER 自动置末），
        # 此处直接遍历，未来可迁移至 PipelineEngine.run() 以启用条件执行与信号验证。
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
                raw_states = list(ctx.metadata["fsm.agent_states"])
                raw_actions = list(ctx.metadata.get("fsm.agent_actions", []))
                # 按 entity_id 建映射表，避免下一轮 agent 淘汰后索引错位
                self._last_fsm_states_map = {
                    entity_ids[i]: raw_states[i] for i in range(len(raw_states))
                } if len(raw_states) == len(entity_ids) else {}
                self._last_fsm_actions_map = {}
                if raw_actions and len(raw_actions) == len(entity_ids):
                    for i in range(len(raw_actions)):
                        if raw_actions[i] is not None:
                            self._last_fsm_actions_map[entity_ids[i]] = dict(raw_actions[i])
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
            _vis = (dec.get("visibility") or "").strip() or "public"
            # 关键词兜底：若 LLM 未输出 visibility，检查行动类型/理由是否暗示秘密行动
            if _vis == "public":
                from .narrative_actions import is_secret_action
                _rationale = dec.get("rationale", "")
                _act_type = dec.get("action_type", "")
                if is_secret_action(_act_type, _rationale):
                    _vis = "private"
            meta["visibility"] = _vis
            # 参与者：发起方 + 目标方，供 _is_event_visible_to 使用
            _participants = "|".join(filter(None, [nm, actor, str(dec.get("target", ""))]))
            sim_round.actions.append(SimulationAction(
                agent_id=actor, action_type=dec["action_type"],
                target_id=dec.get("target", ""), content=content,
                timestamp=datetime.now().isoformat(),
                metadata=meta,
            ))
            # D5: 量化模式写事件类别日志
            self._log_category_event(actor, dec["action_type"], round_number,
                                     dec.get("rationale", ""))
            # W4: 量化轮事件写入 LanceDB 动态表(仅主推演 persist_events=True；优化器隔离不写)
            if self._persist_events and self._preprocessor is not None:
                try:
                    self._preprocessor.add_event_memory(
                        content=content, agent_id=actor,
                        round_number=round_number,
                        event_type=dec["action_type"], priority=0.5,
                        visibility=_vis, participants=_participants)
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
                    for _it in _inters:
                        for _metric, _amount in _it["deltas"].items():
                            self.graph.add_caused(_eid, _it["target"], _metric, float(_amount))
                except Exception as e:
                    logger.debug("[Simulator] 量化因果写入 Kuzu 失败: %s", e)
            evt_suffix = (f"［{alloc_txt}］" if alloc_txt else "") + (f"（{delta_txt}）" if delta_txt else "")
            self._append_event({
                "agent": actor, "agent_name": nm, "action": dec["action_type"],
                "content": content + evt_suffix,
                "round": round_number,
                "event_type": dec["action_type"],
                "target_id": _primary_tid if _inters else "",
                "is_system_event": False,
                "visibility": _vis,
                "participants": _participants,
            })

        # ── 信息传播：将本轮事件按信任度分发至各 agent 知识队列 ──
        self._dispatch_events(round_number)

        # 融合架构·通道②：指标越界自动生成系统事件（在本轮结算后，进入事件流）
        _n_triggers = self._trigger_events_from_metrics(round_number)
        if _n_triggers:
            self._log("simulation",
                      f"指标触发事件(融合): {_n_triggers} 个系统事件生成")
            self._dispatch_events(round_number)

        # ── 共享反思闸门（量化模式：完整 P0-P2 六维反思）──
        if not hasattr(self, "_reflection_baselines"):
            self._reflection_baselines: dict[str, dict[str, float]] = {}
            self._last_reflection_round_n: dict[str, int] = {}
            import random as _random
            for agent in self.agents:
                self._last_reflection_round_n[agent.entity_id] = _random.randint(0, 2)

        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
        from strategy_forge.core.providers import registry as _reg
        _rc = LLMClient()
        _max_conc = max(1, _reg.max_concurrent)
        _sem = asyncio.Semaphore(_max_conc)

        # D4-phase2: 并行纠错
        async def _q_correct_one(eid: str, remaining: list, agent, snapshots) -> None:
            async with _sem:
                for snap in snapshots:
                    if round_number - snap["round"] >= 1:
                        corrected = await self._reflect_correct(agent, snap, round_number, _rc)
                        if corrected and agent.system_prompt_extra:
                            old_rules = agent.system_prompt_extra.split("；")
                            if snap["raw_rule"] in old_rules:
                                new_rules = [r for r in old_rules if r != snap["raw_rule"]]
                                new_rules.append(corrected)
                                agent.system_prompt_extra = "；".join(new_rules)
                        self._log("simulation",
                            f"[D4纠错] {agent.name}: {snap['trigger']} (R{round_number})")
                    else:
                        remaining.append(snap)
                if remaining:
                    self._pending_corrections[eid] = remaining
                else:
                    self._pending_corrections.pop(eid, None)

        d4_tasks = []
        for eid, snapshots in list(self._pending_corrections.items()):
            agent = next((a for a in self.agents if a.entity_id == eid), None)
            if agent:
                remaining: list = []
                d4_tasks.append(_q_correct_one(eid, remaining, agent, snapshots))
        if d4_tasks:
            await asyncio.gather(*d4_tasks)

        # 主要反思循环：并行 + D2/D6.1 路由
        async def _q_reflect_one(agent) -> None:
            async with _sem:
                eid = agent.entity_id
                state = states.get(eid)
                reason = self._should_reflect(eid, round_number, state, re_engine)
                if reason is not None:
                    result = await self._reflect_and_adapt(agent, round_number, _rc,
                                                            mode="strategic" if ("环境" in reason or "累计" in reason or "模式" in reason)
                                                            else "internal" if "内源" in reason else None)
                    tag = (
                        "[D2数值复盘]" if "环境" in reason or "累计" in reason
                        else "[模式预警]" if "模式" in reason
                        else "[内源自省]" if "内源" in reason
                        else "[反思]")
                    if result and "无需调整" not in result:
                        self._reflection_baselines[eid] = dict(self._narrative_env)
                        self._last_reflection_round_n[eid] = round_number
                    self._log("simulation",
                        f"{tag} {agent.name}: {reason} → "
                        f"{'新增准则' if (result and '无需调整' not in result) else '无需调整'} (R{round_number})")

        reflect_tasks = [_q_reflect_one(a) for a in self.agents]
        await asyncio.gather(*reflect_tasks)

        # D6.2: 并行回溯悔悟
        if round_number > 0 and round_number % 5 == 0:
            async def _q_retrospect_one(agent) -> None:
                async with _sem:
                    await self._reflect_retrospect(agent, round_number, _rc)
            retro_tasks = [_q_retrospect_one(a) for a in self.agents]
            await asyncio.gather(*retro_tasks)

        # 保存本轮关系网络快照供下轮对比
        self._prev_rel_map = dict(getattr(self, "_rel_context", {}))

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
                    f"将第 {round_number} 轮量化推演结果改写为一段生动简洁的态势叙事（200 字以内）。\n\n"
            "## 本轮各方行动与数值变化\n" + "\n".join(lines) + "\n\n"
            "## 示例\n"
            "输入：A国 发动进攻(强度0.8) 目标:B国，数值变化: A国strength-8,B国supply-15\n"
            "输出：A国以雷霆之势向B国边境发起猛攻，虽然自身消耗不小，但B国的后勤补给线遭受重创，前线的物资储备已降至警戒水平。\n\n"
            "只输出叙事段落，不要解释或列表。叙事只能基于上方列出的行动和数值变化，不得添加未发生的情节。"
        )
        resp = await client.chat([Message(role="user", content=prompt)],
                                 system="你是推演解说员，把数值变化翻译成简洁叙事。", temperature=0.3)
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
    # Recent events（消上帝视角R2：仅公开事件）
    recent = []
    for e in event_history[-3:]:
        if (e.get("visibility", "") or "public") in _RESTRICTED_VIS:
            continue
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

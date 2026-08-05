"""规则引擎：将 LLM 决策意图映射为量化指标变化，并做存亡/胜负判定。

核心职责：
- 加载规则包（内置领域模板或用户上传的自定义 JSON）
- detect_domain：LLM 领域识别 + 置信度阈值回退叙事
- init_state：按规则包 initial_metrics 创建 EntityState
- resolve_round：基于"轮初快照"统一计算本轮全部 delta（self + target，多方累加），
  由调用方批量应用，避免同轮先手偏差
- is_alive / judge：阈值存亡 + 结构化胜利条件的客观判胜负（解决评估者悖论）

决策契约：
- 单动作（默认，向后兼容 v2.0）：action_type + intensity + target。
- 多动作分配（可选）：budget + actions:[{action_type, weight, target}]，
  按 budget × (weight / Σweight) 把总投入分配给各动作，各动作可带各自 target（多目标）。
  budget=1 时总投入与单动作 intensity=1 等价（量级中性）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from strategy_forge.core.rule_templates import get_template, list_domains

from .models import EntityState

logger = logging.getLogger(__name__)

_CONDITION_RE = re.compile(r'\s*(\w+)\s*(<=|>=|!=|==|<|>)\s*([\d.]+)\s*')
_NOT_RE = re.compile(r'^\s*not\s+')


def _normalize_action(a: Any) -> str:
    """将 action 统一为字符串：dict→取 name 字段，str→原样返回。"""
    if isinstance(a, str):
        return a
    if isinstance(a, dict):
        return str(a.get("name", "") or a.get("description", "") or json.dumps(a, ensure_ascii=False))
    return str(a)


class RuleEngine:
    def __init__(self, rule_pack: dict[str, Any]):
        self.pack = self._with_defaults(rule_pack)
        self.domain = self.pack.get("domain", "generic")
        # Pre-parse static condition strings → structured form for fast eval
        self._parsed_conditions: dict[str, list[list[tuple[str, str, float]]]] = {}
        for cfg in self.pack.get("auto_effects", {}).values():
            c = cfg.get("condition", "")
            if c:
                self._parsed_conditions[c] = self._parse_condition(c)
        for cfg in self.pack.get("conditional_effects", {}).values():
            c = cfg.get("condition", "")
            if c:
                self._parsed_conditions[c] = self._parse_condition(c)

    @staticmethod
    def _parse_condition(condition: str) -> list[list[tuple[str, str, float]]]:
        """Pre-parse condition string into structured form.

        'a<5 and b>2 or c==1' → [[(a,<,5),(b,>,2)], [(c,==,1)]]
        Each atom is (metric, op, value). Negated atoms prefixed with 'not '
        are parsed as (metric, op, value) with op containing '!' marker
        for later evaluation.
        """
        result: list[list[tuple[str, str, float]]] = []
        for or_part in condition.split(" or "):
            atoms: list[tuple[str, str, float]] = []
            for and_part in or_part.strip().split(" and "):
                stripped = and_part.strip()
                negated = bool(_NOT_RE.match(stripped))
                if negated:
                    stripped = _NOT_RE.sub('', stripped).strip()
                m = _CONDITION_RE.match(stripped)
                if m:
                    metric, op, val = m.group(1), m.group(2), float(m.group(3))
                    if negated:
                        op = '!' + op  # marker for eval
                    atoms.append((metric, op, val))
                else:
                    logger.debug("[RuleEngine] condition atom failed to parse: '%s'", stripped)
            if atoms:
                result.append(atoms)
        return result

    # ── 构造 ──
    @classmethod
    def from_domain(cls, domain: str) -> "RuleEngine":
        tpl = get_template(domain)
        if tpl is None:
            raise ValueError(f"未知领域规则包: {domain}")
        return cls(tpl)

    @classmethod
    def from_custom(cls, data: dict[str, Any]) -> "RuleEngine":
        return cls(data)

    @staticmethod
    def _with_defaults(pack: dict[str, Any]) -> dict[str, Any]:
        p = dict(pack)
        p.setdefault("metrics", list(p.get("initial_metrics", {}).keys()))
        p.setdefault("initial_metrics", {m: 50.0 for m in p["metrics"]})
        p.setdefault("metric_ranges", {})
        p.setdefault("thresholds", {})
        p.setdefault("actions", ["observe"])
        p.setdefault("self_effects", {})
        p.setdefault("target_effects", {})
        p.setdefault("conditional_effects", {})
        p.setdefault("delay_effects", {})
        p.setdefault("auto_effects", {})
        # 融合架构：事件→数值冲击 与 数值→事件触发（规则包为空则通道不激活，保持兼容）
        p.setdefault("event_impact", {})
        p.setdefault("event_triggers", [])
        # 实体能力分级：按 SemanticMediator 基础类型限制指标/动作权限
        # Agent=全权限，Subordinate=无军力指标，其余=仅观察
        p.setdefault("entity_capabilities", {
            "Agent": {"metrics": "all", "actions": "all"},
            "Subordinate": {"metrics": "all",
                "exclude_metrics": ["strength", "fatigue", "leadership", "supply",
                                    "morale", "energy_resource", "grain_resource", "chip_stock"],
                "exclude_actions": ["attack", "defend", "siege", "maneuver", "military_offensive",
                                    "defensive_buildup", "electronic_warfare"]},
            "Geography": {"metrics": [], "actions": ["observe"]},
            "Concept": {"metrics": [], "actions": ["observe"]},
            "Resource": {"metrics": [], "actions": ["observe"]},
            "Contract": {"metrics": [], "actions": ["observe"]},
            "Event": {"metrics": [], "actions": ["observe"]},
        })
        return p

    # ── 融合架构：事件/数值双向通道辅助 ──

    def event_impact_map(self) -> dict:
        """事件类型 → 指标冲击映射（通道①）。"""
        return dict(self.pack.get("event_impact", {}) or {})

    def event_triggers(self) -> list:
        """数值越界 → 事件触发规则（通道②）。"""
        return list(self.pack.get("event_triggers", []) or [])

    @staticmethod
    def eval_metric_op(op: str, value: float, threshold: float) -> bool:
        """单指标阈值比较，供 event_triggers 求值。"""
        if op in ("<", "<=", ">", ">=", "==", "!="):
            return {
                "<": value < threshold,
                "<=": value <= threshold,
                ">": value > threshold,
                ">=": value >= threshold,
                "==": value == threshold,
                "!=": value != threshold,
            }[op]
        return False

    def check_event_triggers(self, state: Any, fired: set) -> list[dict]:
        """对单个实体评估 event_triggers，返回未触发过且条件成立的事件配置列表。

        fired: 已触发的 (实体id, 事件名) 集合，用于 once 去重。
        """
        out = []
        for trig in self.event_triggers():
            if not isinstance(trig, dict):
                continue
            m = trig.get("metric", "")
            op = trig.get("op", ">=")
            val = float(trig.get("value", 0))
            name = trig.get("event", "") or str(trig)
            if not m or not name:
                continue
            if trig.get("once") and (state.id, name) in fired:
                continue
            mv = state.get_metric(m)
            if self.eval_metric_op(op, mv, val):
                out.append(trig)
        return out

    # ── 访问器 ──
    def metrics(self) -> list[str]:
        return list(self.pack["metrics"])

    def thresholds(self) -> dict[str, float]:
        return dict(self.pack["thresholds"])

    def ranges(self) -> dict[str, Any]:
        return dict(self.pack.get("metric_ranges", {}))

    def actions(self) -> list[str]:
        return [_normalize_action(a) for a in self.pack["actions"]]

    def action_catalog(self, base_type: str = "Agent") -> str:
        """供决策 prompt 使用的可选动作说明。按 base_type 过滤。
        
        Agent：全量动作
        Subordinate：排除军事/战略动作
        其余类型：仅 observe
        """
        _TARGET_HINTS: dict[str, str] = {
            "partner": "提示：partner 应选择同行业或已知供应链/战略投资关系的实体，不应跨行业随机结盟",
            "diplomacy": "提示：diplomacy 应选择利益相关的可对话方（对手、盟国、冲突方），而非无关第三方",
        }
        allowed = self.get_allowed_actions(base_type)
        lines = []
        for a in self.pack["actions"]:
            a = _normalize_action(a)
            if a not in allowed:
                continue
            eff = self.pack["self_effects"].get(a, {})
            desc = ", ".join(f"{k}{v:+.0f}" for k, v in eff.items()) or "无直接消耗"
            line = f"- {a}（自身效应: {desc}）"
            if a in _TARGET_HINTS:
                line += f"\n  {_TARGET_HINTS[a]}"
            lines.append(line)
        if not lines:
            lines.append("- observe（无动作权限，仅观察）")
        return "\n".join(lines)

    # ── 状态初始化 ──
    def get_capability(self, base_type: str) -> dict:
        """返回指定 base_type 的实体能力配置（指标+动作限制）。"""
        caps = self.pack.get("entity_capabilities", {})
        return caps.get(base_type, caps.get("Agent", {"metrics": "all", "actions": "all"}))

    def get_allowed_actions(self, base_type: str) -> list[str]:
        """返回指定 base_type 可用的动作列表（过滤掉禁止动作）。
        
        - "actions": "all" → 全动作减 exclude_actions
        - "actions": ["observe"] → 仅这些动作
        """
        cap = self.get_capability(base_type)
        allowed = cap.get("actions", "all")
        if isinstance(allowed, list):
            all_actions = self.actions()
            return [a for a in all_actions if a in allowed]
        all_actions = self.actions()
        exclude = set(cap.get("exclude_actions", []))
        return [a for a in all_actions if a not in exclude]

    def init_state(self, entity_id: str, name: str, base_type: str = "Agent") -> EntityState:
        """创建实体初始状态。按 base_type 的能力配置过滤指标。"""
        cap = self.get_capability(base_type)
        if cap.get("metrics") == "all":
            exclude = set(cap.get("exclude_metrics", []))
            metrics = {k: float(v) for k, v in self.pack["initial_metrics"].items()
                       if k not in exclude}
        elif isinstance(cap.get("metrics"), list):
            metrics = {k: float(self.pack["initial_metrics"].get(k, 50.0))
                       for k in cap["metrics"] if k in self.pack["metrics"]}
        else:
            metrics = {}
        return EntityState(id=entity_id, name=name, domain=self.domain, metrics=metrics)

    # ── 单决策 → 增量 ──
    @staticmethod
    def _eval_cond(condition: str, state: Any) -> bool:
        """简单条件表达式求值器（使用预解析结构）。"""
        if not condition or not isinstance(condition, str):
            return True
        for part in condition.split(" or "):
            subs = part.split(" and ")
            if all(RuleEngine._eval_atom(a, state) for a in subs):
                return True
        return False

    def _eval_cond_cached(self, parsed: list[list[tuple[str, str, float]]], state: Any) -> bool:
        """Evaluate a pre-parsed condition against entity state — no string ops.

        Supports 'not' negated atoms (op prefixed with '!').
        Logs unknown metrics at debug level to help catch config errors.
        """
        known_metrics = set(self.pack.get("metrics", []))
        for and_group in parsed:
            ok = True
            for metric, op, val in and_group:
                negated = op.startswith('!')
                real_op = op[1:] if negated else op
                mv = state.get_metric(metric)
                # Debug: flag condition references to non-standard metrics
                if metric not in known_metrics and metric not in ('_intel_exposed',):
                    logger.debug("[RuleEngine] condition references unknown metric '%s' (resolves to 0.0)", metric)
                # Evaluate comparison
                cmp = (
                    (real_op == "<" and mv < val) or
                    (real_op == ">" and mv > val) or
                    (real_op == "<=" and mv <= val) or
                    (real_op == ">=" and mv >= val) or
                    (real_op == "==" and mv == val) or
                    (real_op == "!=" and mv != val)
                )
                if negated:
                    cmp = not cmp
                if not cmp:
                    ok = False
                    break
            if ok:
                return True
        return False

    @staticmethod
    def _eval_atom(atom: str, state: Any) -> bool:
        m = _CONDITION_RE.match(atom.strip())
        if not m:
            return False
        metric, op, val = m.group(1), m.group(2), float(m.group(3))
        mv = state.get_metric(metric)
        return {"<": mv < val, ">": mv > val, "<=": mv <= val, ">=": mv >= val,
                "==": mv == val, "!=": mv != val}[op]

    def compute_deltas(self, action: str, intensity: float,
                        env: dict[str, str] | None = None,
                        state: Any = None,
                        allowed_actions: list[str] | None = None) -> tuple[dict, dict]:
        intensity = max(0.0, min(1.0, float(intensity)))
        # 越权回退：若动作不在允许列表中，回退为 observe
        if allowed_actions is not None and action not in allowed_actions:
            logger.debug("[RuleEngine] action '%s' not allowed, fallback to observe", action)
            action = "observe"
        self_d = {k: v * intensity for k, v in self.pack["self_effects"].get(action, {}).items()}
        tgt_d = {k: v * intensity for k, v in self.pack["target_effects"].get(action, {}).items()}
        # 状态依赖条件效应
        if state is not None:
            for key, cfg in self.pack.get("conditional_effects", {}).items():
                if not key.startswith(action + "_"):
                    continue
                cond = cfg.get("condition", "")
                parsed = self._parsed_conditions.get(cond)
                ok = self._eval_cond_cached(parsed, state) if parsed else self._eval_cond(cond, state)
                if ok:
                    for k, v in cfg.get("self_effects", {}).items():
                        self_d[k] = self_d.get(k, 0.0) + v * intensity
        if env:
            for key, sel in (("weather_modifiers", env.get("weather")),
                             ("terrain_modifiers", env.get("terrain"))):
                mods = self.pack.get(key, {}).get(sel or "", {})
                for k, v in mods.items():
                    self_d[k] = self_d.get(k, 0.0) + v * intensity
        return self_d, tgt_d

    def evaluate_auto_effects(self, states: dict[str, EntityState]) -> dict[str, dict[str, float]]:
        """每轮自动效应：按实体评估条件，返回逐实体增量。"""
        result: dict[str, dict[str, float]] = {}
        auto = self.pack.get("auto_effects", {})
        if not auto:
            return result
        for eid, st in states.items():
            deltas: dict[str, float] = {}
            for _label, cfg in auto.items():
                cond = cfg.get("condition", "")
                parsed = self._parsed_conditions.get(cond)
                if parsed:
                    ok = self._eval_cond_cached(parsed, st)
                elif cond:
                    ok = self._eval_cond(cond, st)
                else:
                    ok = True
                if ok:
                    for metric, delta in cfg.get("effects", {}).items():
                        deltas[metric] = deltas.get(metric, 0.0) + float(delta)
            if deltas:
                result[eid] = deltas
        return result

    # ── 整轮交互解算（基于快照，批量应用由调用方负责） ──
    def resolve_round(self, snapshot_states: dict[str, EntityState],
                      decisions: list[dict[str, Any]], name_to_id: dict[str, str],
                      env: dict[str, str] | None = None,
                      collect_interactions: bool = False):
        """计算本轮全部 delta；collect_interactions=True 时额外返回逐 (actor→target) 归因，
        供因果链(硬档)写入图谱。默认仅返回合并 delta，向后兼容。"""
        result: dict[str, dict[str, float]] = {}
        interactions: list[dict[str, Any]] = []

        # Pre-build O(1) lowercase name→id map for _resolve_target
        lower_map = {n.lower().strip(): eid for n, eid in name_to_id.items()}

        def _add(eid: str, d: dict[str, float]) -> None:
            bucket = result.setdefault(eid, {})
            for k, v in d.items():
                bucket[k] = bucket.get(k, 0.0) + v

        for dec in decisions:
            actor = dec.get("actor_id")
            if actor is None or actor not in snapshot_states:
                continue
            for action, sub_intensity, target in self._iter_subactions(dec):
                if sub_intensity <= 0:
                    continue
                self_d, tgt_d = self.compute_deltas(action, sub_intensity, env,
                                                       state=snapshot_states.get(actor))
                _add(actor, self_d)
                if tgt_d:
                    tid = lower_map.get(target.lower().strip()) if target else None
                    if tid and tid != actor and tid in snapshot_states:
                        _add(tid, tgt_d)
                        if collect_interactions:
                            interactions.append({"actor": actor, "target": tid,
                                                 "action": action, "deltas": dict(tgt_d)})
                    elif target:
                        logger.debug("[RuleEngine] target 未解析/已出局: %s", target)
        if collect_interactions:
            return result, interactions
        return result

    @staticmethod
    def _iter_subactions(dec: dict[str, Any]):  # generator — no return type annotation to avoid typing complexity
        """将决策展开为 [(action_type, sub_intensity, target), ...] 的生成器。"""
        def _legacy():
            try:
                intensity = max(0.0, min(1.0, float(dec.get("intensity", 0.5))))
            except (TypeError, ValueError):
                intensity = 0.5
            yield (str(dec.get("action_type", "observe")), intensity,
                   str(dec.get("target", "") or "").strip())

        actions = dec.get("actions")
        if not isinstance(actions, list) or not actions:
            yield from _legacy()
            return
        try:
            budget = max(0.0, min(1.0, float(dec.get("budget", dec.get("intensity", 0.5)))))
        except (TypeError, ValueError):
            budget = 0.5
        parsed: list[tuple[str, float, str]] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            act = str(a.get("action_type", "observe"))
            try:
                w = max(0.0, float(a.get("weight", 0.0)))
            except (TypeError, ValueError):
                w = 0.0
            parsed.append((act, w, str(a.get("target", "") or "").strip()))
        if not parsed:
            yield from _legacy()
            return
        total = sum(w for _a, w, _t in parsed)
        if total <= 0:
            n = len(parsed)
            for act, _w, tgt in parsed:
                yield (act, budget / n, tgt)
        else:
            for act, w, tgt in parsed:
                yield (act, budget * (w / total), tgt)


    # ── 存亡 ──
    def is_alive(self, state: EntityState) -> bool:
        """存亡判定。支持两种模式：

        weighted_score（需 rules.json 配置 elimination）：
          综合评分 = Σ(metrics[m] × weights[m])
          淘汰条件：score < threshold_score 或 任一 hard_core 指标 ≤ 保底值
          适用：大国博弈等需要多维度综合评估的场景

        threshold（默认，无需配置）：
          任一指标 ≤ thresholds[m] → 死亡
          适用：军事/商业等单维崩盘即淘汰的场景
        """
        elim = self.pack.get("elimination")
        if elim and elim.get("mode") == "weighted_score":
            weights = elim.get("weights", {})
            hard = elim.get("hard_core", {})
            score = sum(state.get_metric(m) * w for m, w in weights.items())
            threshold = float(elim.get("threshold_score", 30.0))
            if score < threshold:
                return False
            for m, floor in hard.items():
                if state.get_metric(m) <= float(floor):
                    return False
            return True
        return state.is_alive(self.pack["thresholds"])

    # ── 结构化胜利条件 → 客观判胜负 ──
    def judge(self, state: EntityState, win_target: dict[str, Any] | None) -> dict[str, Any]:
        alive = self.is_alive(state)
        targets = (win_target or {}).get("metrics") or {}
        logic = (win_target or {}).get("threshold_logic", "all")

        if targets:
            checks, ratios = [], []
            for m, thr in targets.items():
                val = state.get_metric(m)
                thr = float(thr)
                checks.append(val >= thr)
                ratios.append(min(1.0, val / thr) if thr > 0 else (1.0 if val > 0 else 0.0))
            win_score = sum(ratios) / len(ratios) if ratios else 0.0
            if logic == "any":
                success = any(checks)
            elif logic == "weighted_score":
                success = win_score >= 0.5
            else:
                success = all(checks)
        else:
            vals = list(state.metrics.values())
            win_score = (sum(vals) / len(vals) / 100.0) if vals else 0.0
            success = alive

        if not alive:
            success = False
        win_score = max(0.0, min(1.0, win_score))

        # cost：关键指标(阈值约束项)相对初值的损耗均值
        init = self.pack["initial_metrics"]
        losses = []
        for m in self.pack["thresholds"]:
            i = float(init.get(m, 100.0))
            if i > 0:
                losses.append(max(0.0, (i - state.get_metric(m)) / i))
        cost = round(sum(losses) / len(losses), 4) if losses else round(1.0 - win_score, 4)

        return {"success": bool(success), "win_score": round(win_score, 4),
                "cost": cost, "alive": alive}

    # ── 领域识别（LLM） ──
    @staticmethod
    async def detect_domain(text: str, chat_client: Any, confidence_floor: float = 0.6) -> str:
        from ._utils import extract_text, extract_json
        from strategy_forge.core.llm_client import Message

        options = "\n".join(
            f"- {d['domain']}: {d.get('name', d['domain'])}" for d in list_domains())
        _detect_base = (
            "判断以下文本最适合哪个推演领域，并给出 0-1 的置信度。\n\n"
            f"## 可选领域\n{options}\n- narrative: 无明确量化领域 / 纯叙事文学\n\n"
            "## 示例\n"
            '文本："A国与B国在边境爆发激烈交火，双方投入大量装甲部队..."\n'
            '→ {"domain": "military", "confidence": 0.92}\n'
            '文本："市场竞争激烈，新兴品牌通过价格战和社交媒体营销快速占领市场份额"\n'
            '→ {"domain": "business", "confidence": 0.85}\n\n'
            "## 文本\n{text}\n\n"
            '## 输出 JSON（仅 JSON）\n{"domain": "领域标识", "confidence": 0.0到1.0}'
        )
        # 注意：prompt 内含 JSON 示例的花括号，不能用 str.format（会误判为替换字段），
        # 用 replace 注入文本。
        prompt = _detect_base.replace("{text}", text[:4000])
        try:
            resp = await chat_client.chat_json([Message(role="user", content=prompt)],
                                          system="你是领域分类器，只输出 JSON。", schema_name="domain_detect", temperature=0.1)
            data = extract_json(extract_text(resp))
            if isinstance(data, dict):
                dom = str(data.get("domain", "narrative"))
                conf = float(data.get("confidence", 0.0))
                if get_template(dom) is not None and conf >= confidence_floor:
                    return dom
        except Exception as e:
            logger.warning("[RuleEngine] detect_domain 失败，回退叙事: %s", e)
        return "narrative"

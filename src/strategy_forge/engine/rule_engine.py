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

import logging
import re
from typing import Any

from strategy_forge.core.rule_templates import get_template, list_domains

from .models import EntityState

logger = logging.getLogger(__name__)

_CONDITION_RE = re.compile(r'\s*(\w+)\s*(<=|>=|!=|==|<|>)\s*([\d.]+)\s*')


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
        """Pre-parse 'a<5 and b>2 or c==1' → [[(a,<,5),(b,>,2)], [(c,==,1)]]."""
        result: list[list[tuple[str, str, float]]] = []
        for or_part in condition.split(" or "):
            atoms: list[tuple[str, str, float]] = []
            for and_part in or_part.strip().split(" and "):
                m = _CONDITION_RE.match(and_part.strip())
                if m:
                    atoms.append((m.group(1), m.group(2), float(m.group(3))))
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
        return p

    # ── 访问器 ──
    def metrics(self) -> list[str]:
        return list(self.pack["metrics"])

    def thresholds(self) -> dict[str, float]:
        return dict(self.pack["thresholds"])

    def ranges(self) -> dict[str, Any]:
        return dict(self.pack.get("metric_ranges", {}))

    def actions(self) -> list[str]:
        return list(self.pack["actions"])

    def action_catalog(self) -> str:
        """供决策 prompt 使用的可选动作说明。"""
        lines = []
        for a in self.pack["actions"]:
            eff = self.pack["self_effects"].get(a, {})
            desc = ", ".join(f"{k}{v:+.0f}" for k, v in eff.items()) or "无直接消耗"
            lines.append(f"- {a}（自身效应: {desc}）")
        return "\n".join(lines)

    # ── 状态初始化 ──
    def init_state(self, entity_id: str, name: str) -> EntityState:
        return EntityState(id=entity_id, name=name, domain=self.domain,
                           metrics={k: float(v) for k, v in self.pack["initial_metrics"].items()})

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

    @staticmethod
    def _eval_cond_cached(parsed: list[list[tuple[str, str, float]]], state: Any) -> bool:
        """Evaluate a pre-parsed condition against entity state — no string ops."""
        for and_group in parsed:
            ok = True
            for metric, op, val in and_group:
                mv = state.get_metric(metric)
                if op == "<" and not (mv < val):
                    ok = False
                    break
                if op == ">" and not (mv > val):
                    ok = False
                    break
                if op == "<=" and not (mv <= val):
                    ok = False
                    break
                if op == ">=" and not (mv >= val):
                    ok = False
                    break
                if op == "==" and not (mv == val):
                    ok = False
                    break
                if op == "!=" and not (mv != val):
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
                       state: Any = None) -> tuple[dict, dict]:
        intensity = max(0.0, min(1.0, float(intensity)))
        self_d = {k: v * intensity for k, v in self.pack["self_effects"].get(action, {}).items()}
        tgt_d = {k: v * intensity for k, v in self.pack["target_effects"].get(action, {}).items()}
        # 状态依赖条件效应
        if state is not None:
            for key, cfg in self.pack.get("conditional_effects", {}).items():
                if not key.startswith(action):
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
        from ._utils import extract_text
        from .graph_builder import try_extract_json
        from strategy_forge.core.llm_client import Message

        options = "\n".join(f"- {d['domain']}: {d['display_name']}" for d in list_domains())
        _detect_base = (
            "判断以下文本最适合哪个推演领域，并给出 0-1 的置信度。\n\n"
            f"## 可选领域\n{options}\n- narrative: 无明确量化领域 / 纯叙事文学\n\n"
            "## 文本\n{text}\n\n"
            '## 输出 JSON（仅 JSON）\n{"domain": "领域标识", "confidence": 0.0到1.0}'
        )
        prompt = _detect_base.format(text=text[:4000])
        try:
            resp = await chat_client.chat([Message(role="user", content=prompt)],
                                          system="你是领域分类器，只输出 JSON。", temperature=0.1)
            data = try_extract_json(extract_text(resp))
            if isinstance(data, dict):
                dom = str(data.get("domain", "narrative"))
                conf = float(data.get("confidence", 0.0))
                if get_template(dom) is not None and conf >= confidence_floor:
                    return dom
        except Exception as e:
            logger.warning("[RuleEngine] detect_domain 失败，回退叙事: %s", e)
        return "narrative"

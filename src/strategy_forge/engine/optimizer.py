"""策略优化器 (Optimizer) — 蒙特卡洛多方案对比层。

在推演引擎之上增加一层：对同一种子材料，针对多个“策略指令(方案)”各运行 N 次
隔离的并行推演（每次微调温度/随机种子），用 LLM 依据统一“目标条件”评估每次结局，
统计胜率均值/置信区间/成功率/成本，并通过帕累托前沿筛选推荐方案。

设计要点：
- 自建基线：优化器自己跑一次 Phase1-3（本体→图谱→智能体），M 个方案共享该基线。
- 隔离：每次模拟使用 persist_events=False 的 SimulationEngine（纯内存，不写库），
  保证 M×N 次相互隔离、可并发、不污染主会话数据。
- 失败传播：基线构建失败或产出 0 智能体时抛出明确错误，绝不静默返回空统计。
- 统计仅用标准库 statistics，无第三方依赖。
"""
from __future__ import annotations

import asyncio
import copy
import logging
import math
import random
import statistics
from dataclasses import dataclass
from string import Template
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SimulationOutcome:
    success: bool
    win_score: float   # 0-1，达成目标条件的程度
    cost: float        # 0-1，付出的代价/风险（越高越差）
    rationale: str


@dataclass
class RunResult:
    scenario: str
    iteration: int
    seed: int
    temperature: float
    outcome: SimulationOutcome
    action_count: int


_EVAL_PROMPT = """你是推演结果评估专家。请依据“目标条件”评估本次推演结局，并量化打分。

## 目标条件（唯一判定标准）
{win_condition}

## 本方案所采取的策略指令
{directive}

## 本次推演的关键动作（按时间顺序）
{events}

## 输出 JSON（仅输出 JSON，无任何解释/markdown）
{{"success": true 或 false, "win_score": 0.0到1.0之间的小数, "cost": 0.0到1.0之间的小数, "rationale": "30字以内的判定理由"}}
- success: 是否达成目标条件
- win_score: 达成目标条件的程度（越高越好）
- cost: 为达成所付出的代价/承担的风险（越高越差）

## 评分校准示例
场景：目标="市场份额达到50%"，最终份额48%，以持续价格战获得，品牌从35%跌至18%

正确评分：
{{"success": false, "win_score": 0.72, "cost": 0.68, "rationale": "接近目标但品牌代价过大"}}

错误评分（不推荐）：
{{"success": true, "win_score": 0.95, "cost": 0.1, "rationale": "几乎达成目标"}}

- win_score 仅反映距离目标的客观差距，不因\"努力了\"而加分
- cost 反映阈值约束指标的平均损耗比例，品牌腰斩对应高 cost
"""


class StrategyOptimizer:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    async def run_monte_carlo(
        self,
        session_id: str,
        scenarios: list[dict[str, str]],
        win_condition: str,
        iterations: int = 20,
        objective: str = "balanced",
        max_concurrent: int | None = None,
        cancel_event: asyncio.Event | None = None,
        progress_cb: Callable[[int, int, str, SimulationOutcome], None] | None = None,
    ) -> dict[str, Any]:
        from strategy_forge.core.config import config
        from strategy_forge.core.providers import registry as _reg
        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
        from strategy_forge.core.token_counter import _current_session, _current_phase
        from strategy_forge.engine.simulator import SimulationEngine
        from strategy_forge.engine.models import DeductionAgentProfile
        import json as _json

        # 设置 token 统计上下文
        _current_session.set(session_id)
        _current_phase.set("optimizer")

        def olog(msg: str) -> None:
            self.engine.log(session_id, "optimize", msg)

        session = self.engine.get_session(session_id)
        if session is None:
            raise RuntimeError("会话不存在，无法启动优化器")
        source = session.source_material or ""
        if not source.strip():
            raise RuntimeError("种子材料为空，无法进行推演优化")
        total_rounds = session.total_rounds or _reg.default_rounds
        max_concurrent = max_concurrent or _reg.max_concurrent

        # ── 1. 复用主会话基线（实体/图谱/智能体已由正常推演流程构建并经过 sorter 过滤）──
        olog("优化器启动：复用主会话基线...")
        graph = self.engine.get_graph(session_id)
        # 从 Kuzu 读取已持久化的 Agent 节点（经过 agent_factory + intel_list 过滤）
        stored_agents = []
        try:
            stored_agents = graph.get_agents()
        except Exception as e:
            logger.warning("[Optimizer] 读取已存智能体失败: %s", e)
            stored_agents = []
        if stored_agents:
            agents = []
            type_map: dict[str, str] = {}
            try:
                erows = graph.query(f"MATCH (e:{graph.NODE_TABLE}) RETURN e.id, e.type")
                type_map = {r[0]: r[1] for r in erows if r[0] and r[1]}
            except Exception:
                pass
            for a in stored_agents:
                try:
                    goals = _json.loads(a.get("goals") or "[]")
                except (ValueError, TypeError):
                    goals = []
                agents.append(DeductionAgentProfile(
                    entity_id=a.get("id", ""),
                    name=a.get("name", ""),
                    persona=a.get("persona", ""),
                    background=a.get("background", ""),
                    goals=goals if isinstance(goals, list) else [],
                    entity_type=type_map.get(a.get("id", ""), ""),
                ))
            olog(f"复用主会话基线: {len(agents)} 个智能体（已通过 sorter 过滤）")
        else:
            raise RuntimeError("主会话未完成智能体生成，请先完成一次推演再使用优化器")

        if not agents:
            raise RuntimeError("基线构建完成但未产生任何智能体，请检查种子材料是否包含人物(Person)实体")

        # 量化模式：确定规则包 + 建立基线初始状态（只建一次，各次模拟深拷贝隔离）
        rule_engine = None
        base_states: dict[str, Any] = {}
        enable_multi_action = False
        max_actions = 3
        opt_env: dict[str, str] | None = None
        try:
            data = self.engine.session_store.get(session_id)
            cfg = (data or {}).get("config_json", {}) or {}
            if isinstance(cfg, str):
                import json as _json
                cfg = _json.loads(cfg)
            enable_multi_action = bool(cfg.get("enable_multi_action", False))
            try:
                max_actions = int(cfg.get("max_actions", 3))
            except (TypeError, ValueError):
                max_actions = 3
            weather = str(cfg.get("weather", "") or "").strip()
            terrain = str(cfg.get("terrain", "") or "").strip()
            opt_env = {"weather": weather, "terrain": terrain} if (weather or terrain) else None
            domain = (cfg.get("domain") or "narrative").strip()
            if domain not in ("", "narrative"):
                from strategy_forge.engine.rule_engine import RuleEngine
                if domain == "custom" and cfg.get("custom_rules"):
                    rule_engine = RuleEngine.from_custom(cfg["custom_rules"])
                elif domain == "auto":
                    detected = await RuleEngine.detect_domain(source, LLMClient())
                    if detected != "narrative":
                        rule_engine = RuleEngine.from_domain(detected)
                else:
                    rule_engine = RuleEngine.from_domain(domain)
            if rule_engine is not None:
                for a in agents:
                    base_states[a.entity_id] = rule_engine.init_state(a.entity_id, a.name)
                olog(f"量化模式: 领域={rule_engine.domain}，{len(base_states)} 个量化实体，胜负由数值阈值判定")
        except Exception as e:
            logger.warning("[Optimizer] 量化初始化失败，回退 LLM 评估: %s", e)
            rule_engine = None

        total = len(scenarios) * iterations
        olog(f"基线就绪：{len(agents)} 个智能体；开始 {len(scenarios)} 个方案 × {iterations} 次蒙特卡洛（共 {total} 次推演）")

        # ── 2. M×N 隔离模拟 + LLM 评估 ──
        eval_client = LLMClient()
        sem = asyncio.Semaphore(max(1, max_concurrent))
        results: list[RunResult] = []
        done = 0
        best_win = 0.0

        async def one_run(sc: dict[str, str], i: int) -> RunResult | None:
            nonlocal done, best_win
            if cancel_event is not None and cancel_event.is_set():
                return None
            seed = random.randint(1, 10_000_000)
            temp = round(random.uniform(0.3, 0.9), 2)  # 温度抖动，增加结局多样性
            async with sem:
                if cancel_event is not None and cancel_event.is_set():
                    return None
                if rule_engine is not None:
                    # 量化模式：深拷贝基线状态隔离 → 数值阈值客观判胜负
                    states_copy = {eid: copy.deepcopy(st) for eid, st in base_states.items()}
                    sim = SimulationEngine(
                        agents=agents, graph=graph, total_rounds=total_rounds,
                        log_fn=lambda _p, _m: None, preprocessor=None,
                        pre_goals=[sc["directive"]] if sc.get("directive") else [],
                        seed=seed, temperature=temp, persist_events=False, max_concurrent=None,
                        rule_engine=rule_engine, states=states_copy, enable_narrate=False,
                        enable_multi_action=enable_multi_action, max_actions=max_actions,
                        env=opt_env,
                    )
                    actions: list[Any] = []
                    for rnd in range(1, total_rounds + 1):
                        if cancel_event is not None and cancel_event.is_set():
                            break
                        await sim.run_round(rnd)
                    outcome = self._judge_quantified(rule_engine, states_copy, sc)
                else:
                    sim = SimulationEngine(
                        agents=agents, graph=graph, total_rounds=total_rounds,
                        log_fn=lambda _p, _m: None, preprocessor=None,
                        pre_goals=[sc["directive"]] if sc.get("directive") else [],
                        seed=seed, temperature=temp, persist_events=False, max_concurrent=None,
                    )
                    actions = []
                    for rnd in range(1, total_rounds + 1):
                        if cancel_event is not None and cancel_event.is_set():
                            break
                        rd = await sim.run_round(rnd)
                        actions.extend(rd.actions)
                    outcome = await self._evaluate_outcome(
                        eval_client, win_condition, sc.get("directive", ""), actions,
                    )
            done += 1
            best_win = max(best_win, outcome.win_score)
            if progress_cb is not None:
                progress_cb(done, total, sc.get("name", "方案"), outcome)
            olog(
                f"[{done}/{total}] {sc.get('name','方案')} 第{i+1}次: "
                f"{'达成' if outcome.success else '未达成'} 胜分{outcome.win_score:.2f} "
                f"成本{outcome.cost:.2f}（当前最高胜分 {best_win:.2f}）"
            )
            return RunResult(
                scenario=sc.get("name", "方案"), iteration=i, seed=seed,
                temperature=temp, outcome=outcome, action_count=len(actions),
            )

        tasks = []
        for sc in scenarios:
            for i in range(iterations):
                if cancel_event is not None and cancel_event.is_set():
                    break
                tasks.append(asyncio.create_task(one_run(sc, i)))

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for g in gathered:
            if isinstance(g, RunResult):
                results.append(g)
            elif isinstance(g, Exception):
                logger.warning("[Optimizer] run failed: %s", g)

        cancelled = bool(cancel_event is not None and cancel_event.is_set())

        # ── 3. 统计与帕累托分析 ──
        report = self._analyze(results, scenarios, objective, win_condition)
        report["cancelled"] = cancelled
        report["completed_runs"] = len(results)
        report["total_runs"] = total
        report["iterations"] = iterations
        olog(
            ("优化已取消，" if cancelled else "优化完成，")
            + f"有效结果 {len(results)}/{total}"
            + (f"，推荐方案：{report['recommended']['name']}" if report.get("recommended") else "")
        )

        # ── 4. 推荐方案"代表性 run"：persist=True 生成叙事报告 + 点亮时间线/因果页 ──
        if not cancelled and rule_engine is not None and base_states and report.get("recommended"):
            try:
                import json
                rec_name = report["recommended"]["name"]
                rec_sc = next((s for s in scenarios if s.get("name") == rec_name), None)
                if rec_sc is not None:
                    olog(f"为推荐方案「{rec_name}」生成代表性推演报告（报告/时间线/因果页）...")
                    rep_states = {eid: copy.deepcopy(st) for eid, st in base_states.items()}
                    rep_sim = SimulationEngine(
                        agents=agents, graph=graph, total_rounds=total_rounds,
                        log_fn=lambda _p, _m: None, preprocessor=None,
                        pre_goals=[rec_sc["directive"]] if rec_sc.get("directive") else [],
                        seed=20240101, temperature=0.6, persist_events=True, max_concurrent=None,
                        rule_engine=rule_engine, states=rep_states, enable_narrate=False,
                        enable_multi_action=enable_multi_action, max_actions=max_actions,
                        env=opt_env,
                    )
                    rep_rounds = []
                    for rnd in range(1, total_rounds + 1):
                        rep_rounds.append(await rep_sim.run_round(rnd))
                    from strategy_forge.engine.reporter import generate_report
                    session.current_round = total_rounds
                    rep = await generate_report(
                        session=session, graph=graph, rounds=rep_rounds,
                        log_fn=lambda _p, _m: None, preprocessor=None)
                    payload = {
                        "summary": rep.summary,
                        "key_events": rep.key_events,
                        "risk_alerts": rep.risk_alerts,
                        "recommendations": rep.recommendations,
                        "quantified": True,
                        "domain": rule_engine.domain,
                        "optimized_scenario": rec_name,
                        "final_states": {
                            eid: {"name": st.name, "metrics": st.metrics,
                                  "history": st.history[-60:],
                                  "alive": rule_engine.is_alive(st)}
                            for eid, st in rep_states.items()
                        },
                    }
                    self.engine.session_store.update(
                        session_id, report_json=json.dumps(payload, ensure_ascii=False))
                    olog("代表性推演报告已生成")
            except Exception as e:
                logger.warning("[Optimizer] 代表性报告生成失败: %s", e)

        return report

    def _judge_quantified(self, rule_engine: Any, states: dict[str, Any],
                          scenario: dict[str, Any]) -> SimulationOutcome:
        """量化模式：按结构化目标条件读最终 EntityState 数值客观判胜负。

        当指定 entity_ref 时，使用 rule_engine.judge() 的加权成本计算。
        未指定时，使用规则包 elimination.weights（若存在）做加权平均健康度，
        无 weights 时回退等权平均。
        """
        win_target = scenario.get("win_target") or {}
        ref = (win_target.get("entity_ref") or "").strip()
        target_state = None
        if ref:
            for st in states.values():
                nm = getattr(st, "name", "")
                if nm == ref or (len(ref) >= 2 and (ref in nm or nm in ref)):
                    target_state = st
                    break
        if target_state is not None:
            j = rule_engine.judge(target_state, win_target)
            detail = ", ".join(f"{k}={v:.0f}" for k, v in target_state.metrics.items())
            return SimulationOutcome(success=j["success"], win_score=j["win_score"],
                                     cost=j["cost"], rationale=f"{target_state.name}: {detail}")
        # 缺省（未指定我方实体）：用全体存活率 + 加权健康度作客观评分
        if not states:
            return SimulationOutcome(False, 0.0, 1.0, "无量化实体")
        alive = sum(1 for st in states.values() if rule_engine.is_alive(st))
        n = len(states)
        win_score = round(alive / n, 4)
        # 使用规则包 elimination.weights 做加权平均，无 weights 时回退等权
        weights = rule_engine.pack.get("elimination", {}).get("weights", {})
        metric_list = getattr(rule_engine, "_metrics", []) or rule_engine.metrics()
        if not weights:
            weights = {m: 1.0 / len(metric_list) for m in metric_list} if metric_list else {}
        total_weighted = 0.0
        total_weight = 0.0
        for st in states.values():
            for m, w in weights.items():
                val = st.metrics.get(m, 0.0)
                total_weighted += val * w
                total_weight += w
        avg_weighted = (total_weighted / max(1, total_weight)) / 100.0 if total_weight > 0 else 0.0
        cost = round(1.0 - avg_weighted, 4)
        return SimulationOutcome(success=win_score >= 0.5, win_score=win_score,
                                 cost=cost, rationale=f"存活 {alive}/{n}，加权健康度 {avg_weighted:.0%}")

    async def _evaluate_outcome(
        self, client: Any, win_condition: str, directive: str, actions: list[Any],
    ) -> SimulationOutcome:
        from strategy_forge.core.llm_client import Message
        from strategy_forge.engine._utils import extract_text
        from strategy_forge.engine.graph_builder import try_extract_json

        events = "\n".join(
            f"- {getattr(a, 'agent_id', '')[:8]} {getattr(a, 'action_type', '')}: "
            f"{getattr(a, 'content', '')[:80]}"
            for a in actions[-30:]
        ) or "（本次推演未产生有效动作）"

        prompt = Template(_EVAL_PROMPT).substitute(
            win_condition=win_condition or "（未指定，默认评估对主要角色是否有利）",
            directive=directive or "（无特定策略指令，自主行动）",
            events=events,
        )
        try:
            resp = await client.chat(
                [Message(role="user", content=prompt)],
                system="你是推演结果评估专家，只输出 JSON。",
                temperature=0.1,
            )
            data = try_extract_json(extract_text(resp))
            if isinstance(data, dict):
                return SimulationOutcome(
                    success=bool(data.get("success", False)),
                    win_score=_clamp(data.get("win_score", 0.5)),
                    cost=_clamp(data.get("cost", 0.5)),
                    rationale=str(data.get("rationale", ""))[:120],
                )
        except Exception as e:
            logger.warning("[Optimizer] 结局评估失败: %s", e)
        return SimulationOutcome(success=False, win_score=0.5, cost=0.5, rationale="评估失败，取中值")

    def _analyze(
        self, results: list[RunResult], scenarios: list[dict[str, str]],
        objective: str, win_condition: str,
    ) -> dict[str, Any]:
        directive_map = {s.get("name", "方案"): s.get("directive", "") for s in scenarios}
        ref_map = {s.get("name", "方案"): ((s.get("win_target") or {}).get("entity_ref", "") or "") for s in scenarios}
        grouped: dict[str, list[RunResult]] = {}
        for r in results:
            grouped.setdefault(r.scenario, []).append(r)

        stats_list: list[dict[str, Any]] = []
        for name, runs in grouped.items():
            wins = [r.outcome.win_score for r in runs]
            costs = [r.outcome.cost for r in runs]
            n = len(runs)
            succ = sum(1 for r in runs if r.outcome.success)
            mean = statistics.mean(wins) if wins else 0.0
            sd = statistics.pstdev(wins) if n > 1 else 0.0
            ci = 1.96 * sd / math.sqrt(n) if n > 0 else 0.0
            cost_mean = statistics.mean(costs) if costs else 0.0
            stats_list.append({
                "name": name,
                "directive": directive_map.get(name, ""),
                "entity_ref": ref_map.get(name, ""),
                "runs": n,
                "success_rate": round(succ / n, 4) if n else 0.0,
                "win_mean": round(mean, 4),
                "win_stdev": round(sd, 4),
                "win_ci95": [round(max(0.0, mean - ci), 4), round(min(1.0, mean + ci), 4)],
                "cost_mean": round(cost_mean, 4),
                "is_pareto": False,
                "samples": [r.outcome.rationale for r in runs[:3] if r.outcome.rationale],
            })

        # 帕累托前沿：高 win_mean + 低 cost_mean 为优；非被支配者入选
        for a in stats_list:
            dominated = False
            for b in stats_list:
                if b is a:
                    continue
                better_or_equal = b["win_mean"] >= a["win_mean"] and b["cost_mean"] <= a["cost_mean"]
                strictly_better = b["win_mean"] > a["win_mean"] or b["cost_mean"] < a["cost_mean"]
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            a["is_pareto"] = not dominated

        recommended = None
        if stats_list:
            if objective == "max_win_rate":
                best = max(stats_list, key=lambda x: x["win_mean"])
            elif objective == "min_cost":
                best = min(stats_list, key=lambda x: x["cost_mean"])
            else:  # balanced: 帕累托前沿上 (胜率-成本) 最大，同分取更稳定（CI更窄）
                front = [s for s in stats_list if s["is_pareto"]] or stats_list
                best = max(front, key=lambda x: (
                    x["win_mean"] - x["cost_mean"],
                    -(x["win_ci95"][1] - x["win_ci95"][0]),
                ))
            recommended = {
                "name": best["name"],
                "win_mean": best["win_mean"],
                "win_ci95": best["win_ci95"],
                "success_rate": best["success_rate"],
                "cost_mean": best["cost_mean"],
            }

        # 推荐理由
        rationale = ""
        if recommended and stats_list:
            rec = next((s for s in stats_list if s["name"] == recommended["name"]), None)
            others = [s for s in stats_list if s["name"] != recommended["name"]]
            if rec:
                rationale = f"推荐策略「{recommended['name']}」：{rec['runs']}次模拟中达成率{rec['success_rate']*100:.0f}%，平均达成度{rec['win_mean']*100:.0f}%，成本{rec['cost_mean']:.2f}。"
                if others:
                    second = max(others, key=lambda x: x["win_mean"] - x["cost_mean"])
                    win_gap = rec["win_mean"] - second["win_mean"]
                    cost_gap = rec["cost_mean"] - second["cost_mean"]
                    if win_gap > 0 and cost_gap < 0:
                        rationale += f"相比次优方案「{second['name']}」形成支配优势（胜率更高+成本更低）。"
                    elif win_gap > 0:
                        rationale += f"相比次优方案「{second['name']}」胜率高{abs(win_gap)*100:.1f}%，成本高{abs(cost_gap):.2f}，需权衡。"
                    else:
                        rationale += f"相比次优方案「{second['name']}」成本低{abs(cost_gap):.2f}，胜率低{abs(win_gap)*100:.1f}%。"

        return {
            "objective": objective,
            "win_condition": win_condition,
            "scenarios": sorted(stats_list, key=lambda x: x["win_mean"], reverse=True),
            "pareto_front": [s["name"] for s in stats_list if s["is_pareto"]],
            "recommended": recommended,
            "recommendation_rationale": rationale,
        }


def _clamp(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5

"""Deduction Orchestrator — five-stage pipeline coordinator with pause/resume."""
from __future__ import annotations

import json as _json
import logging
from collections.abc import Callable
from typing import Any

from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.storage.session_store import SessionStore

from .models import (
    DeductionPhase,
    DeductionSession,
    SessionStatus,
    SimulationRound,
)

logger = logging.getLogger(__name__)


class _PhaseCancelledError(Exception):
    """用户取消推演（非错误，应持久化进度为 paused）。"""


class DeductionOrchestrator:

    def __init__(
        self,
        session: DeductionSession,
        graph: DeductionGraphStore,
        session_store: SessionStore,
        logger_fn: Callable[[str, str], None] | None = None,
        cancel_event: Any = None,
        round_callback: Callable[[int, int], None] | None = None,
        resume_start_round: int = 0,
    ) -> None:
        self.session = session
        self.graph = graph
        self.store = session_store
        self._log = logger_fn or (lambda p, m: None)
        self._cancel = cancel_event
        self._round_callback = round_callback
        self._resume_start_round = resume_start_round
        # 量化模式状态（rule_engine 非空即量化）
        self._rule_engine: Any = None
        self._states: dict[str, Any] = {}
        self._enable_narrate: bool = True
        self._enable_multi_action: bool = False
        self._max_actions: int = 3

    async def run(self) -> DeductionSession:
        session_id = self.session.id
        try:
            if self._resume_start_round > 0:
                await self._resume_from_pause()
            else:
                await self._phase1_ontology()
                await self._phase1_5_quantify()
                await self._phase2_graph()
                await self._phase3_agents()
            await self._phase4_simulation()
            await self._phase5_report()

            self.store.update(session_id, status=SessionStatus.COMPLETE.value,
                              phase=DeductionPhase.COMPLETE.value)
            self._clear_state_snapshot(session_id)
            self._log("orchestrator", "全部五阶段推演完成")
        except _PhaseCancelledError:
            self._save_pause_snapshot(session_id)
            self._log("orchestrator", "推演已暂停，进度已保存")
        except Exception as e:
            logger.exception("[Deduction] Pipeline failed: %s", e)
            self.store.update(session_id, status=SessionStatus.FAILED.value,
                              error=str(e)[:500])
            self._log("orchestrator", f"推演失败: {e}")
        return self.session

    def _check_cancel(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise _PhaseCancelledError()

    def _save_pause_snapshot(self, session_id: str) -> None:
        """Serialize in-memory state (EntityState metrics/history/delays) into config_json."""
        snapshot: dict[str, Any] = {}
        states = getattr(self, "_states", None)
        if states:
            snapshot["states"] = {
                eid: {
                    "id": st.id,
                    "name": getattr(st, "name", eid),
                    "domain": getattr(st, "domain", ""),
                    "metrics": dict(getattr(st, "metrics", {})),
                    "history": getattr(st, "history", [])[-100:],
                    "pending_delays": getattr(st, "_pending_delays", []),
                }
                for eid, st in states.items()
            }
        data = self.store.get(session_id)
        cfg = (data or {}).get("config_json", {}) or {}
        if isinstance(cfg, str):
            cfg = _json.loads(cfg)
        cfg["state_snapshot"] = snapshot
        self.store.update(session_id, config_json=_json.dumps(cfg, ensure_ascii=False),
                          status=SessionStatus.PAUSED.value)

    @staticmethod
    def _load_state_snapshot(cfg: dict[str, Any]) -> dict[str, Any] | None:
        return cfg.get("state_snapshot")

    def _clear_state_snapshot(self, session_id: str) -> None:
        data = self.store.get(session_id)
        cfg = (data or {}).get("config_json", {}) or {}
        if isinstance(cfg, str):
            cfg = _json.loads(cfg)
        if "state_snapshot" in cfg:
            del cfg["state_snapshot"]
            self.store.update(session_id, config_json=_json.dumps(cfg, ensure_ascii=False))

    async def _resume_from_pause(self) -> None:
        """从 paused 状态续推：恢复内存态，跳过 Phase 1-3。"""
        self._log("orchestrator", "从暂停点恢复推演...")
        data = self.store.get(self.session.id)
        cfg = (data or {}).get("config_json", {}) or {}
        if isinstance(cfg, str):
            cfg = _json.loads(cfg)

        # 1. 恢复配置参数
        self._enable_narrate = bool(cfg.get("enable_narrate", True))
        self._enable_multi_action = bool(cfg.get("enable_multi_action", False))
        try:
            self._max_actions = int(cfg.get("max_actions", 3))
        except (TypeError, ValueError):
            self._max_actions = 3
        self._weather = str(cfg.get("weather", "") or "").strip()
        self._terrain = str(cfg.get("terrain", "") or "").strip()

        # 2. 恢复规则包
        domain = (cfg.get("domain") or "").strip()
        custom = cfg.get("custom_rules")
        if domain and domain != "narrative":
            from .rule_engine import RuleEngine
            try:
                if domain == "custom" and custom:
                    self._rule_engine = RuleEngine.from_custom(custom)
                else:
                    self._rule_engine = RuleEngine.from_domain(domain)
                self._log("orchestrator",
                          f"恢复规则包: {self._rule_engine.pack.get('display_name', domain)}")
            except Exception as e:
                logger.warning("[Orchestrator] 规则包恢复失败，回退叙事: %s", e)
                self._rule_engine = None

        # 3. 恢复预处理器 (打开已有 LanceDB 表)
        from strategy_forge.core.config import config as forge_config
        from .preprocessor import DeductionPreprocessor
        self._preprocessor = DeductionPreprocessor(
            workspace_root=forge_config.project_root,
            session_id=self.session.id,
        )
        self._pre_goals = cfg.get("pre_goals", [])

        # 4. 从 Kuzu 图重建 Agent 列表
        self._log("orchestrator", "从图谱重建智能体...")
        try:
            from .agent_factory import create_agents_from_graph
            agents = await create_agents_from_graph(
                graph=self.graph,
                source_material=self.session.source_material,
                log_fn=self._log,
                preprocessor=self._preprocessor,
            )
            self._agents = agents
            self.session.agent_count = len(agents)
        except Exception as e:
            logger.warning("[Orchestrator] 智能体重建失败: %s", e)

        # 5. 恢复量化状态 (EntityState metrics / history / pending delays)
        snapshot = self._load_state_snapshot(cfg)
        if snapshot and self._rule_engine is not None:
            states_raw = snapshot.get("states", {})
            restored: dict[str, Any] = {}
            for eid, raw in states_raw.items():
                st = self._rule_engine.init_state(
                    raw.get("id", eid),
                    raw.get("name", eid),
                )
                st.metrics = dict(raw.get("metrics", {}))
                st.history = list(raw.get("history", []))
                st._pending_delays = list(raw.get("pending_delays", []))
                restored[eid] = st
            self._states = restored
            self._log("orchestrator",
                      f"恢复量化状态: {len(restored)} 个实体")

        self.store.update(self.session.id,
                          status=SessionStatus.SIMULATING.value,
                          phase=DeductionPhase.SIMULATION.value)
        self._clear_state_snapshot(self.session.id)
        self._log("orchestrator", f"续推就绪，从第 {self._resume_start_round + 1} 轮开始")

    async def _phase1_ontology(self) -> None:
        self._check_cancel()
        self._log("ontology", "阶段1: 本体生成开始")
        self.store.update(self.session.id,
                          status=SessionStatus.ONTOLOGY_RUNNING.value,
                          phase=DeductionPhase.ONTOLOGY.value)

        from .ontology import generate_ontology
        ontology = await generate_ontology(self.session.source_material)
        self.session.ontology = ontology

        self._log("ontology", f"本体生成完成: {len(ontology.entities)} 种实体类型, "
                  f"{len(ontology.relations)} 种关系类型")
        self.store.update(self.session.id,
                          status=SessionStatus.GRAPH_RUNNING.value,
                          phase=DeductionPhase.GRAPH.value)

    async def _phase1_5_quantify(self) -> None:
        """阶段1.5（仅量化模式）：确定规则包。叙事模式或识别失败则保持 _rule_engine=None。"""
        self._check_cancel()

        data = self.store.get(self.session.id)
        cfg = (data or {}).get("config_json", {}) or {}
        if isinstance(cfg, str):
            cfg = _json.loads(cfg)
        self._enable_narrate = bool(cfg.get("enable_narrate", True))
        self._enable_multi_action = bool(cfg.get("enable_multi_action", False))
        try:
            self._max_actions = int(cfg.get("max_actions", 3))
        except (TypeError, ValueError):
            self._max_actions = 3
        self._weather = str(cfg.get("weather", "") or "").strip()
        self._terrain = str(cfg.get("terrain", "") or "").strip()
        domain = (cfg.get("domain") or "narrative").strip()
        custom = cfg.get("custom_rules")
        if domain in ("", "narrative"):
            self._rule_engine = None
            return

        from .rule_engine import RuleEngine
        try:
            if domain == "custom" and custom:
                self._rule_engine = RuleEngine.from_custom(custom)
                self._log("quantify", f"阶段1.5: 使用自定义规则包（{self._rule_engine.domain}）")
            elif domain == "auto":
                self._log("quantify", "阶段1.5: 自动识别推演领域...")
                from strategy_forge.core.llm_client import DeductionLLMClient
                detected = await RuleEngine.detect_domain(
                    self.session.source_material, DeductionLLMClient())
                if detected == "narrative":
                    self._rule_engine = None
                    self._log("quantify", "未识别到明确量化领域，回退叙事模式")
                    return
                self._rule_engine = RuleEngine.from_domain(detected)
                self._log("quantify", f"识别领域: {self._rule_engine.pack.get('display_name', detected)}")
            else:
                self._rule_engine = RuleEngine.from_domain(domain)
                self._log("quantify", f"阶段1.5: 使用领域规则包: {self._rule_engine.pack.get('display_name', domain)}")
        except Exception as e:
            logger.warning("[Orchestrator] 规则包加载失败，回退叙事: %s", e)
            self._rule_engine = None
            self._log("quantify", f"规则包加载失败，回退叙事模式: {e}")

    async def _phase2_graph(self) -> None:
        self._check_cancel()
        self._log("graph", "阶段2: GraphRAG 知识图谱构建开始")

        # 预处理: 语义分块 + 实体提取 + LanceDB 索引
        self._log("graph", "  预处理: 语义分块 + 实体提取 + LanceDB 索引")
        from strategy_forge.core.config import config

        from .preprocessor import DeductionPreprocessor

        preprocessor = DeductionPreprocessor(
            workspace_root=config.project_root,
            session_id=self.session.id,
        )
        preprocessor.preprocess(self.session.source_material)
        self._preprocessor = preprocessor

        from .graph_builder import build_graph
        await build_graph(
            source=self.session.source_material,
            graph=self.graph,
            ontology=self.session.ontology,
            log_fn=self._log,
            preprocessor=preprocessor,
        )

        e_count = self.graph.count_entities()
        r_count = self.graph.count_relations()
        self.session.entity_count = e_count
        self.session.relation_count = r_count

        self._log("graph", f"图谱构建完成: {e_count} 实体, {r_count} 关系")
        self.store.update(self.session.id, entity_count=e_count, relation_count=r_count,
                          status=SessionStatus.AGENTS_RUNNING.value,
                          phase=DeductionPhase.AGENTS.value)

    async def _phase3_agents(self) -> None:
        self._check_cancel()
        self._log("agents", "阶段3: 智能体工厂开始")

        from .agent_factory import create_agents_from_graph
        cfg_data = self.store.get(self.session.id)
        pre_goals: list[str] = []
        if cfg_data:
            cfg = cfg_data.get("config_json", {}) or {}
            if isinstance(cfg, str):
                cfg = _json.loads(cfg)
            pre_goals = cfg.get("pre_goals", [])
        agents = await create_agents_from_graph(
            graph=self.graph,
            source_material=self.session.source_material,
            log_fn=self._log,
            preprocessor=getattr(self, "_preprocessor", None),
            pre_interventions=pre_goals if pre_goals else None,
        )
        self.session.agent_count = len(agents)
        self._agents = agents
        self._pre_goals = pre_goals

        # 将预目标写入 LanceDB 动态事件表 (immutable_goal, priority=0.9)
        # 确保长期推演中智能体"不忘初心"
        pp = getattr(self, "_preprocessor", None)
        if pp and pre_goals:
            for goal in pre_goals:
                try:
                    pp.add_event_memory(
                        content=goal, agent_id="system_user",
                        round_number=1, event_type="immutable_goal",
                        priority=0.9,
                    )
                except Exception:
                    pass
            self._log("agents", f"已注入 {len(pre_goals)} 个不可变目标到 LanceDB")

        self._log("agents", f"智能体工厂完成: {len(agents)} 个智能体生成")
        self.store.update(self.session.id, agent_count=len(agents),
                          status=SessionStatus.SIMULATING.value,
                          phase=DeductionPhase.SIMULATION.value)

    async def _phase4_simulation(self) -> None:
        self._check_cancel()
        total_rounds = self.session.total_rounds
        re_engine = self._rule_engine
        states: dict[str, Any] = {}
        if re_engine is not None:
            for a in self._agents:
                states[a.entity_id] = re_engine.init_state(a.entity_id, a.name)
            self._states = states
            self._log("simulation",
                      f"阶段4: 量化并行模拟开始 ({total_rounds} 轮, {len(states)} 个量化实体, "
                      f"领域={re_engine.domain})")
        else:
            self._log("simulation", f"阶段4: 并行模拟开始 ({total_rounds} 轮)")

        from .simulator import SimulationEngine
        engine = SimulationEngine(
            agents=self._agents,
            graph=self.graph,
            total_rounds=total_rounds,
            log_fn=self._log,
            preprocessor=getattr(self, "_preprocessor", None),
            pre_goals=getattr(self, "_pre_goals", []),
            rule_engine=re_engine,
            states=states if re_engine is not None else None,
            enable_narrate=self._enable_narrate,
            enable_multi_action=self._enable_multi_action,
            max_actions=self._max_actions,
            env={"weather": self._weather, "terrain": self._terrain} if (self._weather or self._terrain) else None,
            cancel_event=self._cancel,
        )

        rounds: list[SimulationRound] = []
        start_rnd = self._resume_start_round + 1
        for rnd in range(start_rnd, total_rounds + 1):
            if self._cancel is not None and self._cancel.is_set():
                self._log("simulation", "推演收到取消信号，提前终止")
                raise _PhaseCancelledError()
            self._log("simulation", f"  第 {rnd}/{total_rounds} 轮开始")
            result = await engine.run_round(rnd)
            rounds.append(result)
            self.session.current_round = rnd
            self.store.update(self.session.id, current_round=rnd)
            self._log("simulation", f"  第 {rnd} 轮完成: {len(result.actions)} 个动作")
            if self._round_callback:
                self._round_callback(rnd, total_rounds)

        self._simulation_rounds = rounds
        self._log("simulation", f"模拟完成: {len(rounds)} 轮, "
                  f"{sum(len(r.actions) for r in rounds)} 个总动作")
        self.store.update(self.session.id,
                          status=SessionStatus.REPORTING.value,
                          phase=DeductionPhase.REPORT.value)

    async def _phase5_report(self) -> None:
        self._log("report", "阶段5: 报告生成开始")

        from .reporter import generate_report
        report = await generate_report(
            session=self.session,
            graph=self.graph,
            rounds=getattr(self, "_simulation_rounds", []),
            log_fn=self._log,
            preprocessor=getattr(self, "_preprocessor", None),
        )
        self.session.report = report

        import json
        report_payload = {
            "summary": report.summary,
            "key_events": report.key_events,
            "risk_alerts": report.risk_alerts,
            "recommendations": report.recommendations,
            "causal_summary": report.causal_summary,
            "stage_narratives": report.stage_narratives,
            "conclusion": report.conclusion,
        }
        if self._rule_engine is not None and self._states:
            report_payload["quantified"] = True
            report_payload["domain"] = self._rule_engine.domain
            report_payload["final_states"] = {
                eid: {"name": st.name, "metrics": st.metrics,
                      "history": st.history[-60:],
                      "alive": self._rule_engine.is_alive(st)}
                for eid, st in self._states.items()
            }
        self.store.update(self.session.id,
                          report_json=json.dumps(report_payload, ensure_ascii=False))
        self._log("report", f"报告生成完成: {report.summary[:100]}...")

    def get_realtime_round(self) -> SimulationRound | None:
        rounds = getattr(self, "_simulation_rounds", None)
        if rounds and self.session.current_round > 0:
            idx = self.session.current_round - 1
            if idx < len(rounds):
                return rounds[idx]
        return None

"""Deduction Orchestrator — five-stage pipeline coordinator with pause/resume."""
from __future__ import annotations

import json as _json
import logging
from collections.abc import Callable
from typing import Any

from strategy_forge.storage.graph_store import DeductionGraphStore
from strategy_forge.storage.session_store import SessionStore
from strategy_forge.core.token_counter import (
    _current_session,
    _current_phase,
    _current_round,
)

from .models import (
    DeductionAgentProfile,
    DeductionPhase,
    DeductionSession,
    EntityState,
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
        fsm_override_store: dict | None = None,
    ) -> None:
        self.session = session
        self.graph = graph
        self.store = session_store
        self._log = logger_fn or (lambda p, m: None)
        self._cancel = cancel_event
        self._round_callback = round_callback
        self._resume_start_round = resume_start_round
        self._fsm_override_store = fsm_override_store if fsm_override_store is not None else {}
        # 量化模式状态（rule_engine 非空即量化）
        self._rule_engine: Any = None
        self._states: dict[str, Any] = {}
        self._enable_narrate: bool = True
        self._enable_multi_action: bool = False
        self._max_actions: int = 3
        self._goal_resolution: str = ""

    async def run(self) -> DeductionSession:
        import time as _time

        session_id = self.session.id
        _current_session.set(session_id)
        _total_start = _time.time()
        _phase_times: dict[str, float] = {}

        async def _timed_phase(name: str, fn):
            t0 = _time.time()
            await fn()
            dt = _time.time() - t0
            _phase_times[name] = dt
            self._log("orchestrator", f"阶段 {name} 耗时 {dt:.1f}s")

        try:
            if self._resume_start_round > 0:
                await self._resume_from_pause()
            else:
                for phase_name, phase_fn in [
                    ("ontology", self._phase1_ontology),
                    ("quantify", self._phase1_5_quantify),
                    ("graph", self._phase2_graph),
                    ("agents", self._phase3_agents),
                ]:
                    await _timed_phase(phase_name, phase_fn)
            await _timed_phase("simulation", self._phase4_simulation)
            await _timed_phase("report", self._phase5_report)

            _total = _time.time() - _total_start
            _sum_phases = sum(_phase_times.values())
            _detail = " | ".join(f"{k}={v:.1f}s" for k, v in _phase_times.items())
            if abs(_total - _sum_phases) > 5.0:
                _detail += f" | 阶段合计={_sum_phases:.1f}s"
            self._log("orchestrator", f"五阶段完成，总耗时 {_total:.1f}s | {_detail}")

            self.store.update(session_id, status=SessionStatus.COMPLETE.value,
                              phase=DeductionPhase.COMPLETE.value)
            self._clear_state_snapshot(session_id)
            if getattr(self, "_preprocessor", None) is not None:
                self._preprocessor.close()
        except _PhaseCancelledError:
            _total = _time.time() - _total_start
            self._log("orchestrator", f"推演已暂停（运行 {_total:.1f}s），进度已保存")
            self._save_pause_snapshot(session_id)
        except Exception as e:
            _total = _time.time() - _total_start
            logger.exception("[Deduction] Pipeline failed: %s", e)
            self.store.update(session_id, status=SessionStatus.FAILED.value,
                              error=str(e)[:500])
            self._log("orchestrator", f"推演失败（运行 {_total:.1f}s）: {e}")
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
        _current_phase.set("resume")
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

        # 4. 从 Kuzu 图恢复 Agent 列表（画像已持久化，免去重新调用 LLM）
        self._agents = []
        try:
            stored_agents = self.graph.get_agents()
        except Exception as e:
            logger.warning("[Orchestrator] 读取已存智能体失败: %s", e)
            stored_agents = []
        if stored_agents:
            self._log("orchestrator", f"从图谱恢复 {len(stored_agents)} 个智能体（复用已存画像）")
            # 预先查询实体类型映射，供恢复时填充 entity_type
            type_map: dict[str, str] = {}
            try:
                erows = self.graph.query(
                    f"MATCH (e:{self.graph.NODE_TABLE}) RETURN e.id, e.type")
                type_map = {r[0]: r[1] for r in erows if r[0] and r[1]}
            except Exception:
                logger.warning("[Orchestrator] 恢复智能体时实体类型查询失败")
                pass
            for a in stored_agents:
                try:
                    goals = _json.loads(a.get("goals") or "[]")
                except (ValueError, TypeError):
                    goals = []
                self._agents.append(DeductionAgentProfile(
                    entity_id=a.get("id", ""),
                    name=a.get("name", ""),
                    persona=a.get("persona", ""),
                    background=a.get("background", ""),
                    goals=goals if isinstance(goals, list) else [],
                    entity_type=type_map.get(a.get("id", ""), ""),
                ))
            self.session.agent_count = len(self._agents)
        else:
            # 兜底: 图中无 Agent 节点(旧会话/损坏)时才重建, 会重新调用 LLM
            self._log("orchestrator", "图谱中无已存智能体，重新生成...")
            try:
                from .agent_factory import create_agents_from_graph
                agents = await create_agents_from_graph(
                    graph=self.graph,
                    source_material=self.session.source_material,
                    log_fn=self._log,
                    preprocessor=self._preprocessor,
                    domain=getattr(self._rule_engine, "domain", "") if self._rule_engine else "",
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
        _current_phase.set("ontology")
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
        _current_phase.set("quantify")
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
        _current_phase.set("graph")
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
        preprocessor.set_progress_callback(
            lambda cur, tot: self._log("preprocess",
                                       f"LLM 实体发现进度 {cur}/{tot}")
        )
        await preprocessor.preprocess(self.session.source_material)
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
        if e_count == 0:
            self._log("graph", "⚠️ 图谱未提取到任何实体——种子材料可能过短或格式无法解析")
        self.store.update(self.session.id, entity_count=e_count, relation_count=r_count,
                          status=SessionStatus.AGENTS_RUNNING.value,
                          phase=DeductionPhase.AGENTS.value)

        # Phase 2.5: Intelligence sorting — classify entities before agent creation
        self._intel_list: list[dict] = []
        # 叙事模式保留全量实体，跳过 IntelSorter 过滤
        if self._rule_engine is not None:
            try:
                from strategy_forge.engine.intel_sorter import sort_entities
                from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
                entity_names = list(self.graph.get_entity_names())
                domain = self._rule_engine.domain
                from strategy_forge.core.rule_templates import get_domain_prompt
                dr = get_domain_prompt(domain, "intel_extra_rules")
                if dr:
                    self._log("graph", f"领域 {domain} 自定义筛选规则已启用（{len(dr)} chars）")
                self._intel_list = await sort_entities(
                    self.session.source_material, entity_names, LLMClient(),
                    domain=domain)
                if self._intel_list:
                    active = sum(1 for e in self._intel_list if e.get("include_in_simulation"))
                    passive = len(self._intel_list) - active
                    self._log("graph", f"情报整理: {len(self._intel_list)} 实体 → {active} 核心博弈者 + {passive} 非战略实体")
                    self._merge_sorter_aliases()
            except Exception as e:
                logger.warning("[Orchestrator] 情报整理失败，使用全部实体: %s", e)
                self._intel_list = []
        else:
            # 叙事模式：用 NarrativeSorter (LLM) 做故事角色分类
            self._log("graph", "叙事模式：故事编辑分类实体...")
            try:
                from strategy_forge.engine.narrative_sorter import sort_narrative_entities
                from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
                entity_names = list(self.graph.get_entity_names())
                # 查询每个实体的图类型（Person/Location/Organization/Position 等），
                # 传入 sorter 供 LLM 判断角色 vs 背景（地点/机构/职衔应排除）
                entity_types: dict[str, str] = {}
                try:
                    rows = self.graph._conn.execute(
                        f"MATCH (e:{self.graph.NODE_TABLE}) RETURN e.name, e.type")
                    while rows.has_next():
                        r = rows.get_next()
                        if r[0] and r[1]:
                            entity_types[str(r[0])] = str(r[1])
                except Exception:
                    logger.warning("[Orchestrator] 叙事模式实体类型查询失败")
                    pass
                # 传递预处理器的实体统计和分块数据，供超长文本构建全局视图
                pp = getattr(self, "_preprocessor", None)
                pp_result = pp.result if pp else None
                entity_freq = getattr(pp_result, "entity_frequencies", None) if pp_result else None
                entity_cov = getattr(pp_result, "entity_chunk_coverage", None) if pp_result else None
                chunk_texts = [c.content for c in pp_result.chunks] if pp_result and pp_result.chunks else None
                self._intel_list = await sort_narrative_entities(
                    self.session.source_material, entity_names, LLMClient(),
                    entity_frequencies=entity_freq,
                    entity_chunk_coverage=entity_cov,
                    chunk_texts=chunk_texts,
                    entity_types=entity_types,
                )
                if self._intel_list:
                    active = sum(1 for e in self._intel_list if e.get("include_in_simulation"))
                    passive = len(self._intel_list) - active
                    self._log("graph", f"故事编辑: {len(self._intel_list)} 实体 → {active} 角色 + {passive} 背景")
                    self._merge_sorter_aliases()
            except Exception as e:
                logger.warning("[Orchestrator] 叙事模式分类失败: %s", e)
                self._intel_list = []

    def _merge_sorter_aliases(self) -> None:
        merged_total = 0
        for e in self._intel_list:
            aliases = e.get("aliases") or []
            if aliases:
                try:
                    merged_total += self.graph.merge_alias_nodes(e.get("name", ""), aliases)
                except Exception as ex:
                    logger.debug("[Orchestrator] 别名合并失败 %s: %s", e.get("name"), ex)
        if merged_total:
            e_count2 = self.graph.count_entities()
            self.store.update(self.session.id, entity_count=e_count2)
            self._log("graph", f"别名合并: 并入 {merged_total} 个别名节点，实体数→{e_count2}")
            # 刷新 intel_list：合并后某些名称可能消失，用图中实际存在的名称替换
            graph_names = set(self.graph.get_entity_names())
            for e in self._intel_list:
                if e.get("name", "") in graph_names:
                    continue
                for alias in e.get("aliases", []):
                    if alias in graph_names:
                        old = e["name"]
                        e["name"] = alias
                        aliases = e.get("aliases", [])
                        if isinstance(aliases, list):
                            aliases = [a for a in aliases if a != alias] + [old]
                        e["aliases"] = aliases
                        break

    async def _phase3_agents(self) -> None:
        _current_phase.set("agents")
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
            intel_list=getattr(self, "_intel_list", None) or None,
            domain=getattr(self._rule_engine, "domain", "") if self._rule_engine else "",
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
                    logger.warning("[Orchestrator] 不可变目标注入 LanceDB 失败")
                    pass
            self._log("agents", f"已注入 {len(pre_goals)} 个不可变目标到 LanceDB")

        self._log("agents", f"智能体工厂完成: {len(agents)} 个智能体生成")
        self.store.update(self.session.id, agent_count=len(agents),
                          status=SessionStatus.SIMULATING.value,
                          phase=DeductionPhase.SIMULATION.value)

    async def _phase4_simulation(self) -> None:
        _current_phase.set("simulation")
        self._check_cancel()
        total_rounds = self.session.total_rounds
        re_engine = self._rule_engine
        states: dict[str, Any] = {}
        if re_engine is not None:
            # Seed data extraction (always on for quantified mode, snapshot-reuse on resume)
            seed_metrics: dict[str, dict[str, float]] = {}
            cfg_data = self.store.get(self.session.id) or {}
            cfg = (cfg_data.get("config_json", {}) or {})
            if isinstance(cfg, str):
                cfg = _json.loads(cfg)
            seed_metrics = cfg.get("seed_metrics", {})
            if seed_metrics:
                self._log("simulation", f"种子数据从快照恢复: {len(seed_metrics)} 个实体")
            else:
                try:
                    from strategy_forge.engine.seed_extractor import extract_seed_metrics
                    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
                    seed_metrics = await extract_seed_metrics(
                        self.session.source_material, re_engine.metrics(), LLMClient())
                    if seed_metrics:
                        detail = ", ".join(f"{n}({len(m)}指标)" for n, m in seed_metrics.items())
                        self._log("simulation", f"种子数据提取: {len(seed_metrics)} 个实体 — {detail}")
                        cfg["seed_metrics"] = seed_metrics
                        self.store.update(self.session.id,
                                          config_json=_json.dumps(cfg, ensure_ascii=False))
                except Exception as e:
                    self._log("simulation", f"种子数据提取失败，使用规则包默认值: {e}")

            for a in self._agents:
                init = dict(re_engine.pack["initial_metrics"])
                overrides = seed_metrics.get(a.name, {})
                for m, v in overrides.items():
                    if m in init:
                        init[m] = float(v)
                states[a.entity_id] = EntityState(
                    id=a.entity_id, name=a.name, domain=re_engine.domain,
                    metrics=init, history=[])
            self._states = states
            self._log("simulation",
                      f"阶段4: 量化并行模拟开始 ({total_rounds} 轮, {len(states)} 个量化实体, "
                      f"领域={re_engine.domain})")
        else:
            self._log("simulation", f"阶段4: 并行模拟开始 ({total_rounds} 轮)")

        from .simulator import SimulationEngine

        # 构建算法模块链（ODE + Physics）
        algorithm_modules = []
        if re_engine is not None:
            from strategy_forge.algorithms.module_utils import build_module_chain
            algorithm_modules = build_module_chain(re_engine)
            self._log("simulation",
                      f"算法模块加载: {', '.join(m.name for m in algorithm_modules)}")

        # 模拟决策温度：接入配置（设置界面/FORGE_LLM_TEMPERATURE 真实生效），钳制到 [0,1.5]
        from strategy_forge.core.providers import registry as _reg
        try:
            _sim_temp = max(0.0, min(1.5, float(_reg.llm_temperature)))
        except Exception:
            _sim_temp = 0.6

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
            max_concurrent=getattr(self, "_max_concurrent", None),
            temperature=_sim_temp,
            seed=abs(hash(self.session.id)) % (2**31),
            algorithm_modules=algorithm_modules,
            fsm_override_store=self._fsm_override_store,
        )

        rounds: list[SimulationRound] = []
        start_rnd = self._resume_start_round + 1
        pre_goals = getattr(self, "_pre_goals", []) or []
        # 设定轮数是上限而非必跑条件：每 conv_interval 轮判定一次目标是否已显现，
        # 已显现则提前收束，不再继续跑满剩余轮数
        conv_interval = max(3, min(10, total_rounds // 10))
        for rnd in range(start_rnd, total_rounds + 1):
            if self._cancel is not None and self._cancel.is_set():
                self._log("simulation", "推演收到取消信号，提前终止")
                raise _PhaseCancelledError()
            _current_round.set(rnd)
            self._log("simulation", f"  第 {rnd}/{total_rounds} 轮开始")
            result = await engine.run_round(rnd)
            rounds.append(result)
            self.session.current_round = rnd
            self.store.update(self.session.id, current_round=rnd)
            self._log("simulation", f"  第 {rnd} 轮完成: {len(result.actions)} 个动作")
            if self._round_callback:
                snapshot = result.state_delta.get("snapshot") if hasattr(result, "state_delta") else None
                self._round_callback(rnd, total_rounds, snapshot)
            # Persist token stats incrementally (survives pause/interrupt)
            from strategy_forge.core.token_counter import accumulator
            stats = accumulator.get_session_stats(self.session.id)
            if stats:
                self.store.update(self.session.id,
                                  token_json=_json.dumps(stats, ensure_ascii=False))
            # ── 目标收敛检查：每 conv_interval 轮判定核心问题是否已可明确回答 ──
            if (pre_goals and rnd < total_rounds and rnd >= conv_interval
                    and rnd % conv_interval == 0):
                resolved, verdict = await self._check_goal_convergence(rounds)
                if resolved:
                    self._goal_resolution = verdict
                    self._log("simulation",
                              f"目标收敛检查(第{rnd}轮): 核心问题已有明确答案，提前收束 — {verdict}")
                    break
                if verdict:
                    self._log("simulation", f"目标收敛检查(第{rnd}轮): 尚未收敛 — {verdict}")
                    # 导演反馈闭环：把"缺失的决定性事件"注入后续轮次，
                    # 推动局势向可判定演进（只提示缺什么，不指定谁该赢）
                    pp = getattr(self, "_preprocessor", None)
                    if pp is not None:
                        try:
                            goals_text = "；".join(pre_goals)
                            pp.add_event_memory(
                                content=(f"【导演提示·第{rnd}轮】核心问题（{goals_text}）尚无法判定，"
                                         f"缺失的决定性进展：{verdict}。后续行动应实质性推动此类摊牌"
                                         f"事件发生——立场与方式由你根据自身利益决定，不要原地观望"
                                         f"或重复既有行为。"),
                                agent_id="director", round_number=rnd,
                                event_type="user_intervention", priority=0.95,
                            )
                            self._log("simulation", f"导演反馈已注入(第{rnd}轮): 推动缺失进展 — {verdict[:60]}")
                        except Exception as e:
                            logger.debug("[Orchestrator] 导演反馈注入失败(忽略): %s", e)

        self._simulation_rounds = rounds
        self._personality_log = list(getattr(engine, "_personality_log", []) or [])
        self._log("simulation", f"模拟完成: {len(rounds)} 轮, "
                  f"{sum(len(r.actions) for r in rounds)} 个总动作")
        self.store.update(self.session.id,
                          status=SessionStatus.REPORTING.value,
                          phase=DeductionPhase.REPORT.value)

    async def _check_goal_convergence(self, rounds: list[SimulationRound]) -> tuple[bool, str]:
        """用 LLM 判定推演核心问题是否已可基于事件给出明确答案。

        返回 (resolved, verdict)。resolved=True 时 verdict 为答案+依据；
        False 时 verdict 为缺失的决定性条件（可为空）。判定失败一律视为未收敛。
        """
        from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
        from strategy_forge.core.llm_client import Message

        from ._utils import extract_json, extract_text

        name_map = {a.entity_id: a.name for a in getattr(self, "_agents", [])}
        events: list[str] = []
        for r in rounds[-12:]:
            for act in r.actions:
                who = name_map.get(act.agent_id, act.agent_id[:8])
                events.append(f"[轮{r.round_number}] {who}: {act.content[:60]}")
        if not events:
            return False, ""

        goals_text = "；".join(getattr(self, "_pre_goals", []))
        prompt = (
            "你是推演裁判。基于近期事件，判断推演核心问题是否已经可以给出明确、可辩护的答案。\n"
            "只有当事件序列显示出决定性的力量对比变化（如某方掌握了压倒性筹码、对手被淘汰或屈服）时才判定为已收敛；"
            "局势仍胶着、各方仅在试探或表态时判定为未收敛。\n\n"
            f"## 核心问题\n{goals_text}\n\n"
            "## 近期事件\n" + "\n".join(events[-40:]) + "\n\n"
            '## 输出 JSON（纯 JSON）\n'
            '{"resolved": true或false, "verdict": "若已收敛：明确答案+关键依据(80字内)；'
            '若未收敛：还缺什么决定性事件(40字内)"}'
        )
        try:
            client = LLMClient()
            resp = await client.chat(
                [Message(role="user", content=prompt)],
                system="你是推演收敛判定裁判，只输出 JSON。",
                temperature=0.2, max_tokens=300)
            data = extract_json(extract_text(resp))
            if isinstance(data, dict):
                return bool(data.get("resolved")), str(data.get("verdict", ""))[:200]
        except Exception as e:
            logger.debug("[Orchestrator] 目标收敛判定失败(忽略): %s", e)
        return False, ""

    async def _phase5_report(self) -> None:
        _current_phase.set("report")
        self._log("report", "阶段5: 报告生成开始")

        from .reporter import generate_report
        report = await generate_report(
            session=self.session,
            graph=self.graph,
            rounds=getattr(self, "_simulation_rounds", []),
            log_fn=self._log,
            preprocessor=getattr(self, "_preprocessor", None),
            pre_goals=getattr(self, "_pre_goals", []),
            states=getattr(self, "_states", None),
            thresholds=self._rule_engine.pack.get("thresholds", {}) if self._rule_engine else None,
            goal_resolution=getattr(self, "_goal_resolution", ""),
            personality_log=getattr(self, "_personality_log", []),
        )
        self.session.report = report

        report_payload = {
            "summary": report.summary,
            "key_events": report.key_events,
            "risk_alerts": report.risk_alerts,
            "recommendations": report.recommendations,
            "causal_summary": report.causal_summary,
            "stage_narratives": report.stage_narratives,
            "deviation_analysis": report.deviation_analysis,
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
                          report_json=_json.dumps(report_payload, ensure_ascii=False))
        self._log("report", f"报告生成完成: {report.summary}")

    def get_realtime_round(self) -> SimulationRound | None:
        rounds = getattr(self, "_simulation_rounds", None)
        if rounds and self.session.current_round > 0:
            idx = self.session.current_round - 1
            if idx < len(rounds):
                return rounds[idx]
        return None

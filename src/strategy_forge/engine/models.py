"""Deduction Engine — 推演引擎数据模型"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DeductionPhase(str, Enum):
    ONTOLOGY = "ontology"
    GRAPH = "graph"
    AGENTS = "agents"
    SIMULATION = "simulation"
    REPORT = "report"
    COMPLETE = "complete"
    FAILED = "failed"


class SessionStatus(str, Enum):
    CREATED = "created"
    ONTOLOGY_RUNNING = "ontology_running"
    GRAPH_RUNNING = "graph_running"
    AGENTS_RUNNING = "agents_running"
    SIMULATING = "simulating"
    REPORTING = "reporting"
    COMPLETE = "complete"
    PAUSED = "paused"
    FAILED = "failed"
    OPTIMIZING = "optimizing"


@dataclass
class EntityTypeDef:
    name: str
    description: str = ""
    properties: list[str] = field(default_factory=list)


@dataclass
class RelationTypeDef:
    name: str
    description: str = ""
    from_type: str = ""
    to_type: str = ""


@dataclass
class Ontology:
    entities: list[EntityTypeDef] = field(default_factory=list)
    relations: list[RelationTypeDef] = field(default_factory=list)


@dataclass
class GraphEntity:
    id: str
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    evidence: str = ""


@dataclass
class DeductionAgentProfile:
    entity_id: str
    name: str
    persona: str
    background: str = ""
    goals: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)
    system_prompt_extra: str = ""
    entity_type: str = ""


@dataclass
class SimulationAction:
    agent_id: str
    action_type: str  # "post", "reply", "decision", "interact", "observe"
    target_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class SimulationRound:
    round_number: int
    actions: list[SimulationAction] = field(default_factory=list)
    state_delta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeductionReport:
    session_id: str
    summary: str = ""
    key_events: list[dict[str, Any]] = field(default_factory=list)
    agent_trajectories: dict[str, list[str]] = field(default_factory=dict)
    risk_alerts: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    raw_graph_stats: dict[str, Any] = field(default_factory=dict)
    causal_summary: list[str] = field(default_factory=list)
    stage_narratives: list[dict[str, Any]] = field(default_factory=list)
    deviation_analysis: list[dict[str, Any]] = field(default_factory=list)
    conclusion: str = ""


@dataclass
class DeductionSession:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    source_material: str = ""
    status: SessionStatus = SessionStatus.CREATED
    phase: DeductionPhase = DeductionPhase.ONTOLOGY
    ontology: Ontology | None = None
    entity_count: int = 0
    relation_count: int = 0
    agent_count: int = 0
    current_round: int = 0
    total_rounds: int = 10
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str = ""
    report: DeductionReport | None = None


@dataclass
class EntityState:
    """通用可量化实体状态（规则包驱动，不绑定具体领域）。

    metrics / 阈值 / 初值 / 取值范围均来自规则包，新增领域只需提供规则包，
    无需修改本类（取代按领域写 Python 子类的做法）。
    """
    id: str
    name: str
    domain: str = "generic"
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    _pending_delays: list[dict[str, Any]] = field(default_factory=list)

    def schedule_delays(self, current_round: int, delay_rounds: int,
                        effects: dict[str, float]) -> None:
        """调度延迟效应：在 current_round + delay_rounds 轮结算。"""
        self._pending_delays.append({
            "apply_round": current_round + int(delay_rounds),
            "effects": dict(effects),
        })

    def resolve_delays(self, current_round: int) -> dict[str, float]:
        """返回本轮到期的延迟效应累加结果，移除已结算项。"""
        result: dict[str, float] = {}
        remaining: list[dict[str, Any]] = []
        for item in self._pending_delays:
            if item["apply_round"] <= current_round:
                for k, v in item["effects"].items():
                    result[k] = result.get(k, 0.0) + v
            else:
                remaining.append(item)
        self._pending_delays = remaining
        return result

    def get_metric(self, name: str) -> float:
        return self.metrics.get(name, 0.0)

    def apply_delta(self, name: str, delta: float, lo: float = 0.0, hi: float = 100.0,
                    round_number: int = 0) -> float:
        old = self.metrics.get(name, 0.0)
        new = max(lo, min(hi, old + delta))
        self.metrics[name] = new
        self.history.append({"round": round_number, "metric": name,
                             "old": round(old, 2), "delta": round(delta, 2),
                             "new": round(new, 2)})
        if len(self.history) > 500:
            self.history = self.history[-500:]
        return new

    def apply_deltas(self, deltas: dict[str, float], round_number: int = 0,
                     ranges: dict[str, Any] | None = None) -> None:
        ranges = ranges or {}
        for name, delta in deltas.items():
            rng = ranges.get(name, [0.0, 100.0])
            lo, hi = float(rng[0]), float(rng[1])
            self.apply_delta(name, delta, lo, hi, round_number)

    def is_alive(self, thresholds: dict[str, float]) -> bool:
        """任一受阈值约束的指标低于(含等于)其阈值则判定为出局。"""
        for name, floor in (thresholds or {}).items():
            if self.metrics.get(name, 0.0) <= float(floor):
                return False
        return True

    def to_prompt_context(self) -> str:
        return f"{self.name}({self.domain}): " + ", ".join(
            f"{k}={v:.1f}" for k, v in self.metrics.items())

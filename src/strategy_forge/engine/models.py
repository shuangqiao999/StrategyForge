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

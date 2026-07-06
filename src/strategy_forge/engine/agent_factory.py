"""Phase 3: Agent Factory — deep persona generation from graph + LanceDB retrieval."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Callable
from string import Template
from typing import Any

from strategy_forge.core.llm_client import LLMConnectionError
from strategy_forge.storage.graph_store import DeductionGraphStore

from ._utils import extract_text
from .models import DeductionAgentProfile
from .preprocessor import DeductionPreprocessor

logger = logging.getLogger(__name__)


_PERSONA_PROMPT = """基于以下实体信息和原文背景，为该人物生成一个独立人格档案。返回 JSON。

## 来自用户的特殊期望（必须严肃考虑）
$user_expectations

## 实体信息
- 名称: $name
- 类型: $type
- 描述: $description
- 战略定位: $role
- 所属组织: $parent_info
- 下属机构: $sub_info

## 全书关键片段（LanceDB 语义检索）
$context

## 高频共现关键词标签
$keywords

## 输出 JSON — 必须是纯 JSON 对象
{
  "persona": "详细的人格描述 (80-150字), 包括性格特征、价值观、行为模式、人物弧光演变",
  "background": "背景故事 (80-150字), 包括关键经历、社会关系、动机、性格变迁",
  "goals": ["目标1", "目标2", "目标3"]
}

【重要】只返回纯JSON对象。不要```json代码块。不要任何解释文字。"""

_PERSONA_PROMPT_FALLBACK = """基于以下实体信息和原文背景，为该人物生成一个独立人格档案。返回 JSON。

## 来自用户的特殊期望（必须严肃考虑）
$user_expectations

## 实体信息
- 名称: $name
- 类型: $type
- 描述: $description
- 战略定位: $role
- 所属组织: $parent_info
- 下属机构: $sub_info

## 原文背景
$context

## 输出 JSON — 必须是纯 JSON 对象
{
  "persona": "详细的人格描述 (50-100字), 包括性格特征、价值观、行为模式",
  "background": "背景故事 (50-100字), 包括关键经历、社会关系、动机",
  "goals": ["目标1", "目标2"]
}

【重要】只返回纯JSON对象。不要```json代码块。不要任何解释文字。"""


async def create_agents_from_graph(
    graph: DeductionGraphStore,
    source_material: str,
    log_fn: Callable[[str, str], None],
    preprocessor: DeductionPreprocessor | None = None,
    pre_interventions: list[str] | None = None,
    chat_fn: Any = None,
    intel_list: list[dict] | None = None,
) -> list[DeductionAgentProfile]:
    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    from strategy_forge.core.llm_client import Message

    # Collect decision-making entities: Person + Organization (nations, armies, alliances)
    persons = graph.get_entities_by_type("Person")
    if not persons:
        persons = []
    # Add Organization-type entities (nations, alliances, institutions, armies)
    orgs = graph.get_entities_by_type("Organization")
    if orgs:
        # Merge: deduplicate by name (an entity may be both Person and Organization)
        seen_names = {p["name"] for p in persons}
        for o in orgs:
            if o["name"] not in seen_names:
                persons.append(o)
                seen_names.add(o["name"])
        log_fn("agents", f"加入 {len(orgs)} 个组织实体作为智能体候选")

    # Fallback: if still no entities, take all
    if not persons:
        result = graph._conn.execute(
            f"MATCH (e:{graph.NODE_TABLE}) RETURN e.id, e.name, e.type, e.description"
        )
        persons = []
        while result.has_next():
            r = result.get_next()
            persons.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3]})

    # Deduplicate using alias map from preprocessor (no substring matching)
    if len(persons) > 1:
        alias_to_std: dict[str, str] = {}
        if preprocessor and preprocessor.result:
            for std, aliases in preprocessor.result.entity_aliases.items():
                alias_to_std[std] = std
                for a in aliases:
                    alias_to_std[a] = std
        # Fold intel_sorter semantic aliases (中英文名/简称) into the dedup map,
        # so e.g. "乌军" 与 "乌克兰军队"、"OECD" 与 "经合组织" 归并为同一规范名
        for e in (intel_list or []):
            canon = (e.get("name") or "").strip()
            if not canon:
                continue
            for a in e.get("aliases", []):
                a = (a or "").strip()
                if a:
                    alias_to_std[a] = canon
        seen: set[str] = set()
        deduped: list[dict] = []
        for p in persons:
            name = p.get("name", "")
            std_name = alias_to_std.get(name, name)
            if std_name in seen:
                continue
            seen.add(std_name)
            if std_name != name:
                p["name"] = std_name
            deduped.append(p)
        if len(deduped) < len(persons):
            log_fn("agents", f"实体去重: {len(persons)} → {len(deduped)}")
        persons = deduped

    # Intelligence sorting: filter non-strategic entities
    if intel_list:
        intel_map = {e["name"]: e for e in intel_list if e.get("name")}
        active_names = {e["name"] for e in intel_list if e.get("include_in_simulation")}
        before = len(persons)
        persons = [p for p in persons if intel_map.get(p["name"], {}).get("include_in_simulation", True)]
        if len(persons) < before:
            log_fn("agents", f"情报过滤: {before} → {len(persons)} 个智能体（排除非战略实体）")
    else:
        intel_map = {}

    max_agents = min(len(persons), config.deduction_max_agents)
    log_fn("agents", f"从 {len(persons)} 个实体中生成最多 {max_agents} 个智能体")

    client = LLMClient()
    agents: list[DeductionAgentProfile] = []

    expected_keys = {"persona", "background", "goals"}

    sem = asyncio.Semaphore(max(1, config.deduction_max_concurrent))

    def _fallback(nm: str) -> dict:
        return {"persona": f"{nm}是一个参与事件的独立个体",
                "background": "来自原文背景", "goals": ["参与互动", "表达观点"]}

    def _build_prompt(person: dict, person_name: str, fragments: list[str] | None) -> str:
        ue = "\n".join(f"- {x}" for x in (pre_interventions or [])) or "无特殊期望"
        im = intel_map.get(person_name, {})
        role = im.get("role", "独立博弈者")
        parent_info = str(im.get("parent") or "无")
        sub_info = ", ".join(str(s) for s in im.get("sub_entities", [])) or "无"
        if fragments:
            from strategy_forge.core.tokenizer import compress_to_keywords
            full_context = "\n---\n".join(fragments)
            keywords = compress_to_keywords(full_context, top_k=10)
            return Template(_PERSONA_PROMPT).substitute(
                name=person_name, type=person.get("type", "Person"),
                description=person.get("description", ""), role=role,
                parent_info=parent_info, sub_info=sub_info,
                context=full_context[:8000],
                keywords=", ".join(keywords) if keywords else "无",
                user_expectations=ue)
        return Template(_PERSONA_PROMPT_FALLBACK).substitute(
            name=person_name, type=person.get("type", "Person"),
            description=person.get("description", ""), role=role,
            parent_info=parent_info, sub_info=sub_info,
            context=source_material[:2000], user_expectations=ue)

    async def gen_one(i: int, person: dict) -> dict:
        person_name = person.get("name", f"Agent-{i}")
        # 召回卸载到线程池，避免阻塞事件循环（与 simulator._recall 一致）
        fragments = None
        if preprocessor and preprocessor.result:
            try:
                fragments = await asyncio.to_thread(
                    preprocessor.retrieve_for_entity, person_name,
                    max(config.deduction_retrieve_top_k, 10), {person_name})
            except Exception as e:
                logger.debug("[Deduction] persona retrieve failed for %s: %s", person_name, e)
        prompt = _build_prompt(person, person_name, fragments)
        system = "You are a JSON-only character profile generator. Output ONLY a valid JSON object. NO markdown, NO explanations."
        messages = [Message(role="user", content=prompt)]
        async with sem:  # 并发上限 = FORGE_MAX_CONCURRENT
            try:
                if chat_fn is not None:
                    content = await asyncio.to_thread(chat_fn, messages, system, 0.7)
                else:
                    response = await client.chat(messages, system=system, temperature=0.7)
                    content = extract_text(response)
                profile_data = _parse_persona_json(content)
                if not isinstance(profile_data, dict) or not expected_keys.intersection(profile_data):
                    profile_data = _fallback(person_name)
            except LLMConnectionError:
                raise  # 连接故障直接传播
            except Exception as e:
                logger.warning("[Deduction] Agent persona gen failed for %s: %s", person_name, e)
                profile_data = _fallback(person_name)
        return {"person": person, "name": person_name, "data": profile_data}

    # 并发生成人设（上限 = FORGE_MAX_CONCURRENT），随后按原顺序构造+写 Kuzu 以保持确定性
    results = await asyncio.gather(
        *(gen_one(i, p) for i, p in enumerate(persons[:max_agents])), return_exceptions=True)
    conn_fails = [r for r in results if isinstance(r, LLMConnectionError)]
    if conn_fails:
        raise conn_fails[0]

    for i, r in enumerate(results):
        person, person_name, profile_data = r["person"], r["name"], r["data"]
        agent_profile = DeductionAgentProfile(
            entity_id=person.get("id", uuid.uuid4().hex[:8]),
            name=person_name,
            persona=profile_data.get("persona", ""),
            background=profile_data.get("background", ""),
            goals=profile_data.get("goals", []),
        )
        agents.append(agent_profile)

        # Store agent node in Kuzu (Agent 节点经 ACTED 时间线查询被读取)
        graph.upsert_agent_node(
            agent_profile.entity_id, agent_profile.name,
            agent_profile.persona, agent_profile.background,
            json.dumps(agent_profile.goals, ensure_ascii=False),
        )

        log_fn("agents", f"  [{i+1}/{max_agents}] {person_name}: {agent_profile.persona[:80]}...")

    return agents


def _parse_persona_json(raw: str) -> dict[str, Any]:
    data = _try_extract_json(raw)
    if not isinstance(data, dict):
        # LLM returned array — take first element
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        else:
            return {}
    return {
        "persona": data.get("persona", ""),
        "background": data.get("background", ""),
        "goals": data.get("goals", []),
    }


def _try_extract_json(raw: str):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', raw)
    cleaned = re.sub(r'\n?```', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    for pat in (r'\{[\s\S]*\}', r'\[[\s\S]*\]'):
        m = re.search(pat, cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                continue
    if raw.strip().startswith('"'):
        try:
            return json.loads("{" + raw.strip() + "}")
        except (json.JSONDecodeError, ValueError):
            pass
    return {} if raw.startswith("{") else []

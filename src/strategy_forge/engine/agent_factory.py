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


_PERSONA_PROMPT = """基于以下实体信息和原文背景，为该$domain_role生成一个独立人格档案。返回 JSON。

## 来自用户的特殊期望（必须严肃考虑）
$user_expectations

## 实体信息
- 名称: $name
- 类型: $type
- 描述: $description
- 战略定位: $role
- 所属组织: $parent_info
- 下属机构: $sub_info
- 实体统计: $entity_stats

## 原文关键片段（LanceDB 语义检索）
$context

## 高频共现关键词标签
$keywords

## 输出 JSON — 必须是纯 JSON 对象
{
  "persona": "详细的人格描述 (80-150字), 包括性格特征、价值观、行为模式、行为演化趋势",
  "background": "背景故事 (80-150字), 包括关键经历、社会关系、动机、性格变迁",
  "goals": ["目标1", "目标2", "目标3"]
}

## persona 质量标准（参考）
好的 persona（具体、有矛盾、可推演行为）：
  "偏执而精明的技术官僚，坚信数据高于直觉。在公开场合沉默寡言，但内部会议中会逐一推翻他人的假设。对失败的容忍度极低，曾因一次供应链延误解雇整个团队。表面追求效率至上，骨子里是对失控的恐惧。"
不好的 persona（抽象、无辨识度，不推荐）：
  "他是一位优秀的领导者，善于团队合作，重视技术创新，致力于推动组织发展。"

【重要】只返回纯JSON对象。不要```json代码块。不要任何解释文字。"""

_PERSONA_PROMPT_FALLBACK = """基于以下实体信息和原文背景，为该$domain_role生成一个独立人格档案。返回 JSON。

## 来自用户的特殊期望（必须严肃考虑）
$user_expectations

## 实体信息
- 名称: $name
- 类型: $type
- 描述: $description
- 战略定位: $role
- 所属组织: $parent_info
- 下属机构: $sub_info
- 实体统计: $entity_stats

## 原文背景
$context

## 输出 JSON — 必须是纯 JSON 对象
{
  "persona": "详细的人格描述 (50-100字), 包括性格特征、价值观、行为模式",
  "background": "背景故事 (50-100字), 包括关键经历、社会关系、动机",
  "goals": ["目标1", "目标2"]
}

## persona 质量标准
好的 persona（具体、有辨识度、可推演行为）：
  "极端务实的成本控制者，将供应链安全视为信仰。因一次芯片断供导致生产线停滞72小时，此后对供应商采取'三择一'策略——任何关键部件必须同时维护至少三个来源。公开场合低调寡言，私下谈判时极具攻击性。"
不好的 persona（模板化、无辨识度，不推荐）：
  "他是一位优秀的领导者，善于团队合作，重视技术创新，致力于推动组织发展。"

【重要】只返回纯JSON对象。不要```json代码块。不要任何解释文字。"""


async def create_agents_from_graph(
    graph: DeductionGraphStore,
    source_material: str,
    log_fn: Callable[[str, str], None],
    preprocessor: DeductionPreprocessor | None = None,
    pre_interventions: list[str] | None = None,
    chat_fn: Any = None,
    intel_list: list[dict] | None = None,
    domain: str = "",
    entity_registry: Any = None,
) -> list[DeductionAgentProfile]:
    from strategy_forge.core.config import config
    from strategy_forge.core.providers import registry as _reg
    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    from strategy_forge.core.llm_client import Message

    # ── 如果传入了 EntityRegistry，直接从中读取 ——
    if entity_registry is not None:
        kept_entities = entity_registry.get_kept()
        log_fn("agents", f"从注册中心读取 {len(kept_entities)} 个博弈实体（共 {entity_registry.total} 个）")
        log_fn("agents", entity_registry.summary()[:200])

        # 构建统一的 persons 列表
        persons: list[dict] = [
            {"id": e.id, "name": e.name, "type": e.type, "description": ""}
            for e in kept_entities
        ]
        # 构建 persona 用的 intel_map
        intel_map: dict[str, dict] = {}
        if intel_list:
            for e in intel_list:
                nm = (e.get("name") or "").strip()
                if nm:
                    intel_map[nm] = e
                for a in e.get("aliases", []):
                    a = str(a).strip()
                    if a and a not in intel_map:
                        intel_map[a] = e

        # freq_map
        freq_map: dict[str, int] = {e.name: e.freq for e in kept_entities}
        if preprocessor and preprocessor.result:
            fm = getattr(preprocessor.result, "entity_frequencies", {}) or {}
            freq_map = {**fm, **freq_map}

        max_agents = len(persons)
        log_fn("agents", f"从 {max_agents} 个注册实体中生成最多 {max_agents} 个智能体")



    client = LLMClient()
    agents: list[DeductionAgentProfile] = []

    expected_keys = {"persona", "background", "goals"}

    _COMPANY_TYPES = {"Company", "Enterprise", "Organization",
                      "公司", "企业", "组织", "机构"}
    def _entity_role(person: dict) -> str:
        from strategy_forge.core.rule_templates import get_domain_prompt
        etype = (person.get("type") or "").strip()
        if etype in _COMPANY_TYPES:
            return "科技企业或行业参与者"
        domain_role = get_domain_prompt(domain, "agent_domain_role")
        if domain_role:
            if domain not in getattr(_entity_role, "_logged", set()):
                logger.info("[AgentFactory] 领域 %s 角色指南已注入 persona prompt", domain)
                _entity_role._logged = getattr(_entity_role, "_logged", set()) | {domain}
        return domain_role or "独立博弈者"

    def _fallback(nm: str) -> dict:
        return {"persona": f"{nm}是一个参与事件的独立个体",
                "background": "来自原文背景", "goals": ["参与互动", "表达观点"]}

    def _build_prompt(person: dict, person_name: str, fragments: list[str] | None) -> str:
        ue = "\n".join(f"- {x}" for x in (pre_interventions or [])) or "无特殊期望"
        im = intel_map.get(person_name, {})
        role = im.get("role", "独立博弈者")
        parent_info = str(im.get("parent") or "无")
        sub_info = ", ".join(str(s) for s in im.get("sub_entities", [])) or "无"
        # 实体统计：频次+覆盖度帮助LLM区分差异，即使短文本LanceDB片段相同
        f = freq_map.get(person_name, 0) if freq_map.get(person_name, 0) > 0 else "?"
        c = preprocessor.result.entity_chunk_coverage.get(person_name, 0) if preprocessor and preprocessor.result else "?"
        entity_stats = f"频次={f}, 覆盖={c}个分块"
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
                user_expectations=ue, domain_role=_entity_role(person),
                entity_stats=entity_stats)
        return Template(_PERSONA_PROMPT_FALLBACK).substitute(
            name=person_name, type=person.get("type", "Person"),
            description=person.get("description", ""), role=role,
            parent_info=parent_info, sub_info=sub_info,
            context=source_material[:2000], user_expectations=ue, domain_role=_entity_role(person),
            entity_stats=entity_stats)

    async def gen_one(i: int, person: dict) -> dict:
        person_name = person.get("name", f"Agent-{i}")
        # 召回卸载到线程池，避免阻塞事件循环（与 simulator._recall 一致）
        fragments = None
        if preprocessor and preprocessor.result:
            try:
                fragments = await asyncio.to_thread(
                    preprocessor.retrieve_for_entity, person_name,
                    max(_reg.retrieve_top_k, 10), {person_name})
            except Exception as e:
                logger.debug("[Deduction] persona retrieve failed for %s: %s", person_name, e)
        prompt = _build_prompt(person, person_name, fragments)
        system = "你是角色档案生成专家，只输出 JSON 对象。不要 markdown，不要解释。"
        messages = [Message(role="user", content=prompt)]
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
        if isinstance(r, Exception):
            logger.warning("[Deduction] Agent persona gen failed (unexpected): %s", r)
            continue
        person, person_name, profile_data = r["person"], r["name"], r["data"]
        agent_profile = DeductionAgentProfile(
            entity_id=person.get("id", uuid.uuid4().hex[:8]),
            name=person_name,
            persona=profile_data.get("persona", ""),
            background=profile_data.get("background", ""),
            goals=profile_data.get("goals", []),
            entity_type=person.get("type", ""),
        )
        agents.append(agent_profile)

        # Store agent node in Kuzu (Agent 节点经 ACTED 时间线查询被读取)
        graph.upsert_agent_node(
            agent_profile.entity_id, agent_profile.name,
            agent_profile.persona, agent_profile.background,
            json.dumps(agent_profile.goals, ensure_ascii=False),
        )

        log_fn("agents", f"  [{i+1}/{max_agents}] {person_name}: {agent_profile.persona}")

    if not agents and results:
        raise RuntimeError(
            f"全部 {len(results)} 个智能体 persona 生成失败，请检查 LLM 连接或模型配置")

    return agents


def _parse_persona_json(raw: str) -> dict[str, Any]:
    from ._utils import extract_json
    data = extract_json(raw)
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



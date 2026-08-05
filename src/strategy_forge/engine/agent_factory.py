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

# 集体决策实体关键词（中英文通用）：政权/组织/国家等不应以个人视角描述
_COLLECTIVE_ENTITY_KW = {
    "国家", "政权", "朝廷", "帝国", "王国", "共和国", "联邦",
    "组织", "机构", "党派", "政党", "团体", "联盟", "协会",
    "公司", "企业", "集团", "财团", "商会",
    "军队", "军团", "舰队", "武装", "部队",
    "country", "nation", "state", "regime", "empire", "kingdom",
    "organization", "company", "enterprise", "corporation", "alliance",
    "army", "military", "fleet", "force", "party", "faction",
}

_COLLECTIVE_PERSPECTIVE = "4. 你是一个集体决策实体（政权/组织/国家/军队），不是个人。禁止以某个统治者或领导人的私人视角描述自己，必须体现该机构的集体利益和制度逻辑。"


def _load_methodology() -> dict:
    try:
        from pathlib import Path
        import yaml
        path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "methodology.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}

_METHODOLOGY = _load_methodology()


_PERSONA_PROMPT = """你是一个客观中立的角色档案生成专家。基于实体信息和原文事实，为该$domain_role生成一个独立人格档案。返回 JSON。

## 铁律（强制执行）
1. 每个实体的 persona 必须同时包含三个要素：
   A. 优势或能力 — 该实体在博弈中的独特筹码
   B. 矛盾或制约 — 该实体面临的内部困境、外部压力、两难选择（必须具体）
   C. 自利逻辑 — 该实体行为的底层利益动机
2. 不赋予任何实体道德高下标签 — 所有参与者都在追逐自身利益
3. 不使用意识形态化形容词 — 用行为描述代替价值判断
$entity_perspective

## 来自用户的特殊期望
$user_expectations

$role_inference

$role_examples

$domain_principles

## 实体信息
- 名称: $name
- 类型: $type
- 描述: $description
- 战略定位: $role
- 所属组织: $parent_info
- 下属机构: $sub_info
- 实体统计: $entity_stats

## 原文关键片段
$context

## 高频共现关键词
$keywords

## 输出 JSON
{
  "persona": "人格描述 (80-150字)。必须铁律ABC三项齐全",
  "background": "背景故事 (80-150字)",
  "goals": ["目标1", "目标2", "目标3"]
}

## 参考
优秀示例（ABC齐全）：
  "偏执而精明的技术官僚，坚信数据高于直觉。在公开场合沉默寡言，但内部会议中逐一推翻他人假设。对失败的容忍度极低，曾因一次供应链延误解雇整个团队。表面追求效率至上，骨子里是对失控的恐惧。"
不合格示例（缺失B或C）：
  "极富远见的战略领袖，深谙国际格局，善于在复杂局势中寻找平衡。" — 抽象赞美，无具体矛盾/制约

只返回 JSON 对象。"""

_PERSONA_PROMPT_FALLBACK = """你是客观中立的角色档案生成专家。基于以下实体信息和原文背景，为该$domain_role生成一个独立人格档案。返回 JSON。

## 铁律
1. 每个 persona 必须同时包含：A.优势/能力 B.矛盾/制约 C.自利逻辑
2. 不赋予道德高下标签，用行为描述代替价值判断
$entity_perspective

## 来自用户的特殊期望
$user_expectations

$role_inference

$role_examples

$domain_principles

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

## 输出 JSON
{
  "persona": "人格描述 (50-100字)。ABC三项齐全",
  "background": "背景故事 (50-100字)",
  "goals": ["目标1", "目标2"]
}

只返回 JSON 对象。"""


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
        # 仅一级核心博弈者生成独立智能体
        tier1_entities = entity_registry.get_tier1()
        tier2_entities = entity_registry.get_tier2()
        log_fn("agents", f"从注册中心读取: tier1={len(tier1_entities)}核心 tier2={len(tier2_entities)}次级 (共{entity_registry.total}个)")
        if tier2_entities:
            log_fn("agents", f"  二级实体保留但不生成智能体: {', '.join(e.name for e in tier2_entities[:20])}")
        log_fn("agents", entity_registry.summary()[:200])

        # 构建统一的 persons 列表（仅 tier1）
        persons: list[dict] = [
            {"id": e.id, "name": e.name, "type": e.type, "description": "",
             "base_type": getattr(e, "base_type", "Agent")}
            for e in tier1_entities
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
        freq_map: dict[str, int] = {e.name: e.freq for e in tier1_entities}
        if preprocessor and preprocessor.result:
            fm = getattr(preprocessor.result, "entity_frequencies", {}) or {}
            freq_map = {**fm, **freq_map}

        max_agents = len(persons)
        log_fn("agents", f"从 {max_agents} 个注册实体中生成最多 {max_agents} 个智能体")

    else:
        # 无 EntityRegistry 时：从 intel_list 回退构建 persons（兼容测试/脚本）
        persons = []
        if intel_list:
            seen: set[str] = set()
            for e in intel_list:
                nm = (e.get("name") or "").strip()
                if nm and nm not in seen:
                    seen.add(nm)
                    persons.append({"id": e.get("id", ""), "name": nm,
                                    "type": e.get("type", "Person"), "description": ""})
        intel_map = {}
        if intel_list:
            for e in intel_list:
                nm = (e.get("name") or "").strip()
                if nm:
                    intel_map[nm] = e
                for a in e.get("aliases", []):
                    a = str(a).strip()
                    if a and a not in intel_map:
                        intel_map[a] = e
        freq_map = {}
        if preprocessor and preprocessor.result:
            freq_map = getattr(preprocessor.result, "entity_frequencies", {}) or {}
        max_agents = len(persons)
        if max_agents == 0:
            log_fn("agents", "无 EntityRegistry 且无 intel_list，智能体工厂返回空")
            return []

    client = LLMClient()
    agents: list[DeductionAgentProfile] = []

    expected_keys = {"persona", "background", "goals"}

    _COMPANY_TYPES = {"Company", "Enterprise", "Organization",
                      "公司", "企业", "组织", "机构"}
    def _entity_role(person: dict) -> str:
        from strategy_forge.engine.domain_adapter import get_adapter
        etype = (person.get("type") or "").strip()
        domain_role = ""
        try:
            domain_role = get_adapter(domain).prompts.agent_domain_role or ""
        except Exception:
            pass
        if not domain_role:
            domain_role = "独立博弈者"
        if domain_role not in getattr(_entity_role, "_logged", set()):
            logger.info("[AgentFactory] 领域 %s 角色指南已注入 persona prompt", domain)
            _entity_role._logged = getattr(_entity_role, "_logged", set()) | {domain}
        return domain_role

    def _fallback(nm: str) -> dict:
        return {"persona": f"{nm}是一个参与事件的独立个体",
                "background": "来自原文背景", "goals": ["参与互动", "表达观点"]}

    def _build_prompt(person: dict, person_name: str, fragments: list[str] | None) -> str:
        ue = "\n".join(f"- {x}" for x in (pre_interventions or [])) or "无特殊期望"
        im = intel_map.get(person_name, {})
        role = im.get("role", "独立博弈者")
        parent_info = str(im.get("parent") or "无")
        sub_info = ", ".join(str(s) for s in im.get("sub_entities", [])) or "无"
        f = freq_map.get(person_name, 0) if freq_map.get(person_name, 0) > 0 else "?"
        c = preprocessor.result.entity_chunk_coverage.get(person_name, 0) if preprocessor and preprocessor.result else "?"
        entity_stats = f"频次={f}, 覆盖={c}个分块"
        role_inference = _METHODOLOGY.get("_role_inference", "") or ""
        role_examples = _METHODOLOGY.get("_entity_role_examples", "") or ""
        domain_principles = _METHODOLOGY.get("_domain_action_principles", "") or ""
        # 检测集体决策实体 → 注入机构视角指令
        etype = (person.get("type") or "").strip()
        ep = ""
        if any(k in etype for k in _COLLECTIVE_ENTITY_KW):
            ep = _COLLECTIVE_PERSPECTIVE
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
                entity_stats=entity_stats,
                role_inference=role_inference,
                role_examples=role_examples,
                domain_principles=domain_principles,
                entity_perspective=ep)
        return Template(_PERSONA_PROMPT_FALLBACK).substitute(
            name=person_name, type=person.get("type", "Person"),
            description=person.get("description", ""), role=role,
            parent_info=parent_info, sub_info=sub_info,
            context=source_material[:2000], user_expectations=ue, domain_role=_entity_role(person),
            entity_stats=entity_stats,
            role_inference=role_inference,
            role_examples=role_examples,
            domain_principles=domain_principles,
            entity_perspective=ep)

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
        system = "你是客观中立的角色档案生成专家。每条 persona 须覆盖 ABC 三项：优势、矛盾、自利逻辑。用行为描述代替价值判断。只输出 JSON。"
        messages = [Message(role="user", content=prompt)]
        try:
            if chat_fn is not None:
                content = await asyncio.to_thread(chat_fn, messages, system, 0.7)
            else:
                response = await client.chat_json(messages, system=system, schema_name="persona", temperature=0)
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
            base_type=person.get("base_type", "Agent"),
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



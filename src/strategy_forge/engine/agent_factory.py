"""Phase 3: Agent Factory — deep persona generation from graph + LanceDB retrieval."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
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
    domain: str = "",
) -> list[DeductionAgentProfile]:
    from strategy_forge.core.config import config
    from strategy_forge.core.providers import registry as _reg
    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    from strategy_forge.core.llm_client import Message

    # Collect ALL Entity nodes — IntelSorter handles filtering later
    result = graph._conn.execute(
        f"MATCH (e:{graph.NODE_TABLE}) RETURN e.id, e.name, e.type, e.description"
    )
    persons: list[dict] = []
    while result.has_next():
        r = result.get_next()
        persons.append({"id": r[0], "name": r[1], "type": r[2], "description": r[3]})
    log_fn("agents", f"收集 {len(persons)} 个实体作为智能体候选")

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
    # 类型常量：用于区分"人物"和"组织/国家/政党"，在 intel_list 分支中按场景决定是否排除组织
    _PERSON_TYPES = {"Person", "person", "人物", "角色", "人",
                     "领导者", "领导", "agent", "Agent"}
    _ORG_TYPES = {"组织", "Organization", "organization", "org",
                  "国家", "Country", "country", "nation",
                  "政党", "PoliticalParty", "party", "党派",
                  "政府", "Government", "government",
                  "媒体", "Media", "Newspaper", "newspaper",
                  "机构", "Institution",
                  "军队", "Military", "military",
                  "团体", "Group", "联盟", "Alliance",
                  "城市", "City", "地点", "Location", "Place",
                  "地理区域", "地区", "Region",
                  "政治实体", "PoliticalEntity",
                  "公司", "Company", "企业", "Enterprise"}
    if intel_list:
        intel_map = {e["name"]: e for e in intel_list if e.get("name")}
        # Build reverse alias mapping from IntelSorter for cross-name matching
        _intel_reverse: dict[str, str] = {}
        for e in intel_list:
            canon = e.get("name", "")
            if not canon:
                continue
            _intel_reverse[canon] = canon
            for a in e.get("aliases", []):
                _intel_reverse[str(a).strip()] = canon
        active_names = {e["name"] for e in intel_list if e.get("include_in_simulation")}
        before = len(persons)
        # Filter: cross-match graph entity names against IntelSorter canonical names via aliases
        filtered: list[dict] = []

        # ── 频率兜底：sorter 遗漏的高频实体仍应纳入智能体 ──
        # 超长文本（长篇小说等）的 sorter 可能因 token 限制漏掉核心角色；
        # 以预处理器统计的全文频次作为兜底阈值，频次够高则不被 sorter 遗漏所杀。
        freq_map: dict[str, int] = {}
        if preprocessor and preprocessor.result:
            freq_map = getattr(preprocessor.result, "entity_frequencies", {}) or {}
        # 动态阈值：以 intel_list 中已标记为 include=true 的实体的频次中位数作为参照
        included_freqs = [
            freq_map.get(e["name"], 0) for e in intel_list if e.get("include_in_simulation")
        ]
        freq_threshold = int(statistics.median(included_freqs) // 3) if included_freqs else 5
        freq_threshold = max(3, freq_threshold)  # 至少出现 3 次才兜底
        fallback_count = 0

        for p in persons:
            pname = p.get("name", "")
            # Resolve to canonical name via reverse alias map
            canon = _intel_reverse.get(pname, pname)
            # (entities NOT in intel_map are excluded by default — IntelSorter must have seen them)
            intel_entry = intel_map.get(canon) or intel_map.get(pname)
            if intel_entry is None:
                # 频率兜底：sorter 未输出的实体，若全文频次足够高则仍纳入
                f = freq_map.get(canon, freq_map.get(pname, 0))
                if f >= freq_threshold:
                    intel_entry = {
                        "name": canon, "aliases": [],
                        "include_in_simulation": True,
                        "role": f"高频角色(频次={f})",
                    }
                    intel_map[canon] = intel_entry
                    fallback_count += 1
                else:
                    continue
            if not intel_entry.get("include_in_simulation", False):
                continue  # IntelSorter显式标记为非战略 → 排除

            filtered.append(p)

        # ── 两遍过滤：组织/国家/政党仅在"人物为主"场景下排除 ──
        # 先分类候选：人物 vs 组织。仅当存在人物型智能体时，才排除其所属组织。
        persons_in = []
        orgs_in = []
        for p in filtered:
            etype = (p.get("type") or "").strip()
            if etype and etype in _PERSON_TYPES:
                persons_in.append(p)
            elif etype and etype in _ORG_TYPES:
                orgs_in.append(p)
            else:
                persons_in.append(p)  # 未知类型视为人物，保守保留
        if persons_in and orgs_in:
            _org_names = {p.get("name") for p in orgs_in}
            log_fn("agents", f"组织类型过滤: 排除 {len(orgs_in)} 个组织/国家/政党")
            filtered = persons_in
        else:
            filtered = persons_in + orgs_in

        persons = filtered
        persons = filtered
        if fallback_count:
            log_fn("agents", f"频率兜底: {fallback_count} 个高频实体（sorter 遗漏）已恢复为智能体，阈值≥{freq_threshold}")
        if len(persons) < before:
            log_fn("agents", f"情报过滤: {before} → {len(persons)} 个智能体（排除非战略实体）")
    else:
        intel_map = {}
        # 叙事模式：基于实体类型的基础过滤，排除非决策者类型
        _NON_AGENT_TYPES = {"地理区域", "地理位置", "地点", "天气", "气象",
                            "文档", "协议", "合同", "批文", "文件",
                            "概念", "现象", "事件", "日期", "时间",
                            "设施", "基础设施", "建筑",
                            "自然景观", "自然现象", "环境要素"}
        before = len(persons)
        persons = [
            p for p in persons
            if p.get("type", "") not in _NON_AGENT_TYPES
        ]
        if len(persons) < before:
            log_fn("agents", f"叙事模式类型过滤: {before} → {len(persons)} 个智能体（排除非决策者类型）")

    max_agents = min(len(persons), _reg.max_agents)
    log_fn("agents", f"从 {len(persons)} 个实体中生成最多 {max_agents} 个智能体")

    client = LLMClient()
    agents: list[DeductionAgentProfile] = []

    expected_keys = {"persona", "background", "goals"}

    sem = asyncio.Semaphore(max(1, _reg.max_concurrent))

    _DOMAIN_ROLES: dict[str, str] = {
        "military": "军事力量或决策实体",
        "business": "企业或行业参与者",
        "politics": "政治实体或政策制定者",
        "ecology": "生态主体或环境利益方",
        "urban": "城市管理机构或市政实体",
        "tech": "科技企业或研究机构",
        "info_war": "信息舆论参与方",
        "geo_strategy": "地缘战略决策主体",
    }
    _domain_role = _DOMAIN_ROLES.get(domain, "独立博弈者")

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
                user_expectations=ue, domain_role=_domain_role)
        return Template(_PERSONA_PROMPT_FALLBACK).substitute(
            name=person_name, type=person.get("type", "Person"),
            description=person.get("description", ""), role=role,
            parent_info=parent_info, sub_info=sub_info,
            context=source_material[:2000], user_expectations=ue, domain_role=_domain_role)

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

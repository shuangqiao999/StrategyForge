"""Intelligence Sorter — LLM reads source material, classifies all entities.

Filters non-strategic entities (regulators, acquired companies, subordinate units)
before the agent factory creates decision-making profiles.
"""
from __future__ import annotations

import logging
from typing import Any

from strategy_forge.core.llm_client import LLMConnectionError

logger = logging.getLogger(__name__)

_INTEL_PROMPT = """你是情报分析师。请根据以下种子材料，整理实体关系清单。

## 所有已提取的实体名称
{entity_names}

## 种子材料（完整上下文）
{source}

## 任务

1. 判断每个实体是独立的战略决策者，还是某个实体的子部分/下属
2. 判断生命周期 —— 如果被收购，在推演时间窗口内是否仍独立存在
3. 过滤非战略实体 —— 监管机构(SEC/证监会)、评级机构(标普/穆迪)、指数、纯媒体 — 不参与推演
4. 建立层级关系 —— 工厂属于企业、部门属于组织
5. 别名合并 —— 识别同一实体的不同名称（中英文名、简称、全称），合并为一条
6. 集合概念过滤 —— 泛指的群体/行业/阵营不是单一决策者

## 输出 JSON（仅 JSON，无 markdown）
{{"entities": [
  {{"name": "特斯拉", "type": "企业", "aliases": ["Tesla", "特斯拉公司"], "parent": null,
    "sub_entities": ["弗里蒙特工厂", "上海超级工厂"], "include_in_simulation": true, "role": "核心博弈者"}},
  {{"name": "乌克兰军队", "type": "军事力量", "aliases": ["乌军", "乌克兰武装部队"], "parent": null,
    "sub_entities": [], "include_in_simulation": true, "role": "防御方"}},
  {{"name": "经合组织", "type": "国际组织", "aliases": ["OECD"], "parent": null, "sub_entities": [],
    "include_in_simulation": false, "role": "协调机构，非独立决策者"}},
  {{"name": "中国科技企业群体", "type": "集合概念", "aliases": [], "parent": null, "sub_entities": [],
    "include_in_simulation": false, "role": "泛指集合，具体成员(华为/中芯/DeepSeek)已单列，不作为单一博弈实体"}},
  {{"name": "俄乌", "type": "关系/对抗", "aliases": [], "parent": null, "sub_entities": ["俄罗斯", "乌克兰"],
    "include_in_simulation": false, "role": "二元对抗关系词，非单一决策者，成员各自为主体"}},
  {{"name": "美国海军第五舰队", "type": "军队编制", "aliases": [], "parent": "美军", "sub_entities": [],
    "include_in_simulation": false, "role": "美军下属编制，归入上级"}},
  {{"name": "北约秘书长", "type": "职务", "aliases": [], "parent": "北约", "sub_entities": [],
    "include_in_simulation": false, "role": "职务头衔，对应人物/组织已单列"}},
  {{"name": "SEC", "type": "监管机构", "aliases": [], "parent": null, "sub_entities": [],
    "include_in_simulation": false, "role": "金融监管者，非商业博弈者"}}
]}}

- include_in_simulation: true = 独立决策者，需要生成智能体
- include_in_simulation: false = 子实体/监管/指数/集合概念/职务/关系词/下属部门 —— 不生成智能体
- parent: null = 独立实体; 填写父实体名 = 从属关系
- 重要：如果某人是某组织的CEO/领导人/代表人物（如马斯克→特斯拉，特朗普→美国，普京→俄罗斯），将其 parent 设为该组织名称，include_in_simulation 设为 false。同时将该人添加到该组织的 sub_entities 列表中。组织本身保留为独立决策者。
- sub_entities: 该实体包含的子部分（工厂、部门、领导人等）
- 重要：论坛/协调机构（G7、G20、OECD、WEF等）和行政下属机构（国台办、美财政部等）不是独立战略决策者 —— 它们不独立发动军事行动或制定外交政策。将其设为 false，或将其设为对应国家/上级组织的一部分。
- 别名合并（重要）：同一实体的多个名称（中英文名如"经合组织/OECD"、简称如"乌军/乌克兰军队"、全称与缩写）只输出一条，选最规范/最完整的名称作为 name，其余全部放入 aliases 数组；不要把同一实体的别名作为独立条目重复输出。规范名称优先选择中文全称（如"美国国防部"而非"DoD"），英文缩写/简称放入 aliases。
- 集合概念过滤（重要）：泛指的群体/行业/阵营/民间/概念集合（如"中国科技企业群体"、"科技行业"、"西方民间"）不是单一战略决策者。若其具体成员（如华为、中芯国际、DeepSeek）已出现在实体列表中，则把该集合设为 include_in_simulation=false，并在 role 说明"泛指集合，成员已单列"；若集合是唯一表述（无具体成员出现），则保留为决策者但在 role 标注"集合概念"。
- 职务/头衔过滤（重要）：纯职务/头衔词（总统、总理、部长、秘书长、司令、主席、领导人等，如"北约秘书长""美国总统"）不是独立实体。将其 include_in_simulation=false，parent 设为对应组织，并把对应的具体人物（如吕特、特朗普）单列为该组织的领导人（按上面的领袖规则处理）。
- 二元关系/对抗词过滤（重要）：由两个及以上主体拼接而成、表示关系或冲突的词（如"俄乌""美伊""中美""巴以""印巴""俄乌冲突"）不是单一决策者。将其 include_in_simulation=false，把各成员（俄罗斯、乌克兰…）放入 sub_entities，成员各自作为独立主体单列。
- 国家内部部门/军队编制过滤（重要）：一国政府的职能部门（国防部、财政部、外交部、商务部、央行、最高法院等）与军队编制（第X舰队、战区、司令部、集团军等）不是独立战略决策者。将其 parent 设为所属国家或军队、include_in_simulation=false；由国家/军队本身作为决策者。
- 不要遗漏任何已提取的实体名（作为别名合并进某条的除外）

## 领域特定示例
以下示例适用于当前领域，帮助你理解该领域下"非独立决策者"的典型模式：
$domain_examples"""


# ── 领域特定示例：帮助 LLM 理解不同领域的过滤模式 ──
_INTEL_DOMAIN_EXAMPLES: dict[str, str] = {
    "business": (
        '  {{"name": "郑州", "type": "地理位置", "include_in_simulation": false, "role": "城市/工厂驻地，非独立决策者"}},\n'
        '  {{"name": "换电", "type": "技术概念", "include_in_simulation": false, "role": "技术方案，非实体决策者"}},\n'
        '  {{"name": "上路", "type": "抽象动词/状态", "include_in_simulation": false, "role": "概念描述，非实体"}}'
    ),
    "politics": (
        '  {{"name": "参议院", "type": "立法机构", "include_in_simulation": false, "parent": "美国", "role": "立法部门，归上级国家"}},\n'
        '  {{"name": "民意调查", "type": "统计工具", "include_in_simulation": false, "role": "数据工具，非决策者"}}'
    ),
    "ecology": (
        '  {{"name": "亚马逊雨林", "type": "地理区域", "include_in_simulation": false, "role": "地理区域，非独立决策者"}},\n'
        '  {{"name": "碳排放", "type": "环境指标", "include_in_simulation": false, "role": "测量指标，非实体"}}'
    ),
    "urban": (
        '  {{"name": "地铁3号线", "type": "基础设施", "include_in_simulation": false, "role": "基建项目，非独立决策者"}},\n'
        '  {{"name": "学区房", "type": "房产概念", "include_in_simulation": false, "role": "市场概念，非实体决策者"}}'
    ),
    "tech": (
        '  {{"name": "5G标准", "type": "技术标准", "include_in_simulation": false, "role": "技术标准规范，非独立决策者"}},\n'
        '  {{"name": "开源社区", "type": "社区集合", "include_in_simulation": false, "role": "社区集合，具体成员已单列"}}'
    ),
    "info_war": (
        '  {{"name": "微博热搜", "type": "媒体平台/工具", "include_in_simulation": false, "role": "信息传播渠道，非独立决策者"}},\n'
        '  {{"name": "假新闻", "type": "信息产物", "include_in_simulation": false, "role": "信息产物，非实体决策者"}}'
    ),
    "geo_strategy": (
        '  {{"name": "联合国", "type": "国际组织", "include_in_simulation": false, "role": "多边协调平台，非独立战略决策者"}},\n'
        '  {{"name": "世界经济论坛", "type": "论坛", "include_in_simulation": false, "role": "讨论平台，非决策实体"}}'
    ),
}


def _as_name(x: Any) -> str:
    """将实体名元素统一转为干净字符串。

    LLM 常把 sub_entities/aliases 返回成对象数组（如 {"name": "华为"}），
    若直接下游 join/strip 会抛 'expected str instance, dict found'。此处集中归一化。
    """
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("name", "entity", "title", "id"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if x is None:
        return ""
    return str(x).strip()


def _as_name_list(raw: Any) -> list[str]:
    """把 aliases/sub_entities 归一化为去空、去重（保序）的字符串列表。"""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        n = _as_name(item)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


async def sort_entities(
    source: str,
    entity_names: list[str],
    client: Any,
    max_source_chars: int = 25000,
    domain: str = "",
) -> list[dict[str, Any]]:
    """LLM reads source material, outputs structured entity relationship list.

    Args:
        source: Full seed text.
        entity_names: All entity names extracted by graph builder.
        client: DeductionLLMClient instance.
        max_source_chars: Max chars of source to send (kept high for context).
        domain: Domain key for injecting domain-specific examples.

    Returns:
        List of entity entries with classification. Empty on failure.
    """
    if not source or not entity_names:
        return []

    domain_examples = _INTEL_DOMAIN_EXAMPLES.get(domain, "")
    prompt = _INTEL_PROMPT.format(
        entity_names=", ".join(entity_names),
        source=source[:max_source_chars],
        domain_examples=domain_examples,
    )

    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import Message
    try:
        resp = await client.chat(
            [Message(role="user", content=prompt)],
            system="你是情报分析师，输出结构化 JSON。只输出 JSON。",
            temperature=0.1,
            max_tokens=config.deduction_intel_max_tokens,
        )
    except LLMConnectionError:
        raise
    except Exception as e:
        logger.warning("[IntelSorter] LLM call failed: %s", e)
        return []

    raw = _extract_text(resp)
    data = _parse_json(raw)
    # 兼容顶层数组：模型偶尔省略 {"entities": ...} 外壳
    if isinstance(data, list):
        data = {"entities": data}
    if not isinstance(data, dict):
        logger.warning("[IntelSorter] Failed to parse LLM output as JSON")
        return []

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []

    result = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        result.append({
            "name": _as_name(e.get("name", "")),
            "type": str(e.get("type", "")).strip(),
            "aliases": _as_name_list(e.get("aliases")),
            "parent": e.get("parent") or None,
            "sub_entities": _as_name_list(e.get("sub_entities")),
            "include_in_simulation": bool(e.get("include_in_simulation", True)),
            "role": str(e.get("role", "")).strip(),
        })

    demoted = _apply_safety_net(result)
    active = sum(1 for e in result if e["include_in_simulation"])
    excluded = [e["name"] for e in result if not e["include_in_simulation"]]
    logger.info("[IntelSorter] 总计 %d 实体 | 活跃 %d | 排除 %d | 安全网降级 %d",
                len(result), active, len(excluded), demoted)
    if excluded:
        logger.info("[IntelSorter] 排除实体: %s", excluded[:15])
    return result


# 二元关系/对抗词（精确集合）——非单一决策者
_DYAD_WORDS = frozenset({
    "俄乌", "美伊", "中美", "美中", "巴以", "以巴", "印巴", "美俄", "俄美",
    "俄乌冲突", "中美关系", "美中关系", "俄乌战争", "巴以冲突",
})
# 军队编制后缀——归上级
_UNIT_SUFFIX = ("舰队", "战区", "司令部", "集团军", "航母战斗群", "特遣队")
# 一国政府职能部门/机关——归上级国家
_DEPT_WORDS = ("国防部", "财政部", "外交部", "商务部", "内政部", "司法部", "央行",
               "中央银行", "最高法院", "最高法", "国务院", "白宫", "国会", "参议院",
               "众议院", "国台办", "发改委", "证监会")
# 纯职务/头衔后缀——对应人物/组织已单列
_TITLE_SUFFIX = ("总统", "总理", "首相", "部长", "秘书长", "司令", "主席", "领导人",
                 "议长", "行长", "总裁", "元首")

# ── 英文关键词（与中文关键词并行）──
_DEPT_WORDS_EN = (
    "defense department", "department of defense", "dod", "treasury",
    "state department", "department of state", "supreme court", "federal reserve",
    "pentagon", "congress", "senate", "white house", "central bank",
    "foreign ministry", "ministry of defence", "ministry of defense",
    "ministry of finance", "ministry of foreign affairs",
    "commerce department", "justice department", "interior department",
    "national security council", "joint chiefs",
)
_UNIT_SUFFIX_EN = ("fleet", "command", "division", "task force", "battalion",
                   "regiment", "brigade", "squadron", "carrier strike group")
_TITLE_SUFFIX_EN = ("president", "prime minister", "secretary", "secretary-general",
                    "chairman", "chief", "minister", "general", "admiral",
                    "commander", "governor", "mayor", "director", "premier")
_DYAD_WORDS_EN = frozenset({
    "russia-ukraine", "us-iran", "us-china", "china-us",
    "israel-palestine", "israel-hamas", "us-russia", "russia-us",
    "india-pakistan", "north korea-south korea",
})

_ALL_DYAD = _DYAD_WORDS | _DYAD_WORDS_EN

# ── 军队名称精确匹配降级（不含后缀的独立军队名）──
_MILITARY_NAMES = frozenset({
    "乌军", "俄军", "美军", "伊朗伊斯兰革命卫队", "以色列国防军",
    "朝鲜人民军", "韩国军队", "日本自卫队",
})


def _any_name_matches(e: dict, keywords: tuple[str, ...], mode: str) -> bool:
    """检查实体 name 及其 aliases 是否命中关键词（大小写不敏感）。"""
    candidates = [e.get("name", "")]
    candidates.extend(e.get("aliases") or [])
    for name in candidates:
        if not isinstance(name, str) or not name:
            continue
        low = name.lower()
        if mode == "exact":
            if low in keywords:
                return True
        elif mode == "suffix":
            if low.endswith(keywords):
                return True
        elif mode == "substring":
            if any(w in low for w in keywords):
                return True
    return False


def _apply_safety_net(result: list[dict[str, Any]]) -> int:
    """保守安全网：对高置信的"非独立决策者"强制 include_in_simulation=false。

    仅在 FORGE_INTEL_SAFETY_NET 开启时生效；只降级、不新增/删除实体，避免误伤唯一代表。
    返回被安全网降级的实体数量。
    """
    import os
    safety_enabled = os.getenv("FORGE_INTEL_SAFETY_NET", "1") == "1"
    if not safety_enabled:
        logger.warning("[IntelSorter] FORGE_INTEL_SAFETY_NET=0，安全网已关闭，完全依赖LLM判断")
        return 0
    logger.info("[IntelSorter] 安全网状态: 启用")

    def _demote(e: dict, note: str) -> bool:
        if not e["include_in_simulation"]:
            return False
        e["include_in_simulation"] = False
        base = e.get("role", "") or ""
        e["role"] = (base + f"｜安全网降级：{note}") if base else f"安全网降级：{note}"
        return True

    demoted = 0
    for e in result:
        if not e.get("name"):
            continue
        if _any_name_matches(e, _ALL_DYAD, mode="exact"):
            demoted += _demote(e, "二元关系词")
        elif _any_name_matches(e, _UNIT_SUFFIX, mode="suffix") or _any_name_matches(e, _UNIT_SUFFIX_EN, mode="suffix"):
            demoted += _demote(e, "军队编制归上级")
        elif _any_name_matches(e, _DEPT_WORDS, mode="substring") or _any_name_matches(e, _DEPT_WORDS_EN, mode="substring"):
            demoted += _demote(e, "政府部门归上级国家")
        elif _any_name_matches(e, _TITLE_SUFFIX, mode="suffix") or _any_name_matches(e, _TITLE_SUFFIX_EN, mode="suffix"):
            demoted += _demote(e, "职务头衔非实体")
        elif e.get("name") in _MILITARY_NAMES:
            demoted += _demote(e, "军队编制归上级")

    return demoted


def _extract_text(resp: Any) -> str:
    if hasattr(resp, "text"):
        return resp.text
    if hasattr(resp, "content"):
        c = resp.content
        if isinstance(c, list):
            from strategy_forge.core.llm_client import TextBlock
            return "".join(b.text for b in c if isinstance(b, TextBlock))
        return str(c)
    if isinstance(resp, dict):
        choices = resp.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
        return str(resp)
    return str(resp)


def _parse_json(raw: str) -> Any:
    from ._utils import extract_json
    return extract_json(raw)

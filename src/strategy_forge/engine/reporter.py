"""Phase 5: Report Generation — analyze simulation results, produce structured report."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from string import Template
from typing import Any

from strategy_forge.storage.graph_store import DeductionGraphStore

from ._utils import extract_text
from .models import DeductionReport, DeductionSession, SimulationRound

logger = logging.getLogger(__name__)

_REPORT_PROMPT = """你是一位高级战略分析专家。你面前的资料，来自模拟推演的完整记录——涵盖军事、商业、科技与政治博弈。你的任务不是罗列数据，而是从这些纷繁的迹象中，剥离出最精准、最克制、最具指向性的方向性判断。言必有据，不夸大，不臆测。

## 核心写作法则（决定报告质量，逐条强制执行）
1. **因果链闭环 + 事件锚定**：每一条战略判断必须引用下方「关键事件序列」中的至少一个[事件N]，并以「动作→机制→结果→反制」的因果链展开。核心判断格式："[事件N]中A的[动作] → 直接导致B的[领域]发生[变化] → 迫使B转而[反制]"。次要事件可简写为"[事件N]中A的[动作] → 导致[结果]，进而使B陷入[困境]"。严禁"A施压B"的扁平句。
2. **段落即战略维度**：正文必须按博弈维度/主题逻辑分段，维度名应从当前推演的实际主题中提炼，不预设固定分类。例如：技术路线之争、供应链重组、市场格局重塑、联盟与分化、成本与效率博弈、生态位争夺等。每段用 `### 维度名` 作为小标题。每段聚焦1-2个主要行为体。这是输出的**强制格式要求**，不是提示词的元指令。
3. **必须写出"战略困境"**：至少为 3 个不同阵营的行为体写出得损失衡分析。句式模板："某方为获取[收益]，不得不承受[代价/风险]"，或"某方虽在[领域A]占优，但在[领域B]已逼近临界点"。注意：模板中的"某方"必须替换为具体实体名，禁止在正文中输出"X"或"某方"等模糊占位符。
4. **行业一致性与实体保真**：描述实体行为时必须遵守两大约束：
   - 研发方向必须与实体行业属性严格一致。车企研发可涉及智驾/三电/制造，芯片企业研发可涉及制程/架构/算力，零售/消费品企业研发只能涉及产品迭代/供应链数字化/用户运营，不得跨行业套用研发术语。
   - 合作关系（如联合、partner、联盟）的描述必须与下方「关键事件序列」中的对应[事件N]完全一致，不得替换合作对象。如果[语义召回]事件标注了合作对象，以该标注为准。
5. **语气与节奏**：克制、事实稠密，但允许使用"倒逼""对冲""临界点"等有张力的动词。每段控制在6行以内，避免堆砌形容词。
6. **多方平衡视角**：避免单向"某方全面胜出/衰落"的叙事。各方均有优势与制约，必须给出对手视角。
7. **区分事实与推测**：只有推演数据/因果归因支持的才作为判断陈述；不确定的用"可能/或将/存在风险"表述，不要写成既成事实。
8. **避免极端断言**：禁止出现"完全孤立""社会崩溃""彻底失败"等表述——此类只能作为风险情景谨慎提及，不作为结论。

## 推演基础信息
- 标题: $title · 领域: $domain · 轮次: $round_count
- 智能体数量: $agent_count · 不可变目标: $immutable_goals

## 智能体概览
$agent_overview

## 关键关系网络（图谱按权重）
$key_relations

## 关键事件序列（精选·必须引用）
$key_events

## 行动时序（Agent→事件，跨轮）
$action_timeline

## 全局态势数据（每实体各指标的定性档位与趋势，及淘汰线参考）
$quantified_context

## 确定性因果归因（源→目标 的影响方向与强度）
$causal_attribution

## 重点转折事件（方向性变化最剧烈的3个事件）
$turning_points

---

## 输出格式（严格遵守 JSON）
返回 JSON，包含四个字段：

1. **"narrative"**：完整推演报告（800-2000字）。
   - **硬性要求**：正文必须至少引用 3 条下方「关键事件序列」中的[事件N]编号，作为因果链锚点。
   - **硬性要求**：正文每个维度段落**必须**以 `### 维度名` 开头（如 `### 科技竞赛`、`### 区域安全`），禁止写成无标题的平铺段落。
   - 开篇（150字内）：作为全文第一个 `###` 段落之前的引入段，直接点明全局核心矛盾与主要博弈轴线。禁用"报告显示""推演表明""第X轮推演中"等废话。
   - 正文：按战略维度分段，每段以 `### ` 开头。因果句格式：`[事件N]中A的[动作] → B的[领域]发生[变化] → 迫使B[反制]`。
   - 结尾（150字内）：总览系统级风险与胜负手临界点。
   - 严禁输出任何具体数值/评分，不得出现 JSON、表格或项目符号列表。

   正确正文格式示例：
   ```
   开篇引入段（无标题，直接进入分析）...

   ### 核心博弈维度一
   [事件4]中A方的关键动作 → 直接导致B方的关键指标发生方向性变化 → 迫使B方转而采取反制策略（[事件7]），试图以优势对冲风险。B方虽在某领域占优，但在另一领域已逼近临界点，不得不承受调整的代价。

   ### 核心博弈维度二
   [事件2]中C方的主动出击 → 直接导致D方的既有优势被稀释 → 迫使E方调整姿态应对连锁反应（[事件10]）...
   ```

2. **"risk_alerts"**：输出 3~5 条。格式：`{风险标题} | {具体触发机制/路径} | {受影响方}`。narrative 中已分析过的风险场景必须在 risk_alerts 中逐一展开触发链，不得遗漏。必须写明"如何触发"的因果链条，而非重复"存在风险"或"导致受损"的表象。
   - 错误示例："供应链风险 | 连续削弱 | 某方"（只写了影响，不是触发路径）
   - 正确示例（供应链/地缘场景）："资源断供风险 | 外部管制扩至关键环节 → 切断核心供应通路 → 产能受限 | 受影响的实体"
   - 正确示例（商业竞争场景）："资金链断裂风险 | 连续价格战消耗现金流 → 融资窗口关闭 → 无法覆盖固定成本 → 流动性枯竭 | 受影响的实体"
   - 直接陈述事实，不前置"可能/或将"。

3. **"recommendations"**：最多5条。格式：`{针对方}→{具体动作}→{预期机制与效果}`。不写"建议""应""需要"等虚词，每条不超过40字。
   - 正确示例："甲方→暂停扩张转向补充资源→预期2轮内恢复至安全水平，避免越过淘汰线"

4. **"conclusion"**：150-250字。首句必须以"虽然…但是…"结构开头，点明全局胜负手与临界变量。不得照抄 narrative 的句子，是更高层的凝练总结。不含具体数值，区分事实与推测。
   - 正确示例（提炼总结，不照抄 narrative）：
     "虽然甲方凭借前期攻势确立了阶段性优势，但其核心资源的持续消耗已逼近临界线。全局胜负手不在于一时的局部得失，而在于谁能率先打破消耗战的僵局。第三方斡旋虽延缓了冲突升级，却无法改变资源对抗的本质逻辑。"
   - 错误示例（照抄 narrative 的事件复述，不推荐）：
     "甲方在第3轮发动猛攻，迫使乙方转入防御，丙方继续外交斡旋，各方陷入僵持。"（这只是复述事件，不是总结）

只返回纯 JSON，不要 markdown 代码块，不要注释。
- 正文中禁止出现"X"、"某方"等模糊占位符，所有战略判断必须指明具体行为体名称。"""

_REPORT_PROMPT_NARRATIVE = """你是一位叙事文学作家。基于以下角色档案、事件序列和行动时序，将推演过程改写为一篇有情感张力、有人物弧光的故事化叙事。

## 推演核心问题（最高优先级——故事必须回答它）
$immutable_goals

## 写作要求
1. 以故事文体写作——有场景氛围、有心理描写、有情感落点。不是分析报告，不是简报，不是总结。
2. 不要逐轮罗列动作。按时间跳跃叙述关键场景——只写转折点、高潮、关键时刻，中间的过程一笔带过或完全省略。
3. 重点刻画角色的心理变化与关系演化——谁在压力下崩溃、谁暗中结盟、谁背叛了谁、谁的信仰动摇了。
4. 对话与内心独白可以虚构，但必须基于下方提供的角色人格档案和事件事实。
5. **结局必须对「推演核心问题」给出明确判定**：基于事件序列中各角色的实际行动、结盟与得失，推断出最合理的答案（如"谁最终掌权""哪一方胜出"），并在故事结尾以具体情节呈现这个结果。禁止用"未来充满不确定性""斗争仍未结束"等模糊表述回避判定。若局势确实胶着，也必须指出当前最占优的一方及其决定性筹码。
6. **证据一致性（硬性规则）**：结局判定必须以事件序列中的具体事件为依据，且不得与事件矛盾——
   - 若某角色在事件中遭到公开指控、证据揭露或攻击且事件序列中不存在其澄清/反制/翻盘事件，该角色**不能**被写成最终赢家；
   - 赢家的胜利必须能追溯到事件序列中其实际做过的行动（结盟、反制、掌握筹码），禁止凭空赋予其"突破""声誉恢复"等未发生的成果；
   - 事件中已出现的重大线索（背叛、秘密录音、通敌、渗透、并购威胁）必须在结局中交代后果，哪怕一句话，禁止悬空蒸发。
7. **现实性（硬性规则）**：叙事和结局必须符合现实事物的发展规律——
   - 禁止超现实桥段：黑客奇迹、凭空出现的巨额资金、一夜掌控他人系统、"底层代码"式的万能筹码；
   - 权力/控制权的转移必须通过现实中可行的路径呈现（股权、投票、法律程序、联盟倒戈、舆论压力），不能靠宣称完成；
   - 已死亡或已退场的人物不得在其死亡/退场之后再有任何行动、对话或表态；
   - 避免"降维打击""绝对控制"等夸张修辞，用克制的现实语言描述力量对比的变化。
8. $output_length 字左右。
9. 避免宏大叙事词汇。禁止使用"胜利""灵魂""升华""永恒""神圣""注定""命运的齿轮"等过度拔高的词。保持克制、冷静的文学语调，像好的新闻特稿而非史诗。
10. 同一角色最多出现 3~4 个关键场景。如果角色在多个轮次做了类似的事，只写最有张力的那一次，不要反复描述同一行为模式。

## 角色档案
$agent_overview

## 人格演变轨迹（推演中真实发生的信念/准则变化，按轮次）
$personality_evolution

## 关键事件序列（按轮次，含开局锚点/高冲突事件/近期事件——结局判定的证据基础）
$key_events

## 行动时序
$action_timeline

## 语义召回关键事件（跨轮重要事件，必须嵌入）
$key_recall

## 输出 JSON（纯 JSON，不要 markdown）
{
  "narrative": "故事文本（按时间线推进，有场景、有心理、有弧光，结尾以具体情节呈现核心问题的答案）",
  "character_arcs": ["角色A: 从天真走向冷酷", "角色B: 在孤独中坚守信念", "角色C: 被背叛后黑化"],
  "conclusion": "故事收束（200字内）。第一句必须直接回答「推演核心问题」——点名具体的人/方及其凭借的关键筹码（筹码必须来自事件序列）；随后给出情感落点。禁止回避式结论。"
}
character_arcs 规则：若「人格演变轨迹」非空，弧光必须以其中该角色真实发生的准则变化为依据（可文学化改写但方向一致）；轨迹中未出现的角色才允许基于事件序列推断弧光。

只返回纯 JSON，不要 markdown 代码块。"""


_CONFLICT_KW = (
    "指控", "揭露", "背叛", "录音", "泄露", "泄漏", "攻击", "对抗", "威胁",
    "并购", "收购", "罢免", "起诉", "诉讼", "摊牌", "决裂", "反制", "曝光",
    "架空", "逼宫", "收买", "渗透", "通敌", "举报", "证据", "秘密资金",
    "情报交易", "联盟", "联合", "翻盘", "夺权", "继承权", "遗嘱",
)


def _sample_arc_events(events: list[str], head_n: int = 5,
                       conflict_n: int = 15, tail_n: int = 15) -> list[str]:
    """全程弧线采样：开局锚点 + 高冲突事件 + 尾部近期事件。

    替代"只取尾部N条"——保证指控/背叛/结盟等因果关键事件必然进入
    收束视野，结局判定才有完整证据链。冲突事件按关键词命中数加权，
    采样后恢复时间顺序。
    """
    if len(events) <= head_n + conflict_n + tail_n:
        return events
    head = events[:head_n]
    tail = events[-tail_n:]
    middle = events[head_n:-tail_n]
    scored: list[tuple[int, int, str]] = []
    for idx, e in enumerate(middle):
        hits = sum(1 for kw in _CONFLICT_KW if kw in e)
        if hits > 0:
            scored.append((hits, idx, e))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = sorted(scored[:conflict_n], key=lambda x: x[1])
    conflict = [e for _, _, e in picked]
    seen: set[str] = set()
    out: list[str] = []
    for e in head + conflict + tail:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _level_label(v: float) -> str:
    """将指标值映射为定性档位（不输出具体数值，避免过度戏剧化极端值）。"""
    if v >= 70:
        return "高位"
    if v >= 45:
        return "中位"
    if v >= 20:
        return "偏低"
    return "低位承压"


def _trend_label(d: float) -> str:
    """将累计变化量映射为趋势词。"""
    if d > 15:
        return "显著上升"
    if d > 3:
        return "上升"
    if d < -15:
        return "显著下降"
    if d < -3:
        return "下降"
    return "趋稳"


def _build_quantified_summary(
    rounds: list[SimulationRound],
    states: dict[str, Any] | None,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Build a structured qualitative trajectory summary for the report prompt.

    每个实体输出其关键指标的快照：档位·趋势·累计变化·淘汰线参考。
    数值仅注入 prompt 供 LLM 理解方向性——LLM 仍以叙事输出趋势，不复制数字。
    """
    if not states:
        return "（叙事模式，无量化指标数据）"

    # 辅助：指标中文化（与 simulator._METRIC_NAME 一致）
    _MN: dict[str, str] = {
        "strength": "军力", "morale": "士气", "supply": "补给", "fatigue": "疲劳度",
        "leadership": "领导力", "market_share": "市场份额", "cash_flow": "现金流",
        "brand": "品牌", "rnd": "研发", "supply_chain": "供应链",
        "support_rate": "支持率", "economy": "经济", "unity": "团结度",
        "intl_relations": "国际关系", "legislative_power": "立法权",
        "population": "人口", "resources": "资源", "pollution": "污染",
        "biodiversity": "生物多样性", "stability": "稳定性",
        "employment": "就业", "infrastructure": "基础设施", "finance": "财政",
        "satisfaction": "满意度", "tech_lead": "技术领先", "chip_stock": "芯片储备",
        "talent_pool": "人才池", "patent_barrier": "专利壁垒",
        "commercialization": "商业化", "narrative_dominance": "舆情主导",
        "public_trust": "公信力", "polarization": "极化度", "media_reach": "媒体触达",
    }
    _mn = _MN.get
    thresholds = thresholds or {}

    parts: list[str] = []
    for eid, st in states.items():
        name = getattr(st, "name", eid[:8])
        metrics = getattr(st, "metrics", {})
        if not metrics:
            parts.append(f"{name}: 无指标")
            continue
        history = getattr(st, "history", []) or []
        deltas_by_metric: dict[str, float] = {}
        for h in history[-30:]:
            m = h.get("metric", "")
            d = h.get("delta", 0)
            if m:
                deltas_by_metric[m] = deltas_by_metric.get(m, 0) + float(d)

        # 选取最关键的指标（按与淘汰线的逼近程度排序）
        scored = []
        for k, v in metrics.items():
            th = thresholds.get(k, 0)
            proximity = float(v) / max(float(th), 1.0) if th > 0 else 10.0
            scored.append((proximity, k, v, th))
        scored.sort()

        segs: list[str] = []
        for proximity, k, v, th in scored[:5]:
            label = _mn(k, k)
            level = _level_label(v)
            trend = _trend_label(deltas_by_metric.get(k, 0.0))
            cum_delta = deltas_by_metric.get(k, 0.0)
            th_text = f" 淘汰线={th:.0f}" if th > 0 else ""
            near_thresh = " ⚠逼近淘汰线" if th > 0 and v <= th * 1.3 else ""
            # 区分战略性投入（现金流下降但研发/市占上升）与危机型下跌
            qualifier = ""
            if k in ("cash_flow", "supply") and cum_delta < -3:
                rnd_d = deltas_by_metric.get("rnd", 0)
                ms_d = deltas_by_metric.get("market_share", 0)
                if rnd_d > 3 or ms_d > 3:
                    qualifier = "（战略性投入，非危机）"
            segs.append(f"{label}({level}·{trend} Δ{cum_delta:+.0f}{th_text}{near_thresh}{qualifier})")

        parts.append(f"{name}: {'; '.join(segs) if segs else '无关键指标'}")
    return "\n".join(parts)


async def generate_report(
    session: DeductionSession,
    graph: DeductionGraphStore,
    rounds: list[SimulationRound],
    log_fn: Callable[[str, str], None],
    preprocessor: Any = None,
    pre_goals: list[str] | None = None,
    states: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
    goal_resolution: str = "",
    personality_log: list[dict[str, Any]] | None = None,
) -> DeductionReport:
    from strategy_forge.core.config import config
    from strategy_forge.core.llm_client import DeductionLLMClient as LLMClient
    from strategy_forge.core.llm_client import Message

    # Collect key events from all rounds (not just last 5)
    key_events: list[str] = []
    agent_trajectories: dict[str, list[str]] = {}
    # Track per-round deltas for turning point detection
    all_deltas: list[tuple[int, str, str, float]] = []  # (round, agent, metric, delta)

    def _agent_name(aid: str) -> str:
        if states and aid in states:
            st = states[aid]
            return getattr(st, "name", "") or aid[:8]
        return aid[:8]

    for rnd in rounds:
        for action in rnd.actions:
            brief = action.content[:60].split("，")[0]
            key_events.append(f"[轮{rnd.round_number}] "
                               f"{_agent_name(action.agent_id)}: "
                               f"{action.action_type} — {brief}")
            agent_trajectories.setdefault(action.agent_id, []).append(action.content[:60])
            if hasattr(action, "metadata") and isinstance(action.metadata, dict):
                for m, v in action.metadata.get("deltas", {}).items():
                    if abs(float(v)) > 3:
                        all_deltas.append((rnd.round_number, action.agent_id[:8], m, float(v)))

    # 跨轮语义召回：从 LanceDB events 表按场景主题召回最相关事件，补足"只看最近5轮"的盲区
    if preprocessor is not None:
        try:
            query = (session.title or session.source_material[:200] or "关键转折与冲突").strip()
            recalled = preprocessor.retrieve_dynamic_events(query, max(config.deduction_retrieve_top_k, 10),
                min_similarity=config.deduction_similarity_threshold)
            for c in recalled:
                line = f"[语义召回] {c[:100]}"
                if line not in key_events:
                    key_events.append(line)
            if recalled:
                log_fn("report", f"LanceDB 语义召回 {len(recalled)} 条跨轮关键事件")
        except Exception as e:
            logger.debug("[Reporter] 语义召回关键事件失败: %s", e)

    if not key_events:
        return DeductionReport(
            session_id=session.id,
            summary="推演未产生足够事件数据以生成报告。",
            raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
        )

    # 从知识图谱取关键关系(按权重)丰富报告
    key_relations = "（无显著关系）"
    if graph is not None:
        try:
            rows = graph.query(
                "MATCH (a:Entity)-[r:RELATES]->(b:Entity) "
                "RETURN a.name, r.relation, b.name, r.weight "
                "ORDER BY r.weight DESC LIMIT 15"
            )
            rels = [f"- {r[0]} --[{r[1]}]--> {r[2]}"
                    for r in rows if r and r[0] and r[2]]
            if rels:
                key_relations = "\n".join(rels)
                log_fn("report", f"图谱关键关系 {len(rels)} 条注入报告")
        except Exception as e:
            logger.debug("[Reporter] 关系查询失败: %s", e)

    # 从 Kuzu 时序行动图(Agent-[ACTED]->Event)取全局事件序列，供因果链分析
    action_timeline = "（无行动时序记录）"
    if graph is not None:
        try:
            seq = graph.get_event_sequence(limit=30)
            lines = [f"- [{e['timestamp'][:19]}] {e['agent_name']} {e['action']}: {e['description'][:60]}"
                     for e in seq if e.get("agent_name")]
            if lines:
                action_timeline = "\n".join(lines)
                log_fn("report", f"Kuzu 行动时序 {len(lines)} 条注入报告")
        except Exception as e:
            logger.debug("[Reporter] 行动时序查询失败: %s", e)

    # 确定性因果归因：从 Kuzu CAUSED 边汇总"源→目标 影响方向"，并附加轮次和变化量级
    causal_attribution = "（无确定性因果数据）"
    if graph is not None:
        try:
            summary = graph.get_causal_summary(limit=15)

            def _causal_dir(amt: float) -> str:
                mag = "大幅" if abs(amt) >= 10 else ("中度" if abs(amt) >= 3 else "小幅")
                return ("助益" if amt > 0 else "削弱") + f"({mag})"

            clines = [f"- {s['source']} → {s['target']}: {s['metric']} {_causal_dir(float(s['amount']))}"
                       f"（{s['amount']:+.0f}）"
                       for s in summary if s.get("metric")]
            if clines:
                causal_attribution = "\n".join(clines)
                log_fn("report", f"Kuzu 确定性因果归因 {len(clines)} 条注入报告")
        except Exception as e:
            logger.debug("[Reporter] 因果归因查询失败: %s", e)

    # ── 重点转折事件：方向性变化最大的 3 个事件 ──
    turning_points = "（无显著转折事件）"
    if all_deltas:
        top_3 = sorted(all_deltas, key=lambda x: abs(x[3]), reverse=True)[:3]
        tp_lines = [f"- [R{r}] {agent} → {metric}: {'激增' if delta > 0 else '骤降'}{abs(delta):.0f}"
                     for r, agent, metric, delta in top_3]
        if tp_lines:
            turning_points = "\n".join(tp_lines)

    # 推演设定上下文
    domain_text = "叙事模式（无量化）"
    if states:
        try:
            first = next(iter(states.values()))
            d = getattr(first, "domain", "")
            if d and d != "generic":
                domain_text = d
        except (StopIteration, AttributeError):
            pass
    agent_overview = "（无智能体数据）"
    if graph is not None:
        try:
            agents = graph.query(
                f"MATCH (a:{graph.AGENT_TABLE}) RETURN a.name, a.persona ORDER BY a.name")
            if agents:
                agent_overview = "\n".join(
                    f"- {r[0]}: {r[1][:60]}" for r in agents[:12] if r[0])
                log_fn("report", f"智能体总览 {len(agents)} 个注入报告")
        except Exception:
            pass

    client = LLMClient()
    _thresholds: dict[str, float] = thresholds or {}
    quantified_context = _build_quantified_summary(rounds, states, _thresholds)
    immutable_goals = "；".join(pre_goals) if pre_goals else "（无）"
    if goal_resolution:
        immutable_goals += f"（收敛判定：{goal_resolution}）"
    numbered = [f"[事件{i+1}] {e}" for i, e in enumerate(key_events[-20:])]

    # ── 模式选择：叙事模式 vs 量化模式 ──
    is_narrative = states is None or len(states) == 0
    if is_narrative:
        # 叙事模式：故事化报告，temperature 更高鼓励创造性
        agent_count = session.agent_count
        output_len = max(2000, min(8000, agent_count * 300 + len(key_events) * 80))
        # 语义召回事件单独注入，帮助 LLM 识别跨轮关键情节
        recall_events = [e for e in key_events if "[语义召回]" in e]
        non_recall = [e for e in key_events if "[语义召回]" not in e]
        narrative_goals = ("；".join(pre_goals) if pre_goals
                           else "（用户未指定核心问题——结局需明确交代各主要角色的最终结局与格局归属，不得含糊收尾）")
        if goal_resolution:
            narrative_goals += f"\n推演中期裁判已判定收敛结果（结局必须与之一致）：{goal_resolution}"
        arc_events = _sample_arc_events(non_recall)
        evolution_lines: list[str] = []
        for p in (personality_log or [])[-30:]:
            old = p.get("old_extra") or "（初始人格）"
            evolution_lines.append(
                f"- [R{p.get('round', '?')}] {p.get('agent', '?')}: {old} → {p.get('new_extra', '')}")
        personality_evolution = "\n".join(evolution_lines) if evolution_lines else "（无人格演变记录）"
        prompt_str = Template(_REPORT_PROMPT_NARRATIVE).substitute(
            output_length=str(output_len),
            immutable_goals=narrative_goals,
            agent_overview=agent_overview,
            personality_evolution=personality_evolution,
            key_events="\n".join(arc_events),
            action_timeline=action_timeline,
            key_recall="\n".join(recall_events) if recall_events else "（无）",
        )
        log_fn("report", f"叙事收束事件采样: 全程{len(non_recall)}条 → 弧线采样{len(arc_events)}条"
                          f" | 人格演变 {len(evolution_lines)} 条注入")
        system = "你是叙事文学作家，撰写故事化推演叙事。只输出 JSON。"
        report_temp = 0.75
    else:
        prompt_str = Template(_REPORT_PROMPT).substitute(
            title=session.title or "推演会话",
            domain=domain_text,
            immutable_goals=immutable_goals,
            agent_count=session.agent_count,
            round_count=session.current_round,
            agent_overview=agent_overview,
            key_relations=key_relations,
            key_events="\n".join(numbered),
            action_timeline=action_timeline,
            quantified_context=quantified_context,
            causal_attribution=causal_attribution,
            turning_points=turning_points,
        )
        system = "你是推演分析专家，撰写自然语言推演报告。只输出 JSON。"
        report_temp = 0.4

    messages = [Message(role="user", content=prompt_str)]

    # ── 中文感知 Token 估算 + 上下文窗口安全上限 ──
    _cn = len(re.findall(r"[\u4e00-\u9fff]", prompt_str))
    _input_est = _cn + max(1, (len(prompt_str) - _cn) // 3)
    _ctx_limit = 262144  # 主流模型上下文窗口上限，400超限时由重试自动修正
    _safe_max = max(2000, _ctx_limit - _input_est - 200)
    report_max_tokens = min(config.deduction_report_max_tokens, _safe_max)
    log_fn("report", f"Token估算: input≈{_input_est} max_tokens={report_max_tokens}")

    default_report = DeductionReport(
        session_id=session.id,
        summary="推演完成，请查看详细事件记录。",
        key_events=[{"description": e} for e in key_events[-20:]],
        agent_trajectories=agent_trajectories,
        raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
    )

    try:
        response = await client.chat(messages, system=system, temperature=report_temp,
                                     max_tokens=report_max_tokens)
        content = extract_text(response)
        report_data = _parse_report_json(content)
    except Exception as e:
        # 上下文超限 → 用减半的 max_tokens 重试一次
        if "400" in str(e) and report_max_tokens > 1500:
            _retry = max(1500, report_max_tokens // 2)
            log_fn("report", f"LLM 调用失败(可能上下文超限)，减半重试(max_tokens={_retry})")
            try:
                response = await client.chat(messages, system=system, temperature=report_temp,
                                             max_tokens=_retry)
                content = extract_text(response)
                report_data = _parse_report_json(content)
            except Exception as e2:
                logger.warning("[Deduction] 报告 LLM 重试失败，使用默认摘要: %s", e2)
                return default_report
        else:
            logger.warning("[Deduction] 报告 LLM 调用失败，使用默认摘要: %s", e)
            return default_report

    log_fn("report", "报告 LLM 生成完成")

    # 归一化：模型可能返回 dict 列表而非 | 分隔字符串，统一转换避免前端崩溃
    raw_risks = report_data.get("risk_alerts", [])
    normalized_risks: list[str] = []
    for item in raw_risks:
        if isinstance(item, str):
            normalized_risks.append(item)
        elif isinstance(item, dict):
            title = item.get("风险标题") or item.get("risk_title") or item.get("标题", "")
            path = item.get("具体触发机制/路径") or item.get("trigger_path") or item.get("触发路径", "")
            target = item.get("受影响方") or item.get("affected_entity") or item.get("受方", "")
            normalized_risks.append(f"{title} | {path} | {target}")
        else:
            normalized_risks.append(str(item))

    raw_recs = report_data.get("recommendations", [])
    normalized_recs: list[str] = []
    for item in raw_recs:
        if isinstance(item, str):
            normalized_recs.append(item)
        elif isinstance(item, dict):
            agent = item.get("针对方") or item.get("agent") or ""
            action = item.get("具体动作") or item.get("action") or ""
            effect = item.get("预期机制与效果") or item.get("effect") or ""
            normalized_recs.append(f"{agent}→{action}→{effect}")
        else:
            normalized_recs.append(str(item))

    # 归一化：确保 summary/conclusion 始终为字符串
    narrative_raw = report_data.get("narrative", "") or report_data.get("summary", default_report.summary)
    conclusion_raw = report_data.get("conclusion", "")
    narrative = str(narrative_raw) if not isinstance(narrative_raw, str) else narrative_raw
    conclusion = str(conclusion_raw) if not isinstance(conclusion_raw, str) else conclusion_raw

    return DeductionReport(
        session_id=session.id,
        summary=narrative,
        key_events=[{"description": e} for e in key_events[-20:]],
        agent_trajectories=default_report.agent_trajectories,
        risk_alerts=normalized_risks,
        recommendations=normalized_recs,
        causal_summary=report_data.get("causal_summary", []),
        stage_narratives=report_data.get("stage_narratives", []) or report_data.get("character_arcs", []),
        deviation_analysis=report_data.get("deviation_analysis", []),
        conclusion=conclusion,
        raw_graph_stats={"entities": session.entity_count, "relations": session.relation_count},
    )


def _parse_report_json(raw: str) -> dict[str, Any]:
    from ._utils import extract_json
    data = extract_json(raw)
    return data if isinstance(data, dict) else {}

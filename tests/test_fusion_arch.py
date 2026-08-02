"""融合架构专项测试 — 事件+数值双重驱动逻辑验证（无 LLM 依赖，确定性）。

覆盖：
  A. 规则包解析：event_impact / event_triggers 默认值 + 求值函数
  B. 通道①事件→数值冲击：agent 主动行动不双重结算；系统事件结算；外部注入事件结算
  C. 通道②数值→事件触发：阈值越界生成系统事件 + once 去重 + LanceDB 持久化
  D. 事件分发去重：二次 dispatch 不重复注入知识队列
  E. 盲点4：DeductionEngine.inject_system_event 队列 → SimulationEngine 通道①消费
  F. 模式边界：叙事分支不调用融合通道（边界不被模糊）

运行：python tests/test_fusion_arch.py  或 pytest tests/test_fusion_arch.py
"""
import os
import sys

sys.path.insert(0, "src")


class TestRuleEngineExt:
    """A. 规则包 event_impact / event_triggers 扩展。"""

    def test_with_defaults(self):
        from strategy_forge.engine.rule_engine import RuleEngine

        re = RuleEngine({"domain": "generic", "metrics": ["m1"], "initial_metrics": {"m1": 50}})
        assert re.event_impact_map() == {}
        assert re.event_triggers() == []

    def test_business_pack(self):
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.rule_engine import RuleEngine

        re = RuleEngine(get_template("business"))
        assert "embargo" in re.event_impact_map()
        assert len(re.event_triggers()) >= 1

    def test_all_domains_fusion_rules(self):
        """所有量化域都应配置 event_impact + event_triggers。"""
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.rule_engine import RuleEngine

        domains = ["military", "politics", "ecology", "urban",
                   "tech", "info_war", "geo_strategy", "business"]
        for dom in domains:
            re = RuleEngine(get_template(dom))
            assert re.event_impact_map(), f"{dom} missing event_impact"
            assert re.event_triggers(), f"{dom} missing event_triggers"

    def test_trigger_threshold_above_death(self):
        """触发阈值应高于死亡阈值（预警型），避免触发永远晚于出局。"""
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.rule_engine import RuleEngine

        domains = ["military", "politics", "ecology", "urban",
                   "tech", "info_war", "geo_strategy", "business"]
        for dom in domains:
            re = RuleEngine(get_template(dom))
            thr = re.thresholds()
            for t in re.event_triggers():
                m = t.get("metric", "")
                op = t.get("op", ">=")
                val = float(t.get("value", 0))
                death = thr.get(m)
                if death is None:
                    continue  # 该指标非死亡阈值指标（如 morale 在 business 中）
                if op == "<":
                    assert val > death, \
                        f"{dom} trigger {m}<{val} must be above death {death}"
                elif op == ">":
                    assert val < death, \
                        f"{dom} trigger {m}>{val} must be below death {death}"

    def test_military_channel2(self):
        """military 域通道②：supply<25 触发后勤补给危机。"""
        from strategy_forge.engine.simulator import SimulationEngine
        from strategy_forge.engine.rule_engine import RuleEngine
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.models import EntityState

        re = RuleEngine(get_template("military"))
        full = {"strength": 50, "morale": 50, "supply": 20, "fatigue": 30, "leadership": 50}
        eng = object.__new__(SimulationEngine)
        eng._quantified = True
        eng._rule_engine = re
        eng._states = {"m1": EntityState(id="m1", name="甲军", metrics=dict(full))}
        eng._event_history = []
        eng._log = lambda p, m: None
        eng._persist_events = False
        eng._preprocessor = None
        n = eng._trigger_events_from_metrics(1)
        assert n >= 1, "military supply crisis should trigger"
        sys_events = [e for e in eng._event_history if e.get("is_system_event")]
        assert sys_events and sys_events[0]["event_type"] == "后勤补给危机"

    def test_eval_metric_op(self):
        from strategy_forge.engine.rule_engine import RuleEngine

        assert RuleEngine.eval_metric_op(">=", 85, 80) is True
        assert RuleEngine.eval_metric_op(">=", 70, 80) is False
        assert RuleEngine.eval_metric_op("<", 5, 15) is True
        assert RuleEngine.eval_metric_op("!=", 5, 15) is True
        assert RuleEngine.eval_metric_op("bad", 5, 15) is False


class TestChannel1:
    """B. 通道①事件→数值冲击。"""

    def _engine(self, injected=None):
        from strategy_forge.engine.simulator import SimulationEngine
        from strategy_forge.engine.rule_engine import RuleEngine
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.models import EntityState

        re = RuleEngine(get_template("business"))
        full = {"market_share": 30, "cash_flow": 50, "brand": 50, "rnd": 50,
                "morale": 50, "supply_chain": 50, "tech_lead": 50}
        eng = object.__new__(SimulationEngine)
        eng._quantified = True
        eng._rule_engine = re
        eng._states = {
            "e1": EntityState(id="e1", name="A公司", metrics=dict(full)),
            "e2": EntityState(id="e2", name="B公司", metrics=dict(full)),
        }
        eng._event_history = []
        eng._log = lambda p, m: None
        eng._injected_events_store = {"pending": injected or []}
        return eng, re

    def test_agent_action_no_double_settle(self):
        """agent 主动行动（is_system_event=False）不应触发事件冲击（避免双重结算）。"""
        eng, _ = self._engine()
        eng._append_event({"agent": "e1", "target_id": "e2", "action": "embargo",
                           "event_type": "embargo", "round": 1, "is_system_event": False})
        n = eng._apply_event_impacts(1)
        assert n == 0, f"agent action must not double-settle, got {n}"
        assert eng._states["e2"].get_metric("supply_chain") == 50

    def test_system_event_applies(self):
        """系统事件（is_system_event=True）应触发事件冲击。"""
        eng, _ = self._engine()
        eng._append_event({"agent": "e1", "target_id": "e2", "action": "system_trigger",
                           "event_type": "embargo", "round": 1, "is_system_event": True})
        n = eng._apply_event_impacts(1)
        assert n == 1, f"system event should apply, got {n}"
        assert eng._states["e2"].get_metric("supply_chain") == 30, \
            f"supply_chain should drop to 30, got {eng._states['e2'].get_metric('supply_chain')}"

    def test_injected_event_consumed(self):
        """外部注入事件（盲点4）经通道①消费。"""
        eng, _ = self._engine(injected=[
            {"event_type": "embargo", "content": "外部制裁B公司",
             "target_id": "e2", "round": 1}])
        n = eng._apply_event_impacts(1)
        assert n == 1, f"injected event should apply, got {n}"
        assert eng._states["e2"].get_metric("supply_chain") == 30
        assert eng._injected_events_store["pending"] == []

    def test_unknown_event_type_skipped(self):
        """未匹配 event_impact 的事件不结算。"""
        eng, _ = self._engine()
        eng._append_event({"agent": "e1", "target_id": "e2", "action": "some_unknown",
                           "event_type": "some_unknown", "round": 1, "is_system_event": True})
        n = eng._apply_event_impacts(1)
        assert n == 0


class TestChannel2:
    """C. 通道②数值→事件触发。"""

    def _engine(self):
        from strategy_forge.engine.simulator import SimulationEngine
        from strategy_forge.engine.rule_engine import RuleEngine
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.models import EntityState

        re = RuleEngine(get_template("business"))
        full = {"market_share": 30, "cash_flow": 50, "brand": 50, "rnd": 50,
                "morale": 5, "supply_chain": 50, "tech_lead": 50}
        eng = object.__new__(SimulationEngine)
        eng._quantified = True
        eng._rule_engine = re
        eng._states = {"e1": EntityState(id="e1", name="A公司", metrics=dict(full))}
        eng._event_history = []
        eng._log = lambda p, m: None
        eng._persist_events = True
        calls = []

        class FakePP:
            def add_event_memory(self, **kw):
                calls.append(kw)

        eng._preprocessor = FakePP()
        eng._pp_calls = calls
        return eng, re

    def test_threshold_trigger(self):
        """morale<20 应触发离职潮事件。"""
        eng, _ = self._engine()
        n = eng._trigger_events_from_metrics(1)
        assert n >= 1, f"expected >=1 trigger, got {n}"
        sys_events = [e for e in eng._event_history if e.get("is_system_event")]
        assert sys_events, "system event should be appended"
        assert sys_events[0]["event_type"] == "员工大规模离职潮"

    def test_persist_to_lancedb(self):
        """系统事件应持久化到 LanceDB。"""
        eng, _ = self._engine()
        eng._trigger_events_from_metrics(1)
        assert len(eng._pp_calls) >= 1, "system event must persist to LanceDB"
        assert any(c.get("event_type", "").startswith("system_") for c in eng._pp_calls)

    def test_once_dedup(self):
        """once 触发在同轮/后续轮去重。"""
        eng, _ = self._engine()
        eng._trigger_events_from_metrics(1)
        n1 = eng._trigger_events_from_metrics(2)
        assert n1 == 0, f"once trigger must not re-fire, got {n1}"


class TestDispatchDedup:
    """D. 事件分发去重。"""

    def test_no_duplicate_dispatch(self):
        from strategy_forge.engine.simulator import SimulationEngine

        eng = object.__new__(SimulationEngine)
        eng._event_history = [{"agent": "e1", "agent_name": "A", "content": "x",
                               "round": 1, "visibility": "public", "participants": ""}]
        eng.agents = [type("Ag", (), {"entity_id": "e2", "name": "B"})()]
        eng._states = {"e2": {}}
        eng._name_to_id = {"A": "e1", "B": "e2"}
        eng.reasoner = type("R", (), {"get_trust": lambda *a, **k: 0.0})()
        eng._intel_bonuses = {}
        eng._agent_knowledge = {}
        eng._dispatched_eids = set()
        eng._dispatch_events(1)
        eng._dispatch_events(1)
        assert len(eng._agent_knowledge.get("e2", [])) == 1, "must not duplicate"


class TestBlindspot4:
    """E. 外部事件注入链路（DeductionEngine → SimulationEngine）。"""

    def test_inject_queue(self):
        from strategy_forge.engine.engine import DeductionEngine

        eng = DeductionEngine.__new__(DeductionEngine)
        eng._injected_events = {}
        # 隔离 log（__new__ 未初始化 session_store）
        eng.log = lambda *a, **k: None
        store = eng.get_injected_events_store("s1")
        assert store == {"pending": []}
        eng.inject_system_event("s1", {"event_type": "embargo", "content": "外部制裁"})
        store2 = eng.get_injected_events_store("s1")
        assert len(store2["pending"]) == 1
        assert store2["pending"][0]["event_type"] == "embargo"


class TestNoiseDefense:
    """G. 量化事件噪音防御：系统事件限幅 + 去重。"""

    def _build_recent_ctx(self, n_sys):
        """复刻 _run_round_quantified 中 _recent_ctxs 构建逻辑，验证限幅。"""
        import inspect
        from strategy_forge.engine.simulator import SimulationEngine

        eng = object.__new__(SimulationEngine)
        eng._event_history = []
        # 构造 n_sys 个不同系统事件 + 1 个自身事件
        for i in range(n_sys):
            eng._event_history.append({
                "agent": f"sys{i}", "agent_name": f"系统{i}", "content": f"系统事件{i}",
                "round": 1, "event_type": f"sys_{i}", "is_system_event": True,
                "visibility": "public", "participants": "",
            })
        eng._event_history.append({
            "agent": "e1", "agent_name": "A", "content": "自身行动",
            "round": 1, "event_type": "invest", "is_system_event": False,
            "visibility": "public", "participants": "",
        })
        eng.agents = [type("Ag", (), {"entity_id": "e1", "name": "A"})()]
        eng._states = {"e1": {}}
        eng._name_to_id = {"A": "e1"}
        eng.reasoner = type("R", (), {"get_trust": lambda *a, **k: 0.0})()
        eng._intel_bonuses = {}
        eng._agent_knowledge = {}
        eng._dispatched_eids = set()
        eng._deliver_ripe_knowledge = lambda *a, **k: []

        items = []
        sys_seen = 0
        _sys_limit = 2
        own_events = [e for e in eng._event_history[-8:]
                      if e.get("is_system_event")
                      or e.get("agent") == "e1"
                      or "A" in e.get("content", "")]
        _shown_sys = set()
        for e in own_events:
            if e.get("is_system_event"):
                sig = (e.get("round"), e.get("event_type"))
                if sig in _shown_sys:
                    continue
                if sys_seen >= _sys_limit:
                    continue
                _shown_sys.add(sig)
                sys_seen += 1
            text = e.get("content", "")[:80]
            items.append(text)
        return items, sys_seen

    def test_system_event_capped(self):
        """系统事件即使很多，每轮每 agent 最多注入 _sys_limit 条。"""
        items, sys_seen = self._build_recent_ctx(n_sys=8)
        assert sys_seen <= 2, f"system events must be capped at 2, got {sys_seen}"
        n_sys_items = sum(1 for t in items if t.startswith("系统事件"))
        assert n_sys_items <= 2, f"only <=2 system events shown, got {n_sys_items}"

    def test_self_event_not_suppressed(self):
        """自身事件应保留（不被系统事件挤掉）。"""
        items, _ = self._build_recent_ctx(n_sys=8)
        assert any("自身行动" in t for t in items), "self event must remain visible"


class TestDefectRegression:
    """H. 深度核查缺陷回归验证。"""

    def test_pause_resume_fired_roundtrip(self):
        """缺陷1：_event_trigger_fired JSON 往返后 restore 不崩溃。"""
        import json
        from strategy_forge.engine.simulator import SimulationEngine

        eng = object.__new__(SimulationEngine)
        eng._event_history = []
        eng._narrative_env = {}
        eng._agent_knowledge = {}
        eng._intel_bonuses = {}
        eng._personality_log = []
        eng._character_journal = {}
        eng._reflection_baselines = {}
        eng._env_snapshots = {}
        eng._last_reflection_round_n = {}
        eng._last_round_outcomes = {}
        eng._prev_rel_map = {}
        eng.agents = []
        eng._event_trigger_fired = {("e1", "员工大规模离职潮")}

        saved = eng.get_state()
        blob = json.dumps(saved)  # 模拟 SQLite 往返
        restored = json.loads(blob)
        eng2 = object.__new__(SimulationEngine)
        eng2.agents = []
        eng2.restore_state(restored)  # 不应抛异常
        assert ("e1", "员工大规模离职潮") in eng2._event_trigger_fired

    def test_dead_entity_not_settled(self):
        """缺陷7：已出局实体不被事件冲击结算。"""
        from strategy_forge.engine.simulator import SimulationEngine
        from strategy_forge.engine.rule_engine import RuleEngine
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.models import EntityState

        re = RuleEngine(get_template("business"))
        full = {"market_share": 30, "cash_flow": 50, "brand": 50, "rnd": 50,
                "morale": 50, "supply_chain": 5, "tech_lead": 50}
        eng = object.__new__(SimulationEngine)
        eng._quantified = True
        eng._rule_engine = re
        eng._states = {"e2": EntityState(id="e2", name="B公司", metrics=dict(full))}
        eng._event_history = []
        eng._log = lambda p, m: None
        eng._injected_events_store = {"pending": [
            {"event_type": "embargo", "content": "制裁", "target_id": "e2", "round": 1}]}
        eng._apply_event_impacts(1)
        assert eng._states["e2"].get_metric("supply_chain") == 5, \
            "dead entity must not be settled"

    def test_name_target_resolution(self):
        """缺陷4：target_id 传实体名可解析到 id。"""
        from strategy_forge.engine.simulator import SimulationEngine
        from strategy_forge.engine.rule_engine import RuleEngine
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.models import EntityState

        re = RuleEngine(get_template("business"))
        full = {"market_share": 30, "cash_flow": 50, "brand": 50, "rnd": 50,
                "morale": 50, "supply_chain": 50, "tech_lead": 50}
        eng = object.__new__(SimulationEngine)
        eng._quantified = True
        eng._rule_engine = re
        eng._states = {"e2": EntityState(id="e2", name="B公司", metrics=dict(full))}
        eng._event_history = []
        eng._log = lambda p, m: None
        eng._injected_events_store = {"pending": [
            {"event_type": "embargo", "content": "制裁", "target_id": "B公司", "round": 1}]}
        eng.agents = [type("Ag", (), {"name": "B公司", "entity_id": "e2"})()]
        eng._apply_event_impacts(1)
        assert eng._states["e2"].get_metric("supply_chain") == 30, \
            "name target_id must resolve to entity"

    def test_custom_impact_applies(self):
        """缺陷5：事件自带自定义 impact（未匹配规则包）应生效。"""
        from strategy_forge.engine.simulator import SimulationEngine
        from strategy_forge.engine.rule_engine import RuleEngine
        from strategy_forge.core.rule_templates import get_template
        from strategy_forge.engine.models import EntityState

        re = RuleEngine(get_template("business"))
        full = {"market_share": 30, "cash_flow": 50, "brand": 50, "rnd": 50,
                "morale": 50, "supply_chain": 50, "tech_lead": 50}
        eng = object.__new__(SimulationEngine)
        eng._quantified = True
        eng._rule_engine = re
        eng._states = {"e1": EntityState(id="e1", name="A公司", metrics=dict(full))}
        eng._event_history = []
        eng._log = lambda p, m: None
        eng._injected_events_store = {"pending": [
            {"event_type": "custom_shock", "content": "自定义冲击",
             "target_id": "e1", "impact": {"cash_flow": -15}, "round": 1}]}
        eng.agents = [type("Ag", (), {"name": "A公司", "entity_id": "e1"})()]
        eng._apply_event_impacts(1)
        assert eng._states["e1"].get_metric("cash_flow") == 35, \
            "custom impact must apply"


class TestReportNumericFilter:
    """I. 报告数值过滤：LLM 自创的具体数字应替换为定性趋势词。"""

    def _f(self, t):
        from strategy_forge.engine.reporter import _strip_numeric_figures
        return _strip_numeric_figures(t)

    def test_percent_stripped(self):
        assert "62%" not in self._f("华为占据62%份额且品牌上升")
        assert "较大比例" in self._f("华为占据62%份额且品牌上升")

    def test_percent_range_stripped(self):
        out = self._f("库迪核心单品提价30%-60%以修复利润")
        assert "30%" not in out and "60%" not in out
        assert "显著幅度" in out

    def test_number_with_trend_word(self):
        out = self._f("现金流下降12")
        assert "12" not in out
        assert "下降明显" in out

    def test_event_ref_preserved(self):
        out = self._f("[事件106]中华为的invest_rnd动作 → 直接导致技术领先度持续高位")
        assert "[事件106]" in out, "event reference must be preserved"

    def test_chinese_fraction(self):
        assert "数成" in self._f("提价3成以修复利润")

    def test_round_number_stripped(self):
        out = self._f("轮次10完成")
        assert "显著" in out

    def test_date_digits_preserved(self):
        """修复1：时间数字（年/月/日/季度）不被替换为'显著'。"""
        out = self._f("2026年7月的商业格局")
        assert "2026年7月" in out, f"date digits must be preserved: {out}"
        assert "显著年" not in out, f"must not become 显著年: {out}"

    def test_quarter_and_day_preserved(self):
        out = self._f("第3季度末完成转型，15日发布新品")
        assert "第3季度" in out and "15日" in out, f"time refs must be preserved: {out}"

    def test_report_prompt_forbids_time(self):
        """修复2：量化报告 prompt 开篇禁止时间描述。"""
        import inspect
        from strategy_forge.engine import reporter
        src = inspect.getsource(reporter)
        assert "严禁以时间开头或描述时间背景" in src, "prompt must forbid time in opening"


class TestModeBoundary:
    """F. 模式边界：叙事模式不调用融合通道（边界不被模糊）。"""

    def test_narrative_uses_agent_decide(self):
        import inspect
        from strategy_forge.engine.simulator import SimulationEngine

        # run_round 仍按 _quantified 分流
        src = inspect.getsource(SimulationEngine.run_round)
        assert "_quantified" in src, "run_round must still branch by _quantified"
        # 叙事分支的 _agent_decide 不应调用融合通道
        decide_src = inspect.getsource(SimulationEngine._agent_decide)
        assert "_apply_event_impacts" not in decide_src, "narrative must not call channel1"
        assert "_trigger_events_from_metrics" not in decide_src, "narrative must not call channel2"

    def test_fusion_channels_only_in_quantified(self):
        import inspect
        from strategy_forge.engine.simulator import SimulationEngine

        q_src = inspect.getsource(SimulationEngine._run_round_quantified)
        assert "_apply_event_impacts" in q_src
        assert "_trigger_events_from_metrics" in q_src


if __name__ == "__main__":
    import traceback

    failed = 0
    total = 0
    for cls in (TestRuleEngineExt, TestChannel1, TestChannel2,
                TestDispatchDedup, TestBlindspot4, TestNoiseDefense,
                TestDefectRegression, TestReportNumericFilter, TestModeBoundary):
        for name in dir(cls):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                getattr(cls(), name)()
                print(f"PASS {cls.__name__}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL {cls.__name__}.{name}")
                traceback.print_exc()
    print(f"\n{'='*50}\nTOTAL: {total}  PASSED: {total - failed}  FAILED: {failed}")
    sys.exit(1 if failed else 0)

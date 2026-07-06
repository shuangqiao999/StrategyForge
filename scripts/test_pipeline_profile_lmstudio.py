"""全流程 5 阶段 Token/耗时 剖析 —— 连接本地 LM Studio（仅测量，不改引擎逻辑）。

用法（项目根目录）：
    python scripts/test_pipeline_profile_lmstudio.py

做法：运行时 monkeypatch DeductionLLMClient.chat 与 preprocessor 的嵌入方法，
把每次 LLM/嵌入调用的 token 与耗时归因到「阶段 / 轮次 / 调用点」，
跑完整 engine.start（~10 agent × 3 轮），最后打印分阶段与热点排行。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
import uuid
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_profile_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")
os.environ.setdefault("FORGE_MAX_AGENTS", "10")   # 控制规模 → ~10 agent

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SOURCE = (
    "2027年，围绕台海与南海的地区博弈进入高强度阶段。美国推动对华军事遏制并强化同盟；"
    "中国坚持维护主权与地区稳定；日本加速防卫转型；韩国在中美之间谨慎平衡；"
    "俄罗斯深化对华协作牵制美国；印度奉行战略自主；澳大利亚强化与美同盟；"
    "菲律宾在南海问题上倚美；越南多方下注；朝鲜以核力量制衡。各方围绕军事部署、"
    "同盟体系、经济制裁与外交博弈展开长期较量，胜负取决于实力、意志与联盟稳固程度。"
)

LLM_CALLS: list[dict] = []
EMBED_CALLS: list[dict] = []


def _check_server() -> bool:
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/models", timeout=5).read()
        return True
    except Exception as e:
        print(f"[致命] 无法连接 LM Studio: {base}/models -> {e}")
        return False


def _discover_chat_model() -> str:
    if os.environ.get("FORGE_LLM_MODEL"):
        return os.environ["FORGE_LLM_MODEL"]
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        chat = [m for m in ids if "embed" not in m.lower()]
        pick = (next((m for m in chat if "9b" in m.lower()), None)
                or next((m for m in chat if "2b" in m.lower()), None)
                or (chat[0] if chat else "local-chat"))
        os.environ["FORGE_LLM_MODEL"] = pick
        return pick
    except Exception:
        os.environ.setdefault("FORGE_LLM_MODEL", "local-chat")
        return os.environ["FORGE_LLM_MODEL"]


def _site_tag(system: str, prompt: str) -> str:
    """从 system/prompt 文案推断调用点标签。"""
    s = (system or "") + " " + (prompt or "")[:120]
    pairs = [
        ("本体", "ontology"), ("知识图谱", "graph.extract"), ("JSON array", "graph.extract"),
        ("情报分析", "intel_sorter"), ("领域分类", "detect_domain"),
        ("character profile", "agent.persona"), ("画像", "agent.persona"),
        ("量化推演中的战略决策", "sim.reason"), ("种子", "seed_extract"),
        ("战略分析师", "report"), ("叙事", "sim.narrate"),
    ]
    for kw, tag in pairs:
        if kw in s:
            return tag
    return "other"


def _install_probes():
    from strategy_forge.core.llm_client import DeductionLLMClient, Message
    from strategy_forge.core.token_counter import _current_phase, _current_round
    from strategy_forge.engine.preprocessor import DeductionPreprocessor

    _orig_chat = DeductionLLMClient.chat

    async def chat_wrap(self, messages, system="", **kw):
        pc = len(system or "")
        for m in messages:
            if isinstance(m, Message):
                pc += len(m.content or "")
            elif isinstance(m, dict):
                pc += len(str(m.get("content", "")))
        prompt0 = ""
        if messages:
            m0 = messages[0]
            prompt0 = m0.content if isinstance(m0, Message) else str(m0.get("content", ""))
        t0 = time.monotonic()
        resp = await _orig_chat(self, messages, system=system, **kw)
        dur = (time.monotonic() - t0) * 1000
        ts = getattr(resp, "token_stats", None)
        LLM_CALLS.append({
            "phase": _current_phase.get(), "round": _current_round.get(),
            "site": _site_tag(system, prompt0), "prompt_chars": pc,
            "ptok": getattr(ts, "prompt_tokens", 0) if ts else 0,
            "ctok": getattr(ts, "completion_tokens", 0) if ts else 0,
            "dur_ms": getattr(ts, "duration_ms", 0) if ts else int(dur),
        })
        return resp

    DeductionLLMClient.chat = chat_wrap

    _orig_single = DeductionPreprocessor._sync_embed_single
    _orig_batch = DeductionPreprocessor._sync_embed_batch

    def single_wrap(self, text):
        t0 = time.monotonic()
        r = _orig_single(self, text)
        EMBED_CALLS.append({"kind": "single", "n": 1, "chars": len(text or ""),
                            "dur_ms": (time.monotonic() - t0) * 1000,
                            "phase": _current_phase.get()})
        return r

    def batch_wrap(self, texts):
        t0 = time.monotonic()
        r = _orig_batch(self, texts)
        EMBED_CALLS.append({"kind": "batch", "n": len(texts), "chars": sum(len(t or "") for t in texts),
                            "dur_ms": (time.monotonic() - t0) * 1000,
                            "phase": _current_phase.get()})
        return r

    DeductionPreprocessor._sync_embed_single = single_wrap
    DeductionPreprocessor._sync_embed_batch = batch_wrap


def _fmt(n):
    return f"{n:,}"


def _report(engine, sid, wall_total):
    from strategy_forge.core.token_counter import accumulator

    print("\n" + "=" * 74)
    print("  全流程 5 阶段 Token/耗时 剖析报告")
    print("=" * 74)

    # 阶段耗时（解析日志 "阶段 X 耗时 Ys"）
    phase_wall: dict[str, float] = {}
    for lg in engine.get_logs(sid, limit=1000):
        m = lg["message"]
        if "阶段" in m and "耗时" in m and "s" in m:
            try:
                name = m.split("阶段", 1)[1].split("耗时")[0].strip()
                secs = float(m.split("耗时")[1].replace("s", "").strip().split()[0])
                phase_wall[name] = secs
            except Exception:
                pass

    # 按阶段聚合 LLM
    byp = defaultdict(lambda: {"calls": 0, "ptok": 0, "ctok": 0, "ms": 0, "maxchars": 0})
    for c in LLM_CALLS:
        d = byp[c["phase"]]
        d["calls"] += 1
        d["ptok"] += c["ptok"]
        d["ctok"] += c["ctok"]
        d["ms"] += c["dur_ms"]
        d["maxchars"] = max(d["maxchars"], c["prompt_chars"])

    print(f"\n总 LLM 调用: {len(LLM_CALLS)}  |  总嵌入调用: {len(EMBED_CALLS)}  |  全程墙钟: {wall_total:.1f}s")
    print(f"{'阶段':<14}{'调用':>5}{'prompt_tok':>12}{'compl_tok':>11}{'LLM耗时s':>10}{'阶段墙钟s':>11}{'最大prompt字符':>14}")
    print("-" * 74)
    order = ["ontology", "quantify", "graph", "agents", "simulation", "report", "unknown"]
    seen = set()
    for ph in order + [p for p in byp if p not in order]:
        if ph in seen or ph not in byp:
            continue
        seen.add(ph)
        d = byp[ph]
        wall = phase_wall.get({"graph": "graph", "simulation": "simulation"}.get(ph, ph), 0)
        # 阶段名与日志名的对应（日志用中文阶段标签的英文 phase 名）
        print(f"{ph:<14}{d['calls']:>5}{_fmt(d['ptok']):>12}{_fmt(d['ctok']):>11}"
              f"{d['ms']/1000:>10.1f}{wall:>11.1f}{_fmt(d['maxchars']):>14}")

    # 模拟阶段逐轮
    byr = defaultdict(lambda: {"calls": 0, "ptok": 0, "ctok": 0})
    for c in LLM_CALLS:
        if c["phase"] == "simulation" and c["round"]:
            d = byr[c["round"]]
            d["calls"] += 1
            d["ptok"] += c["ptok"]
            d["ctok"] += c["ctok"]
    if byr:
        print("\n-- 模拟阶段逐轮 --")
        for r in sorted(byr):
            d = byr[r]
            print(f"  第{r}轮: {d['calls']} 调用, prompt {_fmt(d['ptok'])} tok, 输出 {_fmt(d['ctok'])} tok")

    # 按调用点聚合
    bys = defaultdict(lambda: {"calls": 0, "ptok": 0, "ctok": 0, "chars": 0})
    for c in LLM_CALLS:
        d = bys[c["site"]]
        d["calls"] += 1
        d["ptok"] += c["ptok"]
        d["ctok"] += c["ctok"]
        d["chars"] += c["prompt_chars"]
    print("\n-- 按调用点（site）--")
    print(f"{'调用点':<18}{'调用':>5}{'prompt_tok':>12}{'compl_tok':>11}{'均prompt字符':>13}")
    for site, d in sorted(bys.items(), key=lambda x: -x[1]["ptok"]):
        avg = d["chars"] // max(1, d["calls"])
        print(f"{site:<18}{d['calls']:>5}{_fmt(d['ptok']):>12}{_fmt(d['ctok']):>11}{_fmt(avg):>13}")

    # Top-10 最大单次 prompt
    print("\n-- Top-10 最大单次 prompt --")
    for c in sorted(LLM_CALLS, key=lambda x: -x["prompt_chars"])[:10]:
        print(f"  {c['phase']:<11} {c['site']:<16} {_fmt(c['prompt_chars']):>8} 字符  "
              f"prompt {_fmt(c['ptok'])} tok / 出 {_fmt(c['ctok'])} tok / {c['dur_ms']/1000:.1f}s")

    # 嵌入
    emb_n = sum(e["n"] for e in EMBED_CALLS)
    emb_ms = sum(e["dur_ms"] for e in EMBED_CALLS)
    emb_by_phase = defaultdict(lambda: [0, 0.0])
    for e in EMBED_CALLS:
        emb_by_phase[e["phase"]][0] += e["n"]
        emb_by_phase[e["phase"]][1] += e["dur_ms"]
    print(f"\n-- 嵌入调用 --  批次 {len(EMBED_CALLS)}, 文本 {emb_n} 条, 总耗时 {emb_ms/1000:.1f}s")
    for ph, (n, ms) in sorted(emb_by_phase.items(), key=lambda x: -x[1][1]):
        print(f"  {ph:<12} {n} 条, {ms/1000:.1f}s")

    # 权威汇总（accumulator）
    stats = accumulator.get_session_stats(sid)
    if stats:
        print(f"\n-- accumulator 权威汇总 -- 总 tokens: {_fmt(stats.get('total_tokens', 0))} "
              f"(prompt {_fmt(stats.get('total_prompt_tokens', 0))} / compl {_fmt(stats.get('total_completion_tokens', 0))})")

    tot_p = sum(c["ptok"] for c in LLM_CALLS)
    tot_c = sum(c["ctok"] for c in LLM_CALLS)
    print(f"\n结论速览: 总 prompt {_fmt(tot_p)} tok / 输出 {_fmt(tot_c)} tok；"
          f"prompt 占比 {100*tot_p/max(1, tot_p+tot_c):.0f}%")
    print("按 prompt_tok 排序的阶段热点: " + ", ".join(
        f"{p}({_fmt(byp[p]['ptok'])})" for p in sorted(byp, key=lambda x: -byp[x]["ptok"])[:3]))
    print("按耗时排序的阶段热点: " + ", ".join(
        f"{p}({phase_wall.get(p, byp[p]['ms']/1000):.0f}s)"
        for p in sorted(byp, key=lambda x: -(phase_wall.get(x, byp[x]['ms']/1000)))[:3]))


async def main() -> int:
    print("=== 全流程剖析（LM Studio）===")
    if not _check_server():
        return 2
    chat = _discover_chat_model()
    print("对话模型:", chat, "| 嵌入模型:", os.environ["FORGE_EMBED_MODEL"], "| FORGE_MAX_AGENTS=10")

    _install_probes()
    from strategy_forge.engine.engine import DeductionEngine

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    engine = DeductionEngine(workspace_root=root)
    session = engine.create_session(
        title="亚太安全博弈推演",
        source_material=SOURCE,
        config={"domain": "military", "total_rounds": 3, "enable_narrate": True,
                "enable_multi_action": True, "max_actions": 2},
    )
    sid = session.id
    print(f"会话 {sid} 开始全流程推演（~10 agent × 3 轮）...\n")
    t0 = time.monotonic()
    await engine.start(sid)
    wall = time.monotonic() - t0

    data = engine.session_store.get(sid)
    print(f"最终状态: {data.get('status')}  实体: {data.get('entity_count')}  "
          f"agent: {data.get('agent_count')}")
    _report(engine, sid, wall)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

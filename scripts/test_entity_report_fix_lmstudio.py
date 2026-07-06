"""实体消解增强 + 报告结论去重 全流程验证（连接本地 LM Studio）。

用法（项目根目录）：
    python scripts/test_entity_report_fix_lmstudio.py

验证：
  - 智能体列表不再包含二元关系词(俄乌/美伊)、军队编制(第五舰队)、政府部门(财政部/国防部)、
    职务头衔(北约秘书长)——被 intel 提示词/安全网降级；
  - 报告 conclusion 与 narrative 不再逐字重复（结论是独立凝练文本）；
  - 全流程跑到 complete。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_entrep_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_BASE", "http://127.0.0.1:1234/v1")
os.environ.setdefault("FORGE_EMBED_MODEL", "text-embedding-embeddinggemma-300m-qat")
os.environ.setdefault("FORGE_MAX_AGENTS", "12")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SOURCE = (
    "2027年地区冲突推演。俄乌战场持续消耗：俄罗斯军队在东线进攻，乌克兰军队依托防线抵抗，"
    "外界常以‘俄乌’概括这场对抗。中东方向，美伊紧张升级，伊朗伊斯兰革命卫队对美国海军第五舰队"
    "施加压力；美国国防部与财政部分别负责军事部署与制裁，最高法院审查行政关税令。"
    "北约秘书长吕特推动欧洲防务自主，欧盟承担对乌援助。各方围绕军力、士气、经济与同盟长期博弈。"
)


def _check_server() -> bool:
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/models", timeout=5).read()
        return True
    except Exception as e:
        print(f"[致命] 无法连接 LM Studio: {e}")
        return False


def _discover_chat_model() -> str:
    if os.environ.get("FORGE_LLM_MODEL"):
        return os.environ["FORGE_LLM_MODEL"]
    base = os.environ["FORGE_LLM_BASE"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=5) as resp:
            ids = [m.get("id") for m in json.loads(resp.read()).get("data", []) if m.get("id")]
        chat = [m for m in ids if "embed" not in m.lower()]
        pick = next((m for m in chat if "9b" in m.lower()), chat[0] if chat else "local")
        os.environ["FORGE_LLM_MODEL"] = pick
        return pick
    except Exception:
        os.environ.setdefault("FORGE_LLM_MODEL", "local")
        return os.environ["FORGE_LLM_MODEL"]


async def main() -> int:
    print("=== 实体消解+报告去重 全流程验证（LM Studio）===")
    if not _check_server():
        return 2
    print("对话模型:", _discover_chat_model())

    from strategy_forge.engine.engine import DeductionEngine
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    engine = DeductionEngine(workspace_root=root)
    session = engine.create_session(
        title="地区冲突推演", source_material=SOURCE,
        config={"domain": "military", "total_rounds": 2, "enable_narrate": True,
                "enable_multi_action": True, "max_actions": 2},
    )
    sid = session.id
    print(f"会话 {sid} 全流程推演...")
    await engine.start(sid)

    data = engine.session_store.get(sid)
    graph = engine.get_graph(sid)
    agents = {a["name"] for a in graph.get_agents()}
    report = data.get("report_json", {}) or {}
    if isinstance(report, str):
        report = json.loads(report)
    summary = (report.get("summary") or "").strip()
    conclusion = (report.get("conclusion") or "").strip()

    print(f"\n状态: {data.get('status')}  智能体({len(agents)}): {sorted(agents)}")

    bad = {"俄乌", "美伊", "美国海军第五舰队", "第五舰队", "财政部", "国防部", "最高法院", "北约秘书长"}
    leaked = agents & bad
    p_entity = len(leaked) == 0
    print(f"[实体] 问题实体未成为智能体: {p_entity}" + (f"  泄漏={leaked}" if leaked else ""))

    # 结论与正文不应逐字重复（允许主题重合，但不应是同一段文本）
    p_dedup = bool(conclusion) and conclusion != summary and conclusion not in summary
    print(f"[报告] 结论独立(非照抄正文): {p_dedup}  (summary={len(summary)}字, conclusion={len(conclusion)}字)")
    print(f"  结论预览: {conclusion[:120]}")

    p_complete = data.get("status") == "complete"
    ok = p_entity and p_dedup and p_complete
    print(f"\n[{'PASS' if ok else 'FAIL'}] complete={p_complete} entity={p_entity} dedup={p_dedup}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

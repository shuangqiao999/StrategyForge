"""离线单元测试：intel_sorter 安全网降级逻辑（无需 LLM）。

验证保守安全网把"二元关系词/军队编制/政府部门/职务头衔"强制降级为
include_in_simulation=false，而正常实体不受影响。
用法（项目根目录）：
    python scripts/test_intel_safety_net_offline.py
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ["FORGE_INTEL_SAFETY_NET"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from strategy_forge.engine.intel_sorter import _apply_safety_net  # noqa: E402


def _mk(name, include=True):
    return {"name": name, "type": "", "aliases": [], "parent": None,
            "sub_entities": [], "include_in_simulation": include, "role": ""}


def main() -> int:
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  \u2713 {name} {detail}")
        else:
            failed += 1
            print(f"  \u2717 {name} FAILED {detail}")

    print("=== intel_sorter 安全网离线单测 ===")

    # 应被降级的
    demote_cases = ["俄乌", "美伊", "中美", "巴以",              # 二元关系
                    "美国海军第五舰队", "中部战区", "太平洋司令部",  # 军队编制
                    "国防部", "财政部", "最高法院", "国台办",        # 政府部门
                    "北约秘书长", "美国总统", "欧盟委员会主席"]      # 职务头衔
    ents = [_mk(n) for n in demote_cases]
    _apply_safety_net(ents)
    for e in ents:
        check(f"降级 {e['name']}", e["include_in_simulation"] is False, f"role={e['role'][:24]}")

    # 正常实体不应被降级
    keep_cases = ["美国", "中国", "俄罗斯", "乌克兰军队", "华为", "北约", "欧盟", "特朗普", "DeepSeek"]
    ents2 = [_mk(n) for n in keep_cases]
    _apply_safety_net(ents2)
    for e in ents2:
        check(f"保留 {e['name']}", e["include_in_simulation"] is True)

    # 开关关闭时不降级
    os.environ["FORGE_INTEL_SAFETY_NET"] = "0"
    ents3 = [_mk("俄乌")]
    _apply_safety_net(ents3)
    check("开关关闭时不降级", ents3[0]["include_in_simulation"] is True)
    os.environ["FORGE_INTEL_SAFETY_NET"] = "1"

    print(f"\n结果: {passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

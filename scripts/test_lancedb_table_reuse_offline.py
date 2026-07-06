"""离线单元测试：DeductionPreprocessor._create_or_open 的健壮性。

不联网、不需 LLM/嵌入 —— 直接用 lancedb + pyarrow 在临时目录建表，
验证重复获取同名表不再抛 'already exists'（修复 deduction_events_* 冲突）。

用法（项目根目录）：
    python scripts/test_lancedb_table_reuse_offline.py
"""
from __future__ import annotations

import os
import sys
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_TMP = os.path.join(os.environ.get("TEMP", "."), "sf_tblreuse_" + uuid.uuid4().hex[:6])
os.environ.setdefault("FORGE_DATA_DIR", _TMP)
os.environ.setdefault("FORGE_PROVIDER", "lmstudio")
os.environ.setdefault("FORGE_LLM_BASE", "http://test-offline/v1")
os.environ.setdefault("FORGE_EMBED_BASE", "http://test-offline/v1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pyarrow as pa  # noqa: E402

from strategy_forge.engine.preprocessor import DeductionPreprocessor  # noqa: E402


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

    print("=== _create_or_open 健壮性离线单测 ===")
    pp = DeductionPreprocessor(_TMP, session_id="sess_reuse")
    pp._init_lancedb()

    schema = pa.schema([
        pa.field("event_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 4)),
        pa.field("content", pa.string()),
    ])
    name = "deduction_events_sess_reuse"

    # 1) 首次：创建
    t1 = pp._create_or_open(name, schema)
    t1.add([{"event_id": "e1", "vector": [0.1, 0.2, 0.3, 0.4], "content": "hello"}])
    check("首次创建并写入", t1.count_rows() == 1)

    # 2) 再次获取：应打开已存在表（不抛 already exists）
    t2 = pp._create_or_open(name, schema)
    check("重复获取不抛 already-exists", t2.count_rows() == 1, f"rows={t2.count_rows()}")

    # 3) 模拟老代码崩溃点：直接 create(mode='create') 应抛，证明冲突真实存在
    raised = False
    try:
        pp._db.create_table(name, schema=schema, mode="create")
    except Exception:
        raised = True
    check("原始 create(mode=create) 确会抛 already-exists", raised)

    # 4) _create_or_open 在该冲突下仍安全返回
    t3 = pp._create_or_open(name, schema)
    check("冲突下 _create_or_open 安全返回", t3 is not None and t3.count_rows() == 1)

    print(f"\n结果: {passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""fix_config_env.py — 一次性脚本：把 config.py 的 int/float(os.getenv(...)) 改为容错的 _env_int/_env_float。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "src" / "strategy_forge" / "core" / "config.py"

_INT_RE = re.compile(r'int\(os\.getenv\("([A-Z_]+)", "?(\d+)"?\)\)')
_FLOAT_RE = re.compile(r'float\(os\.getenv\("([A-Z_]+)", "?([\d.]+)"?\)\)')


def main() -> None:
    src = CFG.read_text(encoding="utf-8")
    src = _INT_RE.sub(lambda m: f'_env_int("{m.group(1)}", {m.group(2)})', src)
    src = _FLOAT_RE.sub(lambda m: f'_env_float("{m.group(1)}", {m.group(2)})', src)
    CFG.write_text(src, encoding="utf-8")
    print("config.py env parsing hardened")


if __name__ == "__main__":
    main()

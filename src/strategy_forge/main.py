"""StrategyForge CLI entry point — `strategy-forge serve`."""
from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(prog="strategy-forge", description="StrategyForge — 战略决策推演")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="启动 API 服务器")
    serve_parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口")
    serve_parser.add_argument("--reload", action="store_true", help="开发模式热重载")

    run_parser = sub.add_parser("run", help="执行单次推演任务")
    run_parser.add_argument("text", help="推演材料或文件路径")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        gui_mode = sys.stderr is None or not hasattr(sys.stderr, "isatty")

        if gui_mode:
            # Windows GUI subsystem (PyInstaller console=False): stderr/stdout 为 None,
            # uvicorn 的 DefaultFormatter.__init__ 调用 sys.stderr.isatty() 会崩溃。
            # 改用文件日志，禁用 uvicorn 内部的 dictConfig。
            import os
            log_path = os.path.join(os.path.dirname(sys.executable), "strategy_forge.log")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                handlers=[logging.FileHandler(log_path, encoding="utf-8")],
            )
            logging.getLogger("strategy_forge").info("Backend starting (GUI mode, log=%s)", log_path)

            uvicorn.run(
                "strategy_forge.api:app",
                host=args.host,
                port=args.port,
                reload=False,
                log_config=None,
                access_log=False,
            )
        else:
            uvicorn.run(
                "strategy_forge.api:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
                log_level="info",
            )
    elif args.command == "run":
        print("单次推演模式暂未实现，请使用 serve 模式启动 API 服务")
        sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""StrategyForge dev launcher — starts the API server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "strategy_forge.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )

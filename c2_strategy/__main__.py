"""Allow running with: python -m c2_strategy"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "c2_strategy.strategy_server:app",
        host="0.0.0.0",
        port=9090,
        reload=True,
    )

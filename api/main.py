
from fastapi import FastAPI, Request
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlencode
import nest_asyncio  # type: ignore
from mangum import Mangum
nest_asyncio.apply()

app = FastAPI(
    title="StockBot API",
    description="API for StockBot",
    version="1.0.0"
)

from api.v1.router import router as v1_router
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")

@app.middleware("http")
async def flatten_query_string_lists(request: Request, call_next):
    flattened: list[tuple[str, str]] = []
    for key, value in request.query_params.multi_items():
        flattened.extend((key, entry) for entry in value.split(","))

    request.scope["query_string"] = urlencode(flattened, doseq=True).encode("utf-8")

    return await call_next(request)

@app.get("/health")
def health():
    return {"status": "ok"}

handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        reload= True,
    )
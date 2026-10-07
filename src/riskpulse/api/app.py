"""FastAPI service - the engine's public interface (interactive docs at /docs).

    GET  /health                         liveness + mode
    GET  /signals                        filters: ticker, event_type, min_impact, since, limit
    GET  /signals/latest/{ticker}        latest signal for one ticker (or MARKET / a sector)
    GET  /entities/{ticker}/sentiment    EWMA aggregate sentiment (confidence- and impact-weighted)
    POST /analyze                        text -> signals; inject=true publishes them to the modules
    GET  /stream                         Server-Sent Events stream of new signals
    GET  /stats                          engine, sources, bus, clock
    GET  /module-a/state | /module-a/weights | /module-a/log
    GET  /module-b/portfolio | /module-b/runs ; POST /module-b/what-if
    POST /replay/{pause|resume|restart} ; POST /replay/speed?value=
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from riskpulse import __version__
from riskpulse.clock import ReplayClock
from riskpulse.module_b import stress
from riskpulse.module_b.portfolio import summary
from riskpulse.schema import EVENT_TYPES
from riskpulse.service import RiskPulseService
from riskpulse.universe import tickers


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=4000)
    source_type: str = Field("news", pattern="^(news|social)$")
    inject: bool = False


class WhatIfRequest(BaseModel):
    event_type: str
    impact: float = Field(ge=1, le=10)
    headline: str = ""
    ticker: str = ""
    sector: str = ""
    regions: list[str] = []


def create_app(service: RiskPulseService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = service or RiskPulseService(mode=os.environ.get("RISKPULSE_MODE", "demo"),
                                          speed=float(os.environ["RISKPULSE_SPEED"]) if os.environ.get("RISKPULSE_SPEED") else None,
                                          start=os.environ.get("RISKPULSE_START"), end=os.environ.get("RISKPULSE_END"))
        app.state.svc = svc
        svc.start()
        yield
        svc.stop()

    app = FastAPI(title="RiskPulse API", version=__version__, lifespan=lifespan,
                  description="Real-time AI/NLP financial risk signals and downstream risk modules.")

    def svc() -> RiskPulseService:
        return app.state.svc

    @app.get("/health")
    def health():
        s = svc()
        return {"status": "ok", "mode": s.mode, "engine_ready": s.engine_ready.is_set() and s.engine is not None,
                "clock": s.clock.status()}

    @app.get("/signals")
    def signals(ticker: str | None = None, event_type: str | None = None, min_impact: float | None = Query(None, ge=1, le=10),
                since: str | None = None, since_id: int = 0, limit: int = Query(100, ge=1, le=2000)):
        if event_type and event_type not in EVENT_TYPES:
            raise HTTPException(422, f"event_type must be one of {EVENT_TYPES}")
        return svc().store.signals(since_id=since_id, limit=limit, entity_id=ticker, event_type=event_type,
                                   min_impact=min_impact, since=since)

    @app.get("/signals/latest/{ticker}")
    def latest(ticker: str):
        key = ticker.upper() if ticker.upper() in [*tickers(), "MARKET"] else ticker   # tickers / MARKET / sector name
        rows = svc().store.signals(limit=1, entity_id=key)
        if not rows:
            raise HTTPException(404, f"no signals yet for {ticker}")
        return rows[0]

    @app.get("/entities/{ticker}/sentiment")
    def entity_sentiment(ticker: str):
        t = ticker.upper()
        if t not in tickers():
            raise HTTPException(404, f"{t} is not in the index universe")
        return svc().entities.entity(t, svc().clock.now())

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest):
        try:
            return svc().analyze(req.text, req.source_type, inject=req.inject)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/stream")
    async def stream():
        async def events():
            last = svc().store.max_signal_id()
            while True:
                rows = svc().store.signals(since_id=last, limit=200, ascending=True)
                for r in rows:
                    last = r.pop("id")
                    yield {"event": "signal", "data": json.dumps(r)}
                await asyncio.sleep(0.5)
        return EventSourceResponse(events())

    @app.get("/stats")
    def stats():
        return svc().status()

    @app.get("/module-a/state")
    def module_a_state():
        return svc().module_a.state()

    @app.get("/module-a/weights")
    def module_a_weights(max_snapshots: int = Query(1500, ge=1, le=20000)):
        return svc().store.weights_history(max_snapshots)

    @app.get("/module-a/log")
    def module_a_log(limit: int = Query(200, ge=1, le=5000)):
        return svc().store.rebalance_log(limit)

    @app.get("/module-b/portfolio")
    def module_b_portfolio():
        pos = stress.book()
        return {"summary": summary(pos), "positions": pos.round(6).to_dict(orient="records")}

    @app.get("/module-b/runs")
    def module_b_runs(limit: int = Query(50, ge=1, le=500)):
        return {"monitor": svc().module_b.state(), "runs": svc().store.stress_runs(limit)}

    @app.post("/module-b/what-if")
    def what_if(req: WhatIfRequest):
        try:
            return stress.what_if(req.event_type, req.impact, req.headline, req.ticker, req.sector, req.regions)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/replay/{action}")
    def replay(action: str, value: float | None = None):
        s = svc()
        if not isinstance(s.clock, ReplayClock):
            raise HTTPException(409, "replay controls are only available in demo / replay mode")
        if action == "pause":
            s.clock.pause()
        elif action == "resume":
            s.clock.resume()
        elif action == "speed" and value is not None:
            s.clock.set_speed(value)
        elif action == "restart":
            s.restart_replay()
        else:
            raise HTTPException(422, "action must be pause, resume, restart or speed?value=")
        return s.clock.status()

    return app


app = create_app()

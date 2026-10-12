"""API endpoints via TestClient, running the offline demo service in the lite tier."""
import time

import pytest
from fastapi.testclient import TestClient

from riskpulse.api.app import create_app
from riskpulse.service import RiskPulseService


@pytest.fixture(scope="module")
def client():
    svc = RiskPulseService(mode="demo", speed=5e6, tier="lite")    # whole demo window replays in < 1 s
    with TestClient(create_app(svc)) as c:
        deadline = time.time() + 30
        while time.time() < deadline and not (svc.replay and svc.replay.finished and svc.engine_ready.is_set()):
            time.sleep(0.2)
        yield c


def test_health_and_stats(client):
    assert client.get("/health").json()["status"] == "ok"
    stats = client.get("/stats").json()
    assert stats["mode"] == "demo" and stats["signals_stored"] > 0


def test_signal_queries(client):
    rows = client.get("/signals", params={"min_impact": 7, "limit": 5}).json()
    assert rows and all(r["impact_score"] >= 7 for r in rows)
    assert client.get("/signals", params={"event_type": "Not An Event"}).status_code == 422
    assert client.get("/signals/latest/MARKET").json()["entity"]["id"] == "MARKET"
    sent = client.get("/entities/AAPL/sentiment").json()
    assert -1 <= sent["ewma_sentiment"] <= 1
    assert client.get("/entities/ZZZZ/sentiment").status_code == 404


def test_analyze_preview_and_inject(client):
    text = {"text": "Boeing files for Chapter 11 bankruptcy protection as defaults mount"}
    preview = client.post("/analyze", json=text).json()
    s = preview["signals"][0]
    assert s["entity"]["id"] == "BA" and s["event_type"] == "Credit Event" and not preview["injected"]
    injected = client.post("/analyze", json={**text, "inject": True}).json()
    assert injected["injected"] and injected["signals"][0]["impact_score"] >= 1


def test_modules_endpoints(client):
    state = client.get("/module-a/state").json()
    assert abs(sum(w["weight"] for w in state["weights"]) - 1) < 1e-4   # weights are rounded to 5 dp
    assert client.get("/module-a/weights").json()
    runs = client.get("/module-b/runs").json()
    assert "monitor" in runs
    wi = client.post("/module-b/what-if", json={"event_type": "Geopolitical", "impact": 8.5,
                                                 "headline": "New tariffs on imports"}).json()
    assert wi["scenario_key"] == "trade_war" and wi["value_after"] < wi["value_before"] + abs(wi["pnl"]) + 1
    assert client.post("/module-b/what-if", json={"event_type": "Product Launch", "impact": 9}).status_code == 422


def test_replay_controls(client):
    assert client.post("/replay/pause").json()["paused"] is True
    assert client.post("/replay/resume").json()["paused"] is False
    assert client.post("/replay/speed", params={"value": 100}).json()["speed"] == 100
    assert client.post("/replay/bogus").status_code == 422
    assert client.post("/replay/speed", params={"value": -5}).status_code == 422
    assert client.post("/replay/speed", params={"value": 0}).status_code == 422


def test_input_validation_and_portfolio(client):
    assert client.get("/signals", params={"since": "garbage"}).status_code == 422
    assert client.get("/signals", params={"since": "2025-04-02T00:00:00Z"}).status_code == 200
    pf = client.get("/module-b/portfolio")                     # used to fail on NaN fields
    assert pf.status_code == 200 and pf.json()["summary"]["positions"] > 0

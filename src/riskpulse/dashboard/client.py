"""Thin HTTP client for the RiskPulse API (the dashboard never imports engine internals)."""
from __future__ import annotations

import os

import httpx

from riskpulse import config

API = os.environ.get("RISKPULSE_API_URL") or (f"http://{config.get('api.host', '127.0.0.1')}:"
                                               f"{config.get('api.port', 8000)}")
_client = httpx.Client(base_url=API, timeout=30)


def get(path: str, **params):
    try:
        r = _client.get(path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


def post(path: str, json: dict | None = None, timeout: float = 150, **params):
    try:
        r = _client.post(path, json=json, params=params or None, timeout=timeout)
        if r.status_code >= 400:
            return {"error": r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith(
                "application/json") else r.text}
        return r.json()
    except httpx.HTTPError as exc:
        return {"error": str(exc)}

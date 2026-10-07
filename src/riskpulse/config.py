"""Central configuration: repository paths plus the YAML files in config/.

Every threshold the engine and the downstream modules use lives in config/*.yaml so that
behaviour can be inspected and changed without touching code.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LABELED = DATA / "labeled"
PRICES = DATA / "prices"
PORTFOLIO = DATA / "portfolio"
OUTPUT = DATA / "output"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
RUNTIME = Path(os.environ.get("RISKPULSE_RUNTIME", ROOT / "runtime"))


def _load_dotenv() -> None:
    """Minimal .env reader (KEY=VALUE lines) so no extra dependency is needed."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@cache
def load_yaml(name: str) -> Any:
    with open(CONFIG / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def settings() -> dict[str, Any]:
    return load_yaml("settings.yaml")


def get(path: str, default: Any = None) -> Any:
    """Dotted lookup into settings.yaml, e.g. get("module_b.cooldown_minutes")."""
    node: Any = settings()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def requested_tier() -> str:
    """NLP tier requested by env/config: 'full', 'lite' or 'auto'."""
    return (os.environ.get("RISKPULSE_TIER") or get("engine.tier", "auto")).lower()


def ensure_runtime() -> Path:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    return RUNTIME

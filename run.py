"""RiskPulse - single entry point.

    python run.py demo                 offline demo: API + dashboard + browser (precomputed signals, replay clock)
    python run.py api                  API only (add --mode demo|replay|live)
    python run.py replay --speed 120   stream archived headlines through the live engine
    python run.py live                 poll live news + social sources
    python run.py evaluate             regenerate reports/metrics.json and docs/images/*
    python run.py precompute           re-run the engine over the headline archive (slow in the full tier)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _wait_for(url: str, timeout: float = 90.0) -> bool:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def serve(mode: str, args: argparse.Namespace, dashboard: bool = True) -> None:
    import uvicorn

    from riskpulse import config

    host = config.get("api.host", "127.0.0.1")
    api_port, ui_port = args.port, int(config.get("dashboard.port", 8501))
    os.environ["RISKPULSE_MODE"] = mode
    for key, val in (("RISKPULSE_SPEED", args.speed), ("RISKPULSE_START", getattr(args, "start", None)),
                     ("RISKPULSE_END", getattr(args, "end", None)), ("RISKPULSE_TIER", args.tier)):
        if val:
            os.environ[key] = str(val)
    api_url = f"http://{host}:{api_port}"
    ui = None
    if dashboard:
        env = {**os.environ, "RISKPULSE_API_URL": api_url, "PYTHONPATH": str(ROOT / "src")}
        ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "src/riskpulse/dashboard/app.py"),
                               "--server.port", str(ui_port), "--server.headless", "true",
                               "--browser.gatherUsageStats", "false"], env=env, cwd=ROOT)
        if not args.no_browser:
            def _open() -> None:
                if _wait_for(f"{api_url}/health"):
                    webbrowser.open(f"http://localhost:{ui_port}")
            threading.Thread(target=_open, daemon=True).start()
    print(f"\n  RiskPulse [{mode}]  API {api_url}/docs" + (f"   dashboard http://localhost:{ui_port}" if ui else "") + "\n")
    try:
        uvicorn.run("riskpulse.api.app:app", host=host, port=api_port, log_level="warning")
    finally:
        if ui is not None:
            ui.terminate()


def main() -> None:
    ap = argparse.ArgumentParser(description="RiskPulse real-time AI/NLP risk engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "api", "replay", "live"):
        p = sub.add_parser(name)
        p.add_argument("--speed", type=float, help="replay speed: simulated seconds per second")
        p.add_argument("--tier", choices=["auto", "full", "lite"], help="NLP tier (default: auto)")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--no-browser", action="store_true")
        if name == "api":
            p.add_argument("--mode", choices=["demo", "replay", "live"], default="demo")
        if name in ("replay", "api"):
            p.add_argument("--start", help="replay window start (ISO date)")
            p.add_argument("--end", help="replay window end (ISO date)")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--quick", action="store_true", help="skip the slow full-tier model evaluation")
    pc = sub.add_parser("precompute")
    pc.add_argument("--tier", choices=["full", "lite"], default="full")
    args = ap.parse_args()

    if args.cmd == "demo":
        serve("demo", args)
    elif args.cmd == "api":
        serve(args.mode, args, dashboard=False)
    elif args.cmd in ("replay", "live"):
        serve(args.cmd, args)
    elif args.cmd == "evaluate":
        from riskpulse.evaluation.run_all import main as evaluate
        evaluate(quick=args.quick)
    elif args.cmd == "precompute":
        subprocess.run([sys.executable, str(ROOT / "scripts/precompute_signals.py"), "--tier", args.tier], check=True)


if __name__ == "__main__":
    main()

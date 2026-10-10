"""Capture dashboard screenshots for the docs and deck (needs `python run.py demo` running).

    python scripts/capture_screenshots.py [--out docs/images] [--analyze]

Uses Playwright with the locally installed Edge or Chrome (no browser download).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import _bootstrap  # noqa: F401
from riskpulse.config import DOCS

TABS = ["Live risk feed", "Module A · Index rebalancer", "Module B · Stress test", "Model evaluation"]


def settle(page, seconds: float = 4.0) -> None:
    page.wait_for_load_state("networkidle")
    deadline = time.time() + 60
    while time.time() < deadline and page.locator("[data-testid='stSkeleton']").count():
        time.sleep(0.5)
    time.sleep(seconds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DOCS / "images"))
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--analyze", action="store_true", help="also run the 'Analyze & publish' demo flow")
    ap.add_argument("--fast-forward", action="store_true", help="replay the whole demo window before capturing")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.fast_forward:
        import httpx

        httpx.post(f"{args.api}/replay/restart", timeout=30)
        httpx.post(f"{args.api}/replay/speed", params={"value": 200_000}, timeout=30)
        deadline = time.time() + 300
        while time.time() < deadline and not httpx.get(f"{args.api}/stats", timeout=30).json().get("replay_finished"):
            time.sleep(2)
        print("replay finished:", httpx.get(f"{args.api}/stats", timeout=30).json().get("replay_finished"))
    with sync_playwright() as p:
        for channel in ("msedge", "chrome", None):
            try:
                browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                break
            except Exception:
                continue
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page.goto(args.url)
        page.get_by_text("RiskPulse · AI/NLP Financial Risk Engine").wait_for(timeout=90_000)
        settle(page, 6)
        if args.analyze:
            page.get_by_text("(type your own)").first.click()
            page.get_by_role("option", name="Geopolitical shock (triggers Module B)").click()
            settle(page, 2)
            page.get_by_role("button", name="Analyze & publish to modules").click()
            page.get_by_text("Impact score", exact=True).first.wait_for(timeout=180_000)
            settle(page, 3)
            page.get_by_text("Sentiment score", exact=True).first.scroll_into_view_if_needed()
            page.mouse.wheel(0, -160)                       # keep the headline box in frame
            settle(page, 1)
            page.screenshot(path=str(out / "dashboard_hero.png"))     # one viewport: headline -> scores -> drivers
            print("saved", out / "dashboard_hero.png")
            page.mouse.wheel(0, -10_000)
        for i, tab in enumerate(TABS):
            page.get_by_role("tab", name=tab).click()
            settle(page, 5)
            path = out / f"dashboard_{i + 1}_{tab.split(' ')[0].lower()}{'_' + tab.split(' ')[1] if i in (1, 2) else ''}.png"
            page.screenshot(path=str(path), full_page=True)
            print("saved", path)
        browser.close()


if __name__ == "__main__":
    main()

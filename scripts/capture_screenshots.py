"""Capture dashboard screenshots for the docs and deck (needs `python run.py demo` running).

    python scripts/capture_screenshots.py [--out docs/images] [--analyze]

Uses Playwright with the locally installed Edge or Chrome (no browser download).
"""
from __future__ import annotations

import argparse
import re
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
    if args.analyze:                                   # warm the models so the chip shows steady-state latency
        import httpx

        httpx.post(f"{args.api}/analyze", json={"text": "Apple shares rise after strong iPhone sales"}, timeout=180)
    if args.fast_forward:
        import httpx

        httpx.post(f"{args.api}/replay/restart", timeout=30)
        httpx.post(f"{args.api}/replay/speed", params={"value": 200_000}, timeout=30)
        deadline = time.time() + 300
        while time.time() < deadline and not httpx.get(f"{args.api}/stats", timeout=30).json().get("replay_finished"):
            time.sleep(2)
        print("replay finished:", httpx.get(f"{args.api}/stats", timeout=30).json().get("replay_finished"))
        last, stable_since = -1, time.time()          # the bus may still be delivering the last batches to Module B
        while time.time() - stable_since < 8 and time.time() < deadline + 120:
            n = len((httpx.get(f"{args.api}/module-b/runs", params={"limit": 100}, timeout=30).json() or {}).get("runs", []))
            if n != last:
                last, stable_since = n, time.time()
            time.sleep(1)
        print("module B runs after the replay:", last)
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
            page.mouse.wheel(0, -10_000)                    # back to the top for the tab shots
        for i, tab in enumerate(TABS):
            page.get_by_role("tab", name=tab).click()
            settle(page, 5)
            if i == 2 and args.analyze:                     # inspect the run the typed headline triggered
                box = page.locator("[data-testid='stSelectbox']", has_text="Inspect run")
                box.locator("div[data-baseweb='select']").click()
                page.get_by_role("option", name=re.compile("Russia launches full-scale")).first.click()
                settle(page, 3)
            if i == 0:                                      # the hero already shows the analyze box
                page.get_by_text("Signals stored").first.evaluate("e => e.scrollIntoView({block: 'start'})")
                page.mouse.wheel(0, -40)
                settle(page, 1)
            path = out / f"dashboard_{i + 1}_{tab.split(' ')[0].lower()}{'_' + tab.split(' ')[1] if i in (1, 2) else ''}.png"
            page.screenshot(path=str(path), full_page=True)
            print("saved", path)
            if i == 2:                                      # readable close-up for the deck: trigger -> capital check
                trig = page.get_by_text("Trigger:").locator("visible=true").first     # other tabs stay in the DOM
                trig.evaluate("e => e.scrollIntoView({block: 'start'})")
                page.mouse.wheel(0, -160)                   # clear Streamlit's fixed top bar
                settle(page, 1)
                cap = page.locator(".rp-callout", has_text="CET1").locator("visible=true").first
                t, box = trig.bounding_box(), cap.bounding_box()
                if t and box:
                    top = max(t["y"] - 14, 0)
                    page.screenshot(path=str(out / "dashboard_module_B_strip.png"),
                                    clip={"x": box["x"] - 8, "y": top, "width": box["width"] + 16,
                                          "height": box["y"] + box["height"] + 14 - top})
                    print("saved", out / "dashboard_module_B_strip.png")
        browser.close()


if __name__ == "__main__":
    main()

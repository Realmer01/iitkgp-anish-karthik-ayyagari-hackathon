"""Render docs/architecture.png (3200 x 1800 px) with matplotlib only - no Graphviz or other system tools.

    python scripts/make_architecture_diagram.py
"""
from __future__ import annotations

import matplotlib

import _bootstrap  # noqa: F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from riskpulse.config import DOCS  # noqa: E402

BG, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
LANE = {"src": "#eef4fc", "eng": "#f3f0fb", "bus": "#eef7f1", "mod": "#fdf3ec", "eval": "#f4f4f2"}
ACCENT = {"src": "#2a78d6", "eng": "#4a3aa7", "bus": "#008300", "mod": "#eb6834", "eval": "#898781"}
FONT = "Segoe UI"


def box(ax, x, y, w, h, title, lines=(), lane="src", title_size=11.5, body_size=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", linewidth=1.1,
                                edgecolor=ACCENT[lane], facecolor="white", zorder=3))
    ax.text(x + 0.012, y + h - 0.028, title, fontsize=title_size, fontweight="bold", color=INK, va="top", zorder=4,
            family=FONT)
    for i, line in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.066 - i * 0.034, line, fontsize=body_size, color=INK2, va="top", zorder=4,
                family=FONT)


def lane(ax, x, w, label, key, y=0.17, h=0.72):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.016", linewidth=0,
                                facecolor=LANE[key], zorder=1))
    ax.text(x + 0.012, y + h - 0.012, label.upper(), fontsize=9.5, color=ACCENT[key], fontweight="bold", va="top",
            family=FONT, zorder=2)


def arrow(ax, a, b, label="", curve=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=INK2, zorder=5,
                                 connectionstyle=f"arc3,rad={curve}"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.014, label, fontsize=8.5, color=MUTED, ha="center",
                family=FONT, zorder=6)


def main() -> None:
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.978, "RiskPulse - real-time AI/NLP risk engine", fontsize=19, fontweight="bold", color=INK,
            family=FONT, va="top")
    ax.text(0.02, 0.938, "Unstructured news and social text -> structured risk signals (sentiment, event, impact) "
            "-> index rebalancing and wholesale-bank stress testing", fontsize=11, color=INK2, family=FONT, va="top")

    lane(ax, 0.02, 0.17, "1  Sources", "src", h=0.72)
    lane(ax, 0.205, 0.30, "2  Risk engine", "eng", h=0.72)
    lane(ax, 0.52, 0.16, "3  Publish", "bus", h=0.72)
    lane(ax, 0.695, 0.285, "4  Consumers (schema only)", "mod", h=0.72)

    box(ax, 0.03, 0.66, 0.15, 0.19, "News", ["GDELT DOC 2.0 API", "Yahoo Finance RSS", "Google News RSS",
                                             "NewsAPI (optional key)"], "src")
    box(ax, 0.03, 0.44, 0.15, 0.18, "Social", ["Mastodon (live)", "finance-tweet corpus", "(X/Twitter replay)"], "src")
    box(ax, 0.03, 0.21, 0.15, 0.19, "Replay", ["61k dated headlines", "5k GDELT, exact times", "replay clock (x speed)"],
        "src")

    box(ax, 0.215, 0.21, 0.098, 0.64, "Normaliser", ["RawDocument", "UTC timestamps", "clean text", "slang, emoji",
                                                     "language filter", "dedup", "story clusters", "(confirmation)"],
        "eng", body_size=8.8)
    box(ax, 0.345, 0.725, 0.15, 0.125, "Entity linker", ["cashtags, aliases", "broker + false-positive guards"], "eng",
        body_size=8.6)
    box(ax, 0.345, 0.57, 0.15, 0.125, "Sentiment", ["FinBERT + social head", "lite: VADER + lexicon"], "eng",
        body_size=8.6)
    box(ax, 0.345, 0.415, 0.15, 0.125, "Event class (12)", ["bge-small + LR + rules", "lite: TF-IDF + LR"], "eng",
        body_size=8.6)
    box(ax, 0.345, 0.21, 0.15, 0.175, "Impact 1-10", ["logistic factor model", "9 explainable drivers",
                                                     "event-study validated"], "eng", body_size=8.6)

    box(ax, 0.53, 0.66, 0.14, 0.19, "Sinks", ["JSONL (append-only)", "SQLite (WAL)", "FastAPI REST + SSE"], "bus")
    box(ax, 0.53, 0.44, 0.14, 0.17, "Signal bus", ["in-process pub/sub", "SignalSubscriber API"], "bus")
    box(ax, 0.53, 0.21, 0.14, 0.19, "RiskSignal v1.0", ["sentiment -1..1", "event + probabilities",
                                                       "impact + drivers"], "bus")

    box(ax, 0.705, 0.68, 0.265, 0.17, "Dashboard + clients",
        ["reads REST + SSE", "Streamlit: feed, Module A, Module B, evaluation"], "mod", body_size=8.8)
    box(ax, 0.705, 0.445, 0.265, 0.2, "Module A - tactical index rebalancer",
        ["subscribes to: sentiment", "EWMA sentiment -> w = b * exp(k s)", "caps, drift band, turnover cap, costs",
         "hourly no-look-ahead backtest"], "mod", body_size=8.8)
    box(ax, 0.705, 0.21, 0.265, 0.2, "Module B - event-driven stress test",
        ["subscribes to: event + impact (> 7)", "248-position wholesale book (synthetic)",
         "cited scenarios, IFRS 9, CET1 proxy", "compared with a naive static shock"], "mod", body_size=8.8)

    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.095, boxstyle="round,pad=0.004,rounding_size=0.016",
                                linewidth=0, facecolor=LANE["eval"], zorder=1))
    ax.text(0.032, 0.122, "OFFLINE  VALIDATION + CONFIG", fontsize=9.5, color=ACCENT["eval"], fontweight="bold",
            va="top", family=FONT)
    ax.text(0.032, 0.088, "400-headline gold set  ·  impact event study (abnormal returns)  ·  hourly backtest with "
            "placebos  ·  scenario vs naive comparison  ->  reports/metrics.json      |      config/*.yaml: universe, "
            "taxonomy, impact weights, shock library, thresholds", fontsize=9.6, color=INK2, va="top", family=FONT)

    for y in (0.755, 0.53, 0.30):                      # sources -> ingest
        arrow(ax, (0.182, y), (0.213, y))
    arrow(ax, (0.322, 0.79), (0.336, 0.79))            # normaliser -> entity linker
    for y0, y1 in ((0.725, 0.697), (0.57, 0.542), (0.415, 0.387)):   # NLP chain, top to bottom
        arrow(ax, (0.42, y0), (0.42, y1))
    arrow(ax, (0.504, 0.30), (0.521, 0.30))            # impact -> RiskSignal
    arrow(ax, (0.60, 0.40), (0.60, 0.438))             # RiskSignal -> bus
    arrow(ax, (0.60, 0.61), (0.60, 0.658))             # bus -> sinks
    arrow(ax, (0.672, 0.765), (0.703, 0.765))          # sinks -> dashboard
    arrow(ax, (0.672, 0.545), (0.703, 0.545))          # bus -> Module A
    arrow(ax, (0.672, 0.47), (0.703, 0.33))            # bus -> Module B
    out = DOCS / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()

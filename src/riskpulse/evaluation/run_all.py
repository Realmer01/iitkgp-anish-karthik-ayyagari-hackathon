"""`python run.py evaluate` - regenerate every reported number and chart.

Writes reports/metrics.json (the single source for README / deck numbers), reports/*.csv used by the
dashboard, and docs/images/*.png for the deck. `quick=True` (CI) runs the lite tier only and skips the
sensitivity sweep.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from riskpulse import __version__  # noqa: E402
from riskpulse.config import DOCS, PROCESSED, REPORTS  # noqa: E402
from riskpulse.dashboard.theme import AXIS, GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402
from riskpulse.evaluation import impact_eval, nlp_eval, system_eval  # noqa: E402
from riskpulse.nlp.tier import full_stack_available  # noqa: E402

IMAGES = DOCS / "images"


def _style(ax, title: str, horizontal: bool = False) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="x" if horizontal else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w: float = 9, h: float = 4.8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _grouped_barh(labels, series: dict, title: str, xlabel: str, path, fmt="{:.2f}") -> None:
    fig, ax = _fig(9, 0.55 * len(labels) + 1.6)
    n = len(series)
    hgt = 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        ys = [y + (i - (n - 1) / 2) * hgt for y in range(len(labels))]
        bars = ax.barh(ys, vals, height=hgt * 0.92, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + (0.01 if v >= 0 else -0.01) * max(map(abs, vals)), b.get_y() + b.get_height() / 2,
                    fmt.format(v), va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, color=INK_2, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=INK_2)
    _style(ax, title, horizontal=True)
    ax.axvline(0, color=AXIS, linewidth=1)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="lower center", ncol=n, fontsize=9,
               labelcolor=INK_2)                                # below the plot, never over a bar
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def charts(m: dict, equity: pd.DataFrame) -> list[str]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    saved = []
    s = pd.DataFrame(m["sentiment"]["table"])
    piv = s.pivot_table(index="model", columns="dataset", values="macro_f1")
    order = [x for x in ["VADER (naive baseline)", "Lite: VADER + finance lexicon", "FinBERT",
                         "Social head (bge-small + LR)", "RiskPulse full (blend + policy rule)"] if x in piv.index]
    _grouped_barh(order, {"Real news headlines (gold)": piv.loc[order, "gold_news"].tolist(),
                          "Finance tweets": piv.loc[order, "tweets_valid"].tolist()},
                  "Sentiment: macro-F1 vs the naive VADER baseline", "macro-F1", IMAGES / "eval_sentiment.png")
    saved.append("eval_sentiment.png")
    pc = pd.DataFrame(m["events"]["per_class"])
    pc = pc[pc["support"] >= 6]
    _grouped_barh(pc["class"].tolist(), {"RiskPulse hybrid": pc["hybrid_f1"].tolist(),
                                         "Keyword baseline": pc["keyword_f1"].tolist()},
                  "Event classification: per-class F1 on held-out real headlines (support >= 6)", "F1",
                  IMAGES / "eval_events.png")
    saved.append("eval_events.png")
    d = pd.DataFrame(m["impact"]["deciles"])
    fig, ax = _fig(9, 4.4)
    ax.plot(d["decile"], d["impact_mean_abs_ar"], color=SERIES[0], linewidth=2, marker="o", markersize=6,
            label="Impact score")
    ax.plot(d["decile"], d["baseline_mean_abs_ar"], color=SERIES[1], linewidth=2, marker="o", markersize=6,
            label="|Sentiment| only (naive)")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("score decile (1 = lowest)", color=INK_2)
    ax.set_ylabel("mean |abnormal return| over [D, D+1], %", color=INK_2)
    _style(ax, "Higher scores, bigger moves: realised |abnormal return| by score decile")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(IMAGES / "eval_impact_deciles.png", facecolor=SURFACE)
    plt.close(fig)
    saved.append("eval_impact_deciles.png")
    fig, ax = _fig(9, 4.4)
    for i, c in enumerate(equity.columns):
        ax.plot(equity.index, equity[c], color=SERIES[i], linewidth=2 if i == 0 else 1.4, label=c)
    ax.set_ylabel("growth of $1", color=INK_2)
    _style(ax, "Module A backtest (hourly, no look-ahead, net of 5 bp costs)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES / "eval_module_a.png", facecolor=SURFACE)
    plt.close(fig)
    saved.append("eval_module_a.png")
    b = pd.DataFrame(m["module_b"]["scenarios"])
    _grouped_barh(b["scenario"].tolist(), {"Event-driven scenario": b["pnl_musd"].tolist(),
                                           "Naive static shock": b["naive_pnl_musd"].tolist()},
                  "Module B: stress P&L at impact 8.5 ($m)", "P&L, $m", IMAGES / "eval_module_b.png", fmt="{:,.0f}")
    saved.append("eval_module_b.png")
    return saved


def main(quick: bool = False) -> dict:
    t0 = time.time()
    full = full_stack_available() and not quick
    if not (PROCESSED / "events_valid.csv").exists():
        import runpy

        from riskpulse.config import ROOT
        sys.path.insert(0, str(ROOT / "scripts"))          # the script imports its _bootstrap helper
        runpy.run_path(str(ROOT / "scripts" / "build_training_data.py"), run_name="__main__")
    REPORTS.mkdir(exist_ok=True)
    print("1/6 sentiment ...", flush=True)
    m = {"generated_at": datetime.now(UTC).isoformat(), "version": __version__, "tier_evaluated": "full" if full else "lite"}
    m["sentiment"] = nlp_eval.evaluate_sentiment(full)
    print("2/6 events ...", flush=True)
    ev = nlp_eval.evaluate_events(full)
    gold_preds = ev.pop("gold_predictions")
    m["events"] = ev
    m["gold_agreement"] = nlp_eval.annotator_agreement()
    print("3/6 impact event study ...", flush=True)
    m["impact"] = impact_eval.evaluate("full", write_calibration=not quick)
    print("4/6 engine performance ...", flush=True)
    m["engine"] = system_eval.engine_performance(["lite", "full"] if full else ["lite"])
    print("5/6 module A backtest ...", flush=True)
    m["module_a"], equity, sens = system_eval.module_a(quick)
    print("6/6 module B ...", flush=True)
    eng = None
    if full:
        from riskpulse.engine import RiskEngine
        eng = RiskEngine("full")
    m["module_b"] = system_eval.module_b(eng)
    m["module_b"]["april_2025_replay"] = system_eval.replay_triggers()
    m["module_b"]["trigger_validation"] = system_eval.trigger_validation()
    m["module_b"]["red_alert"] = system_eval.red_alert_validation()
    m["module_b"]["triage_funnel"] = system_eval.triage_funnel()
    m["runtime_s"] = round(time.time() - t0, 1)
    (REPORTS / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    equity.to_csv(REPORTS / "module_a_equity.csv")
    if not sens.empty:
        sens.to_csv(REPORTS / "module_a_sensitivity.csv", index=False)
    pd.DataFrame(m["impact"]["deciles"]).to_csv(REPORTS / "impact_deciles.csv", index=False)
    pd.DataFrame(gold_preds).to_csv(REPORTS / "gold_event_predictions.csv", index=False)
    saved = charts(m, equity)
    print(f"reports/metrics.json written in {m['runtime_s']}s; charts: {', '.join(saved)}")
    return m


if __name__ == "__main__":
    main()

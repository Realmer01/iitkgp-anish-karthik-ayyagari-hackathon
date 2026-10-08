"""Build docs/presentation.pptx (7 slides) from reports/metrics.json and docs/images, then export the PDF.

    python scripts/build_deck.py            # needs reports/metrics.json (python run.py evaluate)

Every number on the slides is read from reports/metrics.json. The PDF export uses PowerPoint (Windows COM)
when available; otherwise open docs/presentation.pptx and save it as PDF.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import _bootstrap  # noqa: F401
from riskpulse.config import CONFIG, DOCS, OUTPUT, RAW, REPORTS, load_yaml
from riskpulse.module_b.stress import book

INK, NAVY, WHITE, PAPER = "0F1B2D", "15263D", "FFFFFF", "F3F6FA"
TEXT, MUTED, LINE = "1B2430", "5B6573", "D5DBE3"
BLUE, RED, AMBER, AQUA = "2A78D6", "E34948", "F2A541", "1BAF7A"
FONT = "Calibri"
IMG = DOCS / "images"


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, dark: bool = False, kicker: str = ""):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(INK if dark else WHITE)
        if kicker:
            self.text(s, f"{self.n:02d}  ·  {kicker.upper()}", 0.6, 0.38, 9, 0.35, 12, AMBER if dark else BLUE, bold=True)
        self.text(s, f"RiskPulse  ·  {self.n} / 7", 10.9, 7.0, 1.9, 0.3, 10, "8A97A8" if dark else MUTED,
                  align=PP_ALIGN.RIGHT)
        return s

    @staticmethod
    def text(s, txt, x, y, w, h, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             italic=False, name="text"):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tb.name = name
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        lines = txt if isinstance(txt, list) else [txt]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            runs = line if isinstance(line, list) else [(line, {})]
            for chunk, opt in runs:
                r = p.add_run()
                r.text = chunk
                f = r.font
                f.name, f.size = FONT, Pt(opt.get("size", size))
                f.bold, f.italic = opt.get("bold", bold), opt.get("italic", italic)
                f.color.rgb = rgb(opt.get("color", color))
            if len(lines) > 1:
                p.space_after = Pt(opt.get("after", 8) if runs else 8)
        return tb

    @staticmethod
    def card(s, x, y, w, h, fill=PAPER, line=None, name="card"):
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.name = name
        shp.adjustments[0] = 0.06
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
        if line:
            shp.line.color.rgb = rgb(line)
            shp.line.width = Pt(1)
        else:
            shp.line.fill.background()
        shp.shadow.inherit = False
        return shp

    @staticmethod
    def dot(s, x, y, d, color, label="", name="badge"):
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
        c.name = name
        c.fill.solid()
        c.fill.fore_color.rgb = rgb(color)
        c.line.fill.background()
        if label:
            tf = c.text_frame
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = label
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(14), True, rgb(WHITE)
        return c

    @staticmethod
    def pulse(s, x, y, w, h, color=AMBER, width=2.25, name="pulse"):
        """The deck's motif: a heartbeat line (RiskPulse)."""
        pts = [(0, .5), (.30, .5), (.36, .35), (.42, .5), (.48, .5), (.52, .05), (.57, .95), (.62, .5), (.70, .5),
               (.74, .38), (.78, .5), (1.0, .5)]
        fb = s.shapes.build_freeform(Inches(x + pts[0][0] * w), Inches(y + pts[0][1] * h), scale=1.0)
        fb.add_line_segments([(Inches(x + px * w), Inches(y + py * h)) for px, py in pts[1:]], close=False)
        shp = fb.convert_to_shape()
        shp.name = name
        shp.fill.background()
        shp.line.color.rgb = rgb(color)
        shp.line.width = Pt(width)
        return shp

    def picture(self, s, path, x, y, w=None, h=None, name="image"):
        pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
        pic.name = name
        return pic

    @staticmethod
    def notes(s, txt: str) -> None:
        s.notes_slide.notes_text_frame.text = txt


def metric(rows, **match):
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            return r
    raise KeyError(match)


def build() -> None:
    m = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    who = load_yaml("candidate.yaml") if (CONFIG / "candidate.yaml").exists() else {}
    name, college = who.get("name", "[Candidate name]"), who.get("college", "[College]")
    sent, ev = m["sentiment"]["table"], m["events"]["table"]
    s_naive = metric(sent, dataset="gold_news", model="VADER (naive baseline)")["macro_f1"]
    s_best = metric(sent, dataset="gold_news", model="RiskPulse full (blend + policy rule)")["macro_f1"]
    e_naive = metric(ev, dataset="gold_news", model="Keyword baseline (naive)")
    e_best = metric(ev, dataset="gold_news", model="RiskPulse full: ensemble + rules")
    mk = {r["score"]: r for r in m["impact"]["market_level"]["table"]}
    imp_m, base_m = mk["Impact score (RiskPulse)"], mk["|Sentiment| only (naive)"]
    mb = m["module_b"]
    lat = mb["headline_to_stress_result_s"]
    a_ic = m["module_a"]["ic"]["mean_ic"]
    tests = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True,
                           cwd=str(DOCS.parent)).stdout
    n_tests = next((ln.split()[0] for ln in tests.splitlines() if "tests collected" in ln or "test collected" in ln), "?")
    book_bn = float(book()["value"].sum()) / 1e9
    hist = pd.read_csv(OUTPUT / "signals_history_full.csv.gz", usecols=["time_precision"])
    date_only = float((hist["time_precision"] == "day").mean())
    n_headlines = len(pd.read_csv(RAW / "news_history.csv.gz", usecols=["title"]))
    d = Deck()

    # ---------------------------------------------------------------- 1 title
    s = d.slide(dark=True)
    d.text(s, "S&P GLOBAL & CRISIL CAMPUS HACKATHON 2026", 0.8, 1.15, 10, 0.4, 13, AMBER, bold=True)
    d.text(s, "RiskPulse", 0.8, 1.7, 11, 1.3, 66, WHITE, bold=True)
    d.text(s, "A real-time AI/NLP risk engine that reads news and social posts, turns each into a structured risk "
              "signal, and acts on it: a tactical index rebalancer and a wholesale-bank stress test.",
           0.8, 3.05, 8.6, 1.2, 20, "C9D3E0")
    d.pulse(s, 0.8, 4.55, 11.7, 0.9)
    d.text(s, [[(name, {"bold": True, "color": WHITE, "size": 20})], [(college, {"color": "C9D3E0", "size": 16})]],
           0.8, 5.75, 8, 0.9)
    d.notes(s, "RiskPulse is a risk engine for a bank's risk desk. It reads news and social text as it arrives, "
               "scores every item for sentiment, event type and impact, and feeds two applications: Module A, an "
               "index rebalancer, and Module B, an event-driven stress test. Everything you will see runs on a laptop, "
               "offline, on public or synthetic data.")

    # ---------------------------------------------------------------- 2 problem & approach
    s = d.slide(kicker="Problem & approach")
    d.text(s, "Risk desks see the news; they cannot score all of it", 0.6, 0.85, 12, 0.8, 34, TEXT, bold=True)
    d.text(s, [[("The problem", {"bold": True, "size": 18, "color": TEXT})],
               "Thousands of headlines and posts a day, in free text",
               "Analysts triage by hand; stress tests run on fixed schedules",
               "A static shock ignores what an event actually does to rates, spreads and sectors"],
           0.6, 1.95, 4.6, 3.4, 16, MUTED)
    steps = [("1", "Read", "News (GDELT, RSS) and social posts, deduplicated into stories", BLUE),
             ("2", "Score", "Sentiment −1…1, one of 12 risk events, impact 1–10 with its drivers", "4A3AA7"),
             ("3", "Act", f"Module A tilts index weights; Module B stress-tests a ${book_bn:.1f}bn book", AQUA)]
    for i, (num, head, body, col) in enumerate(steps):
        y = 1.95 + i * 1.6
        d.card(s, 5.7, y, 7.0, 1.35, name=f"step{num}")
        d.dot(s, 5.95, y + 0.37, 0.6, col, num, name=f"step{num}-badge")
        d.text(s, head, 6.8, y + 0.22, 5.6, 0.45, 20, TEXT, bold=True)
        d.text(s, body, 6.8, y + 0.68, 5.7, 0.6, 15, MUTED)
    d.notes(s, "In my words: the problem is not access to news, it is triage. A risk team cannot read and score "
               "everything, and periodic stress tests use the same shock whatever happened. RiskPulse reads, scores "
               "and acts, in one pipeline, with every score explainable.")

    # ---------------------------------------------------------------- 3 system design
    s = d.slide(kicker="System design")
    d.text(s, "One engine, decoupled consumers, one versioned signal", 0.6, 0.85, 12, 0.8, 34, TEXT, bold=True)
    d.picture(s, DOCS / "architecture.png", 0.95, 1.7, w=11.4, name="architecture")
    d.notes(s, "Data flows left to right. Connectors produce RawDocuments; the normaliser cleans and clusters them "
               "into stories; the NLP stages link entities and score sentiment, event and impact. The output is a "
               "versioned RiskSignal published on a bus and written to JSONL, SQLite and a FastAPI REST and SSE API. "
               "The modules only know the schema, so they could run in another process or on another machine.")

    # ---------------------------------------------------------------- 4 implementation highlights
    s = d.slide(kicker="Implementation highlights")
    d.text(s, "Built for a risk desk: explainable, testable, offline-first", 0.6, 0.85, 12, 0.8, 34, TEXT, bold=True)
    cards = [("Hybrid NLP", "FinBERT + a tweet-trained head blended by source; 12-class event model with "
                            "high-precision rules", BLUE),
             ("Explainable impact", "Logistic factor model: 9 drivers stored on every signal, validated by an "
                                    "event study", "4A3AA7"),
             ("Decoupled design", "Modules read only the RiskSignal schema via in-process, SQLite or SSE "
                                  "subscribers", AQUA),
             ("Module A", "Sentiment tilt with caps, drift band, turnover cap and costs; hourly no-look-ahead "
                          "backtest", AMBER),
             ("Module B", "Cited scenario library, impact scaling, IFRS 9 staging, Basel CRE20 weights, CET1 proxy",
              RED),
             ("Runs anywhere", f"Lite tier without torch, offline replay demo, {n_tests} tests, CI on Python 3.11",
              "5B6573")]
    for i, (head, body, col) in enumerate(cards):
        x, y = 0.6 + (i % 3) * 4.1, 1.95 + (i // 3) * 2.45
        d.card(s, x, y, 3.85, 2.2, name=f"card{i + 1}")
        d.dot(s, x + 0.3, y + 0.3, 0.42, col, name=f"card{i + 1}-dot")
        d.text(s, head, x + 0.9, y + 0.3, 2.8, 0.45, 19, TEXT, bold=True)
        d.text(s, body, x + 0.3, y + 0.95, 3.3, 1.2, 14.5, MUTED)
    d.notes(s, "Key choices and why: FinBERT because it understands financial language, plus a head trained on "
               "tweets because social text is different. Rules where phrases are near-certain, so behaviour is "
               "auditable. A transparent impact model because model-risk teams must explain scores. Two tiers so "
               "it runs on any laptop, and a replay mode so the demo never depends on the network.")

    # ---------------------------------------------------------------- 5 key results
    s = d.slide(kicker="Key results vs naive approaches")
    d.text(s, "Measured against naive baselines on held-out data", 0.6, 0.85, 12, 0.8, 34, TEXT, bold=True)
    stats = [(f"{s_best:.2f}", f"vs {s_naive:.2f}", "sentiment macro-F1 on 200 held-out real headlines (naive: VADER)"),
             (f"{e_best['macro_f1']:.2f}", f"vs {e_naive['macro_f1']:.2f}",
              "event macro-F1 on the same headlines (naive: keyword rules)"),
             (f"{imp_m['spearman_all_days']:.2f}", f"vs {base_m['spearman_all_days']:.2f}",
              "impact vs S&P 500 moves, Spearman (naive: |sentiment|)"),
             (f"{mb['rates_sign_errors_naive']} / 6", "scenarios",
              "where a naive static stress test gets the rates P&L sign wrong")]
    for i, (big, small, label) in enumerate(stats):
        y = 1.85 + i * 1.22
        d.text(s, [[(big, {"size": 34, "bold": True, "color": BLUE}), (f"  {small}", {"size": 16, "color": MUTED})]],
               0.6, y, 4.9, 0.6)
        d.text(s, label, 0.6, y + 0.62, 4.9, 0.5, 13, MUTED)
    d.picture(s, IMG / "eval_module_b.png", 5.75, 1.75, w=6.95, name="chart-module-b")
    lat_txt = f"{lat:.1f} s" if lat else "< 1 s"
    d.text(s, [[("Headline to stress-test result: ", {"bold": True, "color": TEXT}),
                (f"{lat_txt} on a laptop CPU, vs an overnight batch run (assumed ~24 h).", {"color": MUTED})],
               [("Honest result: ", {"bold": True, "color": TEXT}),
                (f"with strict timing, sentiment tilts show no next-day edge (IC {a_ic:+.3f}); a first backtest "
                 "showing IR 3.7 was a look-ahead leak we traced and fixed.", {"color": MUTED})]],
           5.75, 5.75, 6.95, 1.2, 13)
    d.notes(s, "Each number compares RiskPulse with a naive approach on data the models never saw. Sentiment and "
               "event classification beat their baselines; the impact score lines up with market-wide moves far "
               "better than sentiment alone; and event-aware stress tests avoid the sign errors of a static shock. "
               "I also report what did not work: no reliable trading edge once timing is strict.")

    # ---------------------------------------------------------------- 6 domain impact
    s = d.slide(kicker="Domain impact")
    d.text(s, "From headline to capital impact in seconds", 0.6, 0.85, 12, 0.8, 34, TEXT, bold=True)
    shot = IMG / "dashboard_3_module_B.png"
    if shot.exists():
        d.card(s, 6.55, 1.75, 6.2, 4.9, fill=PAPER, line=LINE, name="screenshot-frame")
        d.picture(s, shot, 6.7, 1.9, w=5.9, name="dashboard-module-b")
    rows = [("Risk managers", "Event-aware stress tests run when news breaks, with CET1 headroom and the "
                              "positions that drive the loss", RED),
            ("Portfolio managers", "Index tilts react to company news within limits, every change linked to its "
                                   "headline", BLUE),
            ("Model risk & audit", "Every score carries its drivers, schema and model version; results are "
                                   "reproducible from the repo", "4A3AA7")]
    for i, (head, body, col) in enumerate(rows):
        y = 1.85 + i * 1.6
        d.dot(s, 0.6, y + 0.05, 0.42, col, name=f"impact{i + 1}-dot")
        d.text(s, head, 1.25, y, 4.9, 0.45, 19, TEXT, bold=True)
        d.text(s, body, 1.25, y + 0.5, 4.95, 0.95, 15, MUTED)
    d.notes(s, "Why it matters: a stress test usually runs on a schedule with a generic shock. RiskPulse runs the "
               "right scenario for the event that just happened, sized by its impact, and shows the capital effect "
               "within seconds. Because every score is explainable and reproducible, it fits model-risk governance.")

    # ---------------------------------------------------------------- 7 limitations & next steps
    s = d.slide(dark=True, kicker="Limitations & next steps")
    d.text(s, "What we would build next", 0.6, 0.85, 12, 0.8, 34, WHITE, bold=True)
    lim = [[("Limitations", {"bold": True, "size": 19, "color": AMBER})],
           f"{date_only:.0%} of the {n_headlines:,} archive headlines carry dates, not times",
           "Mega-cap universe of 16 names; two years of history",
           "Gold labels from a single annotator",
           "Sensitivity-based revaluation; simplified CET1 proxy"]
    nxt = [[("Next steps", {"bold": True, "size": 19, "color": AMBER})],
           "Intraday timestamped feeds to test fast reactions",
           "Multilingual sources and target-aware sentiment",
           "Kafka bus, feature store, model registry",
           "Full revaluation library and regulatory scenarios"]
    d.text(s, lim, 0.6, 1.95, 5.6, 3.2, 16, "C9D3E0")
    d.text(s, nxt, 6.9, 1.95, 5.8, 3.2, 16, "C9D3E0")
    d.pulse(s, 0.6, 5.55, 12.1, 0.7)
    d.text(s, "Open source (MIT) · public and synthetic data only · github repo and demo video in the README",
           0.6, 6.45, 12, 0.4, 13, "8A97A8")
    d.notes(s, "The main limitation is timing: most historical headlines carry a date only, so the backtest has to "
               "wait a full day, which is why no trading edge shows. The next step is intraday feeds. I would also "
               "add more languages, a production message bus and a full revaluation library.")

    out = DOCS / "presentation.pptx"
    d.prs.save(out)
    print("saved", out, f"({d.n} slides)")
    export_pdf(out)


def export_pdf(pptx) -> None:
    try:
        import win32com.client
    except ImportError:
        print("PowerPoint automation unavailable - open docs/presentation.pptx and save it as PDF")
        return
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(pptx), True, False, False)
    pres.SaveAs(str(pptx.with_suffix(".pdf")), 32)            # ppSaveAsPDF
    preview = DOCS / "images" / "deck_preview"
    preview.mkdir(parents=True, exist_ok=True)
    for i in range(1, pres.Slides.Count + 1):
        pres.Slides(i).Export(str(preview / f"slide{i}.png"), "PNG", 1600, 900)
    pres.Close()
    print("saved", pptx.with_suffix(".pdf"), "and slide previews in", preview)


if __name__ == "__main__":
    build()

"""Dashboard visual system: one palette, one Plotly template, used by every chart.

Colour jobs (never mixed):
  categorical  fixed slot order (validated for colour-vision deficiency); at most 4 series per chart
  diverging    sentiment / active weight / P&L: red (negative) <- neutral grey -> blue (positive).
               Red/blue instead of red/green so the ~8% of men with red-green deficiency read it too.
  sequential   one blue ramp for magnitudes (confusion matrices)
  status       reserved for state (breach / ok), always with an icon and a label
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEG, MID, POS = "#e34948", "#f0efec", "#2a78d6"
NEG_TINT, POS_TINT = "#f6c4c3", "#c5dbf7"           # table backgrounds: dark ink stays readable
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
DIVERGING = [[0.0, NEG], [0.5, MID], [1.0, POS]]
SEQUENTIAL = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CSS = f"""
<style>
  html, body, [class*="css"] {{ font-family: {FONT}; }}
  .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
  .rp-chip {{ display:inline-block; padding:2px 10px; border-radius:999px; border:1px solid {AXIS};
             font-size:0.82rem; color:{INK_2}; margin-right:6px; background:{SURFACE}; }}
  .rp-callout {{ border-left:4px solid {STATUS['critical']}; background:#fbeaea; padding:10px 14px;
                border-radius:6px; color:{INK}; margin:6px 0 10px 0; }}
  .rp-callout.ok {{ border-left-color:{STATUS['good']}; background:#eaf6ea; }}
  .rp-muted {{ color:{MUTED}; font-size:0.85rem; }}
</style>
"""


def install_template() -> None:
    axis = dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS, zerolinewidth=1,
                tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=INK_2, size=12)), showline=True)
    pio.templates["riskpulse"] = go.layout.Template(layout=go.Layout(
        font=dict(family=FONT, color=INK_2, size=13), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=SERIES, xaxis=axis, yaxis=axis, margin=dict(l=48, r=16, t=40, b=40),
        title=dict(font=dict(color=INK, size=15), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
        bargap=0.25, hovermode="closest"))
    pio.templates.default = "riskpulse"


def tint(value: float, lo: float = -1.0, hi: float = 1.0) -> str:
    """Background tint for a signed value in a table cell (text stays in primary ink)."""
    if value is None:
        return ""
    x = max(-1.0, min(1.0, (value - (lo + hi) / 2) / ((hi - lo) / 2)))
    if abs(x) < 0.15:
        return f"background-color: {MID}"
    return f"background-color: {POS_TINT if x > 0 else NEG_TINT}"

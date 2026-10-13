"""Figure builders. Thin marks, rounded bar ends, 2px surface gaps, hairline grid, hover on every mark."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from riskpulse.dashboard.theme import AXIS, DIVERGING, MUTED, NEG, POS, SEQUENTIAL, SERIES, SURFACE


def _finish(fig: go.Figure, title: str, height: int) -> go.Figure:
    """Title on its own line at the top; a top legend gets a separate row underneath it (never on the title)."""
    legend = fig.layout.showlegend is not False and sum(t.showlegend is not False for t in fig.data) > 1
    fig.update_layout(title=dict(text=title, y=1.0, yref="container", yanchor="top", pad=dict(t=10)), height=height)
    if legend and (fig.layout.legend.y is None or fig.layout.legend.y >= 1):
        fig.update_layout(margin=dict(t=78), legend=dict(y=1.02, yanchor="bottom", x=0, xanchor="left"))
    return fig


def bars_h(values: dict[str, float], title: str, fmt: str = ",.0f", height: int = 320) -> go.Figure:
    """Single-series magnitude bars (one colour), largest at the top."""
    s = pd.Series(values).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    return _finish(fig, title, height)


def signed_bars_h(values: dict[str, float], title: str, fmt: str = "$,.3s", height: int = 360,
                  xtitle: str = "") -> go.Figure:
    """Polarity bars: negative red, positive blue, sorted so the worst is at the top."""
    s = pd.Series(values).sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker=dict(color=[NEG if v < 0 else POS for v in s.values], cornerradius=4),
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    fig.update_xaxes(tickformat=fmt, title=xtitle)
    fig.update_yaxes(type="category")
    return _finish(fig, title, height)


def waterfall(before: float, steps: dict[str, float], title: str, height: int = 360, unit: float = 1e9,
              unit_label: str = "$bn") -> go.Figure:
    """Value before -> each step -> after, in `unit_label` (finance style: $bn, not Plotly's $G)."""
    labels = ["Before", *steps.keys(), "After"]
    b, st_ = before / unit, {k: v / unit for k, v in steps.items()}
    vals = [b, *st_.values(), 0]
    measures = ["absolute", *["relative"] * len(st_), "total"]
    fig = go.Figure(go.Waterfall(x=labels, y=vals, measure=measures, connector=dict(line=dict(color=AXIS, width=1)),
                                 increasing=dict(marker=dict(color=POS)), decreasing=dict(marker=dict(color=NEG)),
                                 totals=dict(marker=dict(color=MUTED)),
                                 hovertemplate="%{x}: %{y:,.3f} " + unit_label + "<extra></extra>"))
    low, high = min(b, b + sum(st_.values())), max(b, b + max(0, sum(v for v in st_.values() if v > 0)))
    pad = (high - low) * 0.25 or abs(b) * 0.01
    fig.update_yaxes(tickformat=",.2f", title=unit_label, range=[low - pad, high + pad * 0.3])
    fig.update_xaxes(type="category")
    return _finish(fig, title, height)


def heatmap_signed(df: pd.DataFrame, title: str, fmt: str = "$,.3s", height: int = 380,
                   colorbar_title: str = "", show_values: bool = False, xtitle: str = "", ytitle: str = "") -> go.Figure:
    lim = float(df.abs().to_numpy().max()) or 1.0
    fig = go.Figure(go.Heatmap(z=df.values, x=[str(c) for c in df.columns], y=[str(i) for i in df.index],
                               colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
                               colorbar=dict(title=colorbar_title, thickness=10, tickformat=fmt),
                               texttemplate="%{z:" + fmt + "}" if show_values else None,
                               hovertemplate="%{y} · %{x}: %{z:" + fmt + "}<extra></extra>"))
    fig.update_layout(plot_bgcolor=SURFACE)
    fig.update_xaxes(showgrid=False, type="category", title=xtitle)     # number-like labels stay evenly spaced
    fig.update_yaxes(showgrid=False, type="category", autorange="reversed", title=ytitle)
    return _finish(fig, title, height)


def heatmap_counts(z, x, y, title: str, height: int = 420) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=SEQUENTIAL, xgap=2, ygap=2, colorbar=dict(thickness=10),
                               texttemplate="%{z}", textfont=dict(size=13),
                               hovertemplate="true %{y} · predicted %{x}: %{z}<extra></extra>"))
    fig.update_xaxes(showgrid=False, type="category", title="predicted", tickangle=-35 if len(x) > 5 else 0)
    fig.update_yaxes(showgrid=False, type="category", title="true label", autorange="reversed")
    return _finish(fig, title, height)


def lines(df: pd.DataFrame, title: str, yfmt: str = ".2f", height: int = 340, reference: pd.Series | None = None,
          reference_name: str = "", ytitle: str = "", xfmt: str | None = None) -> go.Figure:
    """Up to 4 series in fixed slot order, unified crosshair hover, legend always shown for >= 2 series."""
    fig = go.Figure()
    for i, col in enumerate(df.columns[:4]):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=str(col), mode="lines",
                                 line=dict(color=SERIES[i], width=2), hovertemplate="%{y:" + yfmt + "}"))
    if reference is not None:
        fig.add_trace(go.Scatter(x=reference.index, y=reference.values, name=reference_name, mode="lines",
                                 line=dict(color=MUTED, width=1), hovertemplate="%{y:" + yfmt + "}"))
    fig.update_layout(hovermode="x unified", showlegend=len(fig.data) > 1,          # legend under the x axis
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), margin=dict(b=80))
    fig.update_yaxes(tickformat=yfmt, title=ytitle)
    if xfmt:                                                     # one-line date ticks never collide with the legend
        fig.update_xaxes(tickformat=xfmt, hoverformat=xfmt)
    return _finish(fig, title, height)


def grouped_bars(index: list[str], series: dict[str, list[float]], title: str, fmt: str = ".2f",
                 height: int = 340, xtitle: str = "", ytitle: str = "") -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(list(series.items())[:4]):
        fig.add_trace(go.Bar(x=index, y=vals, name=name, marker=dict(color=SERIES[i], cornerradius=4),
                             hovertemplate=name + " · %{x}: %{y:" + fmt + "}<extra></extra>"))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_xaxes(type="category", title=xtitle)              # "1".."10" stay ten labelled categories
    fig.update_yaxes(tickformat=fmt, title=ytitle)
    return _finish(fig, title, height)


def histogram(values, title: str, xtitle: str, bins: int = 18, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=bins, marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
                                 hovertemplate=xtitle + " %{x}: %{y} signals<extra></extra>"))
    fig.update_xaxes(title=xtitle)
    return _finish(fig, title, height)

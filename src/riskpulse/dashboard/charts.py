"""Figure builders. Thin marks, rounded bar ends, 2px surface gaps, hairline grid, hover on every mark."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from riskpulse.dashboard.theme import (AXIS, DIVERGING, MUTED, NEG, POS, SEQUENTIAL, SERIES, SURFACE)


def _finish(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(title=title, height=height)
    return fig


def bars_h(values: dict[str, float], title: str, fmt: str = ",.0f", height: int = 320) -> go.Figure:
    """Single-series magnitude bars (one colour), largest at the top."""
    s = pd.Series(values).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    return _finish(fig, title, height)


def signed_bars_h(values: dict[str, float], title: str, fmt: str = "$,.3s", height: int = 360) -> go.Figure:
    """Polarity bars: negative red, positive blue, sorted so the worst is at the top."""
    s = pd.Series(values).sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker=dict(color=[NEG if v < 0 else POS for v in s.values], cornerradius=4),
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    fig.update_xaxes(tickformat=fmt)
    return _finish(fig, title, height)


def waterfall(before: float, steps: dict[str, float], title: str, height: int = 360) -> go.Figure:
    labels = ["Before", *steps.keys(), "After"]
    vals = [before, *steps.values(), 0]
    measures = ["absolute", *["relative"] * len(steps), "total"]
    fig = go.Figure(go.Waterfall(x=labels, y=vals, measure=measures, connector=dict(line=dict(color=AXIS, width=1)),
                                 increasing=dict(marker=dict(color=POS)), decreasing=dict(marker=dict(color=NEG)),
                                 totals=dict(marker=dict(color=MUTED)),
                                 hovertemplate="%{x}: %{y:$,.4s}<extra></extra>"))
    fig.update_yaxes(tickformat="$,.3s", range=[min(before, before + sum(steps.values())) * 0.97,
                                                max(before, before + max(0, sum(v for v in steps.values() if v > 0))) * 1.005])
    return _finish(fig, title, height)


def heatmap_signed(df: pd.DataFrame, title: str, fmt: str = "$,.3s", height: int = 380,
                   colorbar_title: str = "") -> go.Figure:
    lim = float(df.abs().to_numpy().max()) or 1.0
    fig = go.Figure(go.Heatmap(z=df.values, x=[str(c) for c in df.columns], y=[str(i) for i in df.index],
                               colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
                               colorbar=dict(title=colorbar_title, thickness=10, tickformat=fmt),
                               hovertemplate="%{y} · %{x}: %{z:" + fmt + "}<extra></extra>"))
    fig.update_layout(plot_bgcolor=SURFACE)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _finish(fig, title, height)


def heatmap_counts(z, x, y, title: str, height: int = 420) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=SEQUENTIAL, xgap=2, ygap=2, colorbar=dict(thickness=10),
                               hovertemplate="true %{y} · predicted %{x}: %{z}<extra></extra>"))
    fig.update_xaxes(showgrid=False, tickangle=-35)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _finish(fig, title, height)


def lines(df: pd.DataFrame, title: str, yfmt: str = ".2f", height: int = 340, reference: pd.Series | None = None,
          reference_name: str = "", ytitle: str = "") -> go.Figure:
    """Up to 4 series in fixed slot order, unified crosshair hover, legend always shown for >= 2 series."""
    fig = go.Figure()
    for i, col in enumerate(df.columns[:4]):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=str(col), mode="lines",
                                 line=dict(color=SERIES[i], width=2), hovertemplate="%{y:" + yfmt + "}"))
    if reference is not None:
        fig.add_trace(go.Scatter(x=reference.index, y=reference.values, name=reference_name, mode="lines",
                                 line=dict(color=MUTED, width=1), hovertemplate="%{y:" + yfmt + "}"))
    fig.update_layout(hovermode="x unified", showlegend=len(fig.data) > 1)
    fig.update_yaxes(tickformat=yfmt, title=ytitle)
    return _finish(fig, title, height)


def grouped_bars(index: list[str], series: dict[str, list[float]], title: str, fmt: str = ".2f",
                 height: int = 340) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(list(series.items())[:4]):
        fig.add_trace(go.Bar(x=index, y=vals, name=name, marker=dict(color=SERIES[i], cornerradius=4),
                             hovertemplate=name + " · %{x}: %{y:" + fmt + "}<extra></extra>"))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(tickformat=fmt)
    return _finish(fig, title, height)


def histogram(values, title: str, xtitle: str, bins: int = 18, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=bins, marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
                                 hovertemplate=xtitle + " %{x}: %{y} signals<extra></extra>"))
    fig.update_xaxes(title=xtitle)
    return _finish(fig, title, height)

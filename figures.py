"""All Plotly figures. Every function takes plain DataFrames and returns a go.Figure.

Nothing in here mutates the (shared, cached) DataFrames it receives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import data_utils as du
import geo

# ---- theme -------------------------------------------------------------------------------
ORANGE, CYAN, GREEN, PURPLE = "#ff8c00", "#4cc9f0", "#9be564", "#c77dff"
RED, YELLOW, GREY, TEXT = "#ff5d73", "#ffd166", "#8c8c8c", "#e8e8e8"
PANEL, GRID = "#151515", "#2b2b2b"
WAY = [ORANGE, CYAN, GREEN, PURPLE, RED, YELLOW, "#06d6a0", "#8d99ae"]
SCALE = [[0, "#2a1500"], [0.5, "#b85f00"], [1, "#ffb347"]]
DIVERGING = [[0, CYAN], [0.5, "#1b1b1b"], [1, ORANGE]]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MODEL_COLORS = {"SARIMA": ORANGE, "XGBoost": CYAN, "SARIMA + XGBoost": GREEN}


def style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL, colorway=WAY, height=height,
        font=dict(color=TEXT, size=12), title_text=title or "", title_x=0.01, title_y=0.97, title_yanchor="top",
        title_font_size=15,
        # room above the plot for the legend (which wraps to 2-3 rows) and, if present, the title above it
        margin=dict(l=55, r=20, t=100 if title else 62, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def empty_fig(msg: str = "Load data to begin", height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=GREY), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=height)


def _need(df: pd.DataFrame, *cols: str):
    miss = [c for c in cols if c not in df.columns]
    return empty_fig(f"This chart needs column(s): {', '.join(miss)}") if miss else None


def _top_levels(df: pd.DataFrame, col: str, n: int = 12) -> list:
    return df[col].value_counts().head(n).index.tolist()


# ============================================================================================
# 2. Univariate
# ============================================================================================
def _prep(view: pd.DataFrame, log: bool) -> pd.DataFrame:
    v = view.dropna(subset=["value"])
    if log:
        v = v[v["value"] > 0].copy()
        v["value"] = np.log10(v["value"])
    return v


def hist_fig(view, label, bins, log):
    v = _prep(view, log)
    if v.empty:
        return empty_fig("No values to plot")
    xlabel = f"log10({label})" if log else label
    fig = px.histogram(v, x="value", nbins=int(bins), color_discrete_sequence=[ORANGE])
    fig.update_traces(marker_line_width=0)
    mean, med = float(v["value"].mean()), float(v["value"].median())
    fig.add_vline(x=mean, line_dash="dash", line_color=CYAN, annotation_text=f"mean {mean:,.2f}",
                  annotation_position="top right")
    fig.add_vline(x=med, line_dash="dot", line_color=GREEN, annotation_text=f"median {med:,.2f}",
                  annotation_position="top left")
    fig.update_layout(bargap=0.03, xaxis_title=xlabel, yaxis_title="Count")
    return style(fig, height=340)


def box_fig(view, label, log):
    v = _prep(view, log)
    if v.empty:
        return empty_fig("No values to plot")
    fig = go.Figure(go.Box(x=v["value"], name="", boxpoints="outliers", boxmean="sd", marker_color=ORANGE,
                           line_color=ORANGE, marker=dict(size=3, opacity=0.5)))
    fig.update_layout(xaxis_title=f"log10({label})" if log else label, showlegend=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=340)


def zscore_fig(view, label, log, thr):
    """z-score histogram + z over time. Returns (figure, outlier_count, total)."""
    v = _prep(view, log).copy()
    if len(v) < 3 or v["value"].std(ddof=0) == 0:
        return empty_fig("Not enough variation for z-scores"), 0, len(v)
    v["z"] = (v["value"] - v["value"].mean()) / v["value"].std(ddof=0)
    out = v["z"].abs() > thr

    fig = make_subplots(rows=1, cols=2, column_widths=[0.32, 0.68], horizontal_spacing=0.07,
                        subplot_titles=("Distribution of z", "z-score over time"))
    fig.add_trace(go.Histogram(x=v["z"], nbinsx=40, marker_color=ORANGE, showlegend=False), row=1, col=1)
    for s in (-thr, thr):
        fig.add_vline(x=s, line_dash="dash", line_color=RED, row=1, col=1)
        fig.add_hline(y=s, line_dash="dash", line_color=RED, row=1, col=2)
    fig.add_trace(go.Scatter(x=v.loc[~out, "date"], y=v.loc[~out, "z"], mode="markers", name=f"|z| ≤ {thr:g}",
                             marker=dict(color=GREY, size=4, opacity=0.6)), row=1, col=2)
    fig.add_trace(go.Scatter(x=v.loc[out, "date"], y=v.loc[out, "z"], mode="markers", name=f"|z| > {thr:g}",
                             marker=dict(color=RED, size=6)), row=1, col=2)
    fig.update_xaxes(title_text="z", row=1, col=1)
    fig.update_yaxes(title_text="z", row=1, col=2)
    fig.update_annotations(font_size=12)
    style(fig, height=380)
    # the subplot titles sit just above each plot, so lift the legend clear of them
    fig.update_layout(margin=dict(l=55, r=20, t=100, b=50),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.14, yanchor="bottom"))
    return fig, int(out.sum()), int(len(v))


def stats_records(view: pd.DataFrame) -> list[dict]:
    s = view["value"].dropna()
    if s.empty:
        return []
    q = s.quantile([0.25, 0.5, 0.75])
    rows = [("Count", len(s)), ("Mean", s.mean()), ("Std dev", s.std()), ("Min", s.min()), ("25%", q[0.25]),
            ("Median", q[0.5]), ("75%", q[0.75]), ("Max", s.max()), ("Skewness", s.skew()),
            ("Excess kurtosis", s.kurt())]
    return [{"stat": k, "value": f"{v:,.3f}" if k != "Count" else f"{int(v):,}"} for k, v in rows]


def candle_fig(df, freq):
    """OHLC candles of the daily-sales series: open/close = first/last day of the period."""
    s = du.daily_series(df)
    per = {"Week": "W", "Month": "M", "Quarter": "Q"}.get(freq, "M")
    g = s.groupby(s.index.to_period(per))
    o, c, h, l = g.first(), g.last(), g.max(), g.min()
    fig = go.Figure(go.Candlestick(
        x=o.index.to_timestamp(), open=o.to_numpy(), high=h.to_numpy(), low=l.to_numpy(), close=c.to_numpy(),
        increasing_line_color=ORANGE, increasing_fillcolor=ORANGE,
        decreasing_line_color=CYAN, decreasing_fillcolor=CYAN, name="Daily sales"))
    fig.update_layout(xaxis_rangeslider_visible=False, yaxis_title="Daily sales")
    return style(fig, height=360)


def cat_count_fig(df, col, topn):
    miss = _need(df, col)
    if miss is not None:
        return miss
    vc = df[col].value_counts()
    total = int(vc.sum())
    vc = vc.head(int(topn)).iloc[::-1]
    fig = go.Figure(go.Bar(x=vc.to_numpy(), y=vc.index.astype(str), orientation="h", marker_color=ORANGE,
                           text=[f"{v / total:.1%}" for v in vc.to_numpy()], textposition="outside",
                           cliponaxis=False))
    fig.update_layout(xaxis_title="Order lines", showlegend=False)
    return style(fig, height=max(260, 30 * len(vc) + 90))


# ============================================================================================
# 3. Bivariate
# ============================================================================================
def corr_fig(df, method):
    fr = du.daily_frame(df).select_dtypes("number")
    corr = fr.corr(method=method)
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=DIVERGING, aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(thickness=10))
    return style(fig, height=420)


def pivot_fig(df, row, col, agg):
    miss = _need(df, row, col)
    if miss is not None:
        return miss
    if row == col:
        return empty_fig("Pick two different variables")
    d = df[df[row].isin(_top_levels(df, row, 15)) & df[col].isin(_top_levels(df, col, 15))]
    fn = {"Sum": "sum", "Mean": "mean", "Count": "count"}[agg]
    pv = pd.pivot_table(d, index=row, columns=col, values="Sales", aggfunc=fn)
    fig = px.imshow(pv, text_auto=",.0f", color_continuous_scale=SCALE, aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(thickness=10, title=agg))
    return style(fig, height=420)


def box_by_fig(df, by, kind, log):
    miss = _need(df, by)
    if miss is not None:
        return miss
    d = df[df[by].isin(_top_levels(df, by, 12))]
    groups = [g["Sales"].to_numpy() for _, g in d.groupby(by) if len(g) > 0]
    title = ""
    if len(groups) >= 2:
        try:
            title = f"Kruskal–Wallis across {by}: p = {stats.kruskal(*groups).pvalue:.3g}"
        except ValueError:  # all values identical
            title = ""
    order = d.groupby(by)["Sales"].median().sort_values(ascending=False).index.tolist()
    plot = px.violin if kind == "violin" else px.box
    kwargs = dict(box=True) if kind == "violin" else {}
    fig = plot(d, x=by, y="Sales", color=by, log_y=bool(log), category_orders={by: order}, **kwargs)
    fig.update_layout(showlegend=False)
    return style(fig, title=title, height=420)


def scatter_fig(df, xcol):
    fr = du.daily_frame(df)
    if xcol not in fr.columns:
        return empty_fig(f"'{xcol}' is not available in this dataset")
    d = fr[[xcol, "Daily Sales"]].dropna()
    fig = go.Figure(go.Scatter(x=d[xcol], y=d["Daily Sales"], mode="markers", name="Days",
                               marker=dict(color=ORANGE, size=5, opacity=0.45)))
    title = ""
    if d[xcol].nunique() > 1 and len(d) > 3:
        slope, icpt = np.polyfit(d[xcol], d["Daily Sales"], 1)
        xs = np.array([d[xcol].min(), d[xcol].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", name="OLS fit", line=dict(color=CYAN)))
        pr = stats.pearsonr(d[xcol], d["Daily Sales"])[0]
        sr = stats.spearmanr(d[xcol], d["Daily Sales"])[0]
        title = f"Pearson r = {pr:.2f}  ·  Spearman ρ = {sr:.2f}"
    fig.update_layout(xaxis_title=xcol, yaxis_title="Daily sales")
    return style(fig, title=title, height=380)


def calendar_fig(df):
    s = du.daily_series(df)
    pv = s.groupby([s.index.month, s.index.dayofweek]).mean().unstack()
    pv.index = [MONTHS[m - 1] for m in pv.index]
    pv.columns = [DAYS[d] for d in pv.columns]
    fig = px.imshow(pv, text_auto=",.0f", color_continuous_scale=SCALE, aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(thickness=10, title="Avg"))
    return style(fig, height=420)


# ============================================================================================
# 4. EDA
# ============================================================================================
METRICS = {"Total sales": "sum", "Order lines": "size", "Average line sales": "mean"}


def geo_figs(df, level, metric, coords):
    """(map, top-N bar, note). City level needs `coords` from geo.resolve_city_coords."""
    note = []
    if "Country" in df.columns and df["Country"].nunique() > 1:
        note.append(f"The data has {df['Country'].nunique()} countries; the map shows US locations only.")
    if level == "City" and (coords is None or coords.empty):
        note.append("City-level map needs the `zipcodes` package (pip install zipcodes); showing states instead.")
        level = "State"
    keys = ["State"] if level == "State" else ["City", "State"]
    miss = _need(df, *keys)
    if miss is not None:
        return miss, miss, ""

    how = METRICS.get(metric, "sum")
    g = df.groupby(keys)["Sales"].agg(how).rename("value").reset_index()
    if level == "State":
        g["lat"] = g["State"].map(lambda s: geo.STATE_CENTROIDS.get(s, (np.nan, np.nan))[0])
        g["lon"] = g["State"].map(lambda s: geo.STATE_CENTROIDS.get(s, (np.nan, np.nan))[1])
        g["name"] = g["State"]
    else:
        g = g.merge(coords, on=["City", "State"], how="left")
        g["name"] = g["City"] + ", " + g["State"]
    placed = g.dropna(subset=["lat", "lon"])
    if len(placed) < len(g):
        note.append(f"{len(g) - len(placed)} of {len(g)} {level.lower()}s could not be placed on the map.")
    if placed.empty:
        return empty_fig("No locations could be placed"), empty_fig(""), " ".join(note)

    sizes = placed["value"].clip(lower=0)
    sizeref = 2.0 * max(float(sizes.max()), 1e-9) / (46.0 ** 2)
    fig = go.Figure(go.Scattergeo(
        lat=placed["lat"], lon=placed["lon"], text=placed["name"], customdata=placed["value"],
        hovertemplate="<b>%{text}</b><br>" + metric + ": %{customdata:,.2f}<extra></extra>",
        marker=dict(size=sizes, sizemode="area", sizeref=sizeref, sizemin=3, color=placed["value"],
                    colorscale=SCALE, showscale=True, opacity=0.85, line=dict(width=0.5, color="#000"),
                    colorbar=dict(title=metric, thickness=10, len=0.6))))
    fig.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor="#202020",
                    bgcolor=PANEL, showlakes=False, showsubunits=True, subunitcolor="#4a4a4a")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    style(fig, height=520)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))

    top = placed.nlargest(12, "value").iloc[::-1]
    bar = go.Figure(go.Bar(x=top["value"], y=top["name"], orientation="h", marker_color=ORANGE))
    bar.update_layout(xaxis_title=metric, showlegend=False)
    style(bar, title=f"Top {len(top)} by {metric.lower()}", height=520)
    return fig, bar, " ".join(note)


def drill_fig(df, kind, top_products):
    miss = _need(df, "Category", "Sub-Category")
    if miss is not None:
        return miss
    path = ["Category", "Sub-Category"]
    d = df
    if "Product Name" in df.columns:
        path.append("Product Name")
        d = df.groupby(path, as_index=False)["Sales"].sum()
        rank = d.groupby(["Category", "Sub-Category"])["Sales"].rank(method="first", ascending=False)
        d.loc[rank > int(top_products), "Product Name"] = "Other products"
    d = d.groupby(path, as_index=False)["Sales"].sum()
    d = d[d["Sales"] > 0]  # sunburst/treemap cannot show negative or zero areas
    make = px.treemap if kind == "Treemap" else px.sunburst
    fig = make(d, path=path, values="Sales", color="Sales", color_continuous_scale=SCALE, maxdepth=2)
    fig.update_traces(hovertemplate="<b>%{label}</b><br>Sales: %{value:,.0f}<extra></extra>")
    fig.update_layout(coloraxis_showscale=False)
    style(fig, height=560)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


def trend_fig(df, grain, split, drill):
    """Sales over time. split='split' draws one line per Category, or per Sub-Category once a
    category is selected (click a line to drill in)."""
    sel = drill not in (None, "All") and "Category" in df.columns
    cols = ["Order Date", "Sales"] + [c for c in ("Category", "Sub-Category") if c in df.columns]
    d = df[cols]
    if sel:
        d = d[d["Category"] == drill]
    per = {"Year": "Y", "Quarter": "Q", "Month": "M"}.get(grain, "M")
    d = d.assign(Period=d["Order Date"].dt.to_period(per).dt.to_timestamp())

    by = ("Sub-Category" if sel else "Category") if split == "split" else None
    if by is not None and by not in d.columns:
        by = None

    fig = go.Figure()
    if by is None:
        t = d.groupby("Period")["Sales"].sum()
        fig.add_trace(go.Scatter(x=t.index, y=t.to_numpy(), mode="lines+markers", name="Total sales",
                                 line=dict(color=ORANGE, width=2.5)))
    else:
        pt = d.groupby(["Period", by])["Sales"].sum().unstack(fill_value=0)
        for name in pt.columns:
            fig.add_trace(go.Scatter(x=pt.index, y=pt[name].to_numpy(), mode="lines+markers", name=str(name),
                                     customdata=[str(name)] * len(pt)))
    tickfmt = {"Year": "%Y", "Quarter": "%b %Y", "Month": "%b %Y"}[grain if grain in ("Year", "Quarter") else "Month"]
    fig.update_layout(yaxis_title="Sales", yaxis_tickformat=",.0f", xaxis_tickformat=tickfmt, hovermode="x unified")
    title = f"Sales by {grain.lower()}" + (f" · {drill}" if sel else "")
    return style(fig, title=title, height=400)


def yoy_fig(df, drill):
    sel = drill not in (None, "All") and "Category" in df.columns
    d = df[df["Category"] == drill] if sel else df
    pt = d.groupby([d["Order Date"].dt.month, d["Order Date"].dt.year])["Sales"].sum().unstack()
    fig = go.Figure()
    for yr in pt.columns:
        fig.add_trace(go.Scatter(x=[MONTHS[m - 1] for m in pt.index], y=pt[yr].to_numpy(), mode="lines+markers",
                                 name=str(yr)))
    fig.update_layout(yaxis_title="Sales", yaxis_tickformat=",.0f", hovermode="x unified")
    return style(fig, title="Seasonality: month by month, one line per year" + (f" · {drill}" if sel else ""),
                 height=400)


# ============================================================================================
# 5. Models
# ============================================================================================
def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def forecast_fig(r):
    y, p, n_train = r["y"], r["preds"], r["n_train"]
    tail = y.iloc[max(0, n_train - 90):n_train]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail.index, y=tail.to_numpy(), name="Train (last 90 days)",
                             line=dict(color=GREY, width=1)))
    fig.add_trace(go.Scatter(x=p.index, y=p["Actual"], name="Actual (test)", line=dict(color=TEXT, width=1.4)))
    for m, color in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(x=p.index, y=p[m], name=m + " (one-step)" if m == "SARIMA" else m,
                                 line=dict(color=color, width=1.6)))
    fig.add_trace(go.Scatter(x=p.index, y=p["SARIMA (multi-step)"], name="SARIMA (multi-step)", visible="legendonly",
                             line=dict(color=ORANGE, dash="dot")))
    fig.add_trace(go.Scatter(x=p.index, y=p["Seasonal naive"], name="Seasonal naive", visible="legendonly",
                             line=dict(color=PURPLE, dash="dot")))
    fig.add_shape(type="line", x0=_iso(r["split_date"]), x1=_iso(r["split_date"]), y0=0, y1=1, yref="paper",
                  line=dict(color=YELLOW, dash="dash", width=1))
    fig.add_annotation(x=_iso(r["split_date"]), y=1, yref="paper", text="train | test", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=YELLOW, size=11))
    fig.update_layout(yaxis_title="Daily sales", hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    style(fig, title="Actual vs predicted on the held-out 30%", height=520)
    fig.update_layout(margin=dict(l=55, r=20, t=125, b=50))  # 7 legend entries wrap to several rows
    return fig


def parity_fig(r):
    p = r["preds"]
    fig = go.Figure()
    hi = float(max(p["Actual"].max(), p[list(MODEL_COLORS)].max().max()))
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="perfect", line=dict(color=GREY, dash="dash")))
    for m, color in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(x=p["Actual"], y=p[m], mode="markers", name=m,
                                 marker=dict(color=color, size=5, opacity=0.5)))
    fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
    return style(fig, title="Predicted vs actual (closer to the dashed line is better)", height=400)


def metrics_bar_fig(r):
    m = r["metrics"]
    best = m.loc[m["Model"].isin(MODEL_COLORS), "RMSE"].idxmin()
    colors = [ORANGE if i == best else GREY for i in m.index]
    fig = go.Figure(go.Bar(x=m["Model"], y=m["RMSE"], marker_color=colors, text=m["RMSE"].round(0), textposition="outside"))
    fig.update_layout(yaxis_title="RMSE (lower is better)", showlegend=False)
    return style(fig, title="Test RMSE by model", height=400)


def importance_fig(r):
    imp = r["importance"]
    top = imp.max(axis=1).nlargest(10).index
    imp = imp.loc[top].iloc[::-1]
    fig = go.Figure()
    for m in ("XGBoost", "SARIMA + XGBoost"):
        fig.add_trace(go.Bar(x=imp[m], y=imp.index, orientation="h", name=m, marker_color=MODEL_COLORS[m]))
    fig.update_layout(barmode="group", xaxis_title="Importance")
    return style(fig, title="What XGBoost relies on", height=400)


def future_fig(r):
    y, fut = r["y"], r["future"]
    hist = y.iloc[-30:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.to_numpy(), name="Actual (last 30 days)",
                             line=dict(color=TEXT, width=1.6)))
    fig.add_trace(go.Scatter(
        x=list(fut.index) + list(fut.index[::-1]),
        y=list(fut["SARIMA 95% high"]) + list(fut["SARIMA 95% low"].iloc[::-1]),
        fill="toself", fillcolor="rgba(255,140,0,0.15)", line=dict(width=0), hoverinfo="skip",
        name="SARIMA 95% interval"))
    for m, color in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(x=fut.index, y=fut[m], name=m, mode="lines+markers", line=dict(color=color, width=2)))
    fig.update_layout(yaxis_title="Daily sales", hovermode="x unified")
    return style(fig, title=f"Next {len(fut)} days", height=400)

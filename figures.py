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
ORANGE, CYAN, GREEN, PURPLE = "#F96E2A", "#78B3CE", "#3a8f6e", "#7a5ea8"
RED, YELLOW, GREY, TEXT = "#c0392b", "#d49a00", "#8a9aa3", "#1f3340"
PANEL, GRID = "#ffffff", "#e3edf1"
WAY = [ORANGE, CYAN, GREEN, PURPLE, RED, YELLOW, "#06d6a0", "#8d99ae"]
SCALE = [[0, "#FBF8EF"], [0.5, "#fbb089"], [1, "#F96E2A"]]
MAP_SCALE = [[0, "#f9a273"], [0.5, "#e0561a"], [1, "#7a2606"]]  # darker than SCALE so bubbles stand out on the blue land
DIVERGING = [[0, "#3f86a8"], [0.5, "#FBF8EF"], [1, ORANGE]]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MODEL_COLORS = {"SARIMA": ORANGE, "XGBoost": CYAN, "SARIMA + XGBoost": GREEN}


def style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white", paper_bgcolor=PANEL, plot_bgcolor=PANEL, colorway=WAY, height=height,
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
                    colorscale=MAP_SCALE, showscale=True, opacity=0.9, line=dict(width=0.6, color="#1f3340"),
                    colorbar=dict(title=metric, thickness=10, len=0.6))))
    fig.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor="#9cc3d6",
                    bgcolor=PANEL, showlakes=False, showsubunits=True, subunitcolor="#4f86a3")
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
    train = y.iloc[:n_train]  # the whole training period, so the 70/30 proportions are visible
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.to_numpy(), name="Train (actual)",
                             line=dict(color=GREY, width=1)))
    fig.add_trace(go.Scatter(x=p.index, y=p["Actual"], name="Actual (test)", line=dict(color=TEXT, width=1.4)))
    for m, color in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(x=p.index, y=p[m], name=m + " (one-step)" if m == "SARIMA" else m,
                                 line=dict(color=color, width=1.6)))
    fig.add_trace(go.Scatter(x=p.index, y=p["SARIMA (multi-step)"], name="SARIMA (multi-step)", visible="legendonly",
                             line=dict(color=ORANGE, dash="dot")))
    fig.add_trace(go.Scatter(x=p.index, y=p["Seasonal naive"], name="Seasonal naive", visible="legendonly",
                             line=dict(color=PURPLE, dash="dot")))
    # shade the two regions and label them with their real sizes
    t0, ts, t1 = pd.Timestamp(y.index[0]), pd.Timestamp(r["split_date"]), pd.Timestamp(y.index[-1])
    n = len(y)
    for a, b, fill, label in (
        (t0, ts, "rgba(120,179,206,0.12)", f"TRAIN · {n_train:,} days ({n_train / n:.0%})"),
        (ts, t1, "rgba(249,110,42,0.08)", f"TEST · {n - n_train:,} days ({(n - n_train) / n:.0%})"),
    ):
        fig.add_shape(type="rect", x0=_iso(a), x1=_iso(b), y0=0, y1=1, yref="paper", fillcolor=fill,
                      line_width=0, layer="below")
        fig.add_annotation(x=_iso(a + (b - a) / 2), y=0.98, yref="paper", text=label, showarrow=False,
                           yanchor="top", font=dict(color=GREY if fill.startswith("rgba(120") else ORANGE, size=11))
    fig.add_shape(type="line", x0=_iso(ts), x1=_iso(ts), y0=0, y1=1, yref="paper",
                  line=dict(color=YELLOW, dash="dash", width=1))
    fig.update_layout(yaxis_title="Daily sales", hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    style(fig, title="Train (70%) and held-out test (30%): actual vs predicted", height=520)
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
        fill="toself", fillcolor="rgba(249,110,42,0.15)", line=dict(width=0), hoverinfo="skip",
        name="SARIMA 95% interval"))
    for m, color in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(x=fut.index, y=fut[m], name=m, mode="lines+markers", line=dict(color=color, width=2)))
    fig.update_layout(yaxis_title="Daily sales", hovermode="x unified")
    return style(fig, title=f"Next {len(fut)} days", height=400)


NS_FILL = "rgba(120,179,206,0.22)"  # shaded "not significant" band (|r| < 1.96/sqrt(n))


def _acf_bars(fig, vals, n, m, row, col, nlags):
    """Bars for lags 1..nlags with the 95% non-significance region shaded; seasonal lags are yellow."""
    band = 1.96 / np.sqrt(n)
    lags = np.arange(1, nlags + 1)
    v = vals[1:nlags + 1]
    seasonal = (lags % m == 0) if (m and m > 1) else np.zeros(len(lags), dtype=bool)
    colors = [YELLOW if sf else (ORANGE if abs(x) > band else GREY) for x, sf in zip(v, seasonal)]
    fig.add_hrect(y0=-band, y1=band, fillcolor=NS_FILL, line_width=0, layer="below", row=row, col=col)
    for sgn in (1, -1):
        fig.add_hline(y=sgn * band, line=dict(color=CYAN, dash="dot", width=1), row=row, col=col)
    fig.add_trace(go.Bar(x=lags, y=v, marker_color=colors, showlegend=False, width=0.35,
                         hovertemplate="lag %{x}: %{y:.3f}<extra></extra>"), row=row, col=col)
    fig.update_xaxes(title_text="lag (days)", row=row, col=col)
    fig.update_yaxes(range=[-1, 1] if np.abs(v).max() > 0.5 else [-0.6, 0.6], row=row, col=col)


def acf_pacf_fig(x, d, D, m, nlags=40):
    import models  # local import: figures.py otherwise has no model dependency

    n = len(x)
    nlags = int(max(5, min(nlags, n // 3)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF  → suggests q (and Q)", "PACF  → suggests p (and P)"),
                        horizontal_spacing=0.08)
    _acf_bars(fig, models.acf(x, nlags), n, m, 1, 1, nlags)
    _acf_bars(fig, models.pacf(x, nlags), n, m, 1, 2, nlags)
    desc = f"seasonal diff (lag {m}) ×{int(bool(D))}, regular diff ×{int(d or 0)}, n = {n:,} days"
    return style(fig, title=f"Training series after pre-processing · {desc} · shaded = not significant", height=360)


def rolling_fig(z, window):
    """Rolling mean (top) and rolling standard deviation (bottom) of the pre-processed training series."""
    window = int(max(2, min(window, len(z) // 2)))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f"Rolling mean ({window} days)", f"Rolling standard deviation ({window} days)"))
    fig.add_trace(go.Scatter(x=z.index, y=z.to_numpy(), name="Series", line=dict(color=GREY, width=0.8), opacity=0.6),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=z.index, y=z.rolling(window).mean(), name="Rolling mean", line=dict(color=ORANGE, width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=z.index, y=z.rolling(window).std(), name="Rolling std", line=dict(color=CYAN, width=2)),
                  row=2, col=1)
    return style(fig, title="Does the level or the spread drift over time?", height=480)


def diff_fig(z, d, D, m):
    """Original (pre-processed) series above the same series after the chosen differencing."""
    import models

    x = models.difference(z.to_numpy(), d, D, m)
    idx = z.index[len(z) - len(x):]
    label = f"seasonal (lag {m}) ×{int(bool(D))}, regular ×{int(d or 0)}"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Original (after log / outlier removal)", f"Differenced: {label}"))
    fig.add_trace(go.Scatter(x=z.index, y=z.to_numpy(), name="Original", line=dict(color=ORANGE, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=x, name="Differenced", line=dict(color=CYAN, width=1)), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=GREY, dash="dot", width=1), row=2, col=1)
    return style(fig, title="Has differencing removed the trend / weekly pattern?", height=480)


def periodogram_fig(per, power, peaks, n):
    fig = go.Figure(go.Scatter(x=per, y=power, mode="lines", line=dict(color=ORANGE, width=1.4), name="Power",
                               hovertemplate="period %{x:.1f} days<extra></extra>"))
    for p in (7, 365.25):
        if p < n:
            fig.add_vline(x=p, line=dict(color=GREY, dash="dot", width=1))
    ymax = float(np.max(power)) or 1.0
    for p in peaks:
        i = int(np.argmin(np.abs(per - p)))
        fig.add_annotation(x=np.log10(p), y=float(power[i]), text=f"{p:.1f} d", showarrow=True, arrowhead=0, ay=-24,
                           font=dict(size=11, color=TEXT))
    fig.update_xaxes(type="log", title_text="period (days, log scale)", range=[np.log10(2), np.log10(max(n, 10))])
    fig.update_yaxes(title_text="power", range=[0, ymax * 1.15])
    return style(fig, title="Periodogram: which cycle lengths carry the most power (dotted: 7 and 365 days)", height=380)


def ic_heatmap_fig(res):
    """AIC and BIC over p (rows) and q (columns) for the chosen d, D, P, Q, m. The best cell is outlined."""
    g, pmax, qmax = res["grid"], res["pmax"], res["qmax"]
    P, D, Q, m = res["seasonal"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AIC (lower is better)", "BIC (lower is better)"),
                        horizontal_spacing=0.1)
    for col, crit in ((1, "AIC"), (2, "BIC")):
        z = np.full((pmax + 1, qmax + 1), np.nan)
        for _, row in g.iterrows():
            z[int(row["p"]), int(row["q"])] = row[crit]
        fig.add_trace(go.Heatmap(z=z, x=list(range(qmax + 1)), y=list(range(pmax + 1)), showscale=False,
                                 colorscale=[[0, "#2f6f90"], [0.5, "#C9E6F0"], [1, "#FBF8EF"]],
                                 texttemplate="%{z:.0f}", hovertemplate="p=%{y}, q=%{x}: %{z:.1f}<extra></extra>",
                                 xgap=2, ygap=2), row=1, col=col)
        if np.isfinite(z).any():
            bp, bq = np.unravel_index(np.nanargmin(z), z.shape)
            fig.add_shape(type="rect", x0=bq - 0.5, x1=bq + 0.5, y0=bp - 0.5, y1=bp + 0.5,
                          line=dict(color=ORANGE, width=3), row=1, col=col)
        fig.update_xaxes(title_text="q (MA order)", dtick=1, row=1, col=col)
        fig.update_yaxes(title_text="p (AR order)", dtick=1, autorange="reversed", row=1, col=col)
    return style(fig, title=f"SARIMA(p, {res['d']}, q)({P}, {D}, {Q}, {m}) · orange outline = best", height=400)


def resid_time_fig(resid):
    sd = float(resid.std()) or 1.0
    fig = go.Figure(go.Scatter(x=resid.index, y=resid.to_numpy(), mode="lines+markers", name="Residual",
                               line=dict(color=CYAN, width=0.8), marker=dict(size=3, color=CYAN)))
    for k in (2, -2):
        fig.add_hline(y=k * sd, line=dict(color=RED, dash="dot", width=1))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.update_layout(yaxis_title="standardised residual")
    return style(fig, title="Residuals over time (should look like noise around 0; dotted = ±2 SD)", height=340)


def resid_acf_fig(resid, m):
    import models

    r = resid.dropna().to_numpy()
    n = len(r)
    nlags = int(max(5, min(40, n // 3)))
    fig = make_subplots(rows=1, cols=1)
    _acf_bars(fig, models.acf(r, nlags), n, m, 1, 1, nlags)
    return style(fig, title="Residual ACF (bars inside the shaded band = no autocorrelation left)", height=340)


def resid_dist_fig(resid):
    r = resid.dropna().to_numpy()
    mu, sd = float(r.mean()), float(r.std()) or 1.0
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram vs normal", "Q–Q plot"), horizontal_spacing=0.1)
    fig.add_trace(go.Histogram(x=r, nbinsx=40, histnorm="probability density", marker_color=ORANGE, opacity=0.8,
                               showlegend=False), row=1, col=1)
    xs = np.linspace(r.min(), r.max(), 200)
    fig.add_trace(go.Scatter(x=xs, y=np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), mode="lines",
                             name="Normal", line=dict(color=CYAN, width=2), showlegend=False), row=1, col=1)
    (tq, sq), (slope, icpt, _r) = stats.probplot(r, dist="norm")
    fig.add_trace(go.Scatter(x=tq, y=sq, mode="markers", marker=dict(color=ORANGE, size=4, opacity=0.7),
                             showlegend=False, hovertemplate="theoretical %{x:.2f}, sample %{y:.2f}<extra></extra>"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=[tq.min(), tq.max()], y=[slope * tq.min() + icpt, slope * tq.max() + icpt], mode="lines",
                             line=dict(color=CYAN, dash="dash"), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="standardised residual", row=1, col=1)
    fig.update_xaxes(title_text="theoretical quantile", row=1, col=2)
    fig.update_yaxes(title_text="sample quantile", row=1, col=2)
    return style(fig, title="Are the residuals roughly normal? (tails bending away from the line = heavy tails)", height=360)


def horizon_fig(hz):
    fig = go.Figure()
    colors = {"SARIMA": ORANGE, "XGBoost": CYAN, "Seasonal naive": GREY}
    for c in hz.columns:
        fig.add_trace(go.Scatter(x=hz.index, y=hz[c], mode="lines+markers", name=c,
                                 line=dict(color=colors.get(c, GREEN), width=2, dash="dash" if c == "Seasonal naive" else "solid"),
                                 marker=dict(size=8)))
    fig.update_xaxes(title_text="forecast horizon (days ahead)", dtick=1)
    fig.update_yaxes(title_text="RMSE (sales)")
    return style(fig, title=f"Forecast error by horizon ({hz.attrs.get('origins', '?')} rolling origins in the test period)",
                 height=380)


def k_cv_fig(cv):
    summ, per = cv["summary"], cv["per_fold"]
    best = cv["best"]
    se = summ["SD"] / np.sqrt(summ["folds"].clip(lower=1))
    fig = go.Figure()
    for _, g in per.groupby("fold"):  # one faint line per fold: shows whether the ranking of K is consistent
        fig.add_trace(go.Scatter(x=g["K"], y=g["RMSE"], mode="lines", line=dict(color=CYAN, width=1), opacity=0.35,
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=summ["K"], y=summ["RMSE"], mode="lines+markers", name="Mean RMSE (5-day, CV)",
                             line=dict(color=ORANGE, width=2.5), marker=dict(size=8, color=ORANGE),
                             error_y=dict(type="data", array=se, color=GREY, thickness=1, width=4)))
    b = summ[summ["K"] == best].iloc[0]
    fig.add_trace(go.Scatter(x=[best], y=[b["RMSE"]], mode="markers", name=f"Lowest: K = {best}",
                             marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=TEXT, width=2))))
    base = summ[summ["K"] == 0]
    if len(base):
        fig.add_hline(y=float(base["RMSE"].iloc[0]), line=dict(color=GREY, dash="dash", width=1),
                      annotation_text="K = 0 (no yearly terms)", annotation_position="top right")
    fig.update_xaxes(title_text="K (Fourier harmonic pairs)", dtick=1)
    fig.update_yaxes(title_text="RMSE (sales)")
    return style(fig, title="Forecast validation error vs K", height=400)

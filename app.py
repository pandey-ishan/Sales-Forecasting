"""Sales forecasting studio, built only with Plotly Dash (no extra UI framework, no custom JS).

Tabs follow the analysis order:
  1 Data view -> 2 Univariate -> 3 Bivariate -> 4 EDA -> 5 Forecast (SARIMA, XGBoost, 5-day)

Run:   python app.py        (http://127.0.0.1:8050)
Prod:  gunicorn app:server  (the model run takes a while, so use a long-running host, not serverless)
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

import data_utils as du
import figures as fg
import geo
import models

app = Dash(__name__, title="Sales Forecast Studio", suppress_callback_exceptions=True)
server = app.server


# --------------------------------------------------------------------------- #
# Small layout helpers
# --------------------------------------------------------------------------- #
TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={"backgroundColor": "#1d1d1d", "color": "#ff8c00", "fontWeight": "600", "border": "1px solid #2b2b2b"},
    style_cell={"backgroundColor": "#151515", "color": "#e8e8e8", "border": "1px solid #2b2b2b", "padding": "6px 10px",
                "textAlign": "left", "fontSize": "13px", "maxWidth": "260px", "overflow": "hidden",
                "textOverflow": "ellipsis"},
)
# No "responsive": True here. That makes a graph fill 100% of its parent's height, and a card has no fixed
# height, so the plot collapsed to nothing and drew over the next card. Each figure sets its own height.
CFG = {"displaylogo": False}


def graph(gid: str, height: str | None = None) -> dcc.Graph:
    return dcc.Graph(id=gid, config=CFG, style={"height": height} if height else {})


def card(title: str, *children, note=None, cls: str = "") -> html.Div:
    kids = [html.H3(title)]
    if note:
        kids.append(html.P(note, className="note"))
    return html.Div(className=f"card {cls}".strip(), children=kids + list(children))


def field(label: str, comp) -> html.Div:
    return html.Div(className="field", children=[html.Label(label), comp])


def dd(cid, options, value, **kw):
    return dcc.Dropdown(id=cid, options=options, value=value, clearable=False, **kw)


def num(cid, value, lo=0, hi=5):
    return dcc.Input(id=cid, type="number", value=value, min=lo, max=hi, step=1, debounce=True, className="num")


def opts(values):
    return [{"label": v, "value": v} for v in values]


CAT_OPTS = opts(du.CAT_COLS)


def kpi(label, value):
    return html.Div(className="kpi", children=[html.Div(value, className="kpi-v"), html.Div(label, className="kpi-l")])


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_data():
    return dcc.Tab(label="1 · Data", value="data", className="tab", selected_className="tab-sel", children=html.Div(className="pane", children=[
        html.Div(id="kpis", className="kpis"),
        html.Div(id="data-notes", className="notes"),
        card("First rows", dash_table.DataTable(id="data-table", page_size=12, sort_action="native", **TABLE_STYLE)),
        card("Column profile", dash_table.DataTable(
            id="profile-table", page_size=30, columns=[{"name": c, "id": c} for c in
                                                       ["Column", "Type", "Non-null", "Missing %", "Unique", "Example"]],
            **TABLE_STYLE)),
    ]))


def tab_uni():
    return dcc.Tab(label="2 · Univariate", value="uni", className="tab", selected_className="tab-sel", children=html.Div(className="pane", children=[
        html.Div(className="controls", children=[
            field("Variable", dd("uni-var", [
                {"label": "Sales (order line)", "value": "line"},
                {"label": "Daily sales (total per day)", "value": "daily"},
                {"label": "Ship days (ship − order date)", "value": "ship"}], "line")),
            field("Histogram bins", dcc.Slider(id="uni-bins", min=10, max=100, step=5, value=40, marks=None,
                                               tooltip={"placement": "bottom"})),
            field("z-score threshold", dcc.Slider(id="uni-thr", min=2, max=5, step=0.5, value=3,
                                                  marks={2: "2", 3: "3", 4: "4", 5: "5"})),
            field("Scale", dcc.Checklist(id="uni-log", options=[{"label": " log10 (tames right skew)", "value": "log"}],
                                         value=[])),
        ]),
        html.Div(className="grid2", children=[
            card("Histogram", graph("uni-hist")),
            card("Box plot (quartiles, whiskers, outliers)", graph("uni-box")),
        ]),
        card("z-scores", graph("uni-z"), html.Div(id="uni-ztext", className="callout")),
        html.Div(className="grid2", children=[
            card("Summary statistics", dash_table.DataTable(
                id="uni-stats", columns=[{"name": "Statistic", "id": "stat"}, {"name": "Value", "id": "value"}],
                **TABLE_STYLE)),
            card("Candlestick of daily sales",
                 field("Candle period", dd("candle-freq", opts(["Week", "Month", "Quarter"]), "Month")),
                 graph("uni-candle"),
                 note="Open/close = first/last day of the period, high/low = best/worst day. "
                      "Orange candle: period closed above its open; blue: below."),
        ]),
        card("Categorical frequencies", html.Div(className="controls", children=[
            field("Variable", dd("cat-var", CAT_OPTS, "Segment")),
            field("Show top", dcc.Slider(id="cat-top", min=5, max=20, step=1, value=10, marks=None,
                                         tooltip={"placement": "bottom"})),
        ]), graph("cat-fig")),
    ]))


def tab_biv():
    return dcc.Tab(label="3 · Bivariate", value="biv", className="tab", selected_className="tab-sel", children=html.Div(className="pane", children=[
        html.Div(className="grid2", children=[
            card("Correlation heatmap (daily level)",
                 field("Method", dcc.RadioItems(id="corr-method", options=[
                     {"label": " Pearson", "value": "pearson"}, {"label": " Spearman", "value": "spearman"}],
                     value="pearson", inline=True)),
                 graph("biv-corr"),
                 note="Days without orders count as 0. Lag 1 / Lag 7 are yesterday's / last week's sales: "
                      "their correlation hints at how much a model can learn from history."),
            card("Sales by two categories",
                 html.Div(className="controls tight", children=[
                     field("Rows", dd("piv-row", CAT_OPTS, "Category")),
                     field("Columns", dd("piv-col", CAT_OPTS, "Region")),
                     field("Aggregate", dd("piv-agg", opts(["Sum", "Mean", "Count"]), "Sum")),
                 ]),
                 graph("biv-pivot")),
        ]),
        html.Div(className="grid2", children=[
            card("Sales distribution by group",
                 html.Div(className="controls tight", children=[
                     field("Group by", dd("box-by", CAT_OPTS, "Segment")),
                     field("Plot", dd("box-kind", opts(["box", "violin"]), "box")),
                     field("Scale", dcc.Checklist(id="box-log", options=[{"label": " log y", "value": "log"}],
                                                  value=["log"])),
                 ]),
                 graph("biv-box")),
            card("Daily sales vs another daily variable",
                 field("X variable", dd("sc-x", opts(["Lines", "Orders", "Customers", "Avg Line Sales", "Lag 1",
                                                       "Lag 7", "Day of Week", "Month"]), "Lines")),
                 graph("biv-scatter")),
        ]),
        card("Average daily sales: month × weekday", graph("biv-cal")),
    ]))


def tab_eda():
    return dcc.Tab(label="4 · EDA", value="eda", className="tab", selected_className="tab-sel", children=html.Div(className="pane", children=[
        card("Where sales come from (bubble map)",
             html.Div(className="controls", children=[
                 field("Level", dd("geo-level", opts(["State", "City"]), "State")),
                 field("Bubble size", dd("geo-metric", opts(list(fg.METRICS)), "Total sales")),
                 field("Category", dd("geo-cat", opts(["All"]), "All")),
             ]),
             html.Div(className="grid-map", children=[dcc.Loading(graph("geo-map")), graph("geo-bar")]),
             html.Div(id="geo-note", className="callout")),
        card("Sales per category: click a slice to drill down, click the centre / breadcrumb to go back",
             html.Div(className="controls", children=[
                 field("Chart", dcc.RadioItems(id="drill-kind", options=opts(["Sunburst", "Treemap"]),
                                               value="Sunburst", inline=True)),
                 field("Products kept per sub-category", dcc.Slider(id="drill-top", min=3, max=15, step=1, value=8,
                                                                  marks=None, tooltip={"placement": "bottom"})),
             ]),
             graph("drill-fig")),
        card("Sales over the years",
             html.Div(className="controls", children=[
                 field("Granularity", dcc.RadioItems(id="trend-grain", options=opts(["Year", "Quarter", "Month"]),
                                                     value="Month", inline=True)),
                 field("Lines", dcc.RadioItems(id="trend-split", options=[
                     {"label": " Total", "value": "total"}, {"label": " One line per category", "value": "split"}],
                     value="split", inline=True)),
                 field("Drill into category", dd("trend-drill", opts(["All"]), "All")),
             ]),
             graph("trend-fig"),
             note="Click a category's line to drill into its sub-categories; pick “All” to come back. "
                  "The first and last periods may be partial."),
        card("Seasonality by year", graph("yoy-fig")),
    ]))


def tab_model():
    return dcc.Tab(label="5 · Forecast", value="model", className="tab", selected_className="tab-sel", children=html.Div(className="pane", children=[
        card("SARIMA → can XGBoost improve it?",
             html.P("Daily sales are split chronologically: the first 70% of days train the models, the last 30% test "
                    "them. SARIMA parameters are estimated on the 70% only. XGBoost is tried alone and as a hybrid "
                    "that corrects SARIMA's residuals.", className="note"),
             html.Div(className="controls", children=[
                 field("Forecast target", dd("fc-cat", opts(["All"]), "All")),
                 field("SARIMA order (p, d, q)", html.Div(className="row", children=[
                     num("sar-p", 1), num("sar-d", 1), num("sar-q", 1)])),
                 field("Seasonal (P, D, Q, m)", html.Div(className="row", children=[
                     num("sar-P", 1), num("sar-D", 1), num("sar-Q", 1), num("sar-m", 7, 0, 366)])),
                 html.Button("Train & forecast", id="fc-run", n_clicks=0, className="btn"),
             ])),
        dcc.Loading(type="dot", color="#ff8c00", children=html.Div(id="fc-results", children=[
            html.Div(id="fc-summary", className="callout"),
            html.Div(id="fc-info", className="note pad"),
            html.Div(className="grid2", children=[
                card("Model comparison (held-out 30%)", dash_table.DataTable(
                    id="fc-metrics", columns=[{"name": c, "id": c} for c in ["Model", "Eval", "RMSE", "MAE", "WAPE %", "R²"]],
                    **TABLE_STYLE),
                    note="one-step = each test day predicted from the actual days before it (comparable across "
                         "models). multi-step = SARIMA forecasting all test days blind."),
                card("RMSE", graph("fc-metrics-bar")),
            ]),
            card("Model vs actual", graph("fc-main"),
                 note="Click legend entries to show the dotted multi-step SARIMA and the seasonal-naive baseline."),
            html.Div(className="grid2", children=[card("Predicted vs actual", graph("fc-parity")),
                                                  card("Feature importance", graph("fc-imp"))]),
            html.Div(className="grid2", children=[
                card("Next 5 days", graph("fc-future"),
                     note="The 5 calendar days after the last date in your data."),
                card("Forecast values", dash_table.DataTable(
                    id="fc-future-table", columns=[{"name": c, "id": c} for c in
                                                   ["Date", "Day", "SARIMA", "SARIMA 95% low", "SARIMA 95% high",
                                                    "XGBoost", "SARIMA + XGBoost"]], **TABLE_STYLE)),
            ]),
        ])),
    ]))


def status_text(key):
    if not key:
        return "No data loaded. Upload your CSV/XLSX, drop it in data/superstore.csv, or try the demo."
    try:
        meta = du.get_meta(key)
    except Exception as exc:  # noqa: BLE001
        return f"Could not load data: {exc}"
    tag = "DEMO · synthetic data, not real sales · " if key == "demo" else ""
    return (f"{tag}{meta['rows']:,} rows · {meta['date_min']:%d %b %Y} → {meta['date_max']:%d %b %Y} · "
            f"dates read as {meta['date_format']}")


def serve_layout():
    key = "default" if du.default_available() else None
    return html.Div(className="app", children=[
        dcc.Store(id="data-key", data=key),
        html.Div(className="header", children=[
            html.Div([html.H1("Sales Forecast Studio"), html.Div(id="status", className="status", children=status_text(key))]),
            html.Div(className="load", children=[
                dcc.Upload(id="upload", accept=".csv,.xlsx", multiple=False, className="upload",
                           children=html.Div(["Drop a CSV / XLSX or ", html.Span("browse", className="link")])),
                html.Button("Use demo data", id="demo-btn", n_clicks=0, className="btn ghost"),
            ]),
        ]),
        dcc.Tabs(id="tabs", value="data", className="tabs", children=[
            tab_data(), tab_uni(), tab_biv(), tab_eda(), tab_model()]),
    ])


app.layout = serve_layout


# --------------------------------------------------------------------------- #
# Data access helpers
# --------------------------------------------------------------------------- #
def df_or_none(key):
    if not key:
        return None
    try:
        return du.get_df(key)
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=4)
def city_coords_for(key):
    return geo.resolve_city_coords(du.get_df(key))


def blank(msg="Load data to begin"):
    return fg.empty_fig(msg)


def pick_category(df, cat):
    if cat in (None, "All") or "Category" not in df.columns:
        return df
    return df[df["Category"] == cat]


# --------------------------------------------------------------------------- #
# Loading + control population
# --------------------------------------------------------------------------- #
@app.callback(Output("data-key", "data"), Output("status", "children"),
              Input("upload", "contents"), Input("demo-btn", "n_clicks"),
              State("upload", "filename"), prevent_initial_call=True)
def load_data(contents, _demo_clicks, filename):
    try:
        if ctx.triggered_id == "demo-btn":
            key = "demo"
            du.get_df(key)
        else:
            key = du.save_upload(contents, filename or "")
    except Exception as exc:  # noqa: BLE001
        return no_update, f"Could not load that file: {exc}"
    return key, status_text(key)


@app.callback(Output("geo-cat", "options"), Output("trend-drill", "options"), Output("fc-cat", "options"),
              Input("data-key", "data"))
def populate_categories(key):
    df = df_or_none(key)
    cats = sorted(df["Category"].unique()) if df is not None and "Category" in df.columns else []
    o = opts(["All"] + cats)
    return o, o, o


# --------------------------------------------------------------------------- #
# 1 · Data
# --------------------------------------------------------------------------- #
@app.callback(Output("kpis", "children"), Output("data-notes", "children"), Output("data-table", "data"),
              Output("data-table", "columns"), Output("profile-table", "data"), Input("data-key", "data"))
def render_data(key):
    df = df_or_none(key)
    if df is None:
        return html.Div("Nothing loaded yet.", className="note"), "", [], [], []
    meta = du.get_meta(key)
    k = [kpi("Rows", f"{meta['rows']:,}"),
         kpi("Date range", f"{meta['date_min']:%b %Y} – {meta['date_max']:%b %Y}"),
         kpi("Total sales", f"{df['Sales'].sum():,.0f}"),
         kpi("Avg order line", f"{df['Sales'].mean():,.2f}")]
    for label, col in (("Orders", "Order ID"), ("Customers", "Customer ID"), ("Products", "Product ID"), ("Cities", "City")):
        if col in df.columns:
            k.append(kpi(label, f"{df[col].nunique():,}"))

    notes = [html.Li(f"Dates were read as {meta['date_format']} (detected automatically).")]
    if meta["dropped"]:
        notes.append(html.Li(f"{meta['dropped']:,} row(s) dropped because the date or sales value was unreadable."))
    if meta["missing_postal"]:
        notes.append(html.Li(f"{meta['missing_postal']:,} row(s) have no postal code; the city map falls back to the city name."))
    if key == "demo":
        notes.insert(0, html.Li("This is SYNTHETIC demo data generated by the app. Upload your own file for real results."))

    shown = [c for c in du.CANONICAL if c in df.columns] + [c for c in ("Ship Days",) if c in df.columns]
    view = df[shown].head(500).copy()
    for c in ("Order Date", "Ship Date"):
        if c in view.columns:
            view[c] = view[c].dt.strftime("%Y-%m-%d")
    view = view.astype(object).where(view.notna(), "")
    return k, html.Ul(notes), view.to_dict("records"), [{"name": c, "id": c} for c in view.columns], du.profile(df, shown)


# --------------------------------------------------------------------------- #
# 2 · Univariate
# --------------------------------------------------------------------------- #
@app.callback(Output("uni-hist", "figure"), Output("uni-box", "figure"), Output("uni-z", "figure"),
              Output("uni-stats", "data"), Output("uni-ztext", "children"),
              Input("data-key", "data"), Input("uni-var", "value"), Input("uni-bins", "value"),
              Input("uni-log", "value"), Input("uni-thr", "value"))
def update_uni(key, var, bins, log, thr):
    df = df_or_none(key)
    if df is None:
        b = blank()
        return b, b, b, [], ""
    log = "log" in (log or [])
    view, label = du.numeric_view(df, var)
    zfig, n_out, n_tot = fg.zscore_fig(view, label, log, thr)
    share = f"{n_out / n_tot:.1%}" if n_tot else "–"
    text = (f"{n_out:,} of {n_tot:,} values ({share}) have |z| > {thr:g}. "
            "Sales are usually right-skewed, so large orders flag as outliers; tick log10 to see the picture "
            "after taming the skew. These are unusual values, not necessarily errors.")
    return (fg.hist_fig(view, label, bins, log), fg.box_fig(view, label, log), zfig, fg.stats_records(view), text)


@app.callback(Output("uni-candle", "figure"), Input("data-key", "data"), Input("candle-freq", "value"))
def update_candle(key, freq):
    df = df_or_none(key)
    return blank() if df is None else fg.candle_fig(df, freq)


@app.callback(Output("cat-fig", "figure"), Input("data-key", "data"), Input("cat-var", "value"), Input("cat-top", "value"))
def update_cat(key, col, top):
    df = df_or_none(key)
    return blank() if df is None else fg.cat_count_fig(df, col, top)


# --------------------------------------------------------------------------- #
# 3 · Bivariate
# --------------------------------------------------------------------------- #
@app.callback(Output("biv-corr", "figure"), Input("data-key", "data"), Input("corr-method", "value"))
def update_corr(key, method):
    df = df_or_none(key)
    return blank() if df is None else fg.corr_fig(df, method)


@app.callback(Output("biv-pivot", "figure"), Input("data-key", "data"), Input("piv-row", "value"),
              Input("piv-col", "value"), Input("piv-agg", "value"))
def update_pivot(key, row, col, agg):
    df = df_or_none(key)
    return blank() if df is None else fg.pivot_fig(df, row, col, agg)


@app.callback(Output("biv-box", "figure"), Input("data-key", "data"), Input("box-by", "value"),
              Input("box-kind", "value"), Input("box-log", "value"))
def update_box(key, by, kind, log):
    df = df_or_none(key)
    return blank() if df is None else fg.box_by_fig(df, by, kind, "log" in (log or []))


@app.callback(Output("biv-scatter", "figure"), Input("data-key", "data"), Input("sc-x", "value"))
def update_scatter(key, x):
    df = df_or_none(key)
    return blank() if df is None else fg.scatter_fig(df, x)


@app.callback(Output("biv-cal", "figure"), Input("data-key", "data"))
def update_cal(key):
    df = df_or_none(key)
    return blank() if df is None else fg.calendar_fig(df)


# --------------------------------------------------------------------------- #
# 4 · EDA
# --------------------------------------------------------------------------- #
@app.callback(Output("geo-map", "figure"), Output("geo-bar", "figure"), Output("geo-note", "children"),
              Input("data-key", "data"), Input("geo-level", "value"), Input("geo-metric", "value"),
              Input("geo-cat", "value"))
def update_geo(key, level, metric, cat):
    df = df_or_none(key)
    if df is None:
        return blank(), blank(), ""
    coords = city_coords_for(key) if level == "City" else None
    return fg.geo_figs(pick_category(df, cat), level, metric, coords)


@app.callback(Output("drill-fig", "figure"), Input("data-key", "data"), Input("drill-kind", "value"),
              Input("drill-top", "value"))
def update_drill(key, kind, top):
    df = df_or_none(key)
    return blank() if df is None else fg.drill_fig(df, kind, top)


@app.callback(Output("trend-fig", "figure"), Input("data-key", "data"), Input("trend-grain", "value"),
              Input("trend-split", "value"), Input("trend-drill", "value"))
def update_trend(key, grain, split, drill):
    df = df_or_none(key)
    return blank() if df is None else fg.trend_fig(df, grain, split, drill)


@app.callback(Output("trend-drill", "value"), Input("trend-fig", "clickData"), State("trend-drill", "value"),
              prevent_initial_call=True)
def click_to_drill(click, current):
    """Clicking a category line selects that category (sub-category lines have nothing deeper)."""
    if not click or current not in (None, "All"):
        return no_update
    cd = click["points"][0].get("customdata")
    name = cd[0] if isinstance(cd, (list, tuple)) else cd
    return name if isinstance(name, str) else no_update


@app.callback(Output("yoy-fig", "figure"), Input("data-key", "data"), Input("trend-drill", "value"))
def update_yoy(key, drill):
    df = df_or_none(key)
    return blank() if df is None else fg.yoy_fig(df, drill)


# --------------------------------------------------------------------------- #
# 5 · Forecast
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=6)
def cached_run(key, category, order, seasonal):
    y = du.daily_series(du.get_df(key), None if category in (None, "All") else category)
    return models.run_pipeline(y, order, seasonal)


def _i(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


FC_OUTPUTS = [Output("fc-summary", "children"), Output("fc-info", "children"), Output("fc-metrics", "data"),
              Output("fc-metrics", "style_data_conditional"), Output("fc-metrics-bar", "figure"),
              Output("fc-main", "figure"), Output("fc-parity", "figure"), Output("fc-imp", "figure"),
              Output("fc-future", "figure"), Output("fc-future-table", "data")]


@app.callback(*FC_OUTPUTS, Input("fc-run", "n_clicks"), State("data-key", "data"), State("fc-cat", "value"),
              State("sar-p", "value"), State("sar-d", "value"), State("sar-q", "value"),
              State("sar-P", "value"), State("sar-D", "value"), State("sar-Q", "value"), State("sar-m", "value"),
              prevent_initial_call=True)
def run_models(_n, key, cat, p, d, q, P, D, Q, m):
    fail = lambda msg: (html.Div(msg, className="error"),) + (no_update,) * (len(FC_OUTPUTS) - 1)  # noqa: E731
    if not key:
        return fail("Load data first.")
    order = (_i(p, 1), _i(d, 1), _i(q, 1))
    seasonal = (_i(P, 1), _i(D, 1), _i(Q, 1), _i(m, 7))
    try:
        r = cached_run(key, cat, order, seasonal)
    except ModuleNotFoundError as exc:
        return fail(f"A required package is missing: {exc.name}. Run `pip install -r requirements.txt`.")
    except Exception as exc:  # noqa: BLE001  (bad orders, too little data, non-convergence ...)
        return fail(f"Could not fit the models: {exc}")

    mt = r["metrics"].set_index("Model")
    best, gain = r["best"], r["improvement"]
    rows = [html.Div([html.B(f"Best on the held-out test set: {best}"), f" (RMSE {mt.loc[best, 'RMSE']:,.0f})."])]
    if best == "SARIMA":
        rows.append(html.Div("XGBoost did not improve on SARIMA here, alone or as a hybrid. "
                             "Daily sales are noisy, so that is a legitimate outcome."))
    else:
        rows.append(html.Div(f"That is {gain:.1f}% lower RMSE than SARIMA alone "
                             f"({mt.loc['SARIMA', 'RMSE']:,.0f}), so XGBoost does help."))
    if mt.loc[best, "RMSE"] >= mt.loc["Seasonal naive", "RMSE"]:
        rows.append(html.Div("Caution: no model beat the simple seasonal-naive baseline (same weekday last week).",
                             className="warn"))
    if mt.loc[best, "R²"] < 0.2:
        rows.append(html.Div(f"R² is only {mt.loc[best, 'R²']:.2f}: most day-to-day variation is not predictable "
                             "from the history, so treat point forecasts as a rough guide and use the interval.",
                             className="warn"))

    y, n_tr = r["y"], r["n_train"]
    o, s = r["order"], r["seasonal"]
    info = (f"{len(y):,} days ({y.index[0]:%d %b %Y} → {y.index[-1]:%d %b %Y}) · train {n_tr:,} days to "
            f"{r['split_date'] - pd.Timedelta(days=1):%d %b %Y}, test {len(y) - n_tr:,} days from "
            f"{r['split_date']:%d %b %Y} · SARIMA{o}x{s[:3]}[{s[3]}] (AIC {r['aic']:,.0f}) · "
            f"{r['zero_share']:.1%} of days had no sales (counted as 0)")

    table = r["metrics"].round({"RMSE": 1, "MAE": 1, "WAPE %": 1, "R²": 3}).to_dict("records")
    highlight = [{"if": {"filter_query": f'{{Model}} = "{best}"'}, "backgroundColor": "#3a2200", "fontWeight": "600"}]

    fut = r["future"].copy()
    fut.insert(0, "Day", fut.index.strftime("%a"))
    fut.insert(0, "Date", fut.index.strftime("%Y-%m-%d"))
    fut = fut.round(2)
    return (html.Div(rows), info, table, highlight, fg.metrics_bar_fig(r), fg.forecast_fig(r), fg.parity_fig(r),
            fg.importance_fig(r), fg.future_fig(r), fut.to_dict("records"))


if __name__ == "__main__":
    app.run(debug=True)

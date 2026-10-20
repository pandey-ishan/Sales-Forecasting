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
    style_header={"backgroundColor": "#C9E6F0", "color": "#1f3340", "fontWeight": "600", "border": "1px solid #78B3CE"},
    style_cell={"backgroundColor": "#ffffff", "color": "#1f3340", "border": "1px solid #C9E6F0", "padding": "6px 10px",
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
    return dcc.Input(id=cid, type="number", value=value, min=lo, max=hi, step=1, debounce=False, className="num")


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


def section(num, title, purpose, details, *children, open=False):
    """A collapsible step (native <details>: no JavaScript). `details` is the plots / tests / parameters line."""
    return html.Details(className="step", open=open, children=[
        html.Summary(children=[html.Span(str(num), className="step-n"), html.Span(title, className="step-t")]),
        html.Div(className="step-body", children=[
            html.P(purpose, className="step-purpose"),
            html.P(details, className="note"),
            *children,
        ]),
    ])


def _loading(*kids):
    return dcc.Loading(type="dot", color="#F96E2A", children=list(kids))


def tab_model():
    return dcc.Tab(label="5 · Forecast", value="model", className="tab", selected_className="tab-sel", children=html.Div(className="pane", children=[
        card("Setup: target and pre-processing",
             html.P("Daily sales are split chronologically: the first 70% of days train the models, the last 30% test "
                    "them. Log transform, outlier removal, differencing tests and the order search use the 70% only. "
                    "Work through the steps below in order; each one can be collapsed.", className="note"),
             html.Div(className="controls", children=[
                 field("Forecast target", dd("fc-cat", opts(["All"]), "All")),
                 field("Pre-processing", dcc.Checklist(
                     id="fc-prep", value=["log", "outliers"], className="radio",
                     options=[{"label": "log(1+sales)", "value": "log"},
                              {"label": "remove outliers from training", "value": "outliers"}])),
                 field("Outlier z threshold", dcc.Input(id="fc-z", type="number", value=3, min=0.5, max=10, step=0.1,
                                                        debounce=False, className="num")),
                 field("Outlier side", dd("fc-tail", [{"label": "High only (z > t)", "value": "high"},
                                                      {"label": "Both (|z| > t)", "value": "both"}], "high")),
             ])),

        section(1, "Stationarity Analysis", "Determine whether the series requires differencing.",
                "Plots / tests: rolling mean and variance, ADF test, KPSS test, differencing plots. Parameters identified: d, D.",
                html.Div(className="controls tight", children=[
                    field("Rolling window (days)", dcc.Input(id="st-win", type="number", value=28, min=7, max=365, step=1,
                                                             debounce=False, className="num")),
                    field("Regular differencing d", num("sar-d", 1, 0, 2)),
                    field("Seasonal differencing D", num("sar-D", 1, 0, 1)),
                ]),
                html.Div(id="st-note", className="callout"),
                html.Div(className="grid2", children=[
                    card("Rolling mean and standard deviation", graph("st-roll"),
                         note="A drifting mean means trend (difference it); a spread that grows with the level is what the "
                              "log transform is for."),
                    card("Original vs differenced", graph("st-diff"),
                         note="Uses the d and D above and the season length m from step 2."),
                ]),
                card("ADF and KPSS tests", dash_table.DataTable(
                    id="st-table", columns=[{"name": c, "id": c} for c in ["Series", "ADF p", "KPSS p", "Verdict"]],
                    **TABLE_STYLE),
                     note="ADF: p < 0.05 → stationary. KPSS: p ≥ 0.05 → stationary (statsmodels only reports 0.01 to 0.10). "
                          "Both must agree before you trust the verdict. Needs statsmodels."),
                open=True),

        section(2, "Model Identification", "Identify the suitable ARIMA and seasonal structures.",
                "Plots: ACF, PACF, periodogram. Parameters considered: p, q, P, Q, m.",
                html.Div(className="controls tight", children=[
                    field("AR order p", num("sar-p", 1, 0, 5)),
                    field("MA order q", num("sar-q", 1, 0, 5)),
                    field("Seasonal AR P", num("sar-P", 1, 0, 3)),
                    field("Seasonal MA Q", num("sar-Q", 1, 0, 3)),
                    field("Season length m", num("sar-m", 7, 0, 366)),
                ]),
                card("ACF & PACF: reading p, q, P, Q by eye", graph("fc-acf"),
                     note="Uses the training 70% after the settings above and the d, D, m you chose. Orange bars leave the "
                          "shaded non-significant band; yellow bars are seasonal lags (m, 2m, …). Rule of thumb: PACF cuts "
                          "off after lag p and ACF tails off → AR(p); ACF cuts off after lag q and PACF tails off → MA(q); "
                          "both tail off → try (1,1). Slowly decaying ACF → raise d (or D if the yellow bars decay slowly)."),
                card("Periodogram", graph("id-pgram"),
                     note="Peaks mark dominant cycle lengths. A peak near 7 confirms m = 7. With about 1,000 training days "
                          "a yearly cycle is resolved only coarsely (periods near 340 or 500 days), so use it as a hint and "
                          "model the yearly part with Fourier terms (step 3)."),
                ),

        section(3, "Feature Engineering", "Create additional explanatory variables where useful.",
                "Examples: Fourier terms, holidays, promotions and calendar variables.",
                html.Div(className="controls tight", children=[
                    field("Yearly Fourier pairs (K), 0 = off", dcc.Input(
                        id="fc-fourier", type="number", value=0, min=0, max=6, step=1, debounce=False, className="num")),
                ]),
                html.Div(className="notes", children=html.Ul([
                    html.Li("Fourier terms (this box): K sine/cosine pairs of the day of year, added as SARIMA regressors "
                            "and as XGBoost features. Pick K with the validation plot in step 4."),
                    html.Li("Calendar variables (always on, XGBoost): day of week, day of month, month, month start / end."),
                    html.Li("Lag and rolling features (always on, XGBoost): lags 1, 2, 3, 7, 14, 21, 28 and 7/14/28-day "
                            "rolling mean, plus a 7-day rolling std, all from days strictly before the target day."),
                    html.Li("Holidays and promotions: not in this dataset, so not used. Add a date column of events "
                            "and they can be passed in as extra regressors."),
                ]))),

        section(4, "Model Estimation and Selection", "Fit candidate SARIMA / SARIMAX models and compare their performance.",
                "Methods: maximum likelihood estimation (statsmodels), AIC, BIC and rolling time-series cross-validation.",
                html.Div(className="controls", children=[
                    field("ARIMA order", dcc.RadioItems(
                        id="fc-auto", value="auto", className="radio",
                        options=[{"label": "Auto (stepwise AIC)", "value": "auto"},
                                 {"label": "Manual (steps 1-2 values)", "value": "manual"}])),
                    field("Differencing in Auto", dcc.RadioItems(
                        id="fc-diff", value="tests", className="radio",
                        options=[{"label": "Chosen by tests", "value": "tests"},
                                 {"label": "Use my d and D", "value": "mine"}])),
                    field("Auto-search time limit (s)", dcc.Input(id="fc-time", type="number", value=120, min=10, max=1800,
                                                                   step=10, debounce=True, className="num")),
                    html.Button("Train & forecast", id="fc-run", n_clicks=0, className="btn"),
                ]),
                html.Div(id="fc-info", className="note pad"),
                _loading(card("AIC / BIC heatmap", html.Div(className="controls tight", children=[
                    field("Largest p", num("ic-p", 3, 0, 5)), field("Largest q", num("ic-q", 3, 0, 5)),
                    html.Button("Compute grid", id="ic-run", n_clicks=0, className="btn ghost"),
                ]), graph("ic-fig"), html.Div(id="ic-note", className="callout"),
                    note="Fits SARIMA(p, d, q)(P, D, Q, m) for every p and q with the d, D, P, Q, m and K chosen above, on the "
                         "training 70%. Darker blue = lower = better. AIC favours fit, BIC penalises extra terms harder; "
                         "when they disagree prefer the smaller model.")),
                _loading(card("Choose K: validation RMSE vs K", html.Div(className="controls tight", children=[
                    field("Largest K to try", dcc.Input(id="kcv-max", type="number", value=5, min=1, max=8, step=1,
                                                        debounce=False, className="num")),
                    field("CV folds", dcc.Input(id="kcv-folds", type="number", value=5, min=2, max=10, step=1,
                                                debounce=False, className="num")),
                    html.Button("Compare K", id="kcv-run", n_clicks=0, className="btn ghost"),
                ]), graph("kcv-fig"), html.Div(id="kcv-note", className="callout"),
                    note="Rolling-origin cross-validation on the training 70% only: for each K the SARIMA is refitted at "
                         "several origins and scored on the next 5 days (same origins for every K), using the manual "
                         "order and settings above. Lower is better; if two K are within the error bars, take the smaller.")),
                card("Auto-ARIMA search (lowest AIC wins)", dash_table.DataTable(
                    id="fc-search", columns=[{"name": c, "id": c} for c in ["Order", "Seasonal", "AIC"]], page_size=8,
                    **TABLE_STYLE), note="Only filled when the order is Auto. AIC is measured on the standardised training series."),
                ),

        section(5, "Residual Diagnostics", "Check whether the model has adequately captured patterns in the data.",
                "Plots / tests: residual ACF, Q–Q plot, residual histogram, Ljung–Box test. Click Train & forecast in step 4 to fill them.",
                _loading(html.Div(className="pane", children=[
                    html.Div(id="fc-lb-note", className="callout"),
                    html.Div(className="grid2", children=[
                        card("Residuals over time", graph("fc-rtime"),
                             note="One-step-ahead residuals of the training fit, in the model's own units (log scale if log is on)."),
                        card("Residual ACF", graph("fc-racf")),
                    ]),
                    card("Histogram and Q–Q plot", graph("fc-rdist")),
                    card("Ljung–Box test", dash_table.DataTable(
                        id="fc-lb", columns=[{"name": c, "id": c} for c in ["Lag", "Q", "p-value", "Verdict"]], **TABLE_STYLE),
                         note="H0: no autocorrelation left up to that lag. p < 0.05 means the model missed structure. "
                              "Degrees of freedom are reduced by p + q + P + Q."),
                ]))),

        section(6, "Forecast Validation", "Test forecasting performance on observations not used for training.",
                "Metrics: MAE, RMSE, WAPE, MASE. Plots: actual vs predicted, forecast error by horizon.",
                _loading(html.Div(className="pane", children=[
                    html.Div(id="fc-summary", className="callout"),
                    html.Div(className="grid2", children=[
                        card("Model comparison (held-out 30%)", dash_table.DataTable(
                            id="fc-metrics", columns=[{"name": c, "id": c} for c in
                                                      ["Model", "Eval", "RMSE", "MAE", "WAPE %", "MASE", "R²"]], **TABLE_STYLE),
                             note="one-step = each test day predicted from the actual days before it (comparable across "
                                  "models). multi-step = SARIMA forecasting all test days blind. MASE < 1 beats the "
                                  "in-sample same-weekday-last-week error."),
                        card("RMSE", graph("fc-metrics-bar")),
                    ]),
                    card("Model vs actual", graph("fc-main"),
                         note="Predictions exist only for the test period (the right-hand 30%); the model has not seen those "
                              "days when fitted. Click a legend entry to hide/show a line; the two dotted ones "
                              "(multi-step SARIMA, seasonal naive) start hidden."),
                    card("Forecast error by horizon", graph("fc-horizon"),
                         note="True h-day-ahead forecasts from rolling origins in the test period (SARIMA parameters fixed "
                              "from training; XGBoost uses one direct model per horizon). Errors usually grow with the horizon."),
                    html.Div(className="grid2", children=[card("Predicted vs actual", graph("fc-parity")),
                                                          card("Feature importance", graph("fc-imp"))]),
                ]))),

        section(7, "Final Model Fitting and Forecasting",
                "Refit the selected model using all appropriate available historical data and generate future predictions with prediction intervals.",
                "The 5 calendar days after the last date in your data; the SARIMA line carries a 95% prediction interval.",
                _loading(html.Div(className="grid2", children=[
                    card("Next 5 days", graph("fc-future")),
                    card("Forecast values", dash_table.DataTable(
                        id="fc-future-table", columns=[{"name": c, "id": c} for c in
                                                       ["Date", "Day", "SARIMA", "SARIMA 95% low", "SARIMA 95% high",
                                                        "XGBoost", "SARIMA + XGBoost"]], **TABLE_STYLE)),
                ]))),
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
def cached_run(key, category, order, seasonal, auto, log, outlier_z, tail, time_limit, fixed_dD=None, fourier_k=0):
    y = du.daily_series(du.get_df(key), None if category in (None, "All") else category)
    return models.run_pipeline(y, order, seasonal, auto=auto, log=log, outlier_z=outlier_z, outlier_tail=tail,
                               time_limit=time_limit, fixed_dD=fixed_dD, fourier_k=fourier_k)


def _f(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _prep_series(key, cat, prep, z, tail):
    y = du.daily_series(du.get_df(key), None if cat in (None, "All") else cat)
    if len(y) < models.MIN_DAYS:
        raise ValueError(f"Need at least {models.MIN_DAYS} days of data.")
    prep = prep or []
    return models.prepared_training(y, "log" in prep, _f(z, 3.0) if "outliers" in prep else 0.0, tail or "high")


@app.callback(Output("st-roll", "figure"), Output("st-diff", "figure"), Output("st-table", "data"),
              Output("st-note", "children"),
              Input("data-key", "data"), Input("fc-cat", "value"), Input("fc-prep", "value"), Input("fc-z", "value"),
              Input("fc-tail", "value"), Input("sar-d", "value"), Input("sar-D", "value"), Input("sar-m", "value"),
              Input("st-win", "value"))
def stationarity(key, cat, prep, z, tail, d, D, m, win):
    if not key:
        return blank(), blank(), [], ""
    d, D, m = _i(d, 0), _i(D, 0), _i(m, 7)
    try:
        zs = _prep_series(key, cat, prep, z, tail)
    except Exception as exc:  # noqa: BLE001
        return blank(str(exc)), blank(str(exc)), [], ""
    rows, note = [], []
    try:
        rows = models.stationarity_table(zs, d, D, m)
        sd, sD = models.choose_differencing(zs.to_numpy(), m)
        note = [html.B(f"Tests suggest d = {sd}, D = {sD}"), f" (ADF on the series, weekly-seasonality strength for D). "
                f"You selected d = {d}, D = {D}."]
        final = rows[-1]["Verdict"]
        note.append(html.Div(f"After your differencing the series is judged: {final}.",
                             className="" if final == "stationary" else "warn"))
    except ModuleNotFoundError as exc:
        note = [html.Div(f"The ADF / KPSS tests need {exc.name}: pip install -r requirements.txt", className="warn")]
    except Exception as exc:  # noqa: BLE001
        note = [html.Div(f"Could not run the stationarity tests: {exc}", className="warn")]
    try:
        return (fg.rolling_fig(zs, _i(win, 28)), fg.diff_fig(zs, d, D, m), rows, html.Div(note))
    except Exception as exc:  # noqa: BLE001
        return blank(str(exc)), blank(str(exc)), rows, html.Div(note)


@app.callback(Output("id-pgram", "figure"), Input("data-key", "data"), Input("fc-cat", "value"),
              Input("fc-prep", "value"), Input("fc-z", "value"), Input("fc-tail", "value"))
def periodogram_cb(key, cat, prep, z, tail):
    if not key:
        return blank()
    try:
        zs = _prep_series(key, cat, prep, z, tail)
        per, pw, peaks = models.periodogram(zs.to_numpy())
        return fg.periodogram_fig(per, pw, peaks, len(zs))
    except Exception as exc:  # noqa: BLE001
        return blank(str(exc))


@lru_cache(maxsize=6)
def cached_ic(key, category, d, D, P, Q, m, pmax, qmax, fk, log, outlier_z, tail, time_limit):
    y = du.daily_series(du.get_df(key), None if category in (None, "All") else category)
    return models.ic_grid(y, d, D, P, Q, m, pmax, qmax, fk, log, outlier_z, tail, time_limit=time_limit)


@app.callback(Output("ic-fig", "figure"), Output("ic-note", "children"), Input("ic-run", "n_clicks"),
              State("data-key", "data"), State("fc-cat", "value"), State("sar-d", "value"), State("sar-D", "value"),
              State("sar-P", "value"), State("sar-Q", "value"), State("sar-m", "value"), State("ic-p", "value"),
              State("ic-q", "value"), State("fc-fourier", "value"), State("fc-prep", "value"), State("fc-z", "value"),
              State("fc-tail", "value"), State("fc-time", "value"), prevent_initial_call=True)
def run_ic(_n, key, cat, d, D, P, Q, m, pmax, qmax, fk, prep, z, tail, tlim):
    if not key:
        return blank(), html.Div("Load data first.", className="error")
    prep = prep or []
    try:
        res = cached_ic(key, cat, _i(d, 1), _i(D, 1), _i(P, 1), _i(Q, 1), _i(m, 7), max(0, min(_i(pmax, 3), 5)),
                        max(0, min(_i(qmax, 3), 5)), max(0, min(_i(fk, 0), 6)), "log" in prep,
                        _f(z, 3.0) if "outliers" in prep else 0.0, tail or "high", _f(tlim, 120.0) * 2)
    except ModuleNotFoundError as exc:
        return no_update, html.Div(f"A required package is missing: {exc.name}. Run `pip install -r requirements.txt`.",
                                   className="error")
    except Exception as exc:  # noqa: BLE001
        return no_update, html.Div(f"Could not compute the grid: {exc}", className="error")
    g = res["grid"]
    ba, bb = g.loc[g["AIC"].idxmin()], g.loc[g["BIC"].idxmin()]
    msg = [html.B(f"Lowest AIC: p = {int(ba['p'])}, q = {int(ba['q'])}"), f" ({ba['AIC']:,.0f}) · ",
           html.B(f"lowest BIC: p = {int(bb['p'])}, q = {int(bb['q'])}"), f" ({bb['BIC']:,.0f}) · {res['seconds']:.0f}s."]
    if (ba["p"], ba["q"]) != (bb["p"], bb["q"]):
        msg.append(html.Div("AIC and BIC disagree; BIC prefers the smaller model, so start there unless residuals "
                            "(step 5) show structure left.", className="warn"))
    if res["partial"]:
        msg.append(html.Div("Time limit reached: the grid is incomplete. Lower the largest p / q or raise the limit.",
                            className="warn"))
    return fg.ic_heatmap_fig(res), html.Div(msg)


@app.callback(Output("fc-acf", "figure"), Input("data-key", "data"), Input("fc-cat", "value"),
              Input("fc-prep", "value"), Input("fc-z", "value"), Input("fc-tail", "value"), Input("sar-d", "value"),
              Input("sar-D", "value"), Input("sar-m", "value"))
def acf_pacf(key, cat, prep, z, tail, d, D, m):
    if not key:
        return blank()
    prep = prep or []
    try:
        y = du.daily_series(du.get_df(key), None if cat in (None, "All") else cat)
        if len(y) < models.MIN_DAYS:
            return blank()
        x = models.diagnostic_series(y, _i(d, 0), _i(D, 0), _i(m, 7), "log" in prep,
                                     _f(z, 3.0) if "outliers" in prep else 0.0, tail or "high")
        return fg.acf_pacf_fig(x, _i(d, 0), _i(D, 0), _i(m, 7))
    except Exception:  # noqa: BLE001  (diagnostic only; never break the page)
        return blank()


@lru_cache(maxsize=6)
def cached_kcv(key, category, order, seasonal, log, outlier_z, tail, kmax, folds, time_limit):
    y = du.daily_series(du.get_df(key), None if category in (None, "All") else category)
    return models.k_cv(y, order, seasonal, ks=tuple(range(0, kmax + 1)), folds=folds, log=log, outlier_z=outlier_z,
                       outlier_tail=tail, time_limit=time_limit)


@app.callback(Output("kcv-fig", "figure"), Output("kcv-note", "children"), Input("kcv-run", "n_clicks"),
              State("data-key", "data"), State("fc-cat", "value"), State("sar-p", "value"), State("sar-d", "value"),
              State("sar-q", "value"), State("sar-P", "value"), State("sar-D", "value"), State("sar-Q", "value"),
              State("sar-m", "value"), State("fc-prep", "value"), State("fc-z", "value"), State("fc-tail", "value"),
              State("fc-time", "value"), State("kcv-max", "value"), State("kcv-folds", "value"),
              prevent_initial_call=True)
def run_kcv(_n, key, cat, p, d, q, P, D, Q, m, prep, z, tail, tlim, kmax, folds):
    if not key:
        return blank(), html.Div("Load data first.", className="error")
    prep = prep or []
    order = (_i(p, 1), _i(d, 1), _i(q, 1))
    seasonal = (_i(P, 1), _i(D, 1), _i(Q, 1), _i(m, 7))
    try:
        cv = cached_kcv(key, cat, order, seasonal, "log" in prep, _f(z, 3.0) if "outliers" in prep else 0.0,
                        tail or "high", max(1, min(_i(kmax, 5), 8)), max(2, min(_i(folds, 5), 10)),
                        _f(tlim, 120.0) * 3)
    except ModuleNotFoundError as exc:
        return no_update, html.Div(f"A required package is missing: {exc.name}. Run `pip install -r requirements.txt`.",
                                   className="error")
    except Exception as exc:  # noqa: BLE001
        return no_update, html.Div(f"Could not run the comparison: {exc}", className="error")
    s = cv["summary"].set_index("K")
    msg = [html.B(f"Lowest validation RMSE: K = {cv['best']}"),
           f" ({s.loc[cv['best'], 'RMSE']:,.0f}" + (f" vs {s.loc[0, 'RMSE']:,.0f} with K = 0" if 0 in s.index and cv["best"] else "")
           + f", {cv['n_origins']} folds, SARIMA{order}x{seasonal[:3]}[{seasonal[3]}], {cv['seconds']:.0f}s)."]
    se_best = s.loc[cv["best"], "SD"] / max(s.loc[cv["best"], "folds"], 1) ** 0.5
    simple = [k for k in s.index if s.loc[k, "RMSE"] <= s.loc[cv["best"], "RMSE"] + se_best]
    if simple and min(simple) < cv["best"]:
        msg.append(html.Div(f"K = {min(simple)} is within one standard error of the best, so it is the simpler choice."))
    if cv["partial"]:
        msg.append(html.Div("Time limit reached: larger K values were skipped. Raise the time limit or lower K / folds.",
                            className="warn"))
    msg.append(html.Div("Only a handful of 5-day windows are scored, so differences smaller than the error bars are noise.",
                        className="warn"))
    return fg.k_cv_fig(cv), html.Div(msg)


def _i(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


FC_OUTPUTS = [Output("fc-summary", "children"), Output("fc-info", "children"), Output("fc-metrics", "data"),
              Output("fc-metrics", "style_data_conditional"), Output("fc-metrics-bar", "figure"),
              Output("fc-main", "figure"), Output("fc-parity", "figure"), Output("fc-imp", "figure"),
              Output("fc-future", "figure"), Output("fc-future-table", "data"), Output("fc-search", "data"),
              Output("fc-rtime", "figure"), Output("fc-racf", "figure"), Output("fc-rdist", "figure"),
              Output("fc-lb", "data"), Output("fc-lb-note", "children"), Output("fc-horizon", "figure")]


def residual_outputs(r):
    """Residual plots, Ljung-Box table + note, and error-by-horizon for the Residual / Validation steps."""
    resid, mm = r["resid"], r["seasonal"][3]
    lags = tuple(mm * k for k in (1, 2, 3, 4)) if mm and mm > 1 else (7, 14, 21, 28)
    lb = models.ljung_box(resid, lags=lags, n_params=r["n_params"])
    bad = lb[lb["p-value"] < 0.05]
    if lb.empty:
        note = html.Div("Too few residuals for the Ljung-Box test.")
    elif bad.empty:
        note = html.Div([html.B("No significant autocorrelation left in the residuals. "),
                         "The model has captured the linear time structure at these lags."])
    else:
        note = html.Div([html.B("Autocorrelation remains "), f"(Ljung-Box p < 0.05 at lags {', '.join(map(str, bad['Lag']))}). ",
                         "The model has missed structure: look at the residual ACF for the lags that stick out and try "
                         "adding an AR / MA or seasonal term, a different d / D, or yearly Fourier terms."], className="warn")
    return (fg.resid_time_fig(resid), fg.resid_acf_fig(resid, mm), fg.resid_dist_fig(resid), lb.to_dict("records"),
            note, fg.horizon_fig(r["horizon_err"]))


@app.callback(*FC_OUTPUTS, Input("fc-run", "n_clicks"), State("data-key", "data"), State("fc-cat", "value"),
              State("sar-p", "value"), State("sar-d", "value"), State("sar-q", "value"),
              State("sar-P", "value"), State("sar-D", "value"), State("sar-Q", "value"), State("sar-m", "value"),
              State("fc-prep", "value"), State("fc-z", "value"), State("fc-tail", "value"), State("fc-auto", "value"),
              State("fc-time", "value"), State("fc-diff", "value"), State("fc-fourier", "value"), prevent_initial_call=True)
def run_models(_n, key, cat, p, d, q, P, D, Q, m, prep, z, tail, auto, tlim, diff="tests", fk=0):
    fail = lambda msg: (html.Div(msg, className="error"),) + (no_update,) * (len(FC_OUTPUTS) - 1)  # noqa: E731
    if not key:
        return fail("Load data first.")
    prep = prep or []
    m = _i(m, 7)
    order = (_i(p, 1), _i(d, 1), _i(q, 1))
    seasonal = (_i(P, 1), _i(D, 1), _i(Q, 1), m)
    try:
        r = cached_run(key, cat, order, seasonal, auto == "auto", "log" in prep,
                       _f(z, 3.0) if "outliers" in prep else 0.0, tail or "high", _f(tlim, 120.0),
                       (order[1], seasonal[1]) if diff == "mine" else None, max(0, min(_i(fk, 0), 6)))
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
    ol = r["outliers"]
    steps = ["log(1+sales) with smearing correction ×%.3f" % r["smear"] if r["log"] else "no log transform"]
    if ol["thr"]:
        side = "z > %g" % ol["thr"] if ol["tail"] == "high" else "|z| > %g" % ol["thr"]
        steps.append(f"{ol['n_removed']} of {ol['n_train']:,} training days removed ({side}"
                     + (f"; {ol['n_low']} low-side" if ol["n_low"] else "") + ")")
    else:
        steps.append("no outlier removal")
    if r["fourier_k"]:
        steps.append(f"yearly Fourier terms K={r['fourier_k']} (SARIMA regressors + XGBoost features)")
    if r["auto"]:
        sr = r["search"]
        steps.append(f"auto order, d={sr['d']} D={sr['D']} ({'your choice' if sr['forced'] else 'chosen by tests'})")
    else:
        steps.append("manual order")
    info = (f"{len(y):,} days ({y.index[0]:%d %b %Y} → {y.index[-1]:%d %b %Y}) · train {n_tr:,} days to "
            f"{r['split_date'] - pd.Timedelta(days=1):%d %b %Y}, test {len(y) - n_tr:,} days from "
            f"{r['split_date']:%d %b %Y} · SARIMA{o}x{s[:3]}[{s[3]}] (AIC {r['aic']:,.0f}) · "
            f"{r['zero_share']:.1%} of days had no sales (counted as 0) · " + " · ".join(steps)
            + f" · took {r['seconds']:.0f}s")
    search = []
    if r["search"]:
        search = [{"Order": str(x["order"]), "Seasonal": str(x["seasonal"]), "AIC": round(x["aic"], 1)}
                  for x in r["search"]["tried"]]

    table = r["metrics"].round({"RMSE": 1, "MAE": 1, "WAPE %": 1, "MASE": 2, "R²": 3}).to_dict("records")
    highlight = [{"if": {"filter_query": f'{{Model}} = "{best}"'}, "backgroundColor": "#fde3d4", "fontWeight": "600"}]

    fut = r["future"].copy()
    fut.insert(0, "Day", fut.index.strftime("%a"))
    fut.insert(0, "Date", fut.index.strftime("%Y-%m-%d"))
    fut = fut.round(2)
    return (html.Div(rows), info, table, highlight, fg.metrics_bar_fig(r), fg.forecast_fig(r), fg.parity_fig(r),
            fg.importance_fig(r), fg.future_fig(r), fut.to_dict("records"), search,
            *residual_outputs(r))


if __name__ == "__main__":
    app.run(debug=True)

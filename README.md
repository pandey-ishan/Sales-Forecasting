# Sales Forecast Studio (Plotly Dash)

A Dash-only app (no other UI framework, no custom JavaScript) for the Superstore-style sales table:
data view, univariate and bivariate analysis, EDA, then SARIMA, XGBoost and a 5-day forecast.

## Run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # http://127.0.0.1:8050
```

Load data any of three ways: drag a CSV/XLSX onto the page, save it as `data/superstore.csv`
(or point `SALES_CSV` at it), or click **Use demo data** (synthetic, clearly labelled).

Required columns: `Order Date`, `Sales`. Everything else is optional and matched loosely by name.
Reads UTF-8 and Latin-1 CSVs. `Order Date` format (dd/mm/yyyy vs mm/dd/yyyy) is detected
automatically and shown on the Data tab.

## Tabs

| Tab | What's in it |
|---|---|
| 1 Data | KPIs, first rows, column profile (types, missing values, uniques) |
| 2 Univariate | Histogram, box plot, z-scores (threshold slider, log10 option), summary stats, candlestick of daily sales (week/month/quarter), categorical frequencies |
| 3 Bivariate | Correlation heatmap (Pearson/Spearman, daily level incl. lag 1/7), category × category heatmap, box/violin by group with Kruskal-Wallis p-value, scatter with OLS fit, month × weekday heatmap |
| 4 EDA | Bubble map (state or city), sunburst/treemap drill-down Category → Sub-Category → Product, sales over time with click-to-drill into a category's sub-categories, seasonality by year |
| 5 Forecast | SARIMA vs XGBoost vs hybrid on a chronological 70/30 split, actual-vs-predicted plots, feature importance, next-5-day forecast |

## How the forecast works

- The target is **total sales per calendar day**. Days with no orders count as 0.
- The first 70% of days train, the last 30% test. SARIMA parameters are fitted on the 70% only.
- **One-step-ahead** scores (each test day predicted from the actual days before it) are used for
  SARIMA, XGBoost and the hybrid so they're comparable. SARIMA forecasting the whole test set blind
  (**multi-step**) and a **seasonal-naive** baseline (same weekday last week) are shown as well. If
  nothing beats the baseline, the app says so.
- **XGBoost** uses lags (1, 2, 3, 7, 14, 21, 28), rolling means/std and calendar features, all built
  from days strictly before the target day. **Hybrid** = SARIMA prediction + XGBoost trained on SARIMA's
  residuals.
- The **5-day forecast** refits on all data. XGBoost uses one model per horizon (features lagged
  at least that many days), not recursion, because recursive forecasts drift in testing.
- The default SARIMA order is (1,1,1)(1,1,1,7); edit the boxes on the Forecast tab to try others.

## Things to know

- Daily retail sales are spiky, so expect a low R². The app flags it instead of hiding it.
- The city map uses the offline `zipcodes` package (postal code first, city name as fallback). Without
  it the map shows states. The public Superstore file has a few Burlington, VT rows with no postal code.
- Training takes roughly 10-60 seconds depending on your machine and SARIMA order. Results are cached
  per dataset/category/order. For hosting use a long-running server (`gunicorn app:server`, e.g. Render);
  serverless platforms with short function time limits will time out on the model run.
- Uploaded files are stored in `uploads/` and referenced by a short hash. Data files are git-ignored.

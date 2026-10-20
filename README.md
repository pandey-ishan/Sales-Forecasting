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
| 5 Forecast | Seven collapsible steps (below) from stationarity checks to the final 5-day forecast, on a chronological 70/30 split |

## The Forecast tab, step by step

| Step | What's in it |
|---|---|
| Setup | Target, log transform, outlier removal (z threshold, side) |
| 1 Stationarity Analysis | Rolling mean / std, original vs differenced series, ADF and KPSS tests, suggested d and D. Needs statsmodels |
| 2 Model Identification | p, q, P, Q, m inputs; ACF and PACF with the non-significant band shaded; periodogram |
| 3 Feature Engineering | Yearly Fourier terms (K); lists the calendar / lag features. Holidays and promotions are not in the data |
| 4 Model Estimation and Selection | Auto or manual order, Train & forecast, AIC / BIC heatmap over p and q, validation RMSE vs K (rolling CV), auto-ARIMA search table |
| 5 Residual Diagnostics | Residuals over time, residual ACF, histogram and Q-Q plot, Ljung-Box test |
| 6 Forecast Validation | MAE, RMSE, WAPE, MASE, R²; actual vs predicted; forecast error by horizon (1-5 days); feature importance |
| 7 Final Model Fitting and Forecasting | Refit on all data, next 5 days with a 95% SARIMA interval |

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
- **Yearly Fourier terms (K)**: 0 = off. K > 0 adds K sine/cosine pairs of the day of year as SARIMA regressors and XGBoost features, a smooth yearly curve that works with m = 7 (SARIMA with m = 365 is impractical). Start with K = 2-3 and compare test RMSE with K = 0.
- **Choose K plot** ("Compare K"): rolling-origin cross-validation of the 5-day SARIMA forecast for K = 0..K max, on the training 70% only, using the manual order and the pre-processing settings. Shows mean RMSE per K with error bars, one faint line per fold and the K = 0 baseline. Slow (one SARIMA fit per K per fold), so it's capped by 3x the time-limit box.
- **ACF / PACF plots** above the model controls show the training series after the log/outlier settings and the d, D, m you type, with 95% bands and seasonal lags highlighted, to help pick manual p, q, P, Q.
- **Pre-processing (on by default, each can be switched off and re-run to compare):**
  - **log(1+sales)**: models are fitted on the log scale; predictions are converted back with Duan's
    smearing factor (bounded to 0.5-2) so the back-transform doesn't under-state the mean.
  - **Outlier removal**: training days with z > 3 on raw sales (the same z-score as the Univariate tab) are
    removed. SARIMA sees them as missing values (days aren't deleted, so the weekly rhythm survives);
    XGBoost drops them as training targets. Test days are never touched. The "Both" option also removes
    unusually low days (z < -3). z uses the training mean/std only. After a log, zero-sales days make the
    distribution left-skewed, so z > 3 would flag nothing; that's why the z-score is taken on raw sales.
  - **Auto-ARIMA**: a custom stepwise AIC search (Hyndman-Khandakar style, not `pmdarima`): d from an ADF
    test, D from seasonal strength, then p, q, P, Q around a few starting models, capped by a time limit.
    Choose Manual to type an order instead.

## Things to know

- Daily retail sales are spiky, so expect a low R². The app flags it instead of hiding it.
- The city map uses the offline `zipcodes` package (postal code first, city name as fallback). Without
  it the map shows states. The public Superstore file has a few Burlington, VT rows with no postal code.
- Training takes roughly 10-60 seconds depending on your machine and SARIMA order. Results are cached
  per dataset/category/order. For hosting use a long-running server (`gunicorn app:server`, e.g. Render);
  serverless platforms with short function time limits will time out on the model run.
- Uploaded files are stored in `uploads/` and referenced by a short hash. Data files are git-ignored.

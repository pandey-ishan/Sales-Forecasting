"""SARIMA, XGBoost and a SARIMA+XGBoost hybrid for daily sales.

Design notes
- Chronological 70/30 split of the daily series (no shuffling).
- SARIMA parameters are estimated on the training 70% only.
- "One-step-ahead" evaluation filters the test period through the *train-fitted* SARIMA, so each
  test day is predicted from the actual history before it. XGBoost uses lag features built from
  actual history in the same way, which makes the three models directly comparable.
- "Multi-step" SARIMA (forecast the whole 30% blind) is reported separately as the harsher test.
- A seasonal-naive baseline (same weekday last week) tells you whether any model is worth having.
- The 5-day forecast refits on all data. XGBoost uses a *direct* multi-horizon strategy: one model
  per forecast day h, trained on features lagged by at least h days. (Recursive forecasting, where
  predictions feed the next day's lags, was tried first and drifted to zero within a few steps
  because smooth predictions are off-distribution for lag features built from spiky actuals.)
  The hybrid adds the same kind of XGBoost correction on top of the SARIMA forecast.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 7, 14, 21, 28)
MIN_DAYS = 120
MODEL_NAMES = ["SARIMA", "XGBoost", "SARIMA + XGBoost"]


# --------------------------------------------------------------------------- #
# Features and metrics
# --------------------------------------------------------------------------- #
def make_features(y: pd.Series, base=None, min_lag: int = 1) -> pd.DataFrame:
    """Features for day t built only from days <= t - min_lag (min_lag=1: strictly before t)."""
    idx = y.index
    f = pd.DataFrame(index=idx)
    for lag in sorted({l for l in LAGS if l >= min_lag} | {min_lag}):
        f[f"lag_{lag}"] = y.shift(lag)
    prev = y.shift(min_lag)
    for w in (7, 14, 28):
        f[f"mean_{w}"] = prev.rolling(w).mean()
    f["std_7"] = prev.rolling(7).std()
    f["dow"] = idx.dayofweek
    f["dom"] = idx.day
    f["month"] = idx.month  # day-of-year / week-of-year are left out on purpose: ~4 samples each, trees memorise them
    f["month_end"] = idx.is_month_end.astype(int)
    f["month_start"] = idx.is_month_start.astype(int)
    if base is not None:  # the SARIMA prediction for the same day (hybrid model)
        f["sarima"] = np.asarray(base, dtype=float)
    return f


def metrics(actual, pred) -> dict:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(p))
    a, p = a[ok], p[ok]
    err = a - p
    denom = np.abs(a).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "WAPE %": float(np.abs(err).sum() / denom * 100) if denom else float("nan"),
        "R²": float(1 - (err ** 2).sum() / ss_tot) if ss_tot else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Model wrappers (the heavy libraries are imported lazily)
# --------------------------------------------------------------------------- #
class SarimaModel:
    """Thin wrapper over statsmodels SARIMAX that works on plain numpy arrays."""

    def __init__(self, order, seasonal):
        self.order = tuple(order)
        self.seasonal = tuple(seasonal)
        self.res = None

    def fit(self, endog: np.ndarray, start_params=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(endog, order=self.order, seasonal_order=self.seasonal,
                            enforce_stationarity=False, enforce_invertibility=False)
            kwargs = {} if start_params is None else {"start_params": start_params}
            self.res = model.fit(disp=False, maxiter=200, **kwargs)
        return self

    @property
    def params(self) -> np.ndarray:
        return np.asarray(self.res.params)

    @property
    def aic(self) -> float:
        return float(self.res.aic)

    def one_step(self, endog: np.ndarray) -> np.ndarray:
        """In-sample one-step-ahead predictions over `endog`, parameters held fixed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = self.res.apply(endog, refit=False)
            pred = filtered.get_prediction(start=0, end=len(endog) - 1, dynamic=False)
        return np.asarray(pred.predicted_mean, dtype=float)

    def forecast(self, steps: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.res.get_forecast(steps)
            ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
        return np.asarray(fc.predicted_mean, dtype=float), ci[:, 0], ci[:, 1]


def make_xgb():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=5.0, tree_method="hist", random_state=42, n_jobs=2,
    )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _direct_forecast(factory, y: pd.Series, dates, burn: int, base_hist=None, base_future=None):
    """Forecast `dates` with one model per horizon h, each using only features lagged >= h days.

    Plain XGBoost: target is y. Hybrid (base_* given): target is the SARIMA residual and the
    SARIMA value for the day is a feature and is added back to the prediction.
    Returns (predictions, the horizon-1 model).
    """
    n = len(y)
    ext = pd.concat([y, pd.Series(np.nan, index=dates)])  # NaN placeholders are never read: lags >= h
    hybrid = base_hist is not None
    base_ext = np.concatenate([base_hist, base_future]) if hybrid else None
    target = pd.Series(y.to_numpy() - base_hist, index=y.index) if hybrid else y

    out, first = [], None
    for h, d in enumerate(dates, start=1):
        feats = make_features(ext, base=base_ext, min_lag=h)
        model = factory().fit(feats.iloc[burn:n], target.iloc[burn:])
        pred = float(model.predict(feats.loc[[d]])[0])
        if hybrid:
            pred += float(base_future[h - 1])
        out.append(max(pred, 0.0))
        if h == 1:
            first = model
    return np.asarray(out), first


def run_pipeline(y: pd.Series, order=(1, 1, 1), seasonal=(1, 1, 1, 7), train_frac=0.7, horizon=5,
                 sarima_cls=None, xgb_factory=None) -> dict:
    sarima_cls = sarima_cls or SarimaModel
    xgb_factory = xgb_factory or make_xgb

    y = y.astype(float)
    n = len(y)
    if n < MIN_DAYS:
        raise ValueError(f"Need at least {MIN_DAYS} days of data, got {n}.")
    n_train = int(n * train_frac)
    vals = y.to_numpy()
    seas = tuple(seasonal) if seasonal[3] and seasonal[3] > 1 else (0, 0, 0, 0)
    order = tuple(order)

    scale = float(np.std(vals[:n_train])) or 1.0  # SARIMAX optimises far better on unit-scale data
    burn = max(35, order[1] + seas[1] * seas[3] + 14)  # lag-28 warm-up and SARIMA start-up noise

    # ---- 1. SARIMA, parameters from the training 70% only --------------------------------
    sar = sarima_cls(order, seas).fit(vals[:n_train] / scale)
    base_full = sar.one_step(vals / scale) * scale                 # one-step-ahead, whole series
    multi = sar.forecast(n - n_train)[0] * scale                   # blind forecast of the whole test set

    # ---- 2. XGBoost on lag/calendar features ---------------------------------------------
    feats = make_features(y)
    xgb = xgb_factory().fit(feats.iloc[burn:n_train], y.iloc[burn:n_train])
    xgb_pred = np.clip(xgb.predict(feats.iloc[n_train:]), 0, None)

    # ---- 3. Hybrid: SARIMA + XGBoost correcting SARIMA's residuals -------------------------
    feats_h = make_features(y, base=base_full)
    resid = pd.Series(vals - base_full, index=y.index)
    hyb = xgb_factory().fit(feats_h.iloc[burn:n_train], resid.iloc[burn:n_train])
    hyb_pred = np.clip(base_full[n_train:] + hyb.predict(feats_h.iloc[n_train:]), 0, None)

    # ---- 4. Test-set comparison -------------------------------------------------------------
    test_idx = y.index[n_train:]
    actual = vals[n_train:]
    naive = y.shift(7).to_numpy()[n_train:]
    preds = pd.DataFrame({
        "Actual": actual,
        "Seasonal naive": naive,
        "SARIMA (multi-step)": np.clip(multi, 0, None),
        "SARIMA": np.clip(base_full[n_train:], 0, None),
        "XGBoost": xgb_pred,
        "SARIMA + XGBoost": hyb_pred,
    }, index=test_idx)

    evals = {"Seasonal naive": "baseline", "SARIMA (multi-step)": "multi-step", "SARIMA": "one-step",
             "XGBoost": "one-step", "SARIMA + XGBoost": "one-step"}
    mrows = [{"Model": m, "Eval": e, **metrics(actual, preds[m])} for m, e in evals.items()]
    mdf = pd.DataFrame(mrows)

    rmse = mdf.set_index("Model")["RMSE"]
    best = min(MODEL_NAMES, key=lambda m: rmse[m])
    improvement = float((rmse["SARIMA"] - rmse[best]) / rmse["SARIMA"] * 100) if rmse["SARIMA"] else 0.0

    # ---- 5. Refit on everything and forecast the next `horizon` days ---------------------
    sar_all = sarima_cls(order, seas).fit(vals / scale, start_params=sar.params)
    base_all = sar_all.one_step(vals / scale) * scale
    fut_mean, lo, hi = sar_all.forecast(horizon)
    fut_mean, lo, hi = fut_mean * scale, lo * scale, hi * scale

    dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    xgb_future, xgb_all = _direct_forecast(xgb_factory, y, dates, burn)
    hyb_future, hyb_all = _direct_forecast(xgb_factory, y, dates, burn, base_hist=base_all, base_future=fut_mean)
    feats_h_all = make_features(y, base=base_all)
    future = pd.DataFrame({
        "SARIMA": np.clip(fut_mean, 0, None),
        "SARIMA 95% low": np.clip(lo, 0, None),
        "SARIMA 95% high": np.clip(hi, 0, None),
        "XGBoost": xgb_future,
        "SARIMA + XGBoost": hyb_future,
    }, index=dates)

    importance = pd.concat({
        "XGBoost": pd.Series(np.asarray(xgb_all.feature_importances_, dtype=float), index=feats.columns),
        "SARIMA + XGBoost": pd.Series(np.asarray(hyb_all.feature_importances_, dtype=float),
                                      index=feats_h_all.columns),
    }, axis=1).fillna(0.0)

    return dict(
        y=y, n_train=n_train, split_date=y.index[n_train], preds=preds, metrics=mdf, future=future,
        importance=importance, best=best, improvement=improvement, aic=getattr(sar, "aic", float("nan")),
        order=order, seasonal=seas, zero_share=float((vals == 0).mean()),
    )

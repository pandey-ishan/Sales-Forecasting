"""SARIMA (auto-ARIMA), XGBoost and a SARIMA+XGBoost hybrid for daily sales.

Pre-processing (all optional, all decided from the TRAINING days only)
- log1p transform: models are fitted on log(1 + sales); predictions are converted back with Duan's
  smearing factor, because exp(mean of log) under-states the mean of a skewed variable.
- outlier removal: training days whose raw sales have z > threshold (the same z-score as the Univariate tab; after
  a log the zero-sales days make the distribution left-skewed, so z > 3 would flag nothing) (optionally |z| > threshold)
  are removed. For SARIMA they become missing values (the state-space model handles gaps, whereas
  deleting days would break the weekly rhythm); for XGBoost they are dropped as training targets and
  interpolated when they appear as lag inputs. Test days are never altered.
- auto-ARIMA: d from an ADF test, D from seasonal strength (forecast::nsdiffs rule), then a stepwise
  AIC search over p, q, P, Q (Hyndman-Khandakar style) under a time limit.

Evaluation
- Chronological 70/30 split. SARIMA parameters and the ARIMA order are chosen on the 70% only.
- "One-step-ahead" scores filter the test period through the train-fitted SARIMA, so each test day is
  predicted from the actual days before it. XGBoost uses lag features built the same way, so the
  three models are directly comparable. SARIMA forecasting the whole test set blind ("multi-step") and a
  seasonal-naive baseline (same weekday last week) are reported as well. Metrics are on the original
  sales scale.
- The 5-day forecast refits on all data. XGBoost uses one model per horizon (features lagged >= h days)
  instead of recursion, which drifted to zero in testing.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 7, 14, 21, 28)
MIN_DAYS = 120
MODEL_NAMES = ["SARIMA", "XGBoost", "SARIMA + XGBoost"]


# --------------------------------------------------------------------------- #
# Features and metrics
# --------------------------------------------------------------------------- #
def fourier_terms(index, k: int, period: float = 365.25) -> np.ndarray:
    """k sine/cosine pairs for the yearly cycle, from the calendar day of year (no data needed, so no leakage)."""
    if not k:
        return np.zeros((len(index), 0))
    t = np.asarray(index.dayofyear, dtype=float) / period
    cols = []
    for j in range(1, int(k) + 1):
        cols += [np.sin(2 * np.pi * j * t), np.cos(2 * np.pi * j * t)]
    return np.column_stack(cols)


# Holiday regressors: calendar information is available before the forecast date.
# Install with: pip install holidays
_COUNTRY_ALIASES = {
    "INDIA": "IN", "BHARAT": "IN", "UK": "GB", "UNITED KINGDOM": "GB",
    "GREAT BRITAIN": "GB", "USA": "US", "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US", "UAE": "AE", "UNITED ARAB EMIRATES": "AE",
    "SOUTH KOREA": "KR", "KOREA": "KR", "RUSSIA": "RU",
}


def detect_country(data=None, country_col="Country"):
    """Read one country from a DataFrame's Country column or Series.attrs.

    Raises on multiple countries: a single aggregated sales series must have one
    holiday calendar, or the caller must forecast each country separately.
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        if country_col not in data.columns:
            raise ValueError(f"Country column '{country_col}' not found.")
        values = data[country_col].dropna().astype(str).str.strip()
        values = values[values.ne("")].unique().tolist()
    elif isinstance(data, pd.Series):
        value = data.attrs.get(country_col, data.attrs.get("country"))
        values = [] if value is None else [str(value).strip()]
    elif isinstance(data, str):
        values = [data.strip()]
    else:
        raise TypeError("country_data must be a DataFrame, Series with attrs, or country string")
    norm = {_COUNTRY_ALIASES.get(v.upper(), v.upper()) for v in values if v}
    if len(norm) > 1:
        raise ValueError(f"Multiple countries found in '{country_col}': {sorted(norm)}. Forecast each country separately.")
    return next(iter(norm), None)


def holiday_features(dates, country, subdiv=None):
    """Public holiday, day-before, day-after and holiday-on-weekend indicators."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    columns = ["is_holiday", "pre_holiday", "post_holiday", "holiday_weekend"]
    if not country:
        return pd.DataFrame(0.0, index=dates, columns=columns)
    try:
        import holidays
    except ImportError as exc:
        raise ImportError("Automatic holiday features require: pip install holidays") from exc
    try:
        cal = holidays.country_holidays(country, subdiv=subdiv,
            years=range(dates.min().year - 1, dates.max().year + 2))
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise ValueError(f"Country '{country}' (subdivision {subdiv!r}) is not supported by holidays. ") from exc
    days = set(cal.keys())
    on = np.asarray([d.date() in days for d in dates], dtype=float)
    before = np.asarray([(d + pd.Timedelta(days=1)).date() in days for d in dates], dtype=float)
    after = np.asarray([(d - pd.Timedelta(days=1)).date() in days for d in dates], dtype=float)
    return pd.DataFrame({"is_holiday": on, "pre_holiday": before,
        "post_holiday": after, "holiday_weekend": on * (dates.dayofweek >= 5)}, index=dates)


def calendar_exog(dates, fourier_k=0, country=None, subdiv=None):
    """Identical SARIMAX columns for training, CV and future predictions."""
    fourier = fourier_terms(pd.DatetimeIndex(dates), fourier_k)
    if not country:
        return fourier if fourier_k else None
    hol = holiday_features(dates, country, subdiv).to_numpy(dtype=float)
    return np.column_stack([fourier, hol])


def _exog_kw(exog):
    return {"exog": exog} if exog is not None and np.asarray(exog).shape[1] else {}


def make_features(y: pd.Series, base=None, min_lag: int = 1, fk: int = 0, country=None, subdiv=None) -> pd.DataFrame:
    """Features for day t built only from days <= t - min_lag (min_lag=1: strictly before t)."""
    idx = y.index
    f = pd.DataFrame(index=idx)
    for lag in sorted({l for l in LAGS if l >= min_lag} | {min_lag}):
        f[f"lag_{lag}"] = y.shift(lag)
    prev = y.shift(min_lag)
    for w in (7, 14, 28):
        f[f"mean_{w}"] = prev.rolling(w, min_periods=max(1, w // 2)).mean()
    f["std_7"] = prev.rolling(7, min_periods=3).std()
    f["dow"] = idx.dayofweek
    f["dom"] = idx.day
    f["month"] = idx.month  # day-of-year / week-of-year are left out on purpose: ~4 samples each, trees memorise them
    f["month_end"] = idx.is_month_end.astype(int)
    f["month_start"] = idx.is_month_start.astype(int)
    for j, col in enumerate(fourier_terms(idx, fk).T):  # smooth yearly cycle (sin1, cos1, sin2, ...)
        f[f"{'sin' if j % 2 == 0 else 'cos'}{j // 2 + 1}"] = col
    if country:
        for name, col in holiday_features(idx, country, subdiv).items():
            f[name] = col.to_numpy()
    if base is not None:  # the SARIMA prediction for the same day (hybrid model)
        f["sarima"] = np.asarray(base, dtype=float)
    return f


def metrics(actual, pred, scale=None) -> dict:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(p))
    a, p = a[ok], p[ok]
    err = a - p
    denom = np.abs(a).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    out = {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "WAPE %": float(np.abs(err).sum() / denom * 100) if denom else float("nan"),
        "R²": float(1 - (err ** 2).sum() / ss_tot) if ss_tot else float("nan"),
    }
    if scale:  # MASE: MAE relative to the in-sample seasonal-naive MAE (1.0 = as good as "same weekday last week")
        out["MASE"] = float(out["MAE"] / scale)
    return out


# --------------------------------------------------------------------------- #
# Pre-processing helpers
# --------------------------------------------------------------------------- #
def to_model_space(y, log: bool):
    y = np.asarray(y, dtype=float)
    return np.log1p(y) if log else y


def from_model_space(z, log: bool, smear: float = 1.0):
    z = np.asarray(z, dtype=float)
    return np.clip(np.exp(z) * smear - 1.0, 0.0, None) if log else np.clip(z, 0.0, None)


def flag_outliers(x_train: np.ndarray, thr, tail: str = "high") -> np.ndarray:
    """Boolean mask of training days to remove. z uses the training mean/std only."""
    out = np.zeros(len(x_train), dtype=bool)
    if not thr or thr <= 0 or len(x_train) < 3:
        return out
    sd = float(np.std(x_train))
    if sd == 0:
        return out
    zz = (x_train - float(np.mean(x_train))) / sd
    out |= zz > thr
    if tail == "both":
        out |= zz < -thr
    return out


def smearing_factor(resid: np.ndarray) -> float:
    """Duan's smearing estimator: mean(exp(residual)) of the log-scale model."""
    r = np.asarray(resid, dtype=float)
    r = r[~np.isnan(r)]
    # zero-sales days give huge negative log residuals; callers pass positive days only, and the factor is bounded
    return float(np.clip(np.mean(np.exp(r)), 0.5, 2.0)) if len(r) else 1.0


def seasonal_strength(x: np.ndarray, m: int) -> float:
    """Fs = max(0, 1 - Var(remainder) / Var(detrended)), the measure behind forecast::nsdiffs."""
    x = np.asarray(x, dtype=float)
    if m < 2 or len(x) < 3 * m:
        return 0.0
    s = pd.Series(x)
    detr = (s - s.rolling(m, center=True, min_periods=m).mean()).to_numpy()
    ok = ~np.isnan(detr)
    pos = np.arange(len(x)) % m
    seas = np.array([detr[(pos == k) & ok].mean() if ((pos == k) & ok).any() else 0.0 for k in range(m)])
    rem = detr - seas[pos]
    total = float(np.var(detr[ok]))
    return float(max(0.0, 1.0 - np.var(rem[ok]) / total)) if total > 0 else 0.0


def choose_differencing(x: np.ndarray, m: int):
    """(d, D): D = 1 if the weekly pattern is strong (Fs > 0.64), d = 1 if an ADF test cannot reject a unit root."""
    D = 1 if seasonal_strength(x, m) > 0.64 else 0
    d = 0
    try:
        from statsmodels.tsa.stattools import adfuller

        base = x if D == 0 else (x[m:] - x[:-m])
        d = 0 if adfuller(base, autolag="AIC")[1] < 0.05 else 1
    except Exception:  # noqa: BLE001  (missing package, constant series ...)
        d = 0
    return d, D


# --------------------------------------------------------------------------- #
# Model wrappers (heavy libraries are imported lazily)
# --------------------------------------------------------------------------- #
class SarimaModel:
    """Thin wrapper over statsmodels SARIMAX that works on plain numpy arrays (NaN = missing)."""

    def __init__(self, order, seasonal):
        self.order = tuple(order)
        self.seasonal = tuple(seasonal)
        self.res = None

    def fit(self, endog: np.ndarray, start_params=None, maxiter: int = 200, exog=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(endog, exog=exog, order=self.order, seasonal_order=self.seasonal,
                            enforce_stationarity=False, enforce_invertibility=False)
            kwargs = {} if start_params is None else {"start_params": start_params}
            self.res = model.fit(disp=False, maxiter=maxiter, **kwargs)
        return self

    @property
    def params(self) -> np.ndarray:
        return np.asarray(self.res.params)

    @property
    def aic(self) -> float:
        return float(self.res.aic)

    @property
    def bic(self) -> float:
        return float(self.res.bic)

    def forecast_from(self, endog: np.ndarray, steps: int, exog=None, exog_future=None) -> np.ndarray:
        """Forecast `steps` days after the end of `endog`, with parameters held fixed (used for error-by-horizon)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = self.res.apply(endog, exog=exog, refit=False)
            fc = filtered.get_forecast(steps, exog=exog_future)
        return np.asarray(fc.predicted_mean, dtype=float)

    def one_step(self, endog: np.ndarray, exog=None) -> np.ndarray:
        """One-step-ahead predictions over `endog` with the parameters held fixed (NaN gaps allowed)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = self.res.apply(endog, exog=exog, refit=False)
            pred = filtered.get_prediction(start=0, end=len(endog) - 1, dynamic=False)
        return np.asarray(pred.predicted_mean, dtype=float)

    def forecast(self, steps: int, exog=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.res.get_forecast(steps, exog=exog)
            ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
        return np.asarray(fc.predicted_mean, dtype=float), ci[:, 0], ci[:, 1]


def make_xgb():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=5.0, tree_method="hist", random_state=42, n_jobs=2,
    )


# --------------------------------------------------------------------------- #
# Auto-ARIMA: stepwise AIC search over (p, q, P, Q) with d, D fixed beforehand
# --------------------------------------------------------------------------- #
def _neighbours(pqPQ, bounds):
    p, q, P, Q = pqPQ
    moves = []
    for dp, dq, dP, dQ in [(1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0),
                           (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1),
                           (1, 1, 0, 0), (-1, -1, 0, 0), (0, 0, 1, 1), (0, 0, -1, -1)]:
        cand = (p + dp, q + dq, P + dP, Q + dQ)
        if all(0 <= v <= b for v, b in zip(cand, bounds)):
            moves.append(cand)
    return moves


def auto_arima(endog, m, d, D, sarima_cls, max_p=3, max_q=3, max_P=1, max_Q=1, max_fits=20,
               time_limit=120.0, maxiter=50, exog=None) -> dict:
    """Stepwise search in the spirit of Hyndman-Khandakar. Stops at `max_fits` models or `time_limit` seconds
    and returns the best model found so far."""
    seasonal = bool(m and m > 1)
    bounds = (max_p, max_q, max_P if seasonal else 0, max_Q if seasonal else 0)
    tried: dict = {}
    t0 = time.monotonic()

    def spent() -> bool:
        return len(tried) >= max_fits or (time.monotonic() - t0) > time_limit

    def evaluate(c):
        if c not in tried:
            p, q, P, Q = c
            try:
                aic = float(sarima_cls((p, d, q), (P, D, Q, m) if seasonal else (0, 0, 0, 0))
                            .fit(endog, maxiter=maxiter, **({} if exog is None else {"exog": exog})).aic)
                tried[c] = aic if np.isfinite(aic) else float("inf")
            except Exception:  # noqa: BLE001  (non-invertible, singular matrix ...)
                tried[c] = float("inf")
        return tried[c]

    best, best_aic = None, float("inf")
    starts, seen = [(2, 2, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)], set()
    for s in starts:
        s = tuple(min(v, b) for v, b in zip(s, bounds))
        if s in seen:
            continue
        seen.add(s)
        if tried and spent():
            break
        a = evaluate(s)
        if a < best_aic:
            best, best_aic = s, a

    improved = best is not None
    while improved and not spent():
        improved = False
        for nb in _neighbours(best, bounds):
            if spent():
                break
            if nb in tried:
                continue
            a = evaluate(nb)
            if a < best_aic - 1e-6:
                best, best_aic, improved = nb, a, True
                break  # restart the neighbourhood around the new best

    if best is None or not np.isfinite(best_aic):
        raise ValueError("auto-ARIMA could not fit any candidate model. Try Manual order or a different target.")

    def fmt(c):
        p, q, P, Q = c
        return (p, d, q), ((P, D, Q, m) if seasonal else (0, 0, 0, 0))

    rows = [{"order": fmt(c)[0], "seasonal": fmt(c)[1], "aic": a} for c, a in sorted(tried.items(), key=lambda kv: kv[1])
            if np.isfinite(a)]
    return dict(order=fmt(best)[0], seasonal=fmt(best)[1], aic=best_aic, tried=rows, n_tried=len(tried),
                seconds=time.monotonic() - t0, stopped_early=spent() and improved is False and len(tried) >= max_fits)


# --------------------------------------------------------------------------- #
# Direct multi-horizon XGBoost
# --------------------------------------------------------------------------- #
def _direct_forecast(factory, zf: pd.Series, keep: np.ndarray, dates, burn: int, base_hist=None, base_future=None,
                     fk: int = 0, country=None, subdiv=None):
    """Forecast `dates` (model space) with one model per horizon h, each using only features lagged >= h days.

    `zf` is the (cleaned) series used for features and targets, `keep` marks training rows that may be used as
    targets (removed outliers are False). Hybrid (base_* given): target is the SARIMA residual and the SARIMA
    value for the day is a feature and is added back. Returns (predictions, the horizon-1 model).
    """
    n = len(zf)
    ext = pd.concat([zf, pd.Series(np.nan, index=dates)])  # NaN placeholders are never read: lags >= h
    hybrid = base_hist is not None
    base_ext = np.concatenate([base_hist, base_future]) if hybrid else None
    target = pd.Series(zf.to_numpy() - base_hist, index=zf.index) if hybrid else zf
    rows = np.arange(burn, n)[np.asarray(keep)[burn:n]]

    out, first = [], None
    for h, d in enumerate(dates, start=1):
        feats = make_features(ext, base=base_ext, min_lag=h, fk=fk, country=country, subdiv=subdiv)
        model = factory().fit(feats.iloc[rows], target.iloc[rows])
        pred = float(model.predict(feats.loc[[d]])[0])
        if hybrid:
            pred += float(base_future[h - 1])
        out.append(pred)
        if h == 1:
            first = model
    return np.asarray(out), first


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(y: pd.Series, order=(1, 1, 1), seasonal=(1, 1, 1, 7), auto=False, log=True, outlier_z=3.0,
                 outlier_tail="high", train_frac=0.7, horizon=5, time_limit=120.0, fixed_dD=None, fourier_k=0,
                 sarima_cls=None, xgb_factory=None, country_data=None, country_col="Country",
                 holiday_subdiv=None) -> dict:
    """Fit and evaluate models; detect holidays from country_data or y.attrs.

    Pass the source DataFrame as country_data to read its Country column.
    Without country metadata, holiday features are disabled.
    """
    t_start = time.monotonic()
    sarima_cls = sarima_cls or SarimaModel
    xgb_factory = xgb_factory or make_xgb

    country = detect_country(country_data if country_data is not None else y, country_col)
    y = y.astype(float)
    n = len(y)
    if n < MIN_DAYS:
        raise ValueError(f"Need at least {MIN_DAYS} days of data, got {n}.")
    n_train = int(n * train_frac)
    vals = y.to_numpy()
    m = int(seasonal[3]) if seasonal and seasonal[3] else 0

    ex_full = calendar_exog(y.index, fourier_k, country, holiday_subdiv)
    ex_train = ex_full[:n_train] if ex_full is not None else None
    kx = _exog_kw

    # ---- 0. model space + outliers, decided from the training days only -----------------------
    z = to_model_space(vals, log)
    flags = flag_outliers(vals[:n_train], outlier_z, outlier_tail)  # z on raw sales, as in the Univariate tab
    z_tr = z[:n_train].copy()
    z_tr[flags] = np.nan
    mu, sd = float(np.nanmean(z_tr)), float(np.nanstd(z_tr)) or 1.0  # standardise: SARIMAX likes unit scale
    endog_train = (z_tr - mu) / sd
    endog_full = (np.concatenate([z_tr, z[n_train:]]) - mu) / sd  # NaN only where training outliers were removed

    # ---- 1. SARIMA: order (auto or manual) and parameters from the training 70% -----------------
    search = None
    if auto:
        if fixed_dD is not None:  # user forces d and D; only p, q, P, Q are searched
            d, D = int(fixed_dD[0]), int(fixed_dD[1])
        else:
            d, D = choose_differencing(pd.Series(endog_train).interpolate(limit_direction="both").to_numpy(), m)
        search = auto_arima(endog_train, m, d, D, sarima_cls, time_limit=time_limit, exog=ex_train)
        search["d"], search["D"], search["forced"] = d, D, fixed_dD is not None
        order, seas = search["order"], search["seasonal"]
    else:
        order = tuple(order)
        seas = tuple(seasonal) if m > 1 else (0, 0, 0, 0)
    burn = max(35, order[1] + seas[1] * seas[3] + 14)  # lag-28 warm-up and SARIMA start-up noise

    sar = sarima_cls(order, seas).fit(endog_train, **kx(ex_train))
    base_z = sar.one_step(endog_full, **kx(ex_full)) * sd + mu                   # one-step-ahead, whole series, model space
    multi_z = sar.forecast(n - n_train, **kx(ex_full[n_train:] if ex_full is not None else None))[0] * sd + mu  # blind forecast of the whole test set
    smear = smearing_factor(((z_tr - base_z[:n_train]))[burn:][(vals[:n_train] > 0)[burn:]]) if log else 1.0

    # ---- 2. XGBoost: features from the cleaned series, removed days are not training targets ------
    z_feat = pd.Series(np.concatenate([pd.Series(z_tr).interpolate(limit_direction="both").to_numpy(), z[n_train:]]),
                       index=y.index)
    keep = np.ones(n, dtype=bool)
    keep[:n_train] = ~flags
    rows = np.arange(burn, n_train)[keep[burn:n_train]]
    feats = make_features(z_feat, fk=fourier_k, country=country, subdiv=holiday_subdiv)
    xgb = xgb_factory().fit(feats.iloc[rows], z_feat.iloc[rows])
    xgb_z = xgb.predict(feats.iloc[n_train:])

    # ---- 3. Hybrid: SARIMA + XGBoost correcting SARIMA's residuals -------------------------------
    feats_h = make_features(z_feat, base=base_z, fk=fourier_k, country=country, subdiv=holiday_subdiv)
    resid = pd.Series(z_feat.to_numpy() - base_z, index=y.index)
    hyb = xgb_factory().fit(feats_h.iloc[rows], resid.iloc[rows])
    hyb_z = base_z[n_train:] + hyb.predict(feats_h.iloc[n_train:])

    # ---- 4. Test-set comparison, on the original sales scale ---------------------------------
    back = lambda a, s=smear: from_model_space(a, log, s)  # noqa: E731
    test_idx = y.index[n_train:]
    actual = vals[n_train:]
    naive = y.shift(7).to_numpy()[n_train:]
    preds = pd.DataFrame({
        "Actual": actual,
        "Seasonal naive": naive,
        "SARIMA (multi-step)": back(multi_z),
        "SARIMA": back(base_z[n_train:]),
        "XGBoost": back(xgb_z),
        "SARIMA + XGBoost": back(hyb_z),
    }, index=test_idx)

    evals = {"Seasonal naive": "baseline", "SARIMA (multi-step)": "multi-step", "SARIMA": "one-step",
             "XGBoost": "one-step", "SARIMA + XGBoost": "one-step"}
    lagm = 7
    naive_scale = float(np.mean(np.abs(vals[:n_train][lagm:] - vals[:n_train][:-lagm])))
    mdf = pd.DataFrame([{"Model": k, "Eval": e, **metrics(actual, preds[k], naive_scale)} for k, e in evals.items()])
    rmse = mdf.set_index("Model")["RMSE"]
    best = min(MODEL_NAMES, key=lambda k: rmse[k])
    improvement = float((rmse["SARIMA"] - rmse[best]) / rmse["SARIMA"] * 100) if rmse["SARIMA"] else 0.0

    # ---- 4b. Residuals of the training fit (standardised, model space) and error by forecast horizon ----
    resid_train = pd.Series((z_tr - base_z[:n_train]) / sd, index=y.index[:n_train]).iloc[burn:]
    hz = horizon_errors(sar, endog_full, ex_full, fourier_k, n_train, mu, sd, smear, vals, horizon, log,
                        z_feat, xgb_factory, burn, rows, flags, country=country, subdiv=holiday_subdiv)

    # ---- 5. Refit on all data (same preprocessing rules, same order) and forecast `horizon` days ---
    flags_all = flag_outliers(vals, outlier_z, outlier_tail)
    z_all = z.copy()
    z_all[flags_all] = np.nan
    mu2, sd2 = float(np.nanmean(z_all)), float(np.nanstd(z_all)) or 1.0
    endog_all = (z_all - mu2) / sd2
    dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    ex_future = calendar_exog(dates, fourier_k, country, holiday_subdiv)
    sar_all = sarima_cls(order, seas).fit(endog_all, start_params=sar.params, **kx(ex_full))
    base_all = sar_all.one_step(endog_all, **kx(ex_full)) * sd2 + mu2
    fm, lo, hi = sar_all.forecast(horizon, **kx(ex_future))
    fut_z, lo_z, hi_z = fm * sd2 + mu2, lo * sd2 + mu2, hi * sd2 + mu2
    smear2 = smearing_factor((z_all - base_all)[burn:][(vals > 0)[burn:]]) if log else 1.0

    zf_all = pd.Series(pd.Series(z_all).interpolate(limit_direction="both").to_numpy(), index=y.index)
    xgb_future, xgb_all = _direct_forecast(xgb_factory, zf_all, ~flags_all, dates, burn, fk=fourier_k, country=country, subdiv=holiday_subdiv)
    hyb_future, hyb_all = _direct_forecast(xgb_factory, zf_all, ~flags_all, dates, burn,
                                           base_hist=base_all, base_future=fut_z, fk=fourier_k, country=country, subdiv=holiday_subdiv)
    feats_h_all = make_features(zf_all, base=base_all, fk=fourier_k, country=country, subdiv=holiday_subdiv)
    b2 = lambda a: from_model_space(a, log, smear2)  # noqa: E731
    future = pd.DataFrame({
        "SARIMA": b2(fut_z), "SARIMA 95% low": b2(lo_z), "SARIMA 95% high": b2(hi_z),
        "XGBoost": b2(xgb_future), "SARIMA + XGBoost": b2(hyb_future),
    }, index=dates)

    importance = pd.concat({
        "XGBoost": pd.Series(np.asarray(xgb_all.feature_importances_, dtype=float), index=feats.columns),
        "SARIMA + XGBoost": pd.Series(np.asarray(hyb_all.feature_importances_, dtype=float),
                                      index=feats_h_all.columns),
    }, axis=1).fillna(0.0)

    n_removed = int(flags.sum())
    n_low = 0 if outlier_tail == 'high' else int((vals[:n_train][flags] < vals[:n_train].mean()).sum())
    return dict(
        y=y, n_train=n_train, split_date=y.index[n_train], preds=preds, metrics=mdf, future=future,
        importance=importance, best=best, improvement=improvement, aic=getattr(sar, "aic", float("nan")),
        order=order, seasonal=seas, zero_share=float((vals == 0).mean()), log=bool(log), smear=smear2,
        outliers=dict(n_removed=n_removed, n_low=n_low, n_high=n_removed - n_low, thr=outlier_z,
                      tail=outlier_tail, n_train=n_train),
        search=search, auto=bool(auto), fourier_k=int(fourier_k or 0), holiday_country=country,
        holiday_subdiv=holiday_subdiv, seconds=time.monotonic() - t_start,
        resid=resid_train, horizon_err=hz, n_params=int(order[0] + order[2] + seas[0] + seas[2]),
    )


def horizon_errors(sar, endog_full, ex_full, fk, n_train, mu, sd, smear, vals, horizon, log, z_feat, xgb_factory,
                   burn, rows, flags, max_origins=30, country=None, subdiv=None) -> pd.DataFrame:
    """RMSE of the h-step-ahead forecast (h = 1..horizon) over rolling origins in the TEST period.

    SARIMA: parameters fixed from the training fit, state filtered up to each origin, then a true h-step forecast.
    XGBoost: one direct model per horizon (features lagged >= h), trained on the training rows.
    Seasonal naive: same weekday last week. Same origins and target days for all three."""
    n = len(vals)
    origins = sorted({int(o) for o in np.linspace(n_train, n - horizon, max_origins)})
    err = {"SARIMA": [], "XGBoost": [], "Seasonal naive": []}
    for o in origins:
        fc = sar.forecast_from(endog_full[:o], horizon, exog=ex_full[:o] if ex_full is not None else None,
                               exog_future=ex_full[o:o + horizon] if ex_full is not None else None) * sd + mu
        err["SARIMA"].append(vals[o:o + horizon] - from_model_space(fc, log, smear))
        err["Seasonal naive"].append(vals[o:o + horizon] - vals[o - 7:o - 7 + horizon])
    xg = np.full((len(origins), horizon), np.nan)
    for h in range(1, horizon + 1):
        feats = make_features(z_feat, min_lag=h, fk=fk, country=country, subdiv=subdiv)
        model = xgb_factory().fit(feats.iloc[rows], z_feat.iloc[rows])
        tgt = np.array([o + h - 1 for o in origins])
        xg[:, h - 1] = vals[tgt] - from_model_space(model.predict(feats.iloc[tgt]), log, smear)
    err["XGBoost"] = list(xg)
    out = {k: np.sqrt(np.mean(np.vstack(v) ** 2, axis=0)) for k, v in err.items()}
    df = pd.DataFrame(out, index=pd.RangeIndex(1, horizon + 1, name="Horizon (days)"))
    df.attrs["origins"] = len(origins)
    return df


# --------------------------------------------------------------------------- #
# Stationarity, periodogram, AIC/BIC grid, residual tests
# --------------------------------------------------------------------------- #
def prepared_training(y: pd.Series, log=True, outlier_z=3.0, outlier_tail="high", train_frac=0.7) -> pd.Series:
    """Training 70% after log and outlier removal (removed days interpolated). No differencing, no test data."""
    vals = y.astype(float).to_numpy()[: int(len(y) * train_frac)]
    flags = flag_outliers(vals, outlier_z, outlier_tail)
    z = to_model_space(vals, log)
    z[flags] = np.nan
    z = pd.Series(z, index=y.index[: len(vals)]).interpolate(limit_direction="both")
    return z


def difference(x, d=0, D=0, m=7) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    if D and m and m > 1 and len(z) > m:
        z = z[m:] - z[:-m]
    for _ in range(int(d or 0)):
        z = np.diff(z)
    return z


def stationarity_verdict(adf_p: float, kpss_p: float) -> str:
    """ADF: H0 = unit root (reject, p < .05 → stationary). KPSS: H0 = stationary (reject, p < .05 → not)."""
    adf_ok, kpss_ok = adf_p < 0.05, kpss_p >= 0.05
    if adf_ok and kpss_ok:
        return "stationary"
    if not adf_ok and not kpss_ok:
        return "non-stationary: difference"
    if adf_ok and not kpss_ok:
        return "conflict: trend-stationary?"
    return "inconclusive"


def stationarity_table(z: pd.Series, d, D, m) -> list[dict]:
    """ADF and KPSS on the series as is and after each differencing. Needs statsmodels."""
    from statsmodels.tsa.stattools import adfuller, kpss

    variants = [("Original", 0, 0)]
    if D and m and m > 1:
        variants.append((f"Seasonal diff (lag {m})", 0, 1))
    if d:
        variants.append((f"Regular diff ×{int(d)}", int(d), 0))
    if d and D and m and m > 1:
        variants.append(("Seasonal + regular", int(d), 1))
    rows = []
    for name, dd_, DD in variants:
        x = difference(z.to_numpy(), dd_, DD, m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = float(adfuller(x, autolag="AIC")[1])
            k = float(kpss(x, regression="c", nlags="auto")[1])  # statsmodels clips this p-value to [0.01, 0.10]
        rows.append({"Series": name, "ADF p": round(a, 4), "KPSS p": round(k, 3), "Verdict": stationarity_verdict(a, k)})
    return rows


def periodogram(x, top=5):
    """Power vs period (days) of the linearly detrended series. Returns (periods, power, peaks[list of periods])."""
    from scipy.signal import periodogram as _pg

    x = np.asarray(x, dtype=float)
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    f, p = _pg(x, fs=1.0, window="hann", detrend=False)
    f, p = f[1:], p[1:]
    per = 1.0 / f
    pk = [i for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] > p[i + 1]]
    pk = sorted(pk, key=lambda i: -p[i])[:top]
    return per, p, [float(per[i]) for i in pk]


def ic_grid(y: pd.Series, d, D, P, Q, m, pmax=3, qmax=3, fourier_k=0, log=True, outlier_z=3.0, outlier_tail="high",
            train_frac=0.7, time_limit=180.0, sarima_cls=None, country_data=None, country_col="Country",
            holiday_subdiv=None) -> dict:
    """AIC and BIC of SARIMA(p,d,q)(P,D,Q,m) for p = 0..pmax, q = 0..qmax on the standardised training series."""
    sarima_cls = sarima_cls or SarimaModel
    vals = y.astype(float).to_numpy()[: int(len(y) * train_frac)]
    flags = flag_outliers(vals, outlier_z, outlier_tail)
    z = to_model_space(vals, log)
    z[flags] = np.nan
    endog = (z - np.nanmean(z)) / (np.nanstd(z) or 1.0)
    country = detect_country(country_data if country_data is not None else y, country_col)
    ex = calendar_exog(y.index[: len(vals)], fourier_k, country, holiday_subdiv)
    seas = (int(P), int(D), int(Q), int(m)) if m and m > 1 else (0, 0, 0, 0)
    t0, rows, partial = time.monotonic(), [], False
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            if time.monotonic() - t0 > time_limit:
                partial = True
                break
            try:
                f = sarima_cls((p, int(d), q), seas).fit(endog, maxiter=50, **({} if ex is None else {"exog": ex}))
                a, b = float(f.aic), float(f.bic)
            except Exception:  # noqa: BLE001
                a = b = float("nan")
            rows.append({"p": p, "q": q, "AIC": a, "BIC": b})
        if partial:
            break
    df = pd.DataFrame(rows)
    if df.empty or df["AIC"].notna().sum() == 0:
        raise ValueError("No candidate model could be fitted. Check d, D, P, Q, m.")
    return dict(grid=df, pmax=pmax, qmax=qmax, seasonal=seas, d=int(d), partial=partial, seconds=time.monotonic() - t0)


def ljung_box(resid, lags=(7, 14, 21, 28), n_params=0) -> pd.DataFrame:
    """Ljung-Box test of 'no autocorrelation left in the residuals'. p < 0.05 → structure the model missed."""
    from scipy.stats import chi2

    r = np.asarray(resid, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    rho = acf(r, max(lags))
    rows = []
    for L in lags:
        if L >= n:
            continue
        q = n * (n + 2) * float(np.sum(rho[1:L + 1] ** 2 / (n - np.arange(1, L + 1))))
        df_ = max(L - int(n_params), 1)
        rows.append({"Lag": L, "Q": round(q, 2), "p-value": round(float(chi2.sf(q, df_)), 4),
                     "Verdict": "no autocorrelation left" if chi2.sf(q, df_) >= 0.05 else "autocorrelation remains"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ACF / PACF diagnostics (help choosing manual p, q, P, Q)
# --------------------------------------------------------------------------- #
def acf(x, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom == 0:
        return np.zeros(nlags + 1)
    return np.array([1.0] + [float((x[k:] * x[:-k]).sum() / denom) for k in range(1, nlags + 1)])


def pacf(x, nlags: int) -> np.ndarray:
    """Partial autocorrelations by the Durbin-Levinson recursion on the sample ACF."""
    r = acf(x, nlags)
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    phi_prev = np.zeros(0)
    for k in range(1, nlags + 1):
        if k == 1:
            phi_kk = r[1]
            phi = np.array([phi_kk])
        else:
            num = r[k] - float(np.dot(phi_prev, r[k - 1:0:-1]))
            den = 1.0 - float(np.dot(phi_prev, r[1:k]))
            phi_kk = num / den if abs(den) > 1e-12 else 0.0
            phi = np.concatenate([phi_prev - phi_kk * phi_prev[::-1], [phi_kk]])
        out[k] = phi_kk
        phi_prev = phi
    return np.clip(out, -1, 1)


def diagnostic_series(y: pd.Series, d=0, D=0, m=7, log=True, outlier_z=3.0, outlier_tail="high",
                      train_frac=0.7) -> np.ndarray:
    """The series the ARIMA is fitted on, after the same pre-processing and the chosen differencing.
    Uses the training 70% only, so the plots don't peek at the test period."""
    z = prepared_training(y, log, outlier_z, outlier_tail, train_frac).to_numpy()
    return difference(z, d, D, m)


# --------------------------------------------------------------------------- #
# Choosing K (yearly Fourier pairs) by rolling-origin cross-validation
# --------------------------------------------------------------------------- #
def k_cv(y: pd.Series, order, seasonal, ks=(0, 1, 2, 3, 4, 5), folds=5, horizon=5, log=True, outlier_z=3.0,
         outlier_tail="high", train_frac=0.7, time_limit=300.0, sarima_cls=None, min_train_frac=0.5,
         country_data=None, country_col="Country", holiday_subdiv=None) -> dict:
    """Time-series CV of the SARIMA 5-day forecast for several K, using only the training 70% of the data.

    Fold origins are spread evenly over the second half of the training part; at each origin the model is fitted on
    all days before it (log, outlier removal and standardisation redone from those days only) and scored on the next
    `horizon` days on the original sales scale. The same origins are used for every K, so errors are paired.
    Returns {"per_fold": DataFrame[K, fold, RMSE], "summary": DataFrame[K, RMSE, SD, folds], "best", "partial"}.
    """
    sarima_cls = sarima_cls or SarimaModel
    y = y.astype(float)
    n_train = int(len(y) * train_frac)
    if n_train < MIN_DAYS:
        raise ValueError(f"Need at least {MIN_DAYS} training days, got {n_train}.")
    country = detect_country(country_data if country_data is not None else y, country_col)
    vals = y.to_numpy()[:n_train]
    idx = y.index[:n_train]
    first = int(n_train * min_train_frac)
    origins = sorted({int(o) for o in np.linspace(first, n_train - horizon, int(folds))})
    order, seasonal = tuple(order), tuple(seasonal)
    t0 = time.monotonic()
    rows, partial = [], False
    for k in ks:
        ex = calendar_exog(idx, k, country, holiday_subdiv)
        for fi, o in enumerate(origins):
            if time.monotonic() - t0 > time_limit:
                partial = True
                break
            flags = flag_outliers(vals[:o], outlier_z, outlier_tail)
            z = to_model_space(vals[:o], log)
            z[flags] = np.nan
            mu, sd = float(np.nanmean(z)), float(np.nanstd(z)) or 1.0
            endog = (z - mu) / sd
            ex_hist = ex[:o] if ex is not None else None
            ex_future = ex[o:o + horizon] if ex is not None else None
            try:
                sar = sarima_cls(order, seasonal).fit(endog, **_exog_kw(ex_hist))
                fc = sar.forecast(horizon, **_exog_kw(ex_future))[0] * sd + mu
                smear = 1.0
                if log:
                    res = (z - (sar.one_step(endog, **_exog_kw(ex_hist)) * sd + mu))[35:]
                    smear = smearing_factor(res[(vals[:o] > 0)[35:]])
                pred = from_model_space(fc, log, smear)
                rmse = float(np.sqrt(np.mean((vals[o:o + horizon] - pred) ** 2)))
            except Exception:  # noqa: BLE001  (non-convergence for one fold shouldn't kill the comparison)
                rmse = float("nan")
            rows.append({"K": int(k), "fold": fi + 1, "origin": idx[o], "RMSE": rmse})
        if partial:
            break
    per = pd.DataFrame(rows)
    if per.empty or per["RMSE"].notna().sum() == 0:
        raise ValueError("No model could be fitted for the cross-validation. Check the manual order.")
    full_k = [k for k, g in per.groupby("K") if len(g) == len(origins)]
    per = per[per["K"].isin(full_k)]  # a K cut off by the time limit would be compared on fewer folds: drop it
    summ = per.groupby("K")["RMSE"].agg(RMSE="mean", SD="std", folds="count").reset_index()
    summ["SD"] = summ["SD"].fillna(0.0)
    best = int(summ.loc[summ["RMSE"].idxmin(), "K"])
    return dict(per_fold=per, summary=summ, best=best, partial=partial, n_origins=len(origins),
                seconds=time.monotonic() - t0)

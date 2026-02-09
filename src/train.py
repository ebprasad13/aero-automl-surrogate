# src/train.py
from __future__ import annotations

"""
Stable training + prediction + optimisation backend for the Streamlit app.

Exports used by app.py:
- AutoMLConfig
- TrainResult
- OptimisationConfig
- OptimisationResult
- recommend_settings
- recommend_optimisation
- rank_features_simple
- train_automl
- predict_mean
- predict_with_uncertainty
- optimise_surrogate
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import Callable, Optional

import time
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None  # type: ignore
    ConstantKernel = Matern = WhiteKernel = None  # type: ignore


def _safe_call(cb: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]) -> None:
    """Best-effort callback for UI progress updates."""
    if cb is None:
        return
    try:
        cb(payload)
    except Exception:
        # Never let UI callbacks break core training/optimisation
        return

# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class AutoMLConfig:
    cv_folds: int = 5
    cv_repeats: int = 1
    test_size: float = 0.2
    bootstrap_size: int = 50
    small_data_mode: bool = False
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class TrainResult:
    targets: List[str]
    feature_cols: List[str]
    group_col: Optional[str]
    config: AutoMLConfig

    # trained estimators
    models: Dict[str, Any]                         # per-target fitted estimator
    bootstrap_models: Dict[str, List[Any]]         # per-target list of fitted estimators for uncertainty

    # UI helpers
    feature_ranges: Dict[str, Tuple[float, float]] # exact min/max from input data
    feature_medians: Dict[str, float]              # medians from input data
    baseline_params: Dict[str, float]              # baseline values used for sliders / fixed params

    # Metrics
    best_overall_model: str
    test_table: pd.DataFrame
    cv_table: pd.DataFrame

    # Held-out predictions for fit plots
    test_predictions: pd.DataFrame

    # Dataset summary
    n_rows: int
    n_features: int
    n_groups: Optional[int] = None

    # For OOD distance scaling (uncertainty inflation)
    _x_train_scaled: Optional[np.ndarray] = None
    _scaler_mean: Optional[np.ndarray] = None
    _scaler_scale: Optional[np.ndarray] = None


@dataclass
class OptimisationConfig:
    method: str = "auto"      # auto | bayes | de | anneal | random
    n_evals: int = 120
    pop_size: int = 40        # DE only
    patience: int = 25
    risk_aversion: float = 0.0
    seed: int = 42


@dataclass
class OptimisationResult:
    objective_label: str
    method_used: str
    best_score: float
    best_params: Dict[str, float]
    history: pd.DataFrame


# -----------------------------
# Recommenders (your requested behaviour)
# -----------------------------

def recommend_settings(
    n_rows: int,
    n_features: int,
    grouped: bool = False,
    n_groups: Optional[int] = None,
    n_targets: int = 1,
) -> AutoMLConfig:
    """
    Heuristic defaults.
    Small data mode triggers automatically again.
    """
    spp = n_rows / max(1, n_features)  # samples per feature
    small = (n_rows < 180) or (spp < 8.0) or (grouped and (n_groups is not None and n_groups < 25))

    if grouped and n_groups is not None:
        cv = int(max(3, min(5, n_groups // 5)))  # conservative for grouped data
    else:
        cv = 5 if n_rows >= 120 else 3

    repeats = 3 if n_rows < 120 else 1

    base_boot = 60 if not small else 40
    boot = int(max(20, min(120, base_boot - 5 * max(0, n_targets - 1))))

    test_size = 0.25 if n_rows < 120 else 0.2

    return AutoMLConfig(
        cv_folds=cv,
        cv_repeats=repeats,
        test_size=test_size,
        bootstrap_size=boot,
        small_data_mode=bool(small),
        random_state=42,
        n_jobs=-1,
    )


def recommend_optimisation(n_vars: int, n_rows: int, method_user: str = "auto") -> OptimisationConfig:
    """
    Default should be Bayesian (B) for <=8 vars; else DE.
    """
    method_user = (method_user or "auto").lower()

    if method_user == "auto":
        method = "bayes" if n_vars <= 8 else "de"
    else:
        method = method_user

    if method == "bayes":
        n_evals = int(min(160, max(60, 30 + 12 * n_vars)))
        pop = 0
    elif method == "de":
        pop = int(min(80, max(24, 10 * n_vars)))
        n_evals = int(min(800, max(200, pop * max(8, 4 * n_vars))))
    elif method == "anneal":
        pop = 0
        n_evals = int(min(600, max(200, 60 + 20 * n_vars)))
    else:
        pop = 0
        n_evals = int(min(800, max(200, 80 + 25 * n_vars)))

    patience = int(max(15, min(50, n_evals // 4)))

    return OptimisationConfig(method=method, n_evals=n_evals, pop_size=pop, patience=patience, seed=42)


# -----------------------------
# Training utilities
# -----------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", num_pipe, feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _candidate_models(cfg: AutoMLConfig) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    models["ridge"] = Ridge(alpha=1.0, random_state=cfg.random_state)

    n_est = 250 if cfg.small_data_mode else 350
    max_depth = 10 if cfg.small_data_mode else None
    min_leaf = 2 if cfg.small_data_mode else 1

    models["extra_trees"] = ExtraTreesRegressor(
        n_estimators=n_est,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
    )

    models["random_forest"] = RandomForestRegressor(
        n_estimators=int(max(200, n_est * 0.8)),
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
    )

    models["gbr"] = GradientBoostingRegressor(random_state=cfg.random_state)

    if cfg.small_data_mode:
        models["knn"] = KNeighborsRegressor(n_neighbors=5, weights="distance")

    return models


def _get_cv(cfg: AutoMLConfig, groups: Optional[np.ndarray]) -> Any:
    if groups is not None:
        return GroupKFold(n_splits=max(2, cfg.cv_folds))
    return KFold(n_splits=max(2, cfg.cv_folds), shuffle=True, random_state=cfg.random_state)


def _cv_score_single_target(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[np.ndarray],
    cfg: AutoMLConfig,
) -> Dict[str, float]:
    cv = _get_cv(cfg, groups)
    rmses: List[float] = []
    maes: List[float] = []
    r2s: List[float] = []

    splitter = cv.split(X, y, groups=groups) if groups is not None else cv.split(X, y)

    for tr_idx, va_idx in splitter:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        p = m.predict(X_va)

        maes.append(float(mean_absolute_error(y_va, p)))
        rmses.append(_rmse(y_va.to_numpy(), np.asarray(p)))
        r2s.append(float(r2_score(y_va, p)))

    return {
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes, ddof=0)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses, ddof=0)),
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s, ddof=0)),
    }


def _repeat_cv_scores(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[np.ndarray],
    cfg: AutoMLConfig,
) -> Dict[str, float]:
    if groups is not None or cfg.cv_repeats <= 1:
        return _cv_score_single_target(model, X, y, groups, cfg)

    maes: List[float] = []
    rmses: List[float] = []
    r2s: List[float] = []

    for r in range(cfg.cv_repeats):
        cv = KFold(n_splits=max(2, cfg.cv_folds), shuffle=True, random_state=cfg.random_state + r)
        for tr_idx, va_idx in cv.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            m = clone(model)
            m.fit(X_tr, y_tr)
            p = m.predict(X_va)

            maes.append(float(mean_absolute_error(y_va, p)))
            rmses.append(_rmse(y_va.to_numpy(), np.asarray(p)))
            r2s.append(float(r2_score(y_va, p)))

    return {
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes, ddof=0)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses, ddof=0)),
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s, ddof=0)),
    }


def _bootstrap_fit(base_estimator: Any, X: pd.DataFrame, y: pd.Series, n_models: int, seed: int) -> List[Any]:
    rng = np.random.default_rng(seed)
    n = len(X)
    out: List[Any] = []
    for i in range(n_models):
        idx = rng.integers(0, n, size=n)
        m = clone(base_estimator)
        m.fit(X.iloc[idx], y.iloc[idx])
        out.append(m)
    return out


def train_automl(
    df: pd.DataFrame,
    target_cols: List[str],
    group_col: Optional[str],
    cfg: AutoMLConfig,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> TrainResult:
    df = df.copy()
    target_cols = list(target_cols)

    if not target_cols:
        raise ValueError("No targets selected.")
    if group_col and group_col not in df.columns:
        group_col = None

    drop_cols = set(target_cols)
    if group_col:
        drop_cols.add(group_col)

    # numeric features only
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]

    if not feature_cols:
        raise ValueError("No numeric feature columns found (excluding targets / group).")

    # ranges/medians for UI (exact from input data)
    feature_ranges: Dict[str, Tuple[float, float]] = {}
    feature_medians: Dict[str, float] = {}
    baseline_params: Dict[str, float] = {}

    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        lo = float(s.min())
        hi = float(s.max())
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo == hi:
            lo = float(lo) if np.isfinite(lo) else 0.0
            hi = lo + 1.0
        med = float(s.median())
        feature_ranges[c] = (lo, hi)
        feature_medians[c] = med
        baseline_params[c] = med

    n_groups = int(df[group_col].nunique()) if group_col else None

    X_all = df[feature_cols]
    y_all = df[target_cols]

    # holdout split
    if group_col:
        groups_all = df[group_col].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        tr_idx, te_idx = next(splitter.split(X_all, y_all, groups=groups_all))
        X_tr_raw, X_te_raw = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        groups_tr_raw = df_tr[group_col].to_numpy()
    else:
        X_tr_raw, X_te_raw, df_tr, df_te = train_test_split(
            X_all, df, test_size=cfg.test_size, random_state=cfg.random_state
        )
        groups_tr_raw = None

    pre = _build_preprocessor(feature_cols)
    models = _candidate_models(cfg)

    fitted_models: Dict[str, Any] = {}
    fitted_boot: Dict[str, List[Any]] = {}

    cv_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    test_pred_df = pd.DataFrame(index=df_te.index)

    best_overall = ""
    best_overall_rmse = float("inf")

    t0 = time.time()
    total_steps = max(1, len(target_cols) * max(1, len(models)))
    done_steps = 0
    _safe_call(progress_cb, {
        "phase": "train",
        "event": "start",
        "total": total_steps,
        "done": 0,
        "eta_s": None,
    })

    for t in target_cols:
        y_tr = pd.to_numeric(df_tr[t], errors="coerce")
        y_te = pd.to_numeric(df_te[t], errors="coerce")

        tr_mask = y_tr.notna().to_numpy()
        te_mask = y_te.notna().to_numpy()

        X_tr = X_tr_raw.iloc[np.where(tr_mask)[0]]
        y_tr_clean = y_tr.iloc[np.where(tr_mask)[0]]

        X_te = X_te_raw.iloc[np.where(te_mask)[0]]
        y_te_clean = y_te.iloc[np.where(te_mask)[0]]

        groups_tr = groups_tr_raw[tr_mask] if groups_tr_raw is not None else None

        best_name = ""
        best_rmse = float("inf")
        best_pipe = None

        for name, est in models.items():
            pipe = Pipeline(steps=[("pre", pre), ("model", est)])
            scores = _repeat_cv_scores(pipe, X_tr, y_tr_clean, groups_tr, cfg)
            done_steps += 1
            elapsed = time.time() - t0
            rate = elapsed / max(1, done_steps)
            eta = rate * max(0, total_steps - done_steps)
            _safe_call(progress_cb, {
                "phase": "train",
                "event": "progress",
                "target": t,
                "model": name,
                "total": total_steps,
                "done": done_steps,
                "eta_s": float(eta),
            })

            cv_rows.append({"target": t, "model": name, **scores})
            if scores["RMSE_mean"] < best_rmse:
                best_rmse = scores["RMSE_mean"]
                best_name = name
                best_pipe = pipe

        assert best_pipe is not None

        best_pipe.fit(X_tr, y_tr_clean)
        fitted_models[t] = best_pipe

        p_te = best_pipe.predict(X_te)
        test_rows.append(
            {
                "target": t,
                "winner": best_name,
                "RMSE": _rmse(y_te_clean.to_numpy(), np.asarray(p_te)),
                "MAE": float(mean_absolute_error(y_te_clean, p_te)),
                "R2": float(r2_score(y_te_clean, p_te)),
                "n_test": int(len(y_te_clean)),
            }
        )

        # store heldout predictions for plots
        idx_te = X_te.index
        test_pred_df.loc[idx_te, f"{t}_actual"] = y_te_clean.to_numpy()
        test_pred_df.loc[idx_te, f"{t}_pred"] = np.asarray(p_te)

        if best_rmse < best_overall_rmse:
            best_overall_rmse = best_rmse
            best_overall = best_name

        n_boot = int(max(10, cfg.bootstrap_size))
        fitted_boot[t] = _bootstrap_fit(best_pipe, X_tr, y_tr_clean, n_models=n_boot, seed=cfg.random_state + 1000)

    cv_table = pd.DataFrame(cv_rows).sort_values(["target", "RMSE_mean"], ascending=[True, True])
    test_table = pd.DataFrame(test_rows).sort_values("RMSE", ascending=True)

    # distance scaler for OOD inflation
    dist_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    x_train_scaled = dist_pipe.fit_transform(X_tr_raw)
    scaler = dist_pipe.named_steps["scale"]

    return TrainResult(
        targets=target_cols,
        feature_cols=feature_cols,
        group_col=group_col,
        config=cfg,
        models=fitted_models,
        bootstrap_models=fitted_boot,
        feature_ranges=feature_ranges,
        feature_medians=feature_medians,
        baseline_params=baseline_params,
        best_overall_model=str(best_overall),
        test_table=test_table,
        cv_table=cv_table,
        test_predictions=test_pred_df,
        n_rows=int(len(df)),
        n_features=int(len(feature_cols)),
        n_groups=n_groups,
        _x_train_scaled=np.asarray(x_train_scaled),
        _scaler_mean=np.asarray(getattr(scaler, "mean_", None)),
        _scaler_scale=np.asarray(getattr(scaler, "scale_", None)),
    )


# -----------------------------
# Prediction
# -----------------------------

def _prepare_X(res: TrainResult, X_input: pd.DataFrame) -> pd.DataFrame:
    X = X_input.copy()
    for c in res.feature_cols:
        if c not in X.columns:
            X[c] = res.baseline_params.get(c, res.feature_medians.get(c, 0.0))
    X = X[res.feature_cols]
    for c in res.feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(res.feature_medians.get(c, 0.0))
    return X


def predict_mean(res: TrainResult, X_input: pd.DataFrame) -> pd.DataFrame:
    """
    Fast mean prediction (no bootstrap).
    Used to speed optimisation when risk_aversion = 0.
    """
    X = _prepare_X(res, X_input)
    out: Dict[str, float] = {}
    for t in res.targets:
        out[t] = float(res.models[t].predict(X)[0])
    return pd.DataFrame([out])


def _nn_distance(res: TrainResult, X: pd.DataFrame) -> float:
    if res._x_train_scaled is None or res._scaler_mean is None or res._scaler_scale is None:
        return 0.0
    Xn = _prepare_X(res, X)
    xs = (Xn.to_numpy() - res._scaler_mean) / (res._scaler_scale + 1e-12)
    d2 = np.sum((res._x_train_scaled - xs) ** 2, axis=1)
    return float(np.sqrt(np.min(d2)))


def predict_with_uncertainty(res: TrainResult, X_input: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      mean_df (1 row),
      std_df  (1 row),
      info_df (1 row, diagnostics for UI)
    """
    X = _prepare_X(res, X_input)

    mean_pred: Dict[str, float] = {}
    std_pred: Dict[str, float] = {}

    for t in res.targets:
        models = res.bootstrap_models.get(t, [])
        if not models:
            mu = float(res.models[t].predict(X)[0])
            mean_pred[t] = mu
            std_pred[t] = 0.0
            continue

        preds = np.array([m.predict(X)[0] for m in models], dtype=float)
        mean_pred[t] = float(np.mean(preds))
        std_pred[t] = float(np.std(preds, ddof=0))

    nn = _nn_distance(res, X_input)
    inflate = 1.0 + max(0.0, (nn - 2.5) / 2.5)  # inflate beyond ~2.5 scaled units

    for t in res.targets:
        std_pred[t] *= inflate

    mean_df = pd.DataFrame([mean_pred])
    std_df = pd.DataFrame([std_pred])
    info_df = pd.DataFrame([{"nearest_neighbour_distance": nn, "uncertainty_inflation": inflate}])
    return mean_df, std_df, info_df


# -----------------------------
# Feature ranking (k=6 default in app)
# -----------------------------

def rank_features_simple(df: pd.DataFrame, target: str, feature_cols: List[str], group_col: Optional[str] = None, random_state: int = 42) -> pd.Series:
    """
    Quick ranking using absolute Pearson correlation (numeric-only).
    """
    y = pd.to_numeric(df[target], errors="coerce")
    out: Dict[str, float] = {}
    for c in feature_cols:
        if c not in df.columns:
            out[c] = 0.0
            continue
        x = pd.to_numeric(df[c], errors="coerce")

        if x.isna().all() or y.isna().all():
            out[c] = 0.0
            continue
        xc = x.fillna(x.median())
        yc = y.fillna(y.median())
        v = float(np.corrcoef(xc, yc)[0, 1])
        out[c] = 0.0 if math.isnan(v) else abs(v)
    return pd.Series(out).sort_values(ascending=False)


# -----------------------------
# Optimisation
# -----------------------------

def _random_points(bounds: Dict[str, Tuple[float, float]], n: int, rng: np.random.Generator) -> np.ndarray:
    keys = list(bounds.keys())
    lo = np.array([bounds[k][0] for k in keys], dtype=float)
    hi = np.array([bounds[k][1] for k in keys], dtype=float)
    u = rng.random((n, len(keys)))
    return lo + u * (hi - lo)


def _vec_to_dict(x: np.ndarray, keys: List[str]) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(keys, x)}


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # Vectorised math.erf (safe: no SciPy required)
    erf_v = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_v(z / np.sqrt(2.0)))


def optimise_surrogate(
    res: TrainResult,
    objective: Callable[[pd.DataFrame], Tuple[float, float]],
    bounds: Dict[str, Tuple[float, float]],
    base_params: Dict[str, float],
    cfg: OptimisationConfig,
    objective_label: str,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> OptimisationResult:
    """
    Optimise objective over selected variables.

    objective(X_df) -> (score, score_uncertainty), larger is better.
    risk_aversion penalises uncertainty internally: score_eff = score - risk_aversion * score_unc
    """
    keys = list(bounds.keys())
    d = len(keys)
    rng = np.random.default_rng(cfg.seed)
    opt_t0 = time.time()
    opt_total = int(max(1, cfg.n_evals))
    _safe_call(progress_cb, {
        "phase": "optimise",
        "event": "start",
        "total": opt_total,
        "done": 0,
        "eta_s": None,
    })


    def eval_x(x_vec: np.ndarray) -> Tuple[float, float, float]:
        params = dict(base_params)
        params.update(_vec_to_dict(x_vec, keys))
        X_df = pd.DataFrame([params])
        score, score_unc = objective(X_df)
        eff = float(score - cfg.risk_aversion * score_unc)
        return float(score), float(score_unc), eff

    method = (cfg.method or "auto").lower()
    if method == "auto":
        method = "bayes" if d <= 8 else "de"

    hist_rows: List[Dict[str, Any]] = []

    # ---------------- Bayes (GP + EI) ----------------
    if method == "bayes":
        if GaussianProcessRegressor is None:
            method = "random"

    if method == "bayes":
        n_init = int(min(25, max(12, 2 * d + 4)))
        n_total = int(max(n_init + 5, cfg.n_evals))

        Xs = _random_points(bounds, n_init, rng)
        y_eff: List[float] = []

        for i in range(n_init):
            s, su, eff = eval_x(Xs[i])
            y_eff.append(eff)
            hist_rows.append({"iter": i + 1, "score": s, "score_unc": su, "score_eff": eff})
            done = int(hist_rows[-1]["iter"])
            elapsed = time.time() - opt_t0
            rate = elapsed / max(1, done)
            eta = rate * max(0, opt_total - done)
            _safe_call(progress_cb, {
                "phase": "optimise",
                "event": "progress",
                "method": method,
                "total": opt_total,
                "done": done,
                "eta_s": float(eta),
                "best_score_eff": float(np.max([r["score_eff"] for r in hist_rows])) if hist_rows else None,
            })


        lo = np.array([bounds[k][0] for k in keys], dtype=float)
        hi = np.array([bounds[k][1] for k in keys], dtype=float)
        X01 = (Xs - lo) / (hi - lo + 1e-12)

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        )

        best_eff = float(np.max(y_eff))
        best_x = Xs[int(np.argmax(y_eff))].copy()
        no_improve = 0

        for it in range(n_init, n_total):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=cfg.seed,
                alpha=1e-6,
                n_restarts_optimizer=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X01, np.asarray(y_eff, dtype=float))

            n_cand = int(min(4000, max(1500, 400 * d)))
            C = rng.random((n_cand, d))
            mu, std = gp.predict(C, return_std=True)
            std = np.maximum(std, 1e-9)

            z = (mu - best_eff) / std
            ei = (mu - best_eff) * _norm_cdf(z) + std * _norm_pdf(z)

            x01_next = C[int(np.argmax(ei))]
            x_next = lo + x01_next * (hi - lo)

            s, su, eff = eval_x(x_next)

            X01 = np.vstack([X01, x01_next])
            y_eff.append(eff)
            hist_rows.append({"iter": it + 1, "score": s, "score_unc": su, "score_eff": eff})
            done = int(hist_rows[-1]["iter"])
            elapsed = time.time() - opt_t0
            rate = elapsed / max(1, done)
            eta = rate * max(0, opt_total - done)
            _safe_call(progress_cb, {
                "phase": "optimise",
                "event": "progress",
                "method": method,
                "total": opt_total,
                "done": done,
                "eta_s": float(eta),
                "best_score_eff": float(np.max([r["score_eff"] for r in hist_rows])) if hist_rows else None,
            })

            if eff > best_eff + 1e-12:
                best_eff = eff
                best_x = x_next.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    break

        best_score, best_unc, _ = eval_x(best_x)
        return OptimisationResult(
            objective_label=objective_label,
            method_used="bayes",
            best_score=float(best_score),
            best_params=_vec_to_dict(best_x, keys),
            history=pd.DataFrame(hist_rows),
        )

    # ---------------- Differential Evolution ----------------
    if method == "de":
        pop = int(max(12, cfg.pop_size or (10 * d)))
        pop = int(min(100, pop))
        max_evals = int(max(pop + 1, cfg.n_evals))

        P = _random_points(bounds, pop, rng)
        effs = np.zeros(pop, dtype=float)
        scores = np.zeros(pop, dtype=float)
        uncs = np.zeros(pop, dtype=float)

        for i in range(pop):
            s, su, eff = eval_x(P[i])
            scores[i], uncs[i], effs[i] = s, su, eff
            hist_rows.append({"iter": i + 1, "score": s, "score_unc": su, "score_eff": eff})
            done = int(hist_rows[-1]["iter"])
            elapsed = time.time() - opt_t0
            rate = elapsed / max(1, done)
            eta = rate * max(0, opt_total - done)
            _safe_call(progress_cb, {
                "phase": "optimise",
                "event": "progress",
                "method": method,
                "total": opt_total,
                "done": done,
                "eta_s": float(eta),
                "best_score_eff": float(np.max([r["score_eff"] for r in hist_rows])) if hist_rows else None,
            })

        best_i = int(np.argmax(effs))
        best_x = P[best_i].copy()
        best_eff = float(effs[best_i])

        F = 0.6
        CR = 0.8
        evals = pop
        no_improve_rounds = 0

        while evals < max_evals:
            improved_this_round = False

            for i in range(pop):
                idxs = [j for j in range(pop) if j != i]
                a, b, c = rng.choice(idxs, size=3, replace=False)
                mutant = P[a] + F * (P[b] - P[c])

                # clip
                for j, k in enumerate(keys):
                    lo_k, hi_k = bounds[k]
                    mutant[j] = float(np.clip(mutant[j], lo_k, hi_k))

                trial = P[i].copy()
                j_rand = int(rng.integers(0, d))
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                s, su, eff = eval_x(trial)
                evals += 1
                hist_rows.append({"iter": evals, "score": s, "score_unc": su, "score_eff": eff})
                done = int(hist_rows[-1]["iter"])
                elapsed = time.time() - opt_t0
                rate = elapsed / max(1, done)
                eta = rate * max(0, opt_total - done)
                _safe_call(progress_cb, {
                    "phase": "optimise",
                    "event": "progress",
                    "method": method,
                    "total": opt_total,
                    "done": done,
                    "eta_s": float(eta),
                    "best_score_eff": float(np.max([r["score_eff"] for r in hist_rows])) if hist_rows else None,
                })

                if eff > effs[i]:
                    P[i] = trial
                    effs[i] = eff
                    scores[i] = s
                    uncs[i] = su

                    if eff > best_eff + 1e-12:
                        best_eff = eff
                        best_x = trial.copy()
                        improved_this_round = True

                if evals >= max_evals:
                    break

            if improved_this_round:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                if no_improve_rounds >= cfg.patience:
                    break

        best_score, best_unc, _ = eval_x(best_x)
        return OptimisationResult(
            objective_label=objective_label,
            method_used="de",
            best_score=float(best_score),
            best_params=_vec_to_dict(best_x, keys),
            history=pd.DataFrame(hist_rows),
        )

    # ---------------- Anneal ----------------
    if method == "anneal":
        max_evals = int(max(50, cfg.n_evals))
        x = _random_points(bounds, 1, rng)[0]
        s, su, eff = eval_x(x)

        best_x = x.copy()
        best_eff = eff

        lo = np.array([bounds[k][0] for k in keys], dtype=float)
        hi = np.array([bounds[k][1] for k in keys], dtype=float)
        span = hi - lo
        step = 0.15 * span

        no_improve = 0
        T0 = 1.0

        for it in range(1, max_evals + 1):
            T = T0 * (0.98 ** it)

            x_prop = x + rng.normal(0, 1, size=d) * step
            x_prop = np.minimum(hi, np.maximum(lo, x_prop))

            s2, su2, eff2 = eval_x(x_prop)
            hist_rows.append({"iter": it, "score": s2, "score_unc": su2, "score_eff": eff2})
            done = int(hist_rows[-1]["iter"])
            elapsed = time.time() - opt_t0
            rate = elapsed / max(1, done)
            eta = rate * max(0, opt_total - done)
            _safe_call(progress_cb, {
                "phase": "optimise",
                "event": "progress",
                "method": method,
                "total": opt_total,
                "done": done,
                "eta_s": float(eta),
                "best_score_eff": float(np.max([r["score_eff"] for r in hist_rows])) if hist_rows else None,
            })

            accept = eff2 >= eff or (rng.random() < math.exp((eff2 - eff) / max(1e-9, T)))
            if accept:
                x, s, su, eff = x_prop, s2, su2, eff2

            if eff > best_eff + 1e-12:
                best_eff = eff
                best_x = x.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    break

        best_score, best_unc, _ = eval_x(best_x)
        return OptimisationResult(
            objective_label=objective_label,
            method_used="anneal",
            best_score=float(best_score),
            best_params=_vec_to_dict(best_x, keys),
            history=pd.DataFrame(hist_rows),
        )

    # ---------------- Random fallback ----------------
    max_evals = int(max(50, cfg.n_evals))
    Xs = _random_points(bounds, max_evals, rng)

    best_x = Xs[0].copy()
    best_eff = -float("inf")
    best_score = 0.0

    for i in range(max_evals):
        s, su, eff = eval_x(Xs[i])
        hist_rows.append({"iter": i + 1, "score": s, "score_unc": su, "score_eff": eff})
        done = int(hist_rows[-1]["iter"])
        elapsed = time.time() - opt_t0
        rate = elapsed / max(1, done)
        eta = rate * max(0, opt_total - done)
        _safe_call(progress_cb, {
            "phase": "optimise",
            "event": "progress",
            "method": method,
            "total": opt_total,
            "done": done,
            "eta_s": float(eta),
            "best_score_eff": float(np.max([r["score_eff"] for r in hist_rows])) if hist_rows else None,
        })
        if eff > best_eff:
            best_eff = eff
            best_x = Xs[i].copy()
            best_score = s

    return OptimisationResult(
        objective_label=objective_label,
        method_used="random",
        best_score=float(best_score),
        best_params=_vec_to_dict(best_x, keys),
        history=pd.DataFrame(hist_rows),
    )

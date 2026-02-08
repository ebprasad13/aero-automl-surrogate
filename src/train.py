# src/train.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class AutoMLConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    cv_repeats: int = 10
    bootstrap_size: int = 50
    random_state: int = 42
    small_data_mode: bool = True


@dataclass
class TrainResult:
    targets: List[str]
    feature_cols: List[str]
    group_col: Optional[str]

    split_name: str
    n_used: int
    n_train: int
    n_test: int
    n_features: int
    dropped_id_like: List[str]

    x_mean: np.ndarray
    x_std: np.ndarray
    ood_p90: float

    baseline: Dict[str, float]
    fmin: Dict[str, float]
    fmax: Dict[str, float]
    baseline_y: Dict[str, float]

    best_model_by_target: Dict[str, str]
    best_overall_model: str

    std_p90: Dict[str, float]
    err_p90: Dict[str, float]

    test_metrics: pd.DataFrame
    baseline_comparison: Optional[Dict[str, float]]

    point_models: Dict[str, object]
    bootstrap_models: Dict[str, List[object]]

    train_frame: pd.DataFrame
    feature_rank: Optional[List[str]]
    fit_plots: Dict[str, plt.Figure]


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def recommend_settings(
    n_rows: int,
    n_features: int,
    n_targets: int,
    grouped: bool,
    n_groups: Optional[int],
) -> Dict:
    tiny = n_rows < 60
    small = n_rows < 120

    if tiny:
        cv_folds, cv_repeats, bootstrap, test_size, small_data = 3, 20, 120, 0.25, True
    elif small:
        cv_folds, cv_repeats, bootstrap, test_size, small_data = 5, 12, 80, 0.20, True
    elif n_rows < 600:
        cv_folds, cv_repeats, bootstrap, test_size, small_data = 5, 6, 50, 0.20, False
    else:
        cv_folds, cv_repeats, bootstrap, test_size, small_data = 5, 3, 30, 0.20, False

    if grouped and n_groups is not None and n_groups < 8:
        cv_repeats = max(cv_repeats, 15)
        test_size = min(max(test_size, 0.25), 0.35)
        small_data = True

    return dict(
        test_size=float(test_size),
        cv_folds=int(cv_folds),
        cv_repeats=int(cv_repeats),
        bootstrap_size=int(bootstrap),
        random_state=42,
        small_data_mode=bool(small_data),
    )


def rank_features_simple(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], group_col: Optional[str] = None) -> List[str]:
    d = df.copy()
    scores = {}
    for f in feature_cols:
        if f not in d.columns or not pd.api.types.is_numeric_dtype(d[f]):
            continue
        vals = []
        for t in target_cols:
            if t not in d.columns:
                continue
            try:
                c = d[[f, t]].dropna().corr().iloc[0, 1]
                if np.isfinite(c):
                    vals.append(abs(float(c)))
            except Exception:
                pass
        scores[f] = float(np.mean(vals)) if vals else 0.0
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


def _candidates(cfg: AutoMLConfig) -> Dict[str, List[object]]:
    sd = cfg.small_data_mode

    ridge_alphas = [0.1, 1.0, 10.0]
    en_alphas = [0.001, 0.01, 0.1]
    en_l1 = [0.1, 0.5, 0.9]
    svr_C = [1.0, 10.0]
    svr_eps = [0.01, 0.1]

    rf_leaf = 3 if sd else 1
    rf_depths = [6, 10] if sd else [None, 12]
    et_leaf = 3 if sd else 1
    et_depths = [6, 10] if sd else [None, 14]

    out: Dict[str, List[object]] = {}

    out["ridge"] = [
        Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=a, random_state=cfg.random_state))])
        for a in ridge_alphas
    ]
    out["elasticnet"] = [
        Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=a, l1_ratio=l1, random_state=cfg.random_state, max_iter=20000))])
        for a in en_alphas
        for l1 in en_l1
    ]
    out["svr_rbf"] = [
        Pipeline([("scaler", StandardScaler()), ("model", SVR(C=C, epsilon=eps, kernel="rbf"))])
        for C in svr_C
        for eps in svr_eps
    ]
    out["random_forest"] = [
        RandomForestRegressor(
            n_estimators=400 if not sd else 300,
            max_depth=depth,
            min_samples_leaf=rf_leaf,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        for depth in rf_depths
    ]
    out["extra_trees"] = [
        ExtraTreesRegressor(
            n_estimators=600 if not sd else 400,
            max_depth=depth,
            min_samples_leaf=et_leaf,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        for depth in et_depths
    ]
    out["hist_gbdt"] = [
        HistGradientBoostingRegressor(
            learning_rate=lr,
            max_depth=depth,
            max_leaf_nodes=leaf,
            random_state=cfg.random_state,
        )
        for lr in ([0.05, 0.1] if not sd else [0.05])
        for depth in ([3, None] if not sd else [3])
        for leaf in ([31, 63] if not sd else [31])
    ]

    if sd:
        out["random_forest"] = out["random_forest"][:1]
        out["extra_trees"] = out["extra_trees"][:1]
        out["hist_gbdt"] = out["hist_gbdt"][:1]

    return out


def _iter_cv_splits(X: pd.DataFrame, y: pd.Series, groups: Optional[np.ndarray], cfg: AutoMLConfig):
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=cfg.cv_repeats, test_size=0.2, random_state=cfg.random_state)
        yield from gss.split(X, y, groups=groups)
    else:
        rkf = RepeatedKFold(
            n_splits=max(2, cfg.cv_folds),
            n_repeats=max(1, cfg.cv_repeats),
            random_state=cfg.random_state,
        )
        yield from rkf.split(X, y)


def _cv_score_model(model, X: pd.DataFrame, y: pd.Series, groups: Optional[np.ndarray], cfg: AutoMLConfig):
    rmses, maes, r2s = [], [], []
    residuals = []
    for tr, va in _iter_cv_splits(X, y, groups, cfg):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        m = clone(model)
        m.fit(X_tr, y_tr)
        p = m.predict(X_va)
        maes.append(mean_absolute_error(y_va, p))
        rmses.append(_rmse(y_va, p))
        r2s.append(r2_score(y_va, p))
        residuals.append(y_va.to_numpy() - np.asarray(p).ravel())
    resid = np.concatenate(residuals) if residuals else np.array([], dtype=float)
    return (
        {
            "MAE_mean": float(np.mean(maes)),
            "RMSE_mean": float(np.mean(rmses)),
            "R2_mean": float(np.mean(r2s)),
        },
        resid,
    )


def _fit_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> plt.Figure:
    fig = plt.figure(figsize=(5.4, 4.0), dpi=140)
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=18, alpha=0.85)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _bootstrap_models(model, X: pd.DataFrame, y: pd.Series, B: int, rng: np.random.Generator):
    n = len(X)
    models, boot_idx = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        m = clone(model)
        m.fit(X.iloc[idx], y.iloc[idx])
        models.append(m)
        boot_idx.append(idx)
    return models, boot_idx


def _oob_std_p90(boot_models: List[object], boot_idx: List[np.ndarray], X: pd.DataFrame) -> float:
    n = len(X)
    preds_per_i: List[List[float]] = [[] for _ in range(n)]

    for m, idx in zip(boot_models, boot_idx):
        inbag = np.zeros(n, dtype=bool)
        inbag[idx] = True
        oob = np.where(~inbag)[0]
        if len(oob) == 0:
            continue
        p = np.asarray(m.predict(X.iloc[oob])).ravel()
        for j, pj in zip(oob, p):
            preds_per_i[j].append(float(pj))

    stds = [float(np.std(v, ddof=0)) for v in preds_per_i if len(v) >= 3]
    if not stds:
        P = np.stack([np.asarray(m.predict(X)).ravel() for m in boot_models], axis=0)
        stds = list(np.std(P, axis=0, ddof=0))
    return float(np.quantile(stds, 0.90))


def _ood_calibrate(X_train: pd.DataFrame):
    Xn = X_train.to_numpy(dtype=float)
    mu = Xn.mean(axis=0)
    sd = Xn.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    z = (Xn - mu) / sd
    dist = np.sqrt(np.sum(z ** 2, axis=1))
    thr = float(np.quantile(dist, 0.90))
    return mu, sd, thr


def train_automl(df: pd.DataFrame, target_cols: List[str], group_col: Optional[str], cfg: AutoMLConfig) -> TrainResult:
    df2 = df.copy()
    df2 = df2.dropna(subset=target_cols).copy()

    numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    feature_cols = [c for c in numeric_cols if c not in target_cols and c != group_col]

    # Auto-drop ID-like near-unique columns (prevents “Run” becoming a feature)
    dropped_id_like: List[str] = []
    for c in list(feature_cols):
        name = c.strip().lower()
        if name in {"run", "case", "sample", "geometryid", "geom_id", "id"} or name.endswith("id"):
            nunique = df2[c].nunique(dropna=True)
            if nunique / max(1, len(df2)) > 0.95:
                dropped_id_like.append(c)
    if dropped_id_like:
        feature_cols = [c for c in feature_cols if c not in dropped_id_like]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns left after excluding targets/group/ID columns.")

    keep = feature_cols + target_cols + ([group_col] if group_col else [])
    df2 = df2[keep].dropna().copy()

    n_used = len(df2)

    # Shared split for all targets
    if group_col:
        groups_all = df2[group_col].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        tr_idx, te_idx = next(gss.split(df2[feature_cols], df2[target_cols[0]], groups=groups_all))
        split_name = "group shuffle split"
    else:
        idx = np.arange(n_used)
        tr_idx, te_idx = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.random_state)
        split_name = "random hold-out split"

    train_df = df2.iloc[tr_idx].copy()
    test_df = df2.iloc[te_idx].copy()

    n_train, n_test = len(train_df), len(test_df)

    # Slider ranges MUST match file: use df2 min/max (not train split)
    baseline = {c: float(df2[c].mean()) for c in feature_cols}
    fmin = {c: float(df2[c].min()) for c in feature_cols}
    fmax = {c: float(df2[c].max()) for c in feature_cols}

    x_mean, x_std, ood_p90 = _ood_calibrate(train_df[feature_cols])

    candidates = _candidates(cfg)
    rng = np.random.default_rng(cfg.random_state)

    best_model_by_target: Dict[str, str] = {}
    point_models: Dict[str, object] = {}
    bootstrap_models: Dict[str, List[object]] = {}
    std_p90: Dict[str, float] = {}
    err_p90: Dict[str, float] = {}
    baseline_y: Dict[str, float] = {}
    fit_plots: Dict[str, plt.Figure] = {}

    family_votes: Dict[str, int] = {k: 0 for k in candidates.keys()}
    baseline_rmses, best_rmses = [], []
    metrics_rows = []

    for t in target_cols:
        Xtr, ytr = train_df[feature_cols], train_df[t]
        Xte, yte = test_df[feature_cols], test_df[t]

        # Baseline mean predictor
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(Xtr, ytr)
        base_rmse = _rmse(yte, dummy.predict(Xte))
        baseline_rmses.append(base_rmse)

        groups_tr = train_df[group_col].to_numpy() if group_col else None

        best_name, best_model, best_rmse = None, None, np.inf
        best_resid = None

        for fam, models in candidates.items():
            for m in models:
                scores, resid = _cv_score_model(m, Xtr, ytr, groups_tr, cfg)
                if scores["RMSE_mean"] < best_rmse:
                    best_rmse = scores["RMSE_mean"]
                    best_name, best_model = fam, m
                    best_resid = resid

        assert best_name is not None and best_model is not None

        family_votes[best_name] = family_votes.get(best_name, 0) + 1
        best_model_by_target[t] = best_name

        err_p90[t] = float(np.quantile(np.abs(best_resid), 0.90)) if best_resid is not None and best_resid.size else float("nan")

        pm = clone(best_model)
        pm.fit(Xtr, ytr)
        point_models[t] = pm

        X_base = pd.DataFrame([baseline], columns=feature_cols)
        baseline_y[t] = float(pm.predict(X_base)[0])

        boots, boot_idx = _bootstrap_models(best_model, Xtr, ytr, B=int(cfg.bootstrap_size), rng=rng)
        bootstrap_models[t] = boots
        std_p90[t] = _oob_std_p90(boots, boot_idx, Xtr)

        pred = pm.predict(Xte)
        mae = float(mean_absolute_error(yte, pred))
        rmse = _rmse(yte, pred)
        r2 = float(r2_score(yte, pred))

        best_rmses.append(rmse)

        metrics_rows.append(
            {"target": t, "best_model": best_name, "MAE": mae, "RMSE": rmse, "R2": r2, "STD_p90": std_p90[t], "ERR_p90": err_p90[t]}
        )

        fit_plots[t] = _fit_plot(yte.to_numpy(), np.asarray(pred).ravel(), f"{t}: predicted vs actual")

    best_overall = sorted(family_votes.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    feature_rank = rank_features_simple(train_df, feature_cols, target_cols, group_col=group_col)

    baseline_comparison = None
    if baseline_rmses and best_rmses:
        b = float(np.mean(baseline_rmses))
        m = float(np.mean(best_rmses))
        baseline_comparison = {"baseline_rmse": b, "best_rmse": m, "improvement_frac": float((b - m) / (abs(b) + 1e-12))}

    return TrainResult(
        targets=list(target_cols),
        feature_cols=list(feature_cols),
        group_col=group_col,
        split_name=split_name,
        n_used=int(n_used),
        n_train=int(n_train),
        n_test=int(n_test),
        n_features=int(len(feature_cols)),
        dropped_id_like=list(dropped_id_like),
        x_mean=x_mean,
        x_std=x_std,
        ood_p90=float(ood_p90),
        baseline=baseline,
        fmin=fmin,
        fmax=fmax,
        baseline_y=baseline_y,
        best_model_by_target=best_model_by_target,
        best_overall_model=best_overall,
        std_p90=std_p90,
        err_p90=err_p90,
        test_metrics=pd.DataFrame(metrics_rows),
        baseline_comparison=baseline_comparison,
        point_models=point_models,
        bootstrap_models=bootstrap_models,
        train_frame=train_df,
        feature_rank=feature_rank,
        fit_plots=fit_plots,
    )


def predict_with_uncertainty(res: TrainResult, X: pd.DataFrame):
    Xn = X.to_numpy(dtype=float)
    z = (Xn - res.x_mean) / res.x_std
    ood_dist = float(np.sqrt(np.sum(z ** 2)))

    y_mean: Dict[str, float] = {}
    y_std: Dict[str, float] = {}

    for t in res.targets:
        boots = res.bootstrap_models[t]
        P = np.stack([np.asarray(m.predict(X)).ravel() for m in boots], axis=0)
        y_mean[t] = float(np.mean(P, axis=0)[0])
        y_std[t] = float(np.std(P, axis=0, ddof=0)[0])

    extra = {"ood_distance": ood_dist, "ood_p90": float(res.ood_p90)}
    return y_mean, y_std, extra

# app.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time

from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager as fm

def _setup_plot_fonts() -> None:
    font_dir = Path(__file__).parent / "assets" / "fonts"
    for fp in font_dir.glob("*.ttf"):
        fm.fontManager.addfont(str(fp))

    mpl.rcParams.update({
        "font.family": "Fustat",
        "font.weight": 400,
        "axes.titleweight": 400,
        "axes.labelweight": 400,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

_setup_plot_fonts()

# ---- Matplotlib theme (dark, modern) ----
PLOT_BG   = "#0B0F14"   # page/card background vibe
AX_BG     = "#0F172A"   # slightly lighter than page
TEXT      = "#E5E7EB"
MUTED     = "#94A3B8"
GRID      = "#334155"
ACCENT    = "#FF4B4B"   # your Streamlit red
POINTS    = "#60A5FA"   # modern blue

def _style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

    ax.grid(True, color=GRID, alpha=0.25, linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_alpha(0.35)

from src.train import (
    AutoMLConfig,
    OptimisationConfig,
    TrainResult,
    OptimisationResult,
    recommend_settings,
    recommend_optimisation,
    rank_features_simple,
    train_automl,
    predict_mean,
    predict_with_uncertainty,
    optimise_surrogate,
)

APP_TITLE = "Aero Surrogate: Prediction & Optimization"

def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    rem = s - 60 * m
    return f"{m}m {rem:.0f}s"

# -----------------------------
# UI helpers
# -----------------------------

def _inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fustat:wght@300;400&display=swap');

/* Fustat for main app content */
[data-testid="stAppViewContainer"],
.stMarkdown,
.stText,
button > div > div,
div.stButton > button > div > div,
h1, h2, h3, h4, h5, h6,
label,
input,
textarea,
p {
  font-family: "Fustat", system-ui, -apple-system, "Segoe UI", Arial, sans-serif !important;
  font-weight: 300 !important;
}

/* Icons ONLY - ultra-specific, overrides above */
[data-testid="stIcon"] span,
div[data-testid="stIcon"] span,
button [data-testid="stIcon"] span,
span[class*="material-symbols"],
span[class*="material-icons"] {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-weight: 400 !important;
  font-style: normal !important;
  letter-spacing: normal !important;
  -webkit-font-feature-settings: "liga" !important;
}

h1, h2, h3, h4, h5, h6 { font-weight: 400 !important; }
.small-note { font-size: 0.9rem; opacity: 0.8; }
                
/* Force ALL Streamlit markdown blocks to use Fustat */
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] * {
  font-family: "Fustat", system-ui, -apple-system, "Segoe UI", Arial, sans-serif !important;
}

/* Make bold markdown look like a “heading” */
div[data-testid="stMarkdownContainer"] strong {
  font-weight: 500 !important;  /* try 300 if you want it lighter */
}
                
/* Keep bold markdown from triggering fallback weights */
.stMarkdown strong, .stMarkdown b,
[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] b {
  font-family: "Fustat", system-ui, -apple-system, "Segoe UI", Arial, sans-serif !important;
  font-weight: 400 !important;
}
                
                /* BaseWeb Select / Multiselect: selected “pill” tags */
div[data-baseweb="tag"],
div[data-baseweb="tag"] * ,
div[data-baseweb="tag"] span,
div[data-baseweb="tag"] div {
  font-family: "Fustat", system-ui, -apple-system, "Segoe UI", Arial, sans-serif !important;
  font-weight: 300 !important; /* or 400 */
}

/* The Select input area (covers placeholder + selected text in the field) */
div[data-baseweb="select"] *,
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
  font-family: "Fustat", system-ui, -apple-system, "Segoe UI", Arial, sans-serif !important;
}

/* Dropdown menu options */
ul[role="listbox"] *,
div[role="listbox"] * {
  font-family: "Fustat", system-ui, -apple-system, "Segoe UI", Arial, sans-serif !important;
}



</style>
""", unsafe_allow_html=True)

def _detect_group_candidates(df: pd.DataFrame) -> List[Optional[str]]:
    cands: List[Optional[str]] = [None]
    for c in df.columns:
        if c.lower() in ("run", "group", "case", "id"):
            cands.append(c)
    return cands


def _auto_target_suggestions(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    preferred: List[str] = []
    for key in ("cl", "cd", "cm", "lift", "drag", "eff", "efficiency"):
        for c in numeric:
            if key in c.lower():
                preferred.append(c)
    out: List[str] = []
    for c in preferred + numeric:
        if c not in out:
            out.append(c)
    return out[:6]


def _apply_pending_slider_updates(res: TrainResult) -> None:
    pending = st.session_state.get("pending_slider_update")
    if not pending:
        return
    for f, v in pending.items():
        k = f"sl_{f}"
        st.session_state[k] = float(v)
    st.session_state["pending_slider_update"] = None


def _edge_warning(value: float, lo: float, hi: float) -> bool:
    if hi <= lo:
        return False
    span = hi - lo
    return (value - lo) / span < 0.02 or (hi - value) / span < 0.02


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    fig = plt.figure(facecolor="none")
    ax = fig.add_subplot(111)
    _style_ax(ax)

    ax.scatter(y_true, y_pred, s=22, alpha=0.80, color=POINTS)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.6, color=ACCENT, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return fig



def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    resid = y_pred - y_true
    fig = plt.figure(facecolor="none")
    ax = fig.add_subplot(111)
    _style_ax(ax)

    ax.scatter(y_pred, resid, s=22, alpha=0.80, color=POINTS)
    ax.axhline(0.0, linestyle="--", linewidth=1.6, color=ACCENT, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Pred - Actual)")
    return fig



def _build_objective(
    res: TrainResult,
    objective_mode: str,
    target: Optional[str],
    sign_mode: str,
    need_uncertainty: bool,
):
    """
    Returns (label, objective_fn) where objective_fn(df)->(score, score_unc).
    If need_uncertainty is False, objective_fn returns score_unc=0 and uses predict_mean for speed.
    """
    objective_mode = objective_mode.lower()

    def _predict(df: pd.DataFrame):
        if need_uncertainty:
            mean, std, _info = predict_with_uncertainty(res, df)
            return mean.iloc[0].to_dict(), std.iloc[0].to_dict()
        else:
            mean = predict_mean(res, df)
            return mean.iloc[0].to_dict(), {t: 0.0 for t in res.targets}

    # try identify CL / CD for efficiency
    cl_name = next((c for c in res.targets if c.lower() == "cl" or "cl" in c.lower()), None)
    cd_name = next((c for c in res.targets if c.lower() == "cd" or "cd" in c.lower()), None)

    if objective_mode == "efficiency" and cl_name and cd_name:
        # Determine sign for efficiency
        if sign_mode == "auto":
            base_mu = predict_mean(res, pd.DataFrame([res.baseline_params])).iloc[0].to_dict()
            med_cl = float(base_mu.get(cl_name, 1.0))
            sgn = -1.0 if med_cl < 0 else 1.0
        elif sign_mode == "use -cl/cd":
            sgn = -1.0
        else:
            sgn = 1.0

        label = f"Efficiency ({'-CL/CD' if sgn < 0 else 'CL/CD'})"

        def obj(df: pd.DataFrame):
            mu, sig = _predict(df)
            cl = float(mu.get(cl_name, 0.0))
            cd = float(mu.get(cd_name, 1e-6))
            cl_s = float(sig.get(cl_name, 0.0))
            cd_s = float(sig.get(cd_name, 0.0))
            cd = cd if abs(cd) > 1e-9 else 1e-9
            score = sgn * cl / cd
            if not need_uncertainty:
                return float(score), 0.0
            # delta-method for ratio uncertainty
            score_unc = abs(score) * math.sqrt(
                (cl_s / max(1e-9, abs(cl))) ** 2 + (cd_s / max(1e-9, abs(cd))) ** 2
            )
            return float(score), float(score_unc)

        return label, obj

    if objective_mode == "maximise" and target:
        label = f"Maximise {target}"

        def obj(df: pd.DataFrame):
            mu, sig = _predict(df)
            return float(mu[target]), float(sig.get(target, 0.0))

        return label, obj

    if objective_mode == "minimise" and target:
        label = f"Minimise {target}"

        def obj(df: pd.DataFrame):
            mu, sig = _predict(df)
            return float(-mu[target]), float(sig.get(target, 0.0))

        return label, obj

    t0 = res.targets[0]
    label = f"Maximise {t0}"

    def obj(df: pd.DataFrame):
        mu, sig = _predict(df)
        return float(mu[t0]), float(sig.get(t0, 0.0))

    return label, obj


# -----------------------------
# App
# -----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
_inject_css()

st.title(APP_TITLE)
st.caption("Train a surrogate model on your aero dataset, estimate uncertainty, and optimise for best performance. Built by B.Edupuganti")

# Clean, stable sidebar (no controls jumping around)
with st.sidebar:
    st.header("Help")
    st.markdown("**Defaults:** Bayesian (B) for ≤8 variables, DE for >8.")
    with st.expander("What is Small data mode?"):
        st.write(
            "Auto-enabled when the dataset is small or has low samples-per-feature. "
            "It prefers more conservative modelling settings to reduce overfitting."
        )
    with st.expander("What does uncertainty mean?"):
        st.write(
            "Uncertainty comes from a bootstrap ensemble (many retrains on resampled data). "
            "If your input is far from training data, uncertainty is inflated automatically."
        )
    with st.expander("Bayesian vs DE vs Anneal"):
        st.write(
            "Bayesian is usually fastest and strongest in low dimensions (≤8). "
            "DE is more robust for many variables. Anneal is a simpler alternative sometimes useful between the two."
        )

st.subheader("1) Load data")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded)
# --- Reset model state when a new dataset is uploaded (prevents stale columns like 'Alpha') ---
data_sig = (
    getattr(uploaded, "name", "uploaded"),
    int(df_raw.shape[0]),
    int(df_raw.shape[1]),
    tuple(df_raw.columns),
)
prev_sig = st.session_state.get("_data_sig")

if prev_sig != data_sig:
    st.session_state["_data_sig"] = data_sig

    # Clear training/optimisation state
    st.session_state.pop("res", None)
    st.session_state.pop("opt_res", None)
    st.session_state.pop("pending_slider_update", None)
    st.session_state["trained"] = False

    # Clear old slider widget states to avoid stale keys/values
    for k in list(st.session_state.keys()):
        if k.startswith("sl_"):
            del st.session_state[k]
# -------------------------------------------------------------------
st.write("Preview", df_raw.head())

st.subheader("2) Select targets and optional group split")
group_candidates = _detect_group_candidates(df_raw)
default_group = "Run" if "Run" in df_raw.columns else (group_candidates[1] if len(group_candidates) > 1 else None)

c1, c2 = st.columns([1, 1])
with c1:
    group_col = st.selectbox(
        "Group column (optional)",
        options=group_candidates,
        index=group_candidates.index(default_group) if default_group in group_candidates else 0,
        help="If you have Run/Case grouping, use it to prevent leakage (keeps groups together in splits).",
    )

exclude_cols = [c for c in [group_col] if c]
target_suggestions = _auto_target_suggestions(df_raw, exclude=exclude_cols)

with c2:
    numeric_targets = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c]) and c != group_col]
    targets = st.multiselect(
        "Target columns (outputs to predict)",
        options=numeric_targets,
        default=[c for c in target_suggestions if c in numeric_targets][:2],
    )

if not targets:
    st.warning("Select at least one target column.")
    st.stop()

# Determine numeric feature cols (excluding targets/group)
drop_cols = set(targets)
if group_col:
    drop_cols.add(group_col)

feature_cols = [c for c in df_raw.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_raw[c])]
n_rows = int(len(df_raw))
n_features = int(len(feature_cols))
n_groups = int(df_raw[group_col].nunique()) if group_col else None

cfg_default = recommend_settings(
    n_rows=n_rows,
    n_features=n_features,
    grouped=bool(group_col),
    n_groups=n_groups,
    n_targets=len(targets),
)

with st.expander("Advanced training settings", expanded=False):
    primary = st.get_option("theme.primaryColor") or "#FF4B4B"
    st.markdown(
        f"<div style='color:{primary}; font-weight:400; margin-bottom:8px;'>"
        "These defaults are adjusted automatically based on your dataset. "
        "If you’d like different behaviour, you can adjust them manually."
        "</div>",
        unsafe_allow_html=True,
    )
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        small_data_mode = st.checkbox("Small data mode", value=bool(cfg_default.small_data_mode))
        cv_folds = st.slider(
            "CV folds",
            2, 10, int(cfg_default.cv_folds), 1,
            help=(
                "How many chunks the data is split into to check the model behaves consistently. "
                "More folds gives a steadier estimate but takes longer. "
                "Recommended: 5 for most datasets; 3 if the dataset is very small."
            ),
        )
    with cc2:
        cv_repeats = st.slider(
            "CV repeats",
            1, 30, int(cfg_default.cv_repeats), 1,
            help=(
                "Repeats cross-validation with different random splits. "
                "This helps when you’ve got limited data, but increases training time. "
                "Recommended: 1–3 normally; 3–8 for small datasets."
            ),
        )
        test_size = st.slider(
            "Test split size",
            0.05, 0.40, float(cfg_default.test_size), 0.01,
            help=(
                "How much of the data is held back and never shown during training, "
                "so we can check real-world performance. "
                "Recommended: 0.20 (20%). Use 0.10–0.15 if data is limited."
            ),
        )
    with cc3:
        bootstrap_size = st.slider(
            "Bootstrap ensemble size",
            5, 200, int(cfg_default.bootstrap_size), 5,
            help=(
                "How many re-trained models we build to estimate uncertainty. "
                "Higher is usually a bit more stable, but slower. "
                "Recommended: 30–80 (small data: 50–120)."
            ),
        )
    

cfg = AutoMLConfig(
    cv_folds=int(cv_folds),
    cv_repeats=int(cv_repeats),
    test_size=float(test_size),
    bootstrap_size=int(bootstrap_size),
    small_data_mode=bool(small_data_mode),
    random_state=42,
    n_jobs=-1,
)

st.subheader("3) Train")
train_clicked = st.button("Train model", type="primary")

if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.res = None

if train_clicked:
    prog = st.progress(0, text="Training…")
    eta_ph = st.empty()

    t0 = time.time()

    def _fmt_eta(seconds: float | None) -> str:
        if seconds is None:
            return "—"
        s = max(0.0, float(seconds))
        if s < 60:
            return f"{s:.1f}s"
        m = int(s // 60)
        rem = s - 60 * m
        return f"{m}m {rem:.0f}s"

    def _train_cb(p: dict) -> None:
        if p.get("phase") != "train":
            return
        total = int(p.get("total") or 1)
        done = int(p.get("done") or 0)
        frac = min(1.0, max(0.0, done / max(1, total)))
        target = p.get("target")
        model = p.get("model")
        extra = f" — {target} / {model}" if target and model else ""
        prog.progress(frac, text=f"Training… {done}/{total}{extra}")
        eta_ph.caption(f"ETC: **{_fmt_eta(p.get('eta_s'))}**")

    # Train (now with callback)
    res = train_automl(
        df=df_raw,
        target_cols=targets,
        group_col=group_col,
        cfg=cfg,
        progress_cb=_train_cb,
    )

    prog.progress(1.0, text="Training complete ✅")
    eta_ph.empty()

    st.session_state.res = res
    st.session_state.trained = True
    st.session_state["pending_slider_update"] = dict(res.baseline_params)
    st.rerun()


if not st.session_state.trained or st.session_state.res is None:
    st.info("Train the model to enable prediction and optimisation.")
    st.stop()

res: TrainResult = st.session_state.res
_apply_pending_slider_updates(res)

tab_pred, tab_opt, tab_diag = st.tabs(["Predict", "Optimise", "Diagnostics"])

# -----------------------------
# Predict tab
# -----------------------------
with tab_pred:
    left, right = st.columns([1.1, 1.2], gap="large")

    with left:
        st.subheader("Inputs")
        X_dict: Dict[str, float] = {}
        edge_flags: List[str] = []

        for f in res.feature_cols:
            lo, hi = res.feature_ranges[f]
            key = f"sl_{f}"
            if key not in st.session_state:
                st.session_state[key] = float(res.baseline_params.get(f, lo))

            val = st.slider(
                f,
                min_value=float(lo),
                max_value=float(hi),
                key=key,  # slider will use st.session_state[key]
            )

            X_dict[f] = float(val)
            if _edge_warning(val, lo, hi):
                edge_flags.append(f)

        cA, cB = st.columns(2)
        with cA:
            if st.button("Reset to baseline"):
                st.session_state["pending_slider_update"] = dict(res.baseline_params)
                st.rerun()
        with cB:
            if st.button("Reset to mid-range"):
                mids = {f: float((res.feature_ranges[f][0] + res.feature_ranges[f][1]) / 2.0) for f in res.feature_cols}
                st.session_state["pending_slider_update"] = mids
                st.rerun()

        if edge_flags:
            st.warning(
                "Inputs near training range edge: " + ", ".join(edge_flags) +
                ". Predictions may be less reliable at extremes."
            )

    with right:
        st.subheader("Predictions")
        X = pd.DataFrame([X_dict])
        mean, std, info = predict_with_uncertainty(res, X)

        pred = mean.iloc[0].to_dict()
        unc = std.iloc[0].to_dict()

        rows = []
        for t in res.targets:
            mu = float(pred[t])
            sig = float(unc[t])
            rows.append(
                {
                    "Target": t,
                    "Prediction": mu,
                    "Uncertainty (1σ)": sig,
                    "Approx. 95% range": f"[{mu - 1.96*sig:.4g}, {mu + 1.96*sig:.4g}]",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with st.expander("What does uncertainty mean?"):
            st.write(
                "Uncertainty is estimated from a bootstrap ensemble. "
                "It increases automatically when your input is far from the training set."
            )
            st.dataframe(info, use_container_width=True)

# -----------------------------
# Optimise tab
# -----------------------------
with tab_opt:
    st.subheader("Optimisation")

    # Default k=6 variables (ranked by correlation vs first target)
    rank_target = targets[0]
    feat_rank = rank_features_simple(df_raw, target=rank_target, feature_cols=res.feature_cols, group_col=group_col)
    topk_default = list(feat_rank.index[:6])

    opt_left, opt_right = st.columns([1.05, 0.95], gap="large")

    with opt_left:
        st.markdown("**Variables to optimise (default k = 6)**")
        opt_vars = st.multiselect(
            "Optimisation variables",
            options=res.feature_cols,
            default=topk_default,
            help="The optimiser can change only these. Others stay fixed at baseline.",
        )
        if not opt_vars:
            st.warning("Select at least one optimisation variable.")
            st.stop()

        st.markdown("**Objective**")
        mode = st.selectbox("Objective type", ["Efficiency", "Maximise", "Minimise"], index=0)

        if mode != "Efficiency":
            tgt = st.selectbox("Target", options=res.targets, index=0)
            sign_mode = "auto"
        else:
            tgt = None
            sign_mode = st.selectbox("Efficiency sign", ["Auto", "Use CL/CD", "Use -CL/CD"], index=0).lower()

        st.markdown("**Algorithm**")
        method_choice = st.selectbox(
            "Optimisation algorithm",
            ["Bayesian (B)", "Auto (recommended)", "Differential Evolution (DE)", "Anneal", "Random"],
            index=1,  # default should be B
            help="B is fastest for ≤8 vars; DE is more robust for >8 vars.",
        )
        method_map = {
            "bayesian (b)": "bayes",
            "auto (recommended)": "auto",
            "differential evolution (de)": "de",
            "anneal": "anneal",
            "random": "random",
        }
        method_user = method_map[method_choice.lower()]

        # Auto-switch to DE if >8 vars, unless user forced something else
        if len(opt_vars) > 8 and method_user in ("auto", "bayes"):
            st.info("More than 8 variables selected → default recommendation switches to DE for robustness.")
            if method_user == "auto":
                method_user = "de"

        opt_rec = recommend_optimisation(n_vars=len(opt_vars), n_rows=res.n_rows, method_user=method_user)

        with st.expander("Advanced optimisation settings", expanded=False):
            st.markdown(
            "<div style='color:#FF4B4B; font-weight:400; margin-top:0px; margin-bottom:8px'>"
            "Note: These values are adjusted automatically based on your dataset and how many variables you’re optimising. "
            "You’re welcome to adjust these if you’d like."
            "</div>",
            unsafe_allow_html=True,
        )
            n_evals = st.slider(
                "Evaluation budget",
                40, 1200, int(opt_rec.n_evals), 10,
                help=(
                    "How many candidate designs the optimiser will test. "
                    "Higher usually finds better results but takes longer. "
                    "Recommended: ~80–250 for ≤8 variables; ~200–600 for >8 variables."
                ),
            )
            pop_size = st.slider(
                "Population size (DE only)",
                12, 120, int(opt_rec.pop_size or 40), 2,
                help=(
                    "Only used by Differential Evolution. "
                    "Bigger populations explore more broadly but slow things down. "
                    "Recommended: 20–60 (start at ~40)."
                ),
            )
            patience = st.slider(
                "Early stop patience",
                5, 80, int(opt_rec.patience), 1,
                help=(
                    "Stops optimisation early if there’s no meaningful improvement for this many steps. "
                    "Lower = faster, higher = more thorough. "
                    "Recommended: 10–25 (small data: 15–30)."
                ),
            )
            risk = st.slider(
                "Risk aversion (penalise uncertainty)",
                0.0, 3.0, 0.5, 0.1,
                help=(
                    "Penalises uncertain predictions, nudging the optimiser towards ‘safer’ regions of the data. "
                    "0 = fastest (mean-only). 0.5–1.5 is a sensible range. "
                    "Go higher only if you really want conservative solutions."
                ),
            )


        opt_cfg = OptimisationConfig(
            method=opt_rec.method if method_user == "auto" else method_user,
            n_evals=int(n_evals),
            pop_size=int(pop_size),
            patience=int(patience),
            risk_aversion=float(risk),
            seed=42,
        )

        run_opt = st.button("Run optimisation", type="primary")

    with opt_right:
        st.markdown("**Recommendation summary**")
        st.write(
            f"Vars: **{len(opt_vars)}** · Recommended: **{opt_rec.method.upper()}** · Budget: **{opt_rec.n_evals}**"
        )
        st.caption("Tip: set risk aversion = 0 for speed; add risk aversion only when you need robustness.")

        if "opt_res" not in st.session_state:
            st.session_state.opt_res = None

        if run_opt:
            bounds = {f: res.feature_ranges[f] for f in opt_vars}
            base = dict(res.baseline_params)

            need_unc = opt_cfg.risk_aversion > 0.0
            label, obj_fn = _build_objective(
                res=res,
                objective_mode=mode,
                target=tgt,
                sign_mode=sign_mode,
                need_uncertainty=need_unc,
            )

            with st.spinner("Optimising..."):
                opt_prog = st.progress(0, text="Optimising…")
                opt_eta = st.empty()

                def _opt_cb(p: dict) -> None:
                    if p.get("phase") != "optimise":
                        return
                    total = int(p.get("total") or 1)
                    done = int(p.get("done") or 0)
                    frac = min(1.0, max(0.0, done / max(1, total)))
                    eta_s = p.get("eta_s")
                    method = p.get("method") or ""
                    opt_prog.progress(frac, text=f"Optimising ({method})… {done}/{total}")
                    opt_eta.caption(f"ETC: **{_fmt_eta(eta_s)}**")

                opt_res = optimise_surrogate(
                    res=res,
                    objective=obj_fn,
                    bounds=bounds,
                    base_params=base,
                    cfg=opt_cfg,
                    objective_label=label,
                    progress_cb=_opt_cb,
                )


                opt_prog.progress(1.0, text="Optimisation complete ✅")
                opt_eta.empty()

            st.session_state.opt_res = opt_res
            st.success(f"Optimisation complete. Best score: {opt_res.best_score:.5g}")

        opt_res: Optional[OptimisationResult] = st.session_state.get("opt_res")
        if opt_res:
            st.subheader("Best configuration")
            st.write(f"Objective: **{opt_res.objective_label}** · Method: **{opt_res.method_used.upper()}**")

            best_tbl = pd.DataFrame([opt_res.best_params]).T.reset_index()
            best_tbl.columns = ["Variable", "Value"]
            st.dataframe(best_tbl, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Apply optimum to sliders"):
                    st.session_state["pending_slider_update"] = dict(opt_res.best_params)
                    st.rerun()
            with c2:
                if st.button("Reset sliders to baseline"):
                    st.session_state["pending_slider_update"] = dict(res.baseline_params)
                    st.rerun()

            with st.expander("Optimisation trace (last 50)", expanded=False):
                st.dataframe(opt_res.history.tail(50), use_container_width=True)

# -----------------------------
# Diagnostics tab
# -----------------------------
with tab_diag:
    st.subheader("Training summary")
    st.markdown(f"- Rows: **{res.n_rows}** · Features: **{res.n_features}** · Targets: **{len(res.targets)}**")
    if res.group_col:
        st.markdown(f"- Group split: **{res.group_col}** (groups: {res.n_groups})")

    st.markdown("**Test performance**")
    st.dataframe(res.test_table, use_container_width=True)

    with st.expander("Cross-validation (top rows)", expanded=False):
        st.dataframe(res.cv_table.head(30), use_container_width=True)

    st.markdown("**Fit plots (held-out test split)**")
    for t in res.targets:
        a = f"{t}_actual"
        p = f"{t}_pred"
        if a not in res.test_predictions.columns or p not in res.test_predictions.columns:
            continue
        dfp = res.test_predictions[[a, p]].dropna()
        if len(dfp) < 5:
            continue
        y_true = dfp[a].to_numpy(dtype=float)
        y_pred = dfp[p].to_numpy(dtype=float)

        #st.write(f"**{t}**")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(_plot_pred_vs_actual(y_true, y_pred, f"{t}: Predicted vs Actual"))
        with c2:
            st.pyplot(_plot_residuals(y_true, y_pred, f"{t}: Residuals"))

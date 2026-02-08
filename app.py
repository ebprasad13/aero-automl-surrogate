# app.py
from __future__ import annotations

import hashlib
import io
from dataclasses import asdict
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.train import (
    AutoMLConfig,
    TrainResult,
    predict_with_uncertainty,
    rank_features_simple,
    recommend_settings,
    train_automl,
)

# ============================================================
# Page + styling
# ============================================================
st.set_page_config(page_title="AutoML aero surrogate builder — demo", layout="wide")

st.markdown(
    """
<style>
/* Keep Streamlit top bar, but blend it into the page */
header[data-testid="stHeader"] { background: #0b0c10; }
.stApp { background: #0b0c10; color: #EAEAF2; }
section[data-testid="stSidebar"] { background: #0b0c10; border-right: 1px solid rgba(255,255,255,0.08); }

.block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }

.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 14px;
}

.small { color: rgba(234,234,242,0.72); font-size: 0.92rem; }
hr { border-color: rgba(255,255,255,0.10); }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Session state
# ============================================================
if "trained" not in st.session_state:
    st.session_state.trained = False
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "data_sig" not in st.session_state:
    st.session_state.data_sig = None
if "cfg_sig" not in st.session_state:
    st.session_state.cfg_sig = None
if "offsets" not in st.session_state:
    st.session_state.offsets = {}  # feature -> offset (float)


# ============================================================
# Helpers
# ============================================================
def file_signature(uploaded) -> str:
    b = uploaded.getvalue()
    return hashlib.md5(b).hexdigest()


def safe_read_csv(uploaded) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        txt = uploaded.getvalue().decode("utf-8", errors="ignore")
        sep = "," if txt.count(",") >= txt.count(";") else ";"
        return pd.read_csv(io.StringIO(txt), sep=sep)


def human_int(n: int) -> str:
    return f"{n:,}"


def reset_to_baseline(features: List[str]):
    # IMPORTANT: reset BOTH the stored offsets and the widget keys
    for f in features:
        st.session_state.offsets[f] = 0.0
        st.session_state[f"sl_{f}"] = 0.0


def init_offsets(features: List[str]):
    for f in features:
        st.session_state.offsets.setdefault(f, 0.0)
        st.session_state.setdefault(f"sl_{f}", st.session_state.offsets[f])


# ============================================================
# Header
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("AutoML aero surrogate builder — demo")
st.markdown(
    """
Upload a CSV, pick target columns, and the app will:

- train and compare several regression models,
- select the best option using repeated cross-validation,
- estimate uncertainty (bootstrap + error bands),
- and give you a slider UI to explore predictions.
""".strip()
)
st.caption("Built by ebprasad · Streamlit demo")
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Step 1 — upload + format guidance
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("1) Upload a dataset (CSV)")

col_u1, col_u2 = st.columns([1.1, 0.9])
with col_u1:
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    st.markdown('<div class="small">Tip: one row per design/run.</div>', unsafe_allow_html=True)
with col_u2:
    with st.expander("CSV format checklist (read this once)", expanded=False):
        st.markdown(
            """
**Minimum requirements**
- Header row (column names)
- Numeric feature columns (inputs)
- At least **one numeric target** column (e.g. `cd`, `cl`)

**Optional**
- A *group* column (e.g. `Run`, `Case`, `GeometryID`) to prevent leakage across near-duplicates.

**Good practice**
- Don’t mix units inside a column
- Missing targets → that row is dropped automatically
"""
        )
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded:
    st.stop()

df_raw = safe_read_csv(uploaded)
data_sig = file_signature(uploaded)

# ============================================================
# Step 2 — target + group selection
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("2) Choose targets and (optional) a group column")

st.markdown(
    f"<div class='small'>Loaded <b>{human_int(len(df_raw))}</b> rows and <b>{human_int(df_raw.shape[1])}</b> columns.</div>",
    unsafe_allow_html=True,
)

numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
non_numeric = [c for c in df_raw.columns if c not in numeric_cols]
if non_numeric:
    st.info(
        "Non-numeric columns detected. IDs are fine, but non-numeric features are ignored in the slider explorer."
    )

targets = st.multiselect(
    "Target column(s) (what you want to predict)",
    options=numeric_cols,
    default=[c for c in ["cd", "cl", "Cd", "Cl"] if c in numeric_cols][:2],
    help="Pick one or more numeric columns to predict.",
)

group_col = st.selectbox(
    "Group column (optional)",
    options=["(none)"] + list(df_raw.columns),
    index=(["(none)"] + list(df_raw.columns)).index("Run") if "Run" in df_raw.columns else 0,
    help="Use if multiple rows belong to the same run/geometry. Keeps groups together in splitting.",
)
if group_col == "(none)":
    group_col = None

if not targets:
    st.warning("Pick at least one target column to proceed.")
    st.stop()

# Feature candidates excluding targets + group
feature_candidates = [c for c in df_raw.columns if c not in targets and c != group_col]
numeric_features = [c for c in feature_candidates if c in numeric_cols]
if len(numeric_features) == 0:
    st.error("No numeric input columns remain after excluding targets/group. Add at least one numeric feature column.")
    st.stop()

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Step 3 — training settings (recommended defaults)
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("3) Training settings")

n_rows = len(df_raw.dropna(subset=targets))
n_features = len(numeric_features)
n_targets = len(targets)
n_groups = df_raw[group_col].nunique() if group_col else None

rec = recommend_settings(
    n_rows=n_rows,
    n_features=n_features,
    n_targets=n_targets,
    grouped=group_col is not None,
    n_groups=n_groups,
)

use_recommended = st.checkbox(
    "Use recommended settings (based on this dataset)",
    value=True,
    help="Keeps training stable across different dataset sizes. You can still override advanced settings below.",
)

cfg_default = AutoMLConfig(**rec) if use_recommended else AutoMLConfig()

with st.expander("Advanced settings (optional)", expanded=False):
    st.markdown(
        """
If you're unsure, leave these on the recommended defaults.

- **Hold-out test fraction**: data kept totally unseen until final scoring.
- **CV folds**: how many chunks the training set is split into for validation.
- **CV repeats**: how many times we reshuffle/re-split CV to reduce “lucky split” effects.
- **Bootstrap size**: number of retrains used to estimate prediction uncertainty (std).
"""
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider(
            "Hold-out test fraction",
            0.1,
            0.4,
            float(cfg_default.test_size),
            0.05,
            help="Final test slice reserved for reporting MAE/RMSE/R².",
        )
        small_data_mode = st.checkbox(
            "Small data mode",
            value=bool(cfg_default.small_data_mode),
            help="More conservative: favours simpler models and reduces tree complexity to avoid overfitting.",
        )
    with c2:
        cv_folds = st.slider(
            "CV folds",
            2,
            10,
            int(cfg_default.cv_folds),
            1,
            help="More folds → more reliable estimate, slower training.",
        )
        cv_repeats = st.slider(
            "CV repeats",
            1,
            30,
            int(cfg_default.cv_repeats),
            1,
            help="Repeats CV with different splits. Very helpful for small datasets.",
        )
    with c3:
        bootstrap_size = st.slider(
            "Bootstrap ensemble size",
            10,
            200,
            int(cfg_default.bootstrap_size),
            10,
            help="More bootstraps → smoother uncertainty, slower training.",
        )
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            value=int(cfg_default.random_state),
            help="Makes training repeatable.",
        )

    st.markdown(
        f"<div class='small'>Recommended for this file: CV {rec['cv_folds']} folds × {rec['cv_repeats']} repeats, "
        f"bootstrap {rec['bootstrap_size']}, test fraction {rec['test_size']:.2f}.</div>",
        unsafe_allow_html=True,
    )

cfg = AutoMLConfig(
    test_size=float(test_size),
    cv_folds=int(cv_folds),
    cv_repeats=int(cv_repeats),
    bootstrap_size=int(bootstrap_size),
    random_state=int(random_state),
    small_data_mode=bool(small_data_mode),
)
cfg_sig = hashlib.md5(str(asdict(cfg)).encode("utf-8")).hexdigest()

train_clicked = st.button(
    "Train surrogate",
    type="primary",
    help="Train, validate, select the best model, calibrate uncertainty, then enable the slider explorer.",
)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Training trigger (always retrain if clicked)
# ============================================================
if train_clicked:
    with st.spinner("Training and validating (repeated CV + hold-out test)…"):
        res = train_automl(
            df=df_raw,
            target_cols=targets,
            group_col=group_col,
            cfg=cfg,
        )
    st.session_state.trained = True
    st.session_state.train_result = res
    st.session_state.data_sig = data_sig
    st.session_state.cfg_sig = cfg_sig

if st.session_state.trained and (st.session_state.data_sig != data_sig or st.session_state.cfg_sig != cfg_sig):
    st.warning("Your data/settings changed since the last training. Click **Train surrogate** again to refresh the model.")

if not st.session_state.trained:
    st.stop()

res: TrainResult = st.session_state.train_result

# ============================================================
# Training summary
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Training summary")

st.success(f"Done. Best overall model family (most targets): **{res.best_overall_model}**")

if getattr(res, "dropped_id_like", None):
    st.info("Dropped likely ID columns from features: " + ", ".join(res.dropped_id_like))

perf = res.test_metrics.copy()
perf = perf[["target", "best_model", "MAE", "RMSE", "R2", "STD_p90", "ERR_p90"]].rename(
    columns={"STD_p90": "Std p90", "ERR_p90": "|error| p90"}
)

st.dataframe(
    perf.style.format({"MAE": "{:.6f}", "RMSE": "{:.6f}", "R2": "{:.4f}", "Std p90": "{:.3e}", "|error| p90": "{:.6g}"}),
    width="stretch",
)

st.markdown(
    f"<div class='small'>Split: <b>{res.split_name}</b>. Rows used after dropping missing targets: "
    f"<b>{human_int(res.n_used)}</b> (train {human_int(res.n_train)} / test {human_int(res.n_test)}).</div>",
    unsafe_allow_html=True,
)

# Data adequacy warnings
ratio = res.n_train / max(1, res.n_features)
if ratio < 5:
    st.warning(
        "Few training rows per feature → unstable generalisation likely. "
        "Small data mode + fewer features usually helps."
    )
if res.baseline_comparison is not None and res.baseline_comparison["improvement_frac"] < 0.10:
    st.warning(
        "Model is not much better than predicting the mean (weak signal). Treat predictions as indicative."
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Fit plots
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Fit plots (held-out test set)")

cols = st.columns(min(2, len(res.targets)))
for i, t in enumerate(res.targets):
    c = cols[i % len(cols)]
    fig = res.fit_plots.get(t)
    if fig is not None:
        c.pyplot(fig, clear_figure=False)

st.caption("Predicted vs actual on the held-out test split (45° reference line).")
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Explorer
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Surrogate explorer")
st.markdown("<div class='small'>Adjust inputs in the sidebar, then click <b>Compute</b>.</div>", unsafe_allow_html=True)

feature_cols = res.feature_cols
init_offsets(feature_cols)

ranked = res.feature_rank or rank_features_simple(res.train_frame, feature_cols, res.targets, group_col=group_col)
top_k = ranked[:8]
others = [f for f in feature_cols if f not in top_k]

with st.sidebar:
    st.header("Controls")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Reset to baseline", help="Sets all sliders back to dataset-mean inputs."):
            reset_to_baseline(feature_cols)
            st.rerun()
    with b2:
        compute = st.button("Compute", type="primary")

    st.markdown("---")
    st.subheader("Key parameters")
    st.caption("Top 8 ranked by a simple correlation score.")

    def slider_for_feature(feat: str):
        base = float(res.baseline[feat])
        fmin = float(res.fmin[feat])
        fmax = float(res.fmax[feat])

        left = fmin - base
        right = fmax - base

        if np.isclose(left, right):
            st.caption(f"{feat}: constant in dataset (no slider).")
            st.session_state.offsets[feat] = 0.0
            st.session_state[f"sl_{feat}"] = 0.0
            return

        k = f"sl_{feat}"
        if k not in st.session_state:
            st.session_state[k] = float(st.session_state.offsets.get(feat, 0.0))

        off = st.slider(
            feat,
            float(left),
            float(right),
            float(st.session_state[k]),
            step=float((right - left) / 200.0) if right > left else 0.001,
            key=k,
        )
        st.session_state.offsets[feat] = float(off)
        st.caption(f"Current: **{base + off:.6g}** (baseline {base:.6g})")

    for f in top_k:
        slider_for_feature(f)

    with st.expander("All parameters", expanded=False):
        for f in others:
            slider_for_feature(f)

x_row = {f: float(res.baseline[f]) + float(st.session_state.offsets.get(f, 0.0)) for f in feature_cols}
X_user = pd.DataFrame([x_row], columns=feature_cols)

# --- Edge-of-data flag (near min/max) ---
EDGE_FRAC = 0.03  # 3% of feature range
edge_hits = []

for f in res.feature_cols:
    v = float(X_user.iloc[0][f])
    vmin = float(res.fmin[f])
    vmax = float(res.fmax[f])
    span = vmax - vmin
    if span <= 0:
        continue
    eps = EDGE_FRAC * span
    if v <= vmin + eps or v >= vmax - eps:
        edge_hits.append(f)

edge_flag = len(edge_hits) > 0

if not compute:
    st.info("Use the sidebar sliders, then click **Compute**.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

pred_out = predict_with_uncertainty(res, X_user)
# Backwards/forwards compatible unpacking:
# - newer train.py returns (mean_dict, std_dict, extra_dict)
# - older versions may return (mean_dict, std_dict) only
if isinstance(pred_out, tuple):
    y_mean = pred_out[0]
    y_std = pred_out[1] if len(pred_out) > 1 else {}
    extra = pred_out[2] if len(pred_out) > 2 else {}
elif isinstance(pred_out, dict) and "mean" in pred_out:
    y_mean = pred_out["mean"]
    y_std = pred_out.get("std", {})
    extra = pred_out.get("extra", {})
else:
    raise ValueError("Unexpected return from predict_with_uncertainty().")

st.markdown("<hr/>", unsafe_allow_html=True)

# Predictions
if len(res.targets) <= 4:
    c11, c12 = st.columns(2)
    c21, c22 = st.columns(2)
    cells = [c11, c12, c21, c22]
    for i, t in enumerate(res.targets):
        base = float(res.baseline_y[t])
        pred = float(y_mean[t])
        pct = (pred - base) / (abs(base) + 1e-12) * 100.0
        cells[i].metric(t, f"{pred:.6g}", delta=f"{pct:+.2f}%")
    st.caption("Delta is vs baseline prediction (baseline = model at dataset-mean inputs).")
else:
    out_tbl = pd.DataFrame({"target": res.targets, "predicted": [y_mean[t] for t in res.targets]})
    st.dataframe(out_tbl, width="stretch")

# Uncertainty
st.subheader("Uncertainty / reliability")
with st.expander("What do these mean?", expanded=False):
    st.markdown(
        """
- **Bootstrap std**: variability across bootstrap retrains (model disagreement).
- **Std p90**: calibrated “high uncertainty” threshold (top 10% region).
- **|error| p90**: typical worst-case absolute error estimated from CV residuals.
- **OOD distance**: how far the input is from training coverage (simple z-score norm).
"""
    )

ood = float(extra.get("ood_distance", np.nan))
ood_thr = float(extra.get("ood_p90", np.nan))
ood_flag = (not np.isnan(ood_thr)) and (ood > ood_thr)

rows = []
any_warn = False
for t in res.targets:
    std_p90 = float(res.std_p90.get(t, np.nan))
    err_p90 = float(res.err_p90.get(t, np.nan))
    std = float(y_std.get(t, np.nan))
    low_std = (not np.isnan(std_p90)) and (std > std_p90)
    ood_flag = (not np.isnan(ood_thr)) and (ood > ood_thr)  # you already compute ood/ood_thr
    low = low_std or ood_flag or edge_flag
    any_warn = any_warn or low
    if edge_flag:
     st.warning(
        "⚠️ You’re right near the edge of the training data for: "
        + ", ".join(edge_hits[:6]) + ("…" if len(edge_hits) > 6 else "")
        + ". Predictions can look confident (low std) but still be biased at the boundary."
    )
    rows.append(
        {"target": t, "bootstrap std": std, "std p90": std_p90, "|error| p90": err_p90, "flag": "⚠️ Low" if low else "✅ High"}
    )

df_u = pd.DataFrame(rows)
st.dataframe(
    df_u.style.format({"bootstrap std": "{:.3e}", "std p90": "{:.3e}", "|error| p90": "{:.6g}"}),
    width="stretch",
)

if ood_flag:
    st.warning(f"⚠️ Input looks far from training coverage (OOD distance {ood:.3f} > p90 {ood_thr:.3f}).")
else:
    st.success("✅ Input sits within typical training coverage (based on a simple distance check).")

if any_warn:
    st.warning("At least one target is in a higher-uncertainty region. Treat as indicative and verify if it matters.")
else:
    st.success("All targets are below the std p90 threshold — lower-uncertainty region.")

st.markdown("</div>", unsafe_allow_html=True)

"""
Bank Customer Churn Risk Assessor — Streamlit App
==================================================
Self-contained inference app. No dependency on helpers.py training utilities.

Requirements (runtime only):
    streamlit, pandas, numpy, joblib, shap, matplotlib, plotly,
    scikit-learn, catboost (or whichever model won the tournament)
"""

import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match the notebook exactly
# ─────────────────────────────────────────────────────────────────────────────
SEED      = 21
MODEL_DIR = "models"

NUM_FEATURES         = ["CreditScore", "Age", "Tenure", "Balance",
                        "NumOfProducts", "EstimatedSalary",
                        "BalanceSalaryRatio", "ProductsPerYear"]
CAT_FEATURES         = ["Geography", "Gender", "AgeGroup"]
PASSTHROUGH_FEATURES = ["HasCrCard", "IsActiveMember", "IsActive_by_CreditCard"]

# ─────────────────────────────────────────────────────────────────────────────
# SHAP → Retention Action Map
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_ACTIONS = {
    "IsActiveMember": (
        "Activation outreach",
        "This customer's inactivity is the primary churn driver. "
        "Trigger a personalised call or app re-engagement campaign within 7 days."
    ),
    "IsActive_by_CreditCard": (
        "Activation outreach",
        "Low engagement combined with credit card inactivity signals disengagement. "
        "A targeted usage incentive (e.g. cashback activation) could re-anchor this customer."
    ),
    "NumOfProducts": (
        "Product relationship review",
        "Number of products is a key churn signal here. "
        "If the customer holds only one product, prioritise a cross-sell conversation. "
        "If they hold 3+, schedule a service satisfaction check — friction may be the issue."
    ),
    "ProductsPerYear": (
        "Relationship depth review",
        "Low products-per-year ratio indicates shallow engagement over time. "
        "Review the customer's product history and identify gaps in the relationship."
    ),
    "Balance": (
        "Balance re-engagement",
        "Account balance is driving this prediction. A zero or declining balance "
        "suggests the customer is moving funds elsewhere. "
        "Consider a high-yield savings offer or a dedicated relationship manager call."
    ),
    "BalanceSalaryRatio": (
        "Financial health review",
        "The balance-to-salary ratio is unusually low for this customer's income profile. "
        "This may indicate they are banking primarily elsewhere. "
        "A personalised savings or investment product could deepen the relationship."
    ),
    "CreditScore": (
        "Credit health outreach",
        "Credit score is contributing to churn risk. "
        "If the score is low, proactive credit counselling or a credit-builder product "
        "can improve loyalty. If high, ensure premium products are being offered."
    ),
    "Tenure": (
        "Loyalty recognition",
        "Tenure is influencing churn. New customers (<2 years) need onboarding reinforcement. "
        "Long-tenured customers at risk deserve a loyalty reward or relationship review call "
        "to acknowledge their history with the bank."
    ),
    "Age": (
        "Segment-appropriate outreach",
        "Age is a significant driver. Older customers (45+) respond well to dedicated "
        "relationship managers and personalised service. Younger customers respond better "
        "to digital engagement and product flexibility."
    ),
    "AgeGroup": (
        "Segment-appropriate outreach",
        "Customer age group is influencing this prediction. Review the product and service "
        "offering against what is typical for this segment — mismatched offerings drive churn."
    ),
    "Geography_Germany": (
        "Regional retention action",
        "German geography is a structural churn signal in this dataset. "
        "Investigate local competitor activity and ensure regional pricing and "
        "product offerings are competitive."
    ),
    "Geography_Spain": (
        "Regional engagement check",
        "Geography is influencing this prediction. Review regional service quality "
        "and ensure the customer is aware of locally available products."
    ),
    "Gender_Male": (
        "Personalised engagement",
        "Gender is contributing to the model's prediction in combination with other factors. "
        "Ensure communications and product recommendations are appropriately tailored."
    ),
    "HasCrCard": (
        "Credit card engagement",
        "Credit card ownership is influencing churn. "
        "If the customer has a card but rarely uses it, a rewards activation campaign may help. "
        "If they don't hold one, a card offer could deepen the relationship."
    ),
    "EstimatedSalary": (
        "Income-appropriate product review",
        "Salary band is a factor in this prediction. Ensure the customer is on a product "
        "tier appropriate to their income — under-served high-earners churn to premium competitors."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Risk Assessor", page_icon="", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --c-bg-subtle:      #f8fafc; --c-bg-card:        #f8fafc;
    --c-border:         #e2e8f0; --c-border-subtle:  #f1f5f9;
    --c-text-primary:   #0f172a; --c-text-body:      #334155;
    --c-text-muted:     #64748b; --c-text-faint:     #94a3b8;
    --c-danger-bg:      #fef2f2; --c-danger-border:  #fecaca;
    --c-danger-accent:  #dc2626; --c-danger-text:    #991b1b;
    --c-success-bg:     #f0fdf4; --c-success-border: #bbf7d0;
    --c-success-accent: #16a34a; --c-success-text:   #166534;
    --c-warn-text:      #b45309;
}
@media (prefers-color-scheme: dark) {
    :root {
        --c-bg-subtle:      #1e293b; --c-bg-card:        #1e293b;
        --c-border:         #334155; --c-border-subtle:  #1e293b;
        --c-text-primary:   #f1f5f9; --c-text-body:      #cbd5e1;
        --c-text-muted:     #94a3b8; --c-text-faint:     #64748b;
        --c-danger-bg:      #2d1515; --c-danger-border:  #7f1d1d;
        --c-danger-accent:  #f87171; --c-danger-text:    #fca5a5;
        --c-success-bg:     #14291e; --c-success-border: #14532d;
        --c-success-accent: #4ade80; --c-success-text:   #86efac;
        --c-warn-text:      #fbbf24;
    }
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.block-container            { padding-top: 2rem; padding-bottom: 3rem; }

.app-header { border-bottom:1px solid var(--c-border); padding-bottom:1.5rem; margin-bottom:2rem; text-align:center; }
.app-header h1 { font-size:2.2rem; font-weight:600; color:var(--c-text-primary); letter-spacing:-0.03em; margin:0 0 0.4rem 0; }
.app-header .subtitle { font-size:0.78rem; color:var(--c-text-faint); font-family:'DM Mono',monospace; display:flex; align-items:center; justify-content:center; gap:0.75rem; flex-wrap:wrap; }
.app-header .subtitle-dot { width:3px; height:3px; background:var(--c-text-faint); border-radius:50%; display:inline-block; }

.section-label { font-size:0.7rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:var(--c-text-faint); margin-bottom:0.75rem; }
.form-group-label { font-size:0.65rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:var(--c-text-faint); border-bottom:1px solid var(--c-border-subtle); padding-bottom:0.4rem; margin:1.25rem 0 0.75rem 0; }

.verdict-high { background:var(--c-danger-bg); border:1px solid var(--c-danger-border); border-left:4px solid var(--c-danger-accent); border-radius:6px; padding:1rem 1.25rem; color:var(--c-danger-text); font-weight:500; font-size:0.9rem; }
.verdict-low  { background:var(--c-success-bg); border:1px solid var(--c-success-border); border-left:4px solid var(--c-success-accent); border-radius:6px; padding:1rem 1.25rem; color:var(--c-success-text); font-weight:500; font-size:0.9rem; }

.stat-card { background:var(--c-bg-card); border:1px solid var(--c-border); border-radius:8px; padding:1rem 1.25rem; height:100%; box-sizing:border-box; display:flex; flex-direction:column; justify-content:center; }
.stat-card .label { font-size:0.7rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:var(--c-text-faint); margin-bottom:0.35rem; }
.stat-card .value { font-size:1.6rem; font-weight:600; color:var(--c-text-primary); letter-spacing:-0.02em; line-height:1; }
.stat-card .sub   { font-size:0.75rem; color:var(--c-text-muted); margin-top:0.3rem; font-family:'DM Mono',monospace; }

.stat-card-hero { background:var(--c-text-primary); border:1px solid var(--c-border); border-radius:8px; padding:1rem 1.25rem; height:100%; box-sizing:border-box; display:flex; flex-direction:column; justify-content:center; }
.stat-card-hero .label { font-size:0.7rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:var(--c-text-faint); margin-bottom:0.35rem; }
.stat-card-hero .value { font-size:1.6rem; font-weight:600; color:var(--c-bg-subtle); letter-spacing:-0.02em; line-height:1; }

.card-row { display:flex; gap:1rem; align-items:stretch; margin-bottom:1rem; }
.card-row .stat-card, .card-row .stat-card-hero { flex:1; }

.conf-strong   { color:var(--c-text-primary); font-weight:600; }
.conf-moderate { color:var(--c-text-primary); font-weight:600; }
.conf-border   { color:var(--c-warn-text);    font-weight:600; }

.section-block { background:var(--c-bg-card); border:1px solid var(--c-border); border-radius:10px; padding:1.25rem 1.5rem; margin-bottom:1rem; }

.results-start     { border-top:3px solid var(--c-danger-accent);  padding-top:1.5rem; margin-top:0.5rem; }
.results-start-low { border-top:3px solid var(--c-success-accent); padding-top:1.5rem; margin-top:0.5rem; }

.factor-row { display:flex; align-items:center; gap:0.75rem; padding:0.65rem 0; border-bottom:1px solid var(--c-border-subtle); font-size:0.85rem; }
.factor-row:last-child { border-bottom:none; }
.factor-rank  { font-family:'DM Mono',monospace; font-size:0.7rem; color:var(--c-text-faint); width:1.2rem; flex-shrink:0; }
.factor-name  { font-weight:500; color:var(--c-text-body); flex:1; }
.factor-shap  { font-family:'DM Mono',monospace; font-size:0.72rem; color:var(--c-text-faint); flex-shrink:0; }
.factor-badge-up   { font-size:0.65rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; background:var(--c-danger-bg);  color:var(--c-danger-text);  border:1px solid var(--c-danger-border);  border-radius:4px; padding:0.15rem 0.5rem; flex-shrink:0; }
.factor-badge-down { font-size:0.65rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; background:var(--c-success-bg); color:var(--c-success-text); border:1px solid var(--c-success-border); border-radius:4px; padding:0.15rem 0.5rem; flex-shrink:0; }

.rec-item { padding:0.75rem 0; border-bottom:1px solid var(--c-border-subtle); font-size:0.85rem; color:var(--c-text-body); line-height:1.6; }
.rec-item:last-child { border-bottom:none; }
.rec-item strong { color:var(--c-text-primary); }

.placeholder-card { background:var(--c-bg-subtle); border:1px dashed var(--c-border); border-radius:10px; padding:3rem 2rem; text-align:center; }
.placeholder-card .ph-title { font-size:1.1rem; font-weight:600; color:var(--c-text-muted); }
.placeholder-card p { font-size:0.9rem; margin:0.5rem 0 0 0; color:var(--c-text-muted); }

.batch-sub       { font-size:0.8rem; color:var(--c-text-muted); margin-bottom:1rem; }
.summary-caption { font-size:0.75rem; color:var(--c-text-faint); margin-bottom:0.5rem; }
hr { border:none; border-top:1px solid var(--c-border); margin:1.75rem 0; }

[data-testid="stFormSubmitButton"] > button {
    background:#0f172a !important; color:#ffffff !important; border:none !important;
    border-radius:6px !important; font-weight:500 !important; letter-spacing:0.02em !important;
    padding:0.6rem 1.25rem !important; width:100% !important; transition:background 0.15s ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover { background:#1e293b !important; }
@media (prefers-color-scheme: dark) {
    [data-testid="stFormSubmitButton"] > button { background:#f1f5f9 !important; color:#0f172a !important; }
    [data-testid="stFormSubmitButton"] > button:hover { background:#e2e8f0 !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_threshold():
    pattern = os.path.join(MODEL_DIR, "*_final_pipeline.joblib")
    matches = glob.glob(pattern)
    if not matches:
        st.error(f"No pipeline found in `{MODEL_DIR}/`. Run the notebook first.")
        st.stop()
    pipeline_path       = matches[0]
    model_name          = os.path.basename(pipeline_path).replace("_final_pipeline.joblib", "")
    calibrated_pipeline = joblib.load(pipeline_path)
    threshold_path      = os.path.join(MODEL_DIR, f"{model_name}_threshold.joblib")
    threshold           = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.50
    inner_pipeline      = calibrated_pipeline.estimator
    return calibrated_pipeline, inner_pipeline, float(threshold), model_name

calibrated_pipeline, inner_pipeline, THRESHOLD, MODEL_NAME = load_model_and_threshold()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: SHAP explanation
# ─────────────────────────────────────────────────────────────────────────────
def get_shap_explanation(input_df: pd.DataFrame):
    """Run SHAP on a single customer row through the inner (uncalibrated) pipeline."""
    engineer     = inner_pipeline.named_steps["engineer"]
    preprocessor = inner_pipeline.named_steps["preprocessor"]
    classifier   = inner_pipeline.named_steps["classifier"]
    X_eng        = engineer.transform(input_df)
    X_trans      = preprocessor.transform(X_eng)
    ohe_names    = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
    feat_names   = NUM_FEATURES + list(ohe_names) + PASSTHROUGH_FEATURES
    X_df         = pd.DataFrame(X_trans, columns=feat_names)

    if MODEL_NAME == "Logistic Regression":
        explainer = shap.KernelExplainer(classifier.predict_proba, shap.kmeans(X_df, k=10))
    else:
        explainer = shap.TreeExplainer(classifier)

    sv = explainer.shap_values(X_df)
    if isinstance(sv, list):
        vals, base = sv[1][0], float(np.atleast_1d(explainer.expected_value)[1])
    elif sv.ndim == 3:
        vals, base = sv[0, :, 1], float(np.atleast_1d(explainer.expected_value)[1])
    else:
        vals, base = sv[0], float(np.atleast_1d(explainer.expected_value)[0])

    return shap.Explanation(values=vals, base_values=base,
                            data=X_df.values[0], feature_names=feat_names)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Confidence + risk labels
# ─────────────────────────────────────────────────────────────────────────────
def compute_labels(proba: float):
    margin = abs(proba - THRESHOLD)
    if margin < 0.10:
        conf_label, conf_class = "Borderline", "conf-border"
    elif margin < 0.25:
        conf_label, conf_class = "Moderate",   "conf-moderate"
    else:
        conf_label, conf_class = "Strong",     "conf-strong"
    risk_label = "High" if proba >= THRESHOLD else ("Elevated" if margin <= 0.15 else "Low")
    return conf_label, conf_class, risk_label

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Render SHAP-driven recommendations
# ─────────────────────────────────────────────────────────────────────────────
def render_recommendations(explanation, proba: float):
    if proba < THRESHOLD:
        st.markdown(
            '<div class="verdict-low" style="font-size:0.85rem;">'
            'No immediate retention action required. Continue standard engagement.</div>',
            unsafe_allow_html=True)
        return
    if explanation is None:
        st.caption("Recommendations unavailable — SHAP explanation required.")
        return
    try:
        feat_names    = np.array(explanation.feature_names)
        feat_vals     = explanation.values
        positive_mask = feat_vals > 0
        if positive_mask.sum() == 0:
            positive_mask = np.ones(len(feat_vals), dtype=bool)
        pos_idx    = np.where(positive_mask)[0]
        sorted_pos = pos_idx[np.argsort(feat_vals[pos_idx])[::-1]]

        recommendations, seen = [], set()
        for fi in sorted_pos:
            fname  = feat_names[fi]
            action = FEATURE_ACTIONS.get(fname) or next(
                (v for k, v in FEATURE_ACTIONS.items() if fname.startswith(k)), None)
            if not action:
                continue
            title, text = action
            if title in seen:
                continue
            seen.add(title)
            recommendations.append((title, text, float(feat_vals[fi])))
            if len(recommendations) == 3:
                break

        if not recommendations:
            recommendations = [("General retention",
                "Flag for proactive outreach by the retention team within the next billing cycle.", 0.0)]

        html = "".join(f"""
        <div class="rec-item">
            <strong>{t}</strong>
            <span class="factor-shap" style="margin-left:0.5rem;">SHAP {s:+.3f}</span>
            <br>{txt}
        </div>""" for t, txt, s in recommendations)
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        st.caption("Recommendations unavailable.")

# ─────────────────────────────────────────────────────────────────────────────
# Core: render_prediction
# Shared by Single Customer tab and Batch row detail.
# ─────────────────────────────────────────────────────────────────────────────
def render_prediction(input_df: pd.DataFrame, proba: float, show_shap: bool = True):
    """
    Render the full prediction output for one customer row.

    Parameters
    ----------
    input_df  : Single-row raw feature DataFrame.
    proba     : Calibrated churn probability (0–1).
    show_shap : If True, renders the SHAP waterfall plot.
                Pass False for batch row detail to keep it fast.
    """
    proba_pct                       = proba * 100
    conf_label, conf_class, risk_label = compute_labels(proba)

    # Verdict
    accent_cls = "results-start" if proba >= THRESHOLD else "results-start-low"
    st.markdown(f'<div class="{accent_cls}">', unsafe_allow_html=True)
    if proba >= THRESHOLD:
        st.markdown('<div class="verdict-high">HIGH RISK — This customer is likely to churn.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-low">LOW RISK — This customer is likely to be retained.</div>',
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Churn Probability</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba_pct,
        number={"suffix": "%", "font": {"size": 36, "family": "DM Sans", "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%",
                     "tickcolor": "#64748b", "tickfont": {"size": 11, "color": "#64748b"}},
            "bar": {"color": "#94a3b8", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0,  40], "color": "#16a34a"},
                      {"range": [40, 60], "color": "#d97706"},
                      {"range": [60, 100],"color": "#dc2626"}],
            "threshold": {"line": {"color": "#e2e8f0", "width": 2},
                          "thickness": 0.75, "value": THRESHOLD * 100},
        },
    ))
    fig_gauge.update_layout(height=240, margin=dict(t=20, b=10, l=20, r=20),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"family": "DM Sans"})
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Stat cards
    st.markdown(f"""
    <div class="card-row">
        <div class="stat-card-hero">
            <div class="label">Churn Probability</div>
            <div class="value">{proba:.1%}</div>
        </div>
        <div class="stat-card">
            <div class="label">Risk Level</div>
            <div class="value">{risk_label}</div>
        </div>
        <div class="stat-card">
            <div class="label">Confidence</div>
            <div class="value {conf_class}">{conf_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # SHAP waterfall (optional — skipped for batch row detail)
    explanation = None
    if show_shap:
        st.markdown('<div class="section-block">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Prediction Explanation</div>', unsafe_allow_html=True)
        try:
            explanation = get_shap_explanation(input_df)
            with plt.style.context("default"):
                fig_shap, _ = plt.subplots(figsize=(8, 5))
                fig_shap.patch.set_facecolor("white")
                shap.plots.waterfall(explanation, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close(fig_shap)
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        try:
            explanation = get_shap_explanation(input_df)
        except Exception:
            pass

    # Top Risk Factors
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Top Risk Factors</div>', unsafe_allow_html=True)
    if explanation is not None:
        try:
            f_names = np.array(explanation.feature_names)
            f_vals  = explanation.values
            top_idx = np.argsort(np.abs(f_vals))[::-1][:5]
            rows_html = "".join(f"""
            <div class="factor-row">
                <span class="factor-rank">{rank}</span>
                <span class="factor-name">{f_names[fi]}</span>
                <span class="{'factor-badge-up' if f_vals[fi] > 0 else 'factor-badge-down'}">
                    {'Increases Risk' if f_vals[fi] > 0 else 'Reduces Risk'}
                </span>
                <span class="factor-shap">{f_vals[fi]:+.4f}</span>
            </div>""" for rank, fi in enumerate(top_idx, 1))
            st.markdown(rows_html, unsafe_allow_html=True)
        except Exception:
            st.caption("Risk factor breakdown unavailable.")
    else:
        st.caption("Risk factor breakdown unavailable — SHAP explanation failed.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Retention Recommendations
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Retention Recommendations</div>', unsafe_allow_html=True)
    render_recommendations(explanation, proba)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Customer Input Summary
    st.markdown('<div class="summary-caption">Raw inputs sent to the model for this assessment.</div>',
                unsafe_allow_html=True)
    with st.expander("Customer Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# App Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <h1>Customer Churn Risk Assessor</h1>
    <div class="subtitle">
        <span>{MODEL_NAME}</span>
        <span class="subtitle-dot"></span>
        <span>Threshold {THRESHOLD:.4f}</span>
        <span class="subtitle-dot"></span>
        <span>Isotonic calibration</span>
        <span class="subtitle-dot"></span>
        <span>SHAP interpretability</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Customer", "Batch Prediction"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Customer
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    _, form_col, _ = st.columns([1, 2, 1])

    with form_col:
        st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)
        with st.form("customer_form"):

            st.markdown('<div class="form-group-label">Personal</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age    = st.slider("Age", 18, 92, 40)
            with c2:
                gender = st.selectbox("Gender", ["Male", "Female"])
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"],
                help="German customers show structurally higher churn rates in this dataset.")

            st.markdown('<div class="form-group-label">Financial</div>', unsafe_allow_html=True)
            credit_score = st.slider("Credit Score", 350, 850, 650,
                help="Lower scores can signal financial stress, which correlates with higher churn.")
            c3, c4 = st.columns(2)
            with c3:
                balance = st.number_input("Account Balance", min_value=0.0, max_value=260_000.0,
                    value=0.0, step=1_000.0, format="%.2f",
                    help="A zero balance is a strong churn signal — the customer may already be disengaged.")
            with c4:
                salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200_000.0,
                    value=50_000.0, step=1_000.0, format="%.2f",
                    help="Used to compute the Balance-to-Salary ratio, which captures financial stability.")

            st.markdown('<div class="form-group-label">Banking Relationship</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                tenure      = st.slider("Tenure (years)", 0, 10, 5,
                    help="Used alongside number of products to compute products-per-year engagement.")
                has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"],
                    help="Combined with active membership status as an interaction feature.")
            with c6:
                num_products = st.selectbox("Number of Products", [1, 2, 3, 4],
                    help="1 product = weakest ties. 3+ can paradoxically increase churn via service friction.")
                is_active    = st.selectbox("Is Active Member?", ["Yes", "No"],
                    help="Inactive members churn at significantly higher rates — one of the strongest predictors.")

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "CreditScore":     credit_score,  "Geography":       geography,
            "Gender":          gender,         "Age":             age,
            "Tenure":          tenure,         "Balance":         balance,
            "NumOfProducts":   num_products,   "HasCrCard":       1 if has_cr_card == "Yes" else 0,
            "IsActiveMember":  1 if is_active == "Yes" else 0,
            "EstimatedSalary": salary,
        }])
        proba = float(calibrated_pipeline.predict_proba(input_df)[0, 1])
        _, result_col, _ = st.columns([1, 2, 1])
        with result_col:
            render_prediction(input_df, proba, show_shap=True)
    else:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown("""
            <div class="placeholder-card">
                <div class="ph-title">No prediction run yet</div>
                <p>Fill in the customer profile above and click <strong>Predict Churn Risk</strong>.</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    _, batch_col, _ = st.columns([1, 2, 1])

    with batch_col:
        st.markdown('<div class="section-label">Batch Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="batch-sub">Upload a CSV with the same columns as the original dataset. '
            'RowNumber, CustomerId, Surname, and Exited are ignored if present.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"],
                                         label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                batch_df       = pd.read_csv(uploaded_file)
                drop_cols      = ["RowNumber", "CustomerId", "Surname", "Exited"]
                customer_ids   = batch_df["CustomerId"].values if "CustomerId" in batch_df.columns else None
                batch_features = batch_df.drop(columns=drop_cols, errors="ignore")

                batch_probas = calibrated_pipeline.predict_proba(batch_features)[:, 1]
                batch_preds  = (batch_probas >= THRESHOLD).astype(int)

                # Prediction columns first, then raw features
                elevated_floor = max(0.0, THRESHOLD - 0.15)
                results = pd.DataFrame({
                    "Churn_Probability": batch_probas.round(4),
                    "Churn_Prediction":  np.where(batch_preds == 1, "CHURN", "RETAIN"),
                    "Risk_Level":        pd.cut(batch_probas,
                                                bins=[0, elevated_floor, THRESHOLD, 1.0],
                                                labels=["Low", "Elevated", "High"]),
                })
                if customer_ids is not None:
                    results.insert(0, "CustomerId", customer_ids)
                for col in batch_features.columns:
                    results[col] = batch_features[col].values

                n_total  = len(results)
                n_churn  = int(batch_preds.sum())
                n_retain = n_total - n_churn

                # Summary cards — no percentages
                st.markdown(f"""
                <div class="card-row">
                    <div class="stat-card">
                        <div class="label">Total Customers</div>
                        <div class="value">{n_total:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Predicted Churn</div>
                        <div class="value">{n_churn:,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Predicted Retain</div>
                        <div class="value">{n_retain:,}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.dataframe(
                    results.sort_values("Churn_Probability", ascending=False),
                    use_container_width=True, height=380,
                )

                st.download_button(
                    label="Download Predictions CSV",
                    data=results.to_csv(index=False),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

                # Row detail
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Customer Detail</div>',
                            unsafe_allow_html=True)
                st.caption("Select a row to see the full prediction breakdown for that customer.")

                row_idx = st.selectbox(
                    "Row",
                    options=list(range(n_total)),
                    format_func=lambda i: (
                        f"Row {i}  —  "
                        f"{batch_features.iloc[i].get('Geography', '')}  "
                        f"Age {int(batch_features.iloc[i].get('Age', 0))}  "
                        f"{'CHURN' if batch_preds[i] == 1 else 'RETAIN'}  "
                        f"({batch_probas[i]:.1%})"
                    ),
                    label_visibility="collapsed",
                )

                selected_row   = batch_features.iloc[[row_idx]].reset_index(drop=True)
                selected_proba = float(batch_probas[row_idx])

                render_prediction(selected_row, selected_proba, show_shap=False)

            except Exception as e:
                st.error(f"Error processing file: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Model Information Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
_, footer_col, _ = st.columns([1, 2, 1])

with footer_col:
    with st.expander("Model Information"):
        st.markdown(f"""
        <div style="margin-bottom:1.25rem;">
            <div class="section-label" style="margin-bottom:0.6rem;">What does this model do?</div>
            <div class="rec-item" style="border:none;padding:0;font-size:0.83rem;line-height:1.7;">
                This model scores each customer's likelihood of leaving the bank within the next period.
                A probability above the decision threshold ({THRESHOLD:.1%}) means the model considers
                the customer at risk and recommends retention action. Every prediction comes with a
                feature-level explanation showing exactly which factors drove the score for that customer.
            </div>
        </div>
        <div class="section-label" style="margin-bottom:0.75rem;">How accurate is it?</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.65rem;margin-bottom:1.25rem;">
            <div class="stat-card">
                <div class="label">Recall</div>
                <div class="value" style="font-size:1.3rem;">61.8%</div>
                <div class="sub">Of real churners, the model correctly flags 6 in 10</div>
            </div>
            <div class="stat-card">
                <div class="label">Precision</div>
                <div class="value" style="font-size:1.3rem;">72.0%</div>
                <div class="sub">When the model raises a flag, it is correct 7 in 10 times</div>
            </div>
            <div class="stat-card">
                <div class="label">F1 Score</div>
                <div class="value" style="font-size:1.3rem;">0.665</div>
                <div class="sub">Balanced score across both recall and precision</div>
            </div>
            <div class="stat-card">
                <div class="label">ROC-AUC</div>
                <div class="value" style="font-size:1.3rem;">~0.87</div>
                <div class="sub">Overall ranking ability across all possible thresholds</div>
            </div>
        </div>
        <div class="section-label" style="margin-bottom:0.6rem;">What does that mean in practice?</div>
        <div class="rec-item" style="border:none;padding:0;font-size:0.83rem;line-height:1.7;">
            For every 100 customers who will actually churn, this model catches approximately 62 before
            they leave. Of the customers it flags, roughly 72% genuinely are at risk — meaning about 28%
            of outreach efforts will go to customers who would have stayed anyway. The threshold was
            deliberately set to favour catching more churners (higher recall) over perfect precision,
            since the cost of missing a churner exceeds the cost of an unnecessary retention call.
        </div>
        """, unsafe_allow_html=True)

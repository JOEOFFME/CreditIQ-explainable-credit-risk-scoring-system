import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Explainable Credit Risk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #080B14;
    --bg2:       #0D1120;
    --bg3:       #111827;
    --border:    #1E2D45;
    --accent:    #00D4FF;
    --accent2:   #7C3AED;
    --green:     #10B981;
    --orange:    #F59E0B;
    --red:       #EF4444;
    --text:      #E2E8F0;
    --muted:     #64748B;
    --card:      #0F1729;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* App background */
.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(0,212,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(124,58,237,0.04) 0%, transparent 50%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.4;
}

/* KPI Cards */
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    margin: 8px 0 4px;
}
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Score display */
.score-display {
    text-align: center;
    padding: 32px 0;
}
.score-number {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -2px;
}
.score-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 8px;
}

/* Verdict badges */
.verdict {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 10px 28px;
    border-radius: 50px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 16px;
}
.verdict-approve  { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.4); }
.verdict-review   { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.4); }
.verdict-reject   { background: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid rgba(239,68,68,0.4); }

/* Tier badges */
.tier-high   { color: #EF4444; font-weight: 700; font-family: 'DM Mono', monospace; }
.tier-medium { color: #F59E0B; font-weight: 700; font-family: 'DM Mono', monospace; }
.tier-low    { color: #10B981; font-weight: 700; font-family: 'DM Mono', monospace; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 24px;
    font-family: 'DM Mono', monospace;
}

/* Score bar */
.score-bar-track {
    background: var(--border);
    border-radius: 100px;
    height: 8px;
    width: 100%;
    margin: 12px 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s ease;
}

/* Input styling */
.stSlider > div > div > div { color: var(--accent) !important; }
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}
div[data-baseweb="select"] {
    background: var(--bg3) !important;
}
div[data-baseweb="select"] * {
    background: var(--bg3) !important;
    color: var(--text) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    font-size: 0.85rem !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Metric override */
div[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
div[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

/* Dataframe */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Logo bar */
.logo-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 0 24px;
}
.logo-mark {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00D4FF, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.logo-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    line-height: 1.2;
}

/* Nav items */
.nav-item {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 8px 0;
}

/* Table styling */
thead tr th {
    background: var(--bg3) !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base, 'models'),
        os.path.join(base, '..', 'models'),
    ]
    for p in paths:
        m = os.path.join(p, 'xgboost_model.joblib')
        t = os.path.join(p, 'best_threshold.joblib')
        if os.path.exists(m):
            try:
                return joblib.load(m), joblib.load(t)
            except:
                pass
    return None, 0.59

@st.cache_resource
def get_engine():
    try:
        eng = create_engine(
            'postgresql://postgres:20172008@localhost:5432/credit_risk_db'
        )
        eng.connect()
        return eng
    except:
        return None

@st.cache_data(ttl=300)
def load_portfolio_kpis():
    eng = get_engine()
    if eng is None:
        return None
    try:
        return pd.read_sql("SELECT * FROM v_kpi_summary LIMIT 1", eng)
    except:
        return None

@st.cache_data(ttl=300)
def load_risk_segments():
    eng = get_engine()
    if eng is None:
        return None
    try:
        return pd.read_sql("SELECT * FROM v_risk_by_segment", eng)
    except:
        return None

def probability_to_score(prob):
    return max(300, min(850, round(850 - prob * 550)))

def score_to_color(score):
    if score >= 740: return "#10B981"
    if score >= 670: return "#34D399"
    if score >= 580: return "#F59E0B"
    if score >= 500: return "#F97316"
    return "#EF4444"

def score_to_tier(score):
    if score >= 670: return "LOW RISK",   "tier-low"
    if score >= 550: return "MEDIUM RISK","tier-medium"
    return "HIGH RISK", "tier-high"

def score_to_verdict(score):
    if score >= 670:
        return "APPROVE",  "verdict-approve",  "Application meets credit criteria."
    if score >= 550:
        return "REVIEW",   "verdict-review",   "Manual review required before decision."
    return "REJECT",   "verdict-reject",   "Application does not meet minimum credit criteria."

def make_shap_plot(model, input_df):
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(input_df)
        vals       = shap_vals[0]
        features   = input_df.columns.tolist()

        idx        = np.argsort(np.abs(vals))[-12:]
        top_vals   = vals[idx]
        top_feats  = [features[i] for i in idx]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0F1729')
        ax.set_facecolor('#0F1729')

        colors = ['#EF4444' if v > 0 else '#10B981' for v in top_vals]
        bars   = ax.barh(range(len(top_vals)), top_vals, color=colors,
                         height=0.65, alpha=0.9)

        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(
            [f.replace('_', ' ').title() for f in top_feats],
            fontsize=8.5, color='#94A3B8',
            fontfamily='monospace'
        )
        ax.set_xlabel('SHAP Value (Impact on Default Risk)',
                      fontsize=8, color='#64748B', labelpad=10)
        ax.tick_params(colors='#64748B', labelsize=8)
        ax.axvline(0, color='#1E2D45', linewidth=1.2)
        ax.spines[:].set_visible(False)
        ax.tick_params(left=False)

        for bar, val in zip(bars, top_vals):
            ax.text(
                val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}',
                va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=7.5, color='#94A3B8',
                fontfamily='monospace'
            )

        red_p   = mpatches.Patch(color='#EF4444', alpha=0.9, label='↑ Increases risk')
        green_p = mpatches.Patch(color='#10B981', alpha=0.9, label='↓ Decreases risk')
        ax.legend(handles=[red_p, green_p], loc='lower right',
                  facecolor='#0F1729', edgecolor='#1E2D45',
                  labelcolor='#94A3B8', fontsize=7.5)

        plt.tight_layout(pad=1.5)
        return fig
    except Exception as e:
        return None

def score_gauge_html(score):
    color   = score_to_color(score)
    pct     = (score - 300) / 550 * 100
    tier, _ = score_to_tier(score)
    verdict, vclass, _ = score_to_verdict(score)
    return f"""
    <div class="score-display">
        <div class="score-label">CREDIT SCORE</div>
        <div class="score-number" style="color:{color}">{score}</div>
        <div style="max-width:260px; margin:16px auto 0;">
            <div style="display:flex; justify-content:space-between;
                        font-family:'DM Mono',monospace; font-size:0.65rem;
                        color:#64748B; margin-bottom:4px;">
                <span>300</span><span>580</span><span>670</span><span>850</span>
            </div>
            <div class="score-bar-track">
                <div class="score-bar-fill"
                     style="width:{pct:.1f}%;
                            background:linear-gradient(90deg,#EF4444,{color});">
                </div>
            </div>
        </div>
        <div style="margin-top:16px;">
            <span class="verdict {vclass}">{verdict}</span>
        </div>
    </div>
    """


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-bar">
        <div>
            <div class="logo-mark">◈ CreditIQ</div>
            <div class="logo-sub">Explainable Risk Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["◉  Scoring Terminal",
         "◈  Portfolio Intelligence",
         "◎  Client Lookup"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem;
                color:#334155; line-height:1.8;">
        <div>MODEL · XGBoost + Optuna</div>
        <div>AUC · 0.7741</div>
        <div>THRESHOLD · 0.59</div>
        <div>DATABASE · PostgreSQL 18</div>
        <div style="margin-top:12px; color:#1E2D45;">
            INPT · DATA SCIENCE · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SCORING TERMINAL
# ══════════════════════════════════════════════════════════════════════════════
if page == "◉  Scoring Terminal":

    st.markdown("""
    <div class="section-title">◉ Live Scoring Terminal</div>
    <div class="section-subtitle">
        REAL-TIME CREDIT RISK ASSESSMENT · SHAP-POWERED EXPLANATIONS
    </div>
    """, unsafe_allow_html=True)

    model, threshold = load_model()
    if model is None:
        st.error("Model not found. Ensure `../models/xgboost_model.joblib` exists.")
        st.stop()

    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                    color:#64748B; letter-spacing:0.12em; margin-bottom:20px;">
            CLIENT PARAMETERS
        </div>
        """, unsafe_allow_html=True)

        age           = st.slider("Age (years)", 18, 70, 35)
        income        = st.number_input("Annual Income (MAD)", 50000, 5000000, 180000, step=10000)
        amt_credit    = st.number_input("Loan Amount (MAD)", 50000, 4000000, 500000, step=50000)
        amt_annuity   = st.number_input("Annual Annuity (MAD)", 10000, 500000, 30000, step=5000)
        ext_source_2  = st.slider("External Score 2 (0-1)", 0.0, 1.0, 0.5, 0.01)
        ext_source_1  = st.slider("External Score 1 (0-1)", 0.0, 1.0, 0.5, 0.01)
        ext_source_3  = st.slider("External Score 3 (0-1)", 0.0, 1.0, 0.5, 0.01)
        employment_yr = st.slider("Employment Years", 0, 40, 5)
        cnt_children  = st.slider("Number of Children", 0, 10, 0)

        gender      = st.selectbox("Gender", ["M", "F"])
        income_type = st.selectbox("Income Type",
            ["Working", "Commercial associate", "Pensioner",
             "State servant", "Unemployed"])

        st.markdown('</div>', unsafe_allow_html=True)
        run = st.button("◈  ANALYZE CLIENT")

    with col_result:
        if run:
            ext_mean = np.mean([ext_source_1, ext_source_2, ext_source_3])
            ext_min  = np.min([ext_source_1, ext_source_2, ext_source_3])
            ext_std  = np.std([ext_source_1, ext_source_2, ext_source_3])

            # Map income type to approximate target-encoded value
            income_type_map = {
                "Working": 0.095, "Commercial associate": 0.075,
                "Pensioner": 0.052, "State servant": 0.044, "Unemployed": 0.18
            }
            gender_map = {"M": 0.10, "F": 0.07}

            # Known values from user input
            known = {
                'AGE_YEARS':              age,
                'AMT_INCOME_TOTAL':       income,
                'AMT_CREDIT':             amt_credit,
                'AMT_ANNUITY':            amt_annuity,
                'EXT_SOURCE_1':           ext_source_1,
                'EXT_SOURCE_2':           ext_source_2,
                'EXT_SOURCE_3':           ext_source_3,
                'EXT_SOURCE_MEAN':        ext_mean,
                'EXT_SOURCE_MIN':         ext_min,
                'EXT_SOURCE_STD':         ext_std,
                'CREDIT_INCOME_RATIO':    amt_credit / max(income, 1),
                'ANNUITY_INCOME_RATIO':   amt_annuity / max(income, 1),
                'CREDIT_TERM':            amt_annuity / max(amt_credit, 1),
                'EMPLOYMENT_YEARS':       employment_yr,
                'CNT_CHILDREN':           cnt_children,
                'CNT_FAM_MEMBERS':        cnt_children + 2,
                'INCOME_PER_PERSON':      income / max(cnt_children + 2, 1),
                'CHILDREN_RATIO':         cnt_children / max(cnt_children + 2, 1),
                'EMPLOYMENT_AGE_RATIO':   employment_yr / max(age, 1),
                'IS_UNEMPLOYED':          1 if income_type == "Unemployed" else 0,
                'CODE_GENDER':            gender_map.get(gender, 0.085),
                'NAME_INCOME_TYPE':       income_type_map.get(income_type, 0.09),
                # Bureau defaults (median portfolio values)
                'BUR_NB_CREDITS':         4.0,
                'BUR_NB_ACTIVE':          1.0,
                'BUR_TOTAL_DEBT':         50000.0,
                'BUR_TOTAL_OVERDUE':      0.0,
                'BUR_MAX_OVERDUE_DAYS':   0.0,
                'BUR_AVG_OVERDUE_DAYS':   0.0,
                'BUR_NB_OVERDUE':         0.0,
                'BUR_PROLONGATIONS':      0.0,
                'BUR_TOTAL_CREDIT':       200000.0,
                'BUR_DEBT_CREDIT_RATIO':  0.25,
                'BUR_ACTIVE_RATIO':       0.3,
                'BUR_OVERDUE_FLAG':       0.0,
                'BUR_DEBT_VS_INCOME':     amt_credit / max(income, 1),
                'EXT_SOURCE_X_BUREAU':    ext_source_2 * 0.25,
                # Previous application defaults
                'PREV_NB_APPS':           1.0,
                'PREV_NB_REFUSED':        0.0,
                'PREV_REFUSAL_RATE':      0.0,
                'PREV_AVG_CREDIT_GAP':    0.0,
                'PREV_APPROVAL_RATIO':    1.0,
                'PREV_AVG_INSTALLMENTS':  amt_annuity / 12,
                # Other defaults
                'AMT_GOODS_PRICE':        amt_credit * 0.9,
                'REGION_RATING_CLIENT':   2.0,
                'REG_CITY_NOT_LIVE_CITY': 0.0,
                'REGISTRATION_YEARS':     age - 18,
                'ID_PUBLISHER_YEARS':     age - 20,
                # Categorical target-encoded defaults
                'NAME_CONTRACT_TYPE':     0.08,
                'FLAG_OWN_CAR':           0.07,
                'FLAG_OWN_REALTY':        0.08,
                'NAME_FAMILY_STATUS':     0.08,
                'NAME_HOUSING_TYPE':      0.08,
                'NAME_EDUCATION_TYPE':    0.08,
                'OCCUPATION_TYPE':        0.09,
                'ORGANIZATION_TYPE':      0.09,
            }

            # Get exact feature order from model
            model_features = model.get_booster().feature_names
            input_df = pd.DataFrame([{
                f: known.get(f, 0.0) for f in model_features
            }])

            proba  = float(model.predict_proba(input_df)[0, 1])
            score  = probability_to_score(proba)
            tier, tclass   = score_to_tier(score)
            verdict, vclass, vdesc = score_to_verdict(score)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(score_gauge_html(score), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Default Probability", f"{proba*100:.1f}%")
            with c2:
                st.metric("Risk Tier", tier)
            with c3:
                st.metric("Decision Threshold", f"{threshold:.2f}")

            st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                        color:#64748B; letter-spacing:0.12em; margin-bottom:16px;">
                SHAP EXPLANATION — TOP CONTRIBUTING FEATURES
            </div>
            """, unsafe_allow_html=True)

            fig = make_shap_plot(model, input_df)
            if fig:
                st.pyplot(fig, width="stretch")
            else:
                st.info("SHAP explanation unavailable for this input.")
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:64px 24px;">
                <div style="font-size:3rem; margin-bottom:16px; opacity:0.3;">◈</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.8rem;
                            color:#334155; letter-spacing:0.1em;">
                    CONFIGURE CLIENT PARAMETERS<br>AND PRESS ANALYZE
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PORTFOLIO INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "◈  Portfolio Intelligence":

    st.markdown("""
    <div class="section-title">◈ Portfolio Intelligence</div>
    <div class="section-subtitle">
        LIVE DATA FROM POSTGRESQL · credit_risk_db
    </div>
    """, unsafe_allow_html=True)

    kpis = load_portfolio_kpis()
    segs = load_risk_segments()

    if kpis is None:
        st.warning("Cannot connect to PostgreSQL. Showing demo data.")
        kpis_data = {
            'global_default_rate_pct': 8.07,
            'avg_dti_pct': 18.09,
            'total_clients': 307507,
            'high_risk_count': 12369,
            'medium_risk_count': 24773,
            'low_risk_count': 24360,
        }
    else:
        kpis_data = kpis.iloc[0].to_dict()

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Default Rate",
                  f"{kpis_data.get('global_default_rate_pct', 8.07):.2f}%",
                  delta="-0.3% vs last month", delta_color="inverse")
    with k2:
        st.metric("HIGH RISK",
                  f"{int(kpis_data.get('high_risk_count', 12369)):,}")
    with k3:
        st.metric("MEDIUM RISK",
                  f"{int(kpis_data.get('medium_risk_count', 24773)):,}")
    with k4:
        st.metric("LOW RISK",
                  f"{int(kpis_data.get('low_risk_count', 24360)):,}")
    with k5:
        st.metric("Avg DTI",
                  f"{kpis_data.get('avg_dti_pct', 18.09):.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                    color:#64748B; letter-spacing:0.12em; margin-bottom:16px;">
            RISK TIER DISTRIBUTION
        </div>
        """, unsafe_allow_html=True)

        high   = int(kpis_data.get('high_risk_count',   12369))
        medium = int(kpis_data.get('medium_risk_count', 24773))
        low    = int(kpis_data.get('low_risk_count',    24360))
        total  = high + medium + low

        for label, val, color in [
            ("HIGH",   high,   "#EF4444"),
            ("MEDIUM", medium, "#F59E0B"),
            ("LOW",    low,    "#10B981"),
        ]:
            pct = val / max(total, 1) * 100
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between;
                            font-family:'DM Mono',monospace; font-size:0.72rem;
                            color:#94A3B8; margin-bottom:5px;">
                    <span style="color:{color}; font-weight:600;">{label}</span>
                    <span>{val:,} &nbsp;·&nbsp; {pct:.1f}%</span>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill"
                         style="width:{pct:.1f}%; background:{color}; opacity:0.8;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Score distribution chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                    color:#64748B; letter-spacing:0.12em; margin-bottom:16px;">
            SCORE BAND BREAKDOWN
        </div>
        """, unsafe_allow_html=True)

        bands  = ['300-499', '500-579', '580-669', '670-739', '740-850']
        counts = [
            int(high * 0.6),
            int(high * 0.4),
            int(medium * 0.7),
            int(medium * 0.3 + low * 0.4),
            int(low * 0.6),
        ]
        colors_band = ['#EF4444','#F97316','#F59E0B','#34D399','#10B981']

        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        fig2.patch.set_facecolor('#0F1729')
        ax2.set_facecolor('#0F1729')

        bars2 = ax2.bar(bands, counts, color=colors_band, alpha=0.85, width=0.6)
        ax2.spines[:].set_visible(False)
        ax2.tick_params(colors='#64748B', labelsize=7.5)
        ax2.set_ylabel('Clients', fontsize=7.5, color='#64748B')
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 50,
                     f'{int(bar.get_height()):,}',
                     ha='center', va='bottom',
                     fontsize=7, color='#64748B',
                     fontfamily='monospace')
        plt.xticks(rotation=0)
        plt.tight_layout(pad=1)
        st.pyplot(fig2, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                    color:#64748B; letter-spacing:0.12em; margin-bottom:16px;">
            DEFAULT RATE BY AGE BRACKET
        </div>
        """, unsafe_allow_html=True)

        if segs is not None and 'age_bracket' in segs.columns:
            age_data = (segs.groupby('age_bracket')['default_rate_pct']
                            .mean().reset_index()
                            .sort_values('default_rate_pct', ascending=True))
        else:
            age_data = pd.DataFrame({
                'age_bracket':     ['55+','45-54','35-44','25-34','18-24'],
                'default_rate_pct':[5.2,  6.8,    7.9,    9.1,    10.4],
            })

        fig3, ax3 = plt.subplots(figsize=(5, 3))
        fig3.patch.set_facecolor('#0F1729')
        ax3.set_facecolor('#0F1729')

        bar_colors = ['#EF4444' if v > 9 else '#F59E0B' if v > 7
                      else '#10B981'
                      for v in age_data['default_rate_pct']]
        ax3.barh(age_data['age_bracket'], age_data['default_rate_pct'],
                 color=bar_colors, alpha=0.85, height=0.55)
        ax3.set_xlabel('Default Rate %', fontsize=7.5, color='#64748B')
        ax3.spines[:].set_visible(False)
        ax3.tick_params(colors='#64748B', labelsize=8)
        for i, v in enumerate(age_data['default_rate_pct']):
            ax3.text(v + 0.1, i, f'{v:.1f}%',
                     va='center', fontsize=7.5,
                     color='#94A3B8', fontfamily='monospace')
        plt.tight_layout(pad=1)
        st.pyplot(fig3, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                    color:#64748B; letter-spacing:0.12em; margin-bottom:16px;">
            INCOME TYPE RISK PROFILE
        </div>
        """, unsafe_allow_html=True)

        if segs is not None and 'name_income_type' in segs.columns:
            inc_data = (segs.groupby('name_income_type')['default_rate_pct']
                            .mean().reset_index()
                            .sort_values('default_rate_pct', ascending=False)
                            .head(6))
        else:
            inc_data = pd.DataFrame({
                'name_income_type': ['Unemployed','Maternity leave',
                                     'Working','Commercial associate',
                                     'State servant','Pensioner'],
                'default_rate_pct': [12.1, 10.8, 8.9, 7.6, 5.4, 4.2],
            })

        for _, row in inc_data.iterrows():
            pct   = row['default_rate_pct']
            color = '#EF4444' if pct > 10 else '#F59E0B' if pct > 7 else '#10B981'
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px;
                        margin-bottom:10px;">
                <div style="font-family:'DM Mono',monospace; font-size:0.72rem;
                            color:#94A3B8; width:160px; flex-shrink:0;">
                    {row['name_income_type']}
                </div>
                <div style="flex:1; background:#1E2D45; border-radius:100px;
                            height:6px; overflow:hidden;">
                    <div style="width:{min(pct*5,100):.0f}%; height:100%;
                                background:{color}; border-radius:100px;">
                    </div>
                </div>
                <div style="font-family:'DM Mono',monospace; font-size:0.72rem;
                            color:{color}; width:40px; text-align:right;">
                    {pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLIENT LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "◎  Client Lookup":

    st.markdown("""
    <div class="section-title">◎ Client Lookup</div>
    <div class="section-subtitle">
        SEARCH ANY CLIENT BY ID · FULL RISK PROFILE
    </div>
    """, unsafe_allow_html=True)

    eng = get_engine()

    col_search, _ = st.columns([1, 2])
    with col_search:
        client_id = st.number_input(
            "Client ID (SK_ID_CURR)",
            min_value=100000, max_value=999999,
            value=100038, step=1
        )
        search = st.button("◎  LOOKUP CLIENT")

    if search:
        if eng is None:
            st.error("Cannot connect to PostgreSQL.")
            st.stop()

        query = f"""
            SELECT
                v.sk_id_curr,
                v.age_years,
                v.code_gender,
                v.name_income_type,
                v.name_education_type,
                v.amt_income_total,
                v.amt_credit,
                v.amt_annuity,
                v.credit_income_ratio,
                v.annuity_income_ratio,
                v.default_probability,
                v.risk_tier,
                v.shap_ext_source2,
                v.shap_amt_credit,
                v.score_date,
                r.credit_score
            FROM v_client_risk_profile v
            LEFT JOIN risk_score r ON v.sk_id_curr = r.sk_id_curr
            WHERE v.sk_id_curr = {client_id}
            LIMIT 1
        """
        try:
            df = pd.read_sql(query, eng)
        except Exception as e:
            st.error(f"Query error: {e}")
            st.stop()

        if df.empty:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:40px;">
                <div style="font-family:'DM Mono',monospace; color:#64748B;">
                    NO CLIENT FOUND WITH ID {client_id}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            row = df.iloc[0]
            prob  = float(row.get('default_probability', 0.5) or 0.5)
            score = int(row.get('credit_score') or probability_to_score(prob))
            tier, tclass     = score_to_tier(score)
            verdict, vclass, vdesc = score_to_verdict(score)
            color = score_to_color(score)

            col_a, col_b, col_c = st.columns([1, 1, 1.4], gap="large")

            with col_a:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(score_gauge_html(score), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                            color:#64748B; letter-spacing:0.12em; margin-bottom:16px;">
                    CLIENT PROFILE
                </div>
                """, unsafe_allow_html=True)

                fields = [
                    ("CLIENT ID",    f"{int(row['sk_id_curr'])}"),
                    ("AGE",          f"{float(row['age_years']):.0f} years"),
                    ("GENDER",       str(row['code_gender'])),
                    ("INCOME TYPE",  str(row['name_income_type'])),
                    ("EDUCATION",    str(row['name_education_type'])),
                    ("ANNUAL INCOME",f"{float(row['amt_income_total']):,.0f} MAD"),
                    ("LOAN AMOUNT",  f"{float(row['amt_credit']):,.0f} MAD"),
                    ("ANNUITY",      f"{float(row['amt_annuity']):,.0f} MAD"),
                    ("CREDIT/INCOME",f"{float(row['credit_income_ratio']):.2f}×"),
                    ("DTI RATIO",    f"{float(row['annuity_income_ratio'])*100:.1f}%"),
                ]
                for label, value in fields:
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between;
                                padding:7px 0; border-bottom:1px solid #1E2D45;">
                        <span style="font-family:'DM Mono',monospace;
                                     font-size:0.67rem; color:#475569;">
                            {label}
                        </span>
                        <span style="font-family:'DM Mono',monospace;
                                     font-size:0.72rem; color:#E2E8F0;">
                            {value}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_c:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                            color:#64748B; letter-spacing:0.12em; margin-bottom:16px;">
                    RISK ASSESSMENT
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-bottom:20px; padding:16px;
                            background:rgba(0,0,0,0.2); border-radius:8px;
                            border-left:3px solid {color};">
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem;
                                color:#64748B; margin-bottom:6px;">DECISION</div>
                    <div class="verdict {vclass}" style="margin:0;">{verdict}</div>
                    <div style="font-size:0.78rem; color:#64748B;
                                margin-top:8px; font-family:'DM Sans',sans-serif;">
                        {vdesc}
                    </div>
                </div>

                <div style="display:grid; grid-template-columns:1fr 1fr;
                            gap:12px; margin-bottom:16px;">
                    <div style="background:rgba(0,0,0,0.2); border-radius:8px;
                                padding:12px; text-align:center;">
                        <div style="font-family:'DM Mono',monospace; font-size:0.62rem;
                                    color:#64748B;">DEFAULT PROB</div>
                        <div style="font-family:'Syne',sans-serif; font-size:1.4rem;
                                    font-weight:800; color:{color};">
                            {prob*100:.1f}%
                        </div>
                    </div>
                    <div style="background:rgba(0,0,0,0.2); border-radius:8px;
                                padding:12px; text-align:center;">
                        <div style="font-family:'DM Mono',monospace; font-size:0.62rem;
                                    color:#64748B;">RISK TIER</div>
                        <div class="{tclass}" style="font-size:1rem; margin-top:4px;">
                            {tier}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # SHAP values from DB
                shap_ext = row.get('shap_ext_source2')
                shap_amt = row.get('shap_amt_credit')

                if shap_ext is not None and shap_amt is not None:
                    st.markdown("""
                    <div style="font-family:'DM Mono',monospace; font-size:0.65rem;
                                color:#64748B; margin-bottom:10px;">
                        TOP SHAP CONTRIBUTORS
                    </div>
                    """, unsafe_allow_html=True)

                    for feat, val in [
                        ("EXT_SOURCE_2", float(shap_ext)),
                        ("AMT_CREDIT",   float(shap_amt)),
                    ]:
                        c = '#EF4444' if val > 0 else '#10B981'
                        sign = '+' if val > 0 else ''
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between;
                                    align-items:center; padding:8px 0;
                                    border-bottom:1px solid #1E2D45;">
                            <span style="font-family:'DM Mono',monospace;
                                         font-size:0.7rem; color:#94A3B8;">
                                {feat}
                            </span>
                            <span style="font-family:'DM Mono',monospace;
                                         font-size:0.75rem; color:{c};
                                         font-weight:600;">
                                {sign}{val:.4f}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                score_date = row.get('score_date')
                if score_date:
                    st.markdown(f"""
                    <div style="font-family:'DM Mono',monospace; font-size:0.62rem;
                                color:#334155; margin-top:16px; text-align:right;">
                        SCORED · {str(score_date)[:10]}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
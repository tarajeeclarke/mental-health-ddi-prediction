import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ──────────────────
st.set_page_config(
    page_title="Drug Interaction Classifier",
    page_icon="💊",
    layout="wide",
)

# ── Site color palette ──────────────────────────────────────────
BG      = "#120a0e"
SURFACE = "#1a0e14"
ACCENT  = "#ff6eb0"
MUTED   = "#9c7a88"
TEXT    = "#f5e6ed"

st.markdown(f"""
<style>
  /* Main background */
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  
  /* Sidebar */
  [data-testid="stSidebar"] {{ background-color: {SURFACE}; }}

  /* Metric cards */
  .metric-box {{
    background: {SURFACE};
    border: 1px solid #3a1a28;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
  }}
  .metric-value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {ACCENT};
  }}
  .metric-label {{
    font-size: 0.75rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}

  /* Primary button */
  .stButton > button {{
    background-color: {ACCENT};
    color: {BG};
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2rem;
  }}
  .stButton > button:hover {{
    background-color: #ff9cc8;
    color: {BG};
  }}

  /* Selectbox labels */
  label {{ color: {MUTED} !important; font-size: 0.8rem; letter-spacing: 0.05em; }}

  /* Divider color */
  hr {{ border-color: #3a1a28; }}
</style>
""", unsafe_allow_html=True)


# ── Data loading & model training ──────────────────────────────
@st.cache_data
def load_and_train():
    """
    Loads the pre-filtered mental-health DDI subset, creates labels,
    fits a single LabelEncoder on the union of both drug columns so
    both Drug A and Drug B can be encoded consistently, then trains
    both models.
    """
    # ── Load subset (committed to repo — only ~9,600 rows) ──────
    df = pd.read_csv("data/twosides_mental_health_subset.csv")

    # ── Ensure required columns exist ───────────────────────────
    # Actual column names from the data engineering pipeline:
    #   drug_1_concept_name, drug_2_concept_name, PRR, PRR_float, label
    required = ["drug_1_concept_name", "drug_2_concept_name", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in dataset: {missing}")
        st.stop()

    df = df.dropna(subset=required)

    # ── Single LabelEncoder fit on UNION of both drug columns ───
    # This is critical: encoding drug1 and drug2 separately means
    # a drug that appears only in drug2 crashes at prediction time.
    all_drug_names = pd.concat([
        df["drug_1_concept_name"],
        df["drug_2_concept_name"]
    ]).dropna().unique()
    all_drug_names = sorted(all_drug_names.tolist())

    le = LabelEncoder()
    le.fit(all_drug_names)

    df["drug1_enc"] = le.transform(df["drug_1_concept_name"])
    df["drug2_enc"] = le.transform(df["drug_2_concept_name"])

    X = df[["drug1_enc", "drug2_enc"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train models ─────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    lr = LogisticRegression(
        class_weight="balanced", max_iter=500, random_state=42
    )
    lr.fit(X_train, y_train)

    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    n_records = len(df)

    return rf, lr, le, auc_rf, auc_lr, all_drug_names, n_records


rf_model, lr_model, le, auc_rf, auc_lr, drug_names, n_records = load_and_train()


# ── Header ──────────────────────────────────────────────────────
st.markdown("## 💊 Drug Interaction Classifier")
st.markdown(
    "*Predict clinically significant drug-drug interactions in mental-health "
    "polypharmacy using the TwoSIDES pharmacovigilance dataset.*"
)
st.divider()

# ── Model performance metrics ────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">{n_records:,}</div>'
    f'<div class="metric-label">Records Trained On</div>'
    f'</div>',
    unsafe_allow_html=True,
)
col2.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">AUC {auc_rf:.2f}</div>'
    f'<div class="metric-label">Random Forest</div>'
    f'</div>',
    unsafe_allow_html=True,
)
col3.markdown(
    f'<div class="metric-box">'
    f'<div class="metric-value">AUC {auc_lr:.2f}</div>'
    f'<div class="metric-label">Logistic Regression</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Drug pair selection ──────────────────────────────────────────
st.markdown("### Select a Drug Pair")
st.caption(
    f"Showing {len(drug_names):,} unique mental-health medications from the TwoSIDES dataset."
)

c1, c2 = st.columns(2)
drug1 = c1.selectbox("Drug A", drug_names, index=0)
drug2 = c2.selectbox("Drug B", drug_names, index=min(1, len(drug_names) - 1))

# ── Prediction ───────────────────────────────────────────────────
if st.button("Predict Interaction", type="primary"):

    if drug1 == drug2:
        st.warning("Please select two different drugs.")
    else:
        try:
            d1_enc = le.transform([drug1])[0]
            d2_enc = le.transform([drug2])[0]
            X_input = pd.DataFrame([[d1_enc, d2_enc]], columns=["drug1_enc", "drug2_enc"])

            prob_rf  = rf_model.predict_proba(X_input)[0][1]
            prob_lr  = lr_model.predict_proba(X_input)[0][1]
            avg_prob = (prob_rf + prob_lr) / 2

            if avg_prob > 0.60:
                risk       = "🔴 High Risk"
                risk_color = "#ff4d4d"
                risk_note  = (
                    "This drug combination shows a strong adverse interaction signal "
                    "in pharmacovigilance data. Clinical review is strongly advised."
                )
            elif avg_prob > 0.35:
                risk       = "🟡 Moderate Risk"
                risk_color = "#ffcc00"
                risk_note  = (
                    "This combination shows an elevated interaction signal. "
                    "Monitor closely if co-prescribed."
                )
            else:
                risk       = "🟢 Low Risk"
                risk_color = "#4dff88"
                risk_note  = (
                    "No strong interaction signal detected for this pair "
                    "in the TwoSIDES pharmacovigilance dataset."
                )

            st.divider()
            st.markdown(
                f"<h3 style='color:{risk_color};'>{risk}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:{MUTED}; font-size:0.9rem;'>{risk_note}</p>",
                unsafe_allow_html=True,
            )

            r1, r2, r3 = st.columns(3)
            r1.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{avg_prob:.1%}</div>'
                f'<div class="metric-label">Avg Interaction Probability</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            r2.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{prob_rf:.1%}</div>'
                f'<div class="metric-label">Random Forest</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            r3.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{prob_lr:.1%}</div>'
                f'<div class="metric-label">Logistic Regression</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Model agreement callout ──────────────────────────
            model_diff = abs(prob_rf - prob_lr)
            if model_diff > 0.20:
                st.info(
                    f"⚠️ **Model disagreement:** Random Forest ({prob_rf:.1%}) and "
                    f"Logistic Regression ({prob_lr:.1%}) differ by {model_diff:.1%}. "
                    f"Predictions for rare or borderline drug pairs may be less reliable."
                )

            st.divider()
            st.caption(
                "⚠️ **Disclaimer:** This tool is for portfolio demonstration only. "
                "Predictions are based on PRR-derived signals from FDA FAERS spontaneous "
                "reports — not controlled clinical evidence. Not for clinical use."
            )

        except ValueError as e:
            st.error(
                f"Encoding error: {e}. "
                "One or both drug names could not be found in the training data."
            )


# ── Sidebar: about ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### About")
    st.markdown(
        f"""
        <div style='color:{MUTED}; font-size:0.85rem; line-height:1.7;'>
        This classifier predicts adverse drug-drug interaction signals in 
        mental-health polypharmacy using the <b style='color:{TEXT};'>TwoSIDES</b> 
        dataset derived from FDA FAERS spontaneous adverse event reports.
        <br><br>
        <b style='color:{TEXT};'>Signal definition:</b> A drug pair is labeled 
        as a signal (label = 1) when its Proportional Reporting Ratio (PRR) > 1, 
        indicating the adverse event is reported more often than expected.
        <br><br>
        <b style='color:{TEXT};'>Models:</b> Logistic Regression and Random Forest, 
        both trained with class balancing to handle the heavily imbalanced 
        TwoSIDES dataset.
        <br><br>
        <b style='color:{TEXT};'>Scope:</b> Mental-health medications only — SSRIs, 
        SNRIs, antipsychotics, benzodiazepines, mood stabilizers, and stimulants.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f"<div style='color:{MUTED}; font-size:0.75rem;'>"
        f"Built by <b style='color:{TEXT};'>TaraJee Clarke</b><br>"
        f"MS Health Informatics — Hofstra University"
        f"</div>",
        unsafe_allow_html=True,
    )

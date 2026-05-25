"""Streamlit dashboard for diabetes risk prediction.

Run with:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
IMG_DIR = ROOT / "images"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "final_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    defaults = joblib.load(MODELS_DIR / "feature_defaults.pkl")
    return model, feature_names, defaults


model, feature_names, defaults = load_artifacts()

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ---------- Section 1 — Introduction ----------
st.title("Diabetes Risk Predictor")
st.markdown(
    """
This dashboard predicts an individual's diabetes risk using a **Random Forest**
classifier trained on the Pima Indians Diabetes dataset (768 patients, 8 health
indicators). Adjust the sliders in the sidebar to see the predicted risk update
in real time.

> **Disclaimer.** This is an educational project. It is not a medical device and
> must not be used for clinical decision-making.
"""
)

# ---------- Section 2 — Patient inputs ----------
st.sidebar.header("Patient inputs")

SLIDER_CONFIG = {
    "Pregnancies":              {"min": 0,    "max": 17,   "step": 1,    "help": "Number of times pregnant"},
    "Glucose":                  {"min": 40,   "max": 250,  "step": 1,    "help": "Plasma glucose, 2-h oral glucose tolerance test (mg/dL)"},
    "BloodPressure":            {"min": 30,   "max": 130,  "step": 1,    "help": "Diastolic blood pressure (mm Hg)"},
    "SkinThickness":            {"min": 5,    "max": 100,  "step": 1,    "help": "Triceps skinfold thickness (mm)"},
    "Insulin":                  {"min": 10,   "max": 850,  "step": 1,    "help": "2-h serum insulin (mu U/ml)"},
    "BMI":                      {"min": 15.0, "max": 70.0, "step": 0.1,  "help": "Body mass index (kg/m²)"},
    "DiabetesPedigreeFunction": {"min": 0.05, "max": 2.5,  "step": 0.01, "help": "Genetic risk score from family history"},
    "Age":                      {"min": 18,   "max": 90,   "step": 1,    "help": "Age in years"},
}

user_input = {}
for feat in feature_names:
    cfg = SLIDER_CONFIG[feat]
    default = float(defaults[feat])
    # clamp default into the slider range
    default = max(cfg["min"], min(cfg["max"], default))
    if isinstance(cfg["step"], float):
        user_input[feat] = st.sidebar.slider(
            feat, float(cfg["min"]), float(cfg["max"]), float(default),
            step=cfg["step"], help=cfg["help"],
        )
    else:
        user_input[feat] = st.sidebar.slider(
            feat, int(cfg["min"]), int(cfg["max"]), int(default),
            step=cfg["step"], help=cfg["help"],
        )

X_user = pd.DataFrame([user_input], columns=feature_names)

# ---------- Section 3 — Prediction ----------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Prediction")
    prob = float(model.predict_proba(X_user)[0, 1])
    pred = int(prob >= 0.5)

    pct = prob * 100
    st.metric(label="Predicted diabetes risk", value=f"{pct:.1f}%")
    st.progress(min(prob, 1.0))

    if pred == 1:
        st.error(
            f"**High risk** — the model predicts diabetic at probability {pct:.1f}%."
        )
    else:
        st.success(
            f"**Lower risk** — the model predicts non-diabetic at probability "
            f"{(1 - prob) * 100:.1f}%."
        )

    with st.expander("Input summary"):
        st.dataframe(X_user.T.rename(columns={0: "value"}))

with col_right:
    st.subheader("Where this risk sits")
    fig, ax = plt.subplots(figsize=(5, 1.4))
    ax.barh([""], [100], color="#E6E6E6")
    ax.barh([""], [pct], color="#DD8452" if pred else "#4C72B0")
    ax.axvline(50, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Predicted probability of diabetes (%)")
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    st.pyplot(fig)

    st.caption(
        "Threshold is 0.5 — patients above the dashed line are predicted diabetic. "
        "In a clinical deployment you would tune this threshold to trade off "
        "false negatives against false positives."
    )

st.divider()

# ---------- Section 4 — Model insights ----------
st.subheader("Model insights")

tab_imp, tab_shap, tab_roc, tab_cm = st.tabs(
    ["Feature importance", "SHAP summary", "ROC curves", "Confusion matrix"]
)

with tab_imp:
    st.markdown(
        "Random Forest impurity-based importance. Higher = the feature reduces "
        "prediction uncertainty more when split on."
    )
    st.image(str(IMG_DIR / "feature_importance_rf.png"), use_container_width=True)

with tab_shap:
    st.markdown(
        "SHAP values explain how each feature pushes a prediction up or down. "
        "Red = high feature value, blue = low. Glucose is the dominant driver."
    )
    st.image(str(IMG_DIR / "shap_summary.png"), use_container_width=True)

with tab_roc:
    st.markdown(
        "ROC curves let us compare models across all classification thresholds. "
        "Higher AUC = better separation between diabetic and non-diabetic."
    )
    st.image(str(IMG_DIR / "roc_curves.png"), use_container_width=True)

with tab_cm:
    st.markdown(
        "Confusion matrix on the held-out test set (Random Forest, default 0.5 "
        "threshold). The off-diagonals are the errors — bottom-left is the "
        "clinically expensive one (a missed diabetic)."
    )
    st.image(str(IMG_DIR / "confusion_matrix_rf.png"), use_container_width=True)

st.divider()
st.caption(
    "Built with Streamlit · scikit-learn Random Forest · Pima Indians Diabetes Dataset"
)

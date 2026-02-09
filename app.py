# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# ---------- Page setup ----------
st.set_page_config(page_title="Heart Disease Classification", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease – ML Classification Demo")

st.markdown("""
Upload a **test CSV** with the same feature columns used in training.
- If your CSV includes **`target`** (or **`num`**) we will compute **metrics** and show a **confusion matrix** and **classification report**.
- If your CSV has **no target**, we will output **predictions** only.
""")

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "artifacts"

# ---------- Cached loaders ----------
@st.cache_resource(show_spinner=False)
def load_feature_cols():
    with open(MODEL_DIR / "feature_columns.json", "r") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_scaler():
    # Only used for LR, KNN, NB. Keep as separate loader in case it's missing.
    return joblib.load(MODEL_DIR / "scaler.pkl")

@st.cache_resource(show_spinner=False)
def load_models():
    return {
        "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.pkl"),
        "Decision Tree":       joblib.load(MODEL_DIR / "decision_tree.pkl"),
        "KNN":                 joblib.load(MODEL_DIR / "knn.pkl"),
        "Naive Bayes":         joblib.load(MODEL_DIR / "naive_bayes.pkl"),
        "Random Forest":       joblib.load(MODEL_DIR / "random_forest.pkl"),
        "XGBoost":             joblib.load(MODEL_DIR / "xgboost.pkl"),
    }

# ---------- Try to load artifacts with helpful errors ----------
try:
    feature_cols = load_feature_cols()
except FileNotFoundError:
    st.error("`feature_columns.json` not found in `model/artifacts/`. "
             "Please run training to generate artifacts (or upload them).")
    st.stop()

try:
    MODELS = load_models()
except FileNotFoundError as e:
    st.error(f"Model artifact missing: {e}. "
             "Please ensure all pickles exist in `model/artifacts/`.")
    st.stop()

# Scaler is needed only for some models; we will load it lazily later.

# ---------- UI controls ----------
model_choice = st.selectbox("Select a trained model", list(MODELS.keys()))
uploaded = st.file_uploader("Upload test CSV", type=["csv"])

# ---------- Show training metrics if available ----------
metrics_path = MODEL_DIR / "metrics.csv"
with st.expander("📊 Training Metrics (saved)", expanded=False):
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path))
    else:
        st.info("No `metrics.csv` found yet. Train models to populate this table.")

# ---------- Helpers ----------
def prepare_input(df_raw: pd.DataFrame):
    df = df_raw.copy()
    target_col = None
    if "target" in df.columns:
        target_col = "target"
    elif "num" in df.columns:
        target_col = "num"
        # plain Python '>' (not HTML entity)
        df["target"] = (df["num"] > 0).astype(int)
        target_col = "target"

    # Keep only expected features (ignore extra cols)
    X = df[[c for c in feature_cols if c in df.columns]].copy()

    # Ensure column order & fill missing
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    y = None
    if target_col is not None and target_col in df.columns:
        y = df["target"].astype(int)
    return X, y

def compute_and_show_metrics(y_true, y_pred, y_prob):
    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    # AUC if both classes present and proba available
    auc = np.nan
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
    cols[1].metric("AUC", "N/A" if np.isnan(auc) else f"{auc:.4f}")
    cols[2].metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
    cols[3].metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
    cols[4].metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
    cols[5].metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T)

# ---------- Main inference ----------
if uploaded:
    df_u = pd.read_csv(uploaded)
    st.write("🔎 **Preview**", df_u.head())

    X, y_true = prepare_input(df_u)

    model = MODELS[model_choice]
    use_scaled = model_choice in {"Logistic Regression", "KNN", "Naive Bayes"}

    if use_scaled:
        try:
            scaler = load_scaler()
        except FileNotFoundError:
            st.error("`scaler.pkl` not found in `model/artifacts/`. "
                     "Please retrain (LogReg/KNN/NB require scaling).")
            st.stop()
        X_infer = scaler.transform(X)
    else:
        X_infer = X.values

    y_pred = model.predict(X_infer)
    y_prob = model.predict_proba(X_infer)[:, 1] if hasattr(model, "predict_proba") else None

    st.subheader("Predictions")
    out = df_u.copy()
    out["prediction"] = y_pred
    if y_prob is not None:
        out["prob_1"] = y_prob
    st.dataframe(out.head())

    st.download_button(
        "⬇️ Download predictions as CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

    if y_true is not None:
        st.markdown("---")
        st.subheader("Evaluation on Uploaded Data")
        compute_and_show_metrics(y_true, y_pred, y_prob)
else:
    st.info("Upload a CSV to run predictions / view metrics.")
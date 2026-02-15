import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
import joblib

# Defaults (can be overridden via environment variables)
DATA_PATH = os.getenv("DATA_PATH", "data/heart.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "model/artifacts")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV, coerce to numeric, ensure binary target (0/1), impute missing.
    Accepts either 'target' (0/1) or 'num' (0..4) which will be binarized as (num>0).
    """
    df = pd.read_csv(DATA_PATH)

    # Determine target column
    if "target" in df.columns:
        target_col = "target"
    elif "num" in df.columns:
        # Create a binary target
        df["target"] = (df["num"] > 0).astype(int)
        target_col = "target"
        df.drop(columns=["num"],inplace=True)
    else:
        raise ValueError("Expected a 'target' or 'num' column in the dataset.")

    # Keep numeric columns only (the common heart datasets are numeric/encoded)
    df = df.select_dtypes(include=[np.number])

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Median impute any missing numeric values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Save feature schema so the Streamlit app can validate uploads
    save_feature_columns(list(X.columns))

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split with fixed seed."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit a StandardScaler on X_train and transform both splits.
    Save a single shared scaler (for LR, KNN, NB).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    save_scaler(scaler)
    return X_train_s, X_test_s, scaler


def compute_metrics(y_true, y_pred, y_prob: Optional[np.ndarray] = None):
    """
    Return Accuracy, AUC (if available), Precision, Recall, F1, MCC.
    If AUC cannot be computed (e.g., single-class), returns NaN.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = np.nan
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan

    return acc, auc, prec, rec, f1, mcc


def save_model(model_name: str, model_obj):
    """Serialize a model under model/artifacts/<model_name>.pkl"""
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(model_obj, path)
    return path


def save_scaler(scaler):
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, path)
    return path


def save_feature_columns(cols):
    path = os.path.join(MODEL_DIR, "feature_columns.json")
    try:
        with open(path, "w") as f:
            json.dump(cols, f, indent=2)
    except Exception:
        # Don't block training if schema save fails for any reason
        pass


def append_metrics(model_name: str, acc, auc, prec, rec, f1, mcc):
    """
    Append/update a CSV of metrics at model/artifacts/metrics.csv.
    Re-running individual models will keep appending rows; you can
    start fresh by deleting the metrics file or using a wrapper script.
    """
    path = os.path.join(MODEL_DIR, "metrics.csv")
    row = pd.DataFrame(
        [[model_name, acc, auc, prec, rec, f1, mcc]],
        columns=["model", "accuracy", "auc", "precision", "recall", "f1", "mcc"]
    )
    if os.path.exists(path):
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)
    return path

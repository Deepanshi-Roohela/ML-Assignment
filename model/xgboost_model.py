# Train & save XGBoost (Ensemble)
from xgboost import XGBClassifier
from model.utils import (
    load_and_prepare, split_data,
    compute_metrics, save_model, append_metrics
)

MODEL_NAME = "xgboost"

def main():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = split_data(X, y)

    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc, auc, prec, rec, f1, mcc = compute_metrics(y_test, y_pred, y_prob)
    save_model(MODEL_NAME, clf)
    append_metrics(MODEL_NAME, acc, auc, prec, rec, f1, mcc)

    print(f"[{MODEL_NAME}] "
          f"Acc={acc:.4f} | AUC={'nan' if auc!=auc else f'{auc:.4f}'} | "
          f"P={prec:.4f} | R={rec:.4f} | F1={f1:.4f} | MCC={mcc:.4f}")

if __name__ == "__main__":
    main()
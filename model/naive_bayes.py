# Train & save Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from model.utils import (
    load_and_prepare, split_data, scale_features,
    compute_metrics, save_model, append_metrics
)

MODEL_NAME = "naive_bayes"

def main():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # NB benefits from scaling here (numeric continuous features)
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)

    clf = GaussianNB()
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc, auc, prec, rec, f1, mcc = compute_metrics(y_test, y_pred, y_prob)
    save_model(MODEL_NAME, clf)
    append_metrics(MODEL_NAME, acc, auc, prec, rec, f1, mcc)

    print(f"[{MODEL_NAME}] "
          f"Acc={acc:.4f} | AUC={'nan' if auc!=auc else f'{auc:.4f}'} | "
          f"P={prec:.4f} | R={rec:.4f} | F1={f1:.4f} | MCC={mcc:.4f}")

if __name__ == "__main__":
    main()
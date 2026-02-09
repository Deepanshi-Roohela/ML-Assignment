# Train & save K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from model.utils import (
    load_and_prepare, split_data, scale_features,
    compute_metrics, save_model, append_metrics
)

MODEL_NAME = "knn"

def main():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # KNN needs scaling
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)

    clf = KNeighborsClassifier(n_neighbors=7)  # adjust k if desired
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]  # available for KNN

    acc, auc, prec, rec, f1, mcc = compute_metrics(y_test, y_pred, y_prob)
    save_model(MODEL_NAME, clf)
    append_metrics(MODEL_NAME, acc, auc, prec, rec, f1, mcc)

    print(f"[{MODEL_NAME}] "
          f"Acc={acc:.4f} | AUC={'nan' if auc!=auc else f'{auc:.4f}'} | "
          f"P={prec:.4f} | R={rec:.4f} | F1={f1:.4f} | MCC={mcc:.4f}")

if __name__ == "__main__":
    main()
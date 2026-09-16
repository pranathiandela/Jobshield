from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "cleaned_dataset.csv"
MODEL_FILE = BASE_DIR / "model" / "fake_job_model.joblib"
VECTORIZER_FILE = BASE_DIR / "model" / "tfidf_vectorizer.joblib"


def main():
    df = pd.read_csv(DATA_FILE)

    X = df["job_text"].fillna("").astype(str)
    y = df["fraudulent"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    X_test_tfidf = vectorizer.transform(X_test)

    probabilities = model.predict_proba(
        X_test_tfidf
    )[:, 1]

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print()
    print(
        f"PR-AUC / Average Precision: "
        f"{pr_auc:.4f}"
    )

    print()
    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-score':<12}"
        f"{'Predicted Fraud':<18}"
    )

    print("-" * 70)

    for threshold in [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        predicted_fraud = int(
            predictions.sum()
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{predicted_fraud:<18}"
        )


if __name__ == "__main__":
    main()

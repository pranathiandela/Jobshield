from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "cleaned_dataset.csv"
MODEL_FILE = BASE_DIR / "model" / "fake_job_model.joblib"
VECTORIZER_FILE = BASE_DIR / "model" / "tfidf_vectorizer.joblib"


def main():
    df = pd.read_csv(DATA_FILE)

    X = df["job_text"].fillna("").astype(str)
    y = df["fraudulent"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    X_test_tfidf = vectorizer.transform(X_test)

    predictions = model.predict(X_test_tfidf)
    probabilities = model.predict_proba(X_test_tfidf)[:, 1]

    results = pd.DataFrame({
        "job_text": X_test.values,
        "actual": y_test.values,
        "predicted": predictions,
        "fraud_probability": probabilities,
    })

    false_positives = results[
        (results["actual"] == 0)
        & (results["predicted"] == 1)
    ].sort_values(
        "fraud_probability",
        ascending=False,
    )

    false_negatives = results[
        (results["actual"] == 1)
        & (results["predicted"] == 0)
    ].sort_values(
        "fraud_probability",
        ascending=True,
    )

    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print()
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")

    print()
    print("=" * 70)
    print("TOP FALSE POSITIVES")
    print("=" * 70)

    for i, (_, row) in enumerate(
        false_positives.head(10).iterrows(),
        start=1,
    ):
        print()
        print(f"--- False Positive #{i} ---")
        print(
            f"Fraud probability: "
            f"{row['fraud_probability']:.4f}"
        )
        print(row["job_text"][:1500])

    print()
    print("=" * 70)
    print("TOP FALSE NEGATIVES")
    print("=" * 70)

    for i, (_, row) in enumerate(
        false_negatives.head(10).iterrows(),
        start=1,
    ):
        print()
        print(f"--- False Negative #{i} ---")
        print(
            f"Fraud probability: "
            f"{row['fraud_probability']:.4f}"
        )
        print(row["job_text"][:1500])


if __name__ == "__main__":
    main()

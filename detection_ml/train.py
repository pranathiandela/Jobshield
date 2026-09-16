from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "cleaned_dataset.csv"
MODEL_DIR = BASE_DIR / "model"

VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_FILE = MODEL_DIR / "fake_job_model.joblib"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    required_columns = {"job_text", "fraudulent"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    X = df["job_text"].fillna("").astype(str)
    y = df["fraudulent"].astype(int)

    print("Dataset:")
    print(f"Total samples: {len(df)}")
    print(f"Legitimate: {(y == 0).sum()}")
    print(f"Fraudulent: {(y == 1).sum()}")
    print()

    # ---------------------------------------------------------
    # Stratified train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Train/Test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print()

    print("Training class distribution:")
    print(y_train.value_counts().sort_index())
    print()

    print("Testing class distribution:")
    print(y_test.value_counts().sort_index())
    print()

    # ---------------------------------------------------------
    # TF-IDF
    #
    # Fit ONLY on training text.
    # ---------------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=100000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF:")
    print(f"Training matrix: {X_train_tfidf.shape}")
    print(f"Testing matrix: {X_test_tfidf.shape}")
    print()

    # ---------------------------------------------------------
    # Logistic Regression
    #
    # class_weight='balanced' compensates for the minority
    # fraudulent class.
    # ---------------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    y_pred = model.predict(X_test_tfidf)
    y_probability = model.predict_proba(X_test_tfidf)[:, 1]

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print()

    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Fraudulent",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    # ---------------------------------------------------------
    # Save model artifacts
    # ---------------------------------------------------------

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print("Model artifacts saved:")
    print(f"Vectorizer: {VECTORIZER_FILE}")
    print(f"Model     : {MODEL_FILE}")


if __name__ == "__main__":
    main()

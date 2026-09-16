"""Machine-learning fake-job detector.

Loads the trained TF-IDF vectorizer and Logistic Regression model
and returns a fraud probability for a supplied job posting.
"""

from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_FILE = MODEL_DIR / "fake_job_model.joblib"


class MLDetectorError(Exception):
    """Raised when the ML detector cannot be loaded or used."""


_vectorizer = None
_model = None


def _load_model():
    """Load the vectorizer and classifier once and cache them."""

    global _vectorizer, _model

    if _vectorizer is not None and _model is not None:
        return _vectorizer, _model

    if not VECTORIZER_FILE.exists():
        raise MLDetectorError(
            f"TF-IDF vectorizer not found: {VECTORIZER_FILE}"
        )

    if not MODEL_FILE.exists():
        raise MLDetectorError(
            f"Fake-job model not found: {MODEL_FILE}"
        )

    try:
        _vectorizer = joblib.load(VECTORIZER_FILE)
        _model = joblib.load(MODEL_FILE)
    except Exception as exc:
        raise MLDetectorError(
            "The fake-job ML model could not be loaded."
        ) from exc

    return _vectorizer, _model


def predict_fake_probability(job_text):
    """Return the model's probability that the posting is fraudulent.

    Parameters
    ----------
    job_text : str
        Job title/description text to analyze.

    Returns
    -------
    float
        Fraud probability between 0.0 and 1.0.
    """

    text = (job_text or "").strip()

    if not text:
        raise MLDetectorError(
            "Job text is required for ML prediction."
        )

    vectorizer, model = _load_model()

    try:
        features = vectorizer.transform([text])
        probability = model.predict_proba(features)[0, 1]
    except Exception as exc:
        raise MLDetectorError(
            "The fake-job ML model could not analyze the supplied text."
        ) from exc

    return float(probability)


def predict_fake_job(job_text):
    """Return a structured ML prediction for a job posting."""

    probability = predict_fake_probability(job_text)

    return {
        "fraud_probability": probability,
        "fraud_percentage": round(probability * 100, 2),
        "prediction": (
            "Fraudulent"
            if probability >= 0.50
            else "Legitimate"
        ),
    }

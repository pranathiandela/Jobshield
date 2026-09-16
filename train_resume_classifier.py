"""One-time training script for JobShield's resume domain classifier.

Run manually to retrain:
    python train_resume_classifier.py
"""
import json
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

RESUME_CSV_PATH = "Resume.csv"
JOB_CSV_PATH = "job_dataset.csv"
MODEL_PATH = "resume_classifier.joblib"
KEYWORDS_PATH = "domain_keywords.json"

NON_TECH_EXCLUDE = {
    "content writer", "copywriter", "business analyst", "product manager",
    "project manager", "marketing specialist", "seo specialist",
    "sales executive", "operations manager", "digital marketing specialist",
    "market research analyst", "technical writer",
}


def _classify_tech_title(title):
    t = title.lower()
    if t in NON_TECH_EXCLUDE:
        return None

    # Priority 1: Data Science & AI/ML
    if any(k in t for k in [
        "data scientist", "data science", "machine learning", "ml engineer",
        "ai engineer", "deep learning", "nlp", "computer vision",
        "artificial intelligence", "data analyst", "bi analyst", "big data"
    ]):
        return "DATA-SCIENCE-ML"

    # Priority 2: Cybersecurity
    if any(k in t for k in [
        "cyber", "security analyst", "soc analyst", "ethical hacker",
        "incident response", "infosec", "penetration tester", "vulnerability"
    ]):
        return "CYBERSECURITY"

    # Priority 3: Cloud & DevOps
    if any(k in t for k in [
        "cloud", "devops", "site reliability", "sre", "platform engineer",
        "solutions architect", "infrastructure engineer"
    ]):
        return "CLOUD-DEVOPS"

    # Priority 4: QA & Testing
    if any(k in t for k in [
        "qa", "test automation", "tester", "quality assurance", "sdet", "test engineer"
    ]):
        return "QA-TESTING"

    # Priority 5: Mobile Development
    if any(k in t for k in [
        "android", "ios", "swift developer", "flutter", "mobile engineer", "react native"
    ]):
        return "MOBILE-DEVELOPMENT"

    # Priority 6: UI/UX Design
    if any(k in t for k in [
        "ux", "ui designer", "product designer", "interaction designer", "user experience"
    ]):
        return "UX-UI-DESIGN"

    # Priority 7: Network Engineering
    if "network" in t:
        return "NETWORK-ENGINEERING"

    # Priority 8: Generic Software Development fallback
    if any(k in t for k in [
        "software developer", "software engineer", "full stack", "backend",
        "frontend", "web developer", "java developer", "python developer",
        ".net developer", "golang developer"
    ]):
        return "SOFTWARE-DEVELOPMENT"

    return None


def _load_merged_dataset():
    resumes = pd.read_csv(RESUME_CSV_PATH)[["Resume_str", "Category"]]

    jobs = pd.read_csv(JOB_CSV_PATH)
    jobs = jobs.dropna(subset=["Title"])
    jobs["Category"] = jobs["Title"].apply(_classify_tech_title)
    jobs = jobs.dropna(subset=["Category"])

    jobs["Resume_str"] = (
        jobs["Title"].fillna("") + ". " +
        jobs["Skills"].fillna("").str.replace(";", " ") + ". " +
        jobs["Responsibilities"].fillna("").str.replace(";", " ") + ". " +
        jobs["Keywords"].fillna("").str.replace(";", " ")
    )
    jobs = jobs[["Resume_str", "Category"]]

    merged = pd.concat([resumes, jobs], ignore_index=True)
    print(
        f"Merged dataset: {len(resumes)} resumes + {len(jobs)} tech postings "
        f"= {len(merged)} total records across {merged['Category'].nunique()} categories."
    )
    return merged


def _build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=7000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=4000, C=1.0),
            cv=3
        )),
    ])


def main():
    df = _load_merged_dataset()

    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(_build_pipeline(), df["Resume_str"], df["Category"], cv=cv, scoring="accuracy")
    print(f"Cross-validated accuracy: {cv_scores.mean():.1%}\n")

    print("Training on full dataset...")
    final_pipeline = _build_pipeline()
    final_pipeline.fit(df["Resume_str"], df["Category"])

    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
"""One-time training script for JobShield's resume domain classifier.

Run this manually (not part of the Flask app) whenever you want to
(re)train the model:

    python train_resume_classifier.py

MERGES TWO DATA SOURCES:
  1. Resume.csv - the Kaggle "Resume Dataset": 2,484 real, labeled resumes
     across 24 broad job categories (Chef, Teacher, HR, Finance, etc.)
  2. job_dataset.csv - 1,068 tech job postings (Title/Skills/Responsibilities/
     Keywords) covering modern software/tech roles this first dataset is
     missing (Software Development, Data Science/ML, Cloud/DevOps,
     Cybersecurity, Mobile Development, Network Engineering, QA/Testing,
     UX/UI Design). Each posting is converted into pseudo-resume text and
     added as additional labeled training data.

Both CSVs are gitignored (large, and job_dataset.csv especially is
easy to re-obtain) - only this script's OUTPUT is committed:
  - resume_classifier.joblib   - the trained TF-IDF + Logistic Regression
                                  pipeline, trained on the MERGED data
  - domain_keywords.json       - for each of the 32 categories, the words
                                  most strongly associated with it, learned
                                  directly from the merged training data
                                  (not hand-picked) - used for missing-skill
                                  suggestions.
"""
import json
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

RESUME_CSV_PATH = "Resume.csv"
JOB_CSV_PATH = "job_dataset.csv"
MODEL_PATH = "resume_classifier.joblib"
KEYWORDS_PATH = "domain_keywords.json"

TERMS_PER_DOMAIN = 18

BOILERPLATE_TERMS = {
    "summary", "experience", "objective", "skills", "education", "references",
    "highlights", "accomplishments", "professional", "company", "name",
    "high school", "city", "state", "present", "current", "various",
    "other", "related", "general", "school diploma",
    "basics", "advanced", "fundamentals", "familiarity", "knowledge",
    "understanding", "exposure", "intermediate", "proficiency",
}

# Non-tech job titles in job_dataset.csv - already reasonably covered by
# Resume.csv's own categories, so excluded from the tech-domain merge.
NON_TECH_EXCLUDE = {
    "content writer", "copywriter", "business analyst", "product manager",
    "project manager", "marketing specialist", "seo specialist",
    "sales executive", "operations manager", "digital marketing specialist",
    "market research analyst", "technical writer",
}


def _classify_tech_title(title):
    """Groups job_dataset.csv's 218 specific titles (e.g. "Senior iOS
    Engineer", "DevOps Engineer - Fresher") into a manageable set of new
    tech domain categories."""
    t = title.lower()
    if t in NON_TECH_EXCLUDE:
        return None
    if "android" in t or "ios " in t or t.startswith("ios") or "swift developer" in t or "mobile" in t:
        return "MOBILE-DEVELOPMENT"
    if any(k in t for k in ["cyber", "security analyst", "soc analyst", "ethical hacker", "incident response", "information security"]):
        return "CYBERSECURITY"
    if any(k in t for k in ["data scientist", "data science", "machine learning", "ml engineer", "ml/ai", "ml infrastructure", "ai engineer", "ai prompt", "data engineer", "data analyst", "bi analyst", "big data"]):
        return "DATA-SCIENCE-ML"
    if any(k in t for k in ["qa engineer", "test automation", "software tester", "sdet"]):
        return "QA-TESTING"
    if any(k in t for k in ["ux", "ui designer", "product designer", "graphic designer", "interaction designer"]):
        return "UX-UI-DESIGN"
    if "network" in t:
        return "NETWORK-ENGINEERING"
    if any(k in t for k in ["cloud", "devops", "site reliability", "system engineer", "solutions architect"]):
        return "CLOUD-DEVOPS"
    if any(k in t for k in ["software developer", "software engineer", "full stack", "backend developer", "frontend developer", "web developer", "java developer", "python developer", "javascript developer", ".net developer", "game developer", "blockchain developer", "ar/vr developer", "vibe coder", "fintech engineer", "iot engineer", "robotics"]):
        return "SOFTWARE-DEVELOPMENT"
    return None  # left ungrouped rather than guessed


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
    print(f"Merged dataset: {len(resumes)} real resumes + {len(jobs)} tech job postings "
          f"= {len(merged)} total rows across {merged['Category'].nunique()} categories.")
    return merged


def _is_clean_term(term):
    if re.search(r"\d", term):
        return False
    if term in BOILERPLATE_TERMS:
        return False
    if any(word in BOILERPLATE_TERMS for word in term.split()):
        return False
    return True


def main():
    df = _load_merged_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["Resume_str"], df["Category"],
        test_size=0.2, random_state=42, stratify=df["Category"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english", max_features=6000,
            ngram_range=(1, 2), min_df=3,
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    print("Training classifier...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {accuracy:.1%}\n")
    print(classification_report(y_test, preds))

    print("Retraining on the FULL merged dataset for the version we ship...")
    pipeline.fit(df["Resume_str"], df["Category"])

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    print("Extracting data-driven skill keywords per category...")
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()

    domain_keywords = {}
    for category in clf.classes_:
        category_words = set(w.lower() for w in category.replace("-", " ").split())
        idx = list(clf.classes_).index(category)
        ranked_indices = clf.coef_[idx].argsort()[::-1]

        terms = []
        for i in ranked_indices:
            term = feature_names[i]
            if set(term.split()).issubset(category_words):
                continue
            if not _is_clean_term(term):
                continue
            terms.append(term)
            if len(terms) >= TERMS_PER_DOMAIN:
                break

        domain_keywords[category] = terms

    with open(KEYWORDS_PATH, "w") as f:
        json.dump(domain_keywords, f, indent=2)
    print(f"Saved skill keywords to {KEYWORDS_PATH}")

    print(f"\nDone. Commit {MODEL_PATH} and {KEYWORDS_PATH}")
    print(f"(do NOT commit {RESUME_CSV_PATH} or {JOB_CSV_PATH} - they stay local/gitignored).")


if __name__ == "__main__":
    main()
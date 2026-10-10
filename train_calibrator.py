"""Teach the scorer YOUR standard of "good resume".

The rule engine in resume_analyzer.py works out of the box.  This script learns a
correction on top of it from resumes that a human has scored, so the final score
tracks real judgement instead of guessed weights.

HOW TO USE
  1. Put resume texts in a folder, e.g. data/resumes/ (one .txt per resume).
  2. Make labels.csv with two columns:   file,score
        data/resumes/priya.txt,88
        data/resumes/john.txt,32
     score = what a good recruiter would give (0-100).  50+ resumes is useful,
     200+ is good.  Include clearly good, average and bad ones, in all your domains.
  3. python train_calibrator.py labels.csv
  4. It writes resume_calibrator.joblib next to resume_analyzer.py; the analyzer
     loads it automatically.  Delete that file to go back to rules-only.

SAFETY: the model is only saved if it beats the rule engine in cross-validation.
It also prints the resumes where rules and humans disagree most, so you can fix
real parsing/logic bugs instead of hiding them behind a model.
"""
import csv
import os
import sys

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from resume_analyzer import FEATURE_NAMES, extract_features


def load(labels_csv):
    base = os.path.dirname(os.path.abspath(labels_csv))
    X, y, rule, names = [], [], [], []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row["file"] if os.path.isabs(row["file"]) else os.path.join(base, row["file"])
            if not os.path.exists(path):
                path = row["file"]
            text = open(path, encoding="utf-8", errors="ignore").read()
            feats = extract_features(text)
            X.append([feats[k] for k in FEATURE_NAMES])
            rule.append(feats["rule_score"])
            y.append(float(row["score"]))
            names.append(row["file"])
    return np.array(X), np.array(y), np.array(rule), names


def main(labels_csv):
    X, y, rule, names = load(labels_csv)
    n = len(y)
    if n < 20:
        sys.exit(f"Only {n} labelled resumes. Need at least 20 (ideally 50+) for a trustworthy model.")
    if n < 60:
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        kind = "Ridge (small data)"
    else:
        model = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, random_state=0)
        kind = "GradientBoosting"
    k = min(5, n)
    cv_pred = cross_val_predict(model, X, y, cv=KFold(k, shuffle=True, random_state=0))
    mae_rule = float(np.mean(np.abs(rule - y)))
    mae_model = float(np.mean(np.abs(cv_pred - y)))
    mae_blend = float(np.mean(np.abs(0.4 * rule + 0.6 * cv_pred - y)))
    print(f"resumes: {n}   model: {kind}")
    print(f"mean abs error vs human score  | rules only: {mae_rule:.1f}   model only: {mae_model:.1f}   blend: {mae_blend:.1f}")

    gap = np.abs(rule - y)
    print("\nBiggest rule-vs-human disagreements (inspect these for logic/parsing bugs):")
    for i in np.argsort(-gap)[:8]:
        print(f"  human {y[i]:5.0f}  rules {rule[i]:5.1f}  {names[i]}")

    if mae_blend >= mae_rule:
        print("\nModel does NOT beat the rule engine on held-out data -> not saved. Add more/cleaner labels.")
        return
    model.fit(X, y)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resume_calibrator.joblib")
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "blend": 0.6}, out)
    print(f"\nSaved {out}  (blend 60% model / 40% rules)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python train_calibrator.py labels.csv")
    main(sys.argv[1])

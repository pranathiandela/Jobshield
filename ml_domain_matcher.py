"""Runtime resume domain detection + missing-skill suggestions.

Loads the pre-trained classifier (resume_classifier.joblib) and the
data-driven skill keywords (domain_keywords.json) produced by
train_resume_classifier.py. Both are committed to the repo, so this loads
instantly — no dataset, no training, no download needed to run the app.

- predict_domain(text)         -> best-fit category + confidence + ranked list
- missing_skills_for_domain()  -> stem-aware, negation-aware gap analysis
                                   against that category's real, data-driven
                                   keyword list
- screen_for_domain()          -> combines both; if a domain is explicitly
                                   chosen, skips prediction and screens
                                   against that domain directly
"""
import json
import re

import joblib

from resume_analyzer import _is_negated_context
from stemmer import stems_match

MODEL_PATH = "resume_classifier.joblib"
KEYWORDS_PATH = "domain_keywords.json"

_model = None
_domain_keywords = None


def _load():
    global _model, _domain_keywords
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _domain_keywords is None:
        with open(KEYWORDS_PATH) as f:
            _domain_keywords = json.load(f)
    return _model, _domain_keywords


def get_domain_names():
    _, keywords = _load()
    return list(keywords.keys())


CONFIDENCE_THRESHOLD = 20  # below this, the top prediction is too uncertain to present as fact


def predict_domain(resume_text):
    """Returns the model's best-fit category for this resume, plus a
    confidence score and the full ranked list of categories. `confident`
    is a soft hint for the UI (e.g. show a "double check this" note) —
    it should NEVER be used to hide the ranked_domains list. Some
    categories genuinely overlap in the training data, so the model can
    be confidently wrong; always let the user see alternatives."""
    model, _ = _load()

    if not (resume_text or "").strip():
        return None

    probabilities = model.predict_proba([resume_text])[0]
    classes = model.classes_

    ranked = sorted(zip(classes, probabilities), key=lambda p: p[1], reverse=True)
    ranked_domains = [
        {"domain": name, "score": round(float(prob) * 100)}
        for name, prob in ranked
    ]

    best = ranked_domains[0]
    runner_up = ranked_domains[1] if len(ranked_domains) > 1 else None
    gap = best["score"] - runner_up["score"] if runner_up else best["score"]

    confident = best["score"] >= CONFIDENCE_THRESHOLD and gap >= 8

    return {
        "best_domain": best["domain"],
        "best_score": best["score"],
        "confident": confident,
        "ranked_domains": ranked_domains,
    }


def _skill_status(skill, resume_lower):
    """Stem-aware, negation-aware check for whether a skill/keyword
    (possibly multi-word, e.g. "unit testing") appears in the resume."""
    skill_words = skill.lower().split()
    resume_tokens = [(m.group(0), m.start()) for m in re.finditer(r"[a-zA-Z]+", resume_lower)]

    present_unnegated = False
    present_negated = False
    span = len(skill_words)

    for i in range(len(resume_tokens) - span + 1):
        window = resume_tokens[i:i + span]
        if all(stems_match(window[j][0], skill_words[j]) for j in range(span)):
            match_start = window[0][1]
            context_before = resume_lower[max(0, match_start - 60):match_start]
            if _is_negated_context(context_before):
                present_negated = True
            else:
                present_unnegated = True
                break

    return present_unnegated, present_negated


def missing_skills_for_domain(resume_text, domain):
    _, domain_keywords = _load()
    skills = domain_keywords.get(domain, [])
    if not skills:
        return None

    resume_lower = (resume_text or "").lower()
    present, missing, negated = [], [], []

    for skill in skills:
        present_unnegated, present_negated = _skill_status(skill, resume_lower)
        if present_unnegated:
            present.append(skill)
        elif present_negated:
            missing.append(skill)
            negated.append(skill)
        else:
            missing.append(skill)

    coverage = round((len(present) / len(skills)) * 100) if skills else 0

    return {
        "domain": domain,
        "coverage": coverage,
        "present_skills": present,
        "missing_skills": missing,
        "negated_skills": negated,
        "skill_total": len(skills),
    }


def screen_for_domain(resume_text, chosen_domain=None):
    _, domain_keywords = _load()
    detection = None

    if chosen_domain and chosen_domain in domain_keywords:
        domain = chosen_domain
    else:
        detection = predict_domain(resume_text)
        if not detection:
            return None
        domain = detection["best_domain"]

    skill_report = missing_skills_for_domain(resume_text, domain)

    return {
        "domain": domain,
        "auto_detected": chosen_domain is None,
        "detection": detection,
        "skills": skill_report,
    }
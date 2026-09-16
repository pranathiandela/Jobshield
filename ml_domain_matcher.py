"""Runtime resume domain detection + hybrid skill matching.

Loads the pre-trained classifier (resume_classifier.joblib) and the
data-driven skill keywords (domain_keywords.json). Combines ML probabilities
with authentic domain keyword density to prevent false detections.
"""
import json
import re
from collections import defaultdict
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
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            _domain_keywords = json.load(f)
    return _model, _domain_keywords


def get_domain_names():
    _, keywords = _load()
    return list(keywords.keys())


CONFIDENCE_THRESHOLD = 35


def _get_keyword_density(resume_lower, resume_tokens, domain_keywords):
    """Counts how many distinct keywords each domain naturally hits in the resume."""
    scores = defaultdict(int)
    for domain, skills in domain_keywords.items():
        hit_count = 0
        for skill in skills:
            skill_clean = skill.lower().strip()
            # Direct symbol comparison
            if skill_clean in {"c++", "c#"}:
                pattern = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(skill_clean) + r"(?![a-zA-Z0-9])")
                if pattern.search(resume_lower):
                    hit_count += 1
                continue
            # Token comparison
            for token, start_pos in resume_tokens:
                if stems_match(token, skill_clean):
                    ctx = resume_lower[max(0, start_pos - 50):start_pos]
                    if not _is_negated_context(ctx):
                        hit_count += 1
                        break
        scores[domain] = hit_count
    return scores


def predict_domain(resume_text):
    """Predicts domain probabilities using a hybrid of ML + Keyword Density."""
    model, domain_keywords = _load()

    if not (resume_text or "").strip():
        return None

    resume_lower = resume_text.lower()
    resume_tokens = [(m.group(0), m.start()) for m in re.finditer(r"[a-zA-Z0-9+#]+", resume_lower)]

    # 1. Base ML model probabilities
    probabilities = model.predict_proba([resume_text])[0]
    classes = model.classes_
    ml_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

    # 2. Keyword density counts
    density_dict = _get_keyword_density(resume_lower, resume_tokens, domain_keywords)
    max_hits = max(density_dict.values()) if density_dict else 1
    if max_hits == 0:
        max_hits = 1

    # 3. Hybrid scoring: 55% ML model, 45% keyword density
    combined_scores = []
    for domain in classes:
        ml_weight = ml_dict.get(domain, 0.0)
        density_weight = density_dict.get(domain, 0) / max_hits
        # Boost domains that have strong evidence (multiple keywords)
        final_score = (ml_weight * 0.55) + (density_weight * 0.45)
        combined_scores.append((domain, final_score))

    # Normalize scores to sum to 100%
    total_val = sum(score for _, score in combined_scores) or 1.0
    ranked = sorted(combined_scores, key=lambda p: p[1], reverse=True)
    ranked_domains = [
        {"domain": name, "score": max(1, round((score / total_val) * 100))}
        for name, score in ranked
    ]

    best = ranked_domains[0]
    runner_up = ranked_domains[1] if len(ranked_domains) > 1 else None
    gap = best["score"] - (runner_up["score"] if runner_up else 0)

    # Stricter confidence criteria: must be solid score & clear lead
    confident = best["score"] >= CONFIDENCE_THRESHOLD and gap >= 10

    return {
        "best_domain": best["domain"],
        "best_score": best["score"],
        "confident": confident,
        "ranked_domains": ranked_domains,
    }


def _skill_status(skill, resume_lower, resume_tokens):
    """Accurate single-word/symbol skill match with negation awareness."""
    skill_clean = skill.lower().strip()

    if skill_clean in {"c++", "c#"}:
        pattern = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(skill_clean) + r"(?![a-zA-Z0-9])")
        match = pattern.search(resume_lower)
        if match:
            context_before = resume_lower[max(0, match.start() - 50):match.start()]
            return (False, True) if _is_negated_context(context_before) else (True, False)
        return False, False

    for token, start_pos in resume_tokens:
        if stems_match(token, skill_clean):
            context_before = resume_lower[max(0, start_pos - 50):start_pos]
            if _is_negated_context(context_before):
                return False, True
            return True, False

    return False, False


def calculate_calibrated_coverage(matched_count, total_count):
    """Realistic career scoring: 6-9 matched core skills indicates solid expertise."""
    if matched_count == 0 or total_count == 0:
        return 0
    # Scaled benchmark: 6 skills = 75%, 8 skills = 90%, 9+ skills = 95-100%
    benchmark_target = 8
    ratio = min(1.0, matched_count / benchmark_target)
    return max(15, round(ratio * 95))


def missing_skills_for_domain(resume_text, domain):
    _, domain_keywords = _load()
    skills = domain_keywords.get(domain, [])
    if not skills:
        return None

    resume_lower = (resume_text or "").lower()
    resume_tokens = [(m.group(0), m.start()) for m in re.finditer(r"[a-zA-Z0-9+#]+", resume_lower)]

    present, missing, negated = [], [], []

    for skill in skills:
        present_unnegated, present_negated = _skill_status(skill, resume_lower, resume_tokens)
        if present_unnegated:
            present.append(skill)
        elif present_negated:
            missing.append(skill)
            negated.append(skill)
        else:
            missing.append(skill)

    coverage = calculate_calibrated_coverage(len(present), len(skills))

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
    detection = predict_domain(resume_text)
    if not detection:
        return None

    if chosen_domain and chosen_domain in domain_keywords:
        domain = chosen_domain
    else:
        domain = detection["best_domain"]

    skill_report = missing_skills_for_domain(resume_text, domain)

    return {
        "domain": domain,
        "auto_detected": chosen_domain is None,
        "detection": detection,
        "skills": skill_report,
    }
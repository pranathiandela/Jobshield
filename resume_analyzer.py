"""Rule-based JobShield resume screening engine.

Mirrors the style of job_analyzer.py: a pure, modular rule-based engine so a
stronger ML model can be swapped in later without changing the UI or routes.

Two responsibilities:
  1. score_resume(text)              -> overall score + line-by-line bullet feedback
  2. match_job_description(text, jd) -> relevancy score + matched/missing keywords
analyze_resume(text, job_description=None) combines both into one result dict.
"""
import re

WEAK_VERBS = {
    "helped", "worked", "responsible", "assisted", "involved", "participated",
    "handled", "did", "tasked", "supported",
}

STRONG_VERBS = {
    "led", "built", "designed", "launched", "grew", "reduced", "increased",
    "created", "managed", "delivered", "improved", "negotiated", "automated",
    "optimized", "drove", "implemented", "architected", "scaled", "cut",
    "saved", "achieved", "developed", "coordinated", "redesigned", "analyzed",
}

VAGUE_PHRASES = [
    "various projects", "various tasks", "day to day", "day-to-day",
    "responsible for", "helped with", "worked on", "in charge of",
    "duties included", "general support",
]

METRIC_PATTERN = re.compile(r"(\d+(\.\d+)?\s*%|\$\s?\d|₹\s?\d|\b\d+[kKmM]?\+?\b)")

STOPWORDS = {
    "the", "and", "for", "with", "a", "an", "to", "of", "in", "on", "at",
    "is", "are", "be", "as", "or", "will", "we", "you", "your", "our",
    "this", "that", "must", "have", "has", "role", "job", "experience",
    "years", "year", "work", "working", "team", "including", "etc",
    "strong", "ability", "skills", "using", "who", "their", "they",
}


def _split_bullets(text):
    lines = [ln.strip(" \t-•*").strip() for ln in (text or "").splitlines()]
    return [ln for ln in lines if len(ln) > 8]


def _first_word(line):
    match = re.match(r"[A-Za-z]+", line)
    return match.group(0).lower() if match else ""


def score_resume(text):
    """Score bullets/lines for verb strength, quantified impact, and vagueness."""
    bullets = _split_bullets(text)
    if not bullets:
        return {
            "score": 0,
            "bullet_count": 0,
            "line_feedback": [],
            "summary": "No resume content was found to review. Paste your resume text or upload a file with readable content.",
        }

    line_feedback = []
    points = 0
    max_points = len(bullets) * 3  # up to 3 points per bullet: verb, metric, non-vague

    for line in bullets:
        flags = []
        first = _first_word(line)
        has_metric = bool(METRIC_PATTERN.search(line))
        is_vague = any(phrase in line.lower() for phrase in VAGUE_PHRASES)

        if first in STRONG_VERBS:
            points += 1
            flags.append("strong verb")
        elif first in WEAK_VERBS:
            flags.append("weak verb")
        else:
            flags.append("no clear action verb")

        if has_metric:
            points += 1
            flags.append("quantified")
        else:
            flags.append("missing metric")

        if is_vague:
            flags.append("vague phrasing")
        else:
            points += 1

        if "strong verb" in flags and "quantified" in flags and "vague phrasing" not in flags:
            status = "strong"
        elif "vague phrasing" in flags or "no clear action verb" in flags:
            status = "needs work"
        else:
            status = "good"

        line_feedback.append({"line": line, "status": status, "flags": flags})

    score = round((points / max_points) * 100) if max_points else 0

    if score >= 80:
        summary = "Most bullets use strong action verbs and quantified impact. Minor polish remaining."
    elif score >= 50:
        summary = "Several bullets are missing metrics or use weak/vague language. Fixing those will raise your score noticeably."
    else:
        summary = "Most bullets need stronger action verbs and measurable results. Rewrite the flagged lines first."

    return {
        "score": score,
        "bullet_count": len(bullets),
        "line_feedback": line_feedback,
        "summary": summary,
    }


def _keywords_from_jd(jd_text):
    words = re.findall(r"[A-Za-z][A-Za-z+\-.#]{1,}", jd_text or "")
    seen, keywords = set(), []
    for w in words:
        wl = w.lower().strip(".")
        if wl in STOPWORDS or len(wl) < 3:
            continue
        if wl not in seen:
            seen.add(wl)
            keywords.append(wl)
    return keywords[:40]  # cap so relevancy scoring stays meaningful


def match_job_description(resume_text, jd_text):
    """Compare resume text against a pasted job description's keywords."""
    keywords = _keywords_from_jd(jd_text)
    if not keywords:
        return None

    resume_lower = (resume_text or "").lower()
    matched = [k for k in keywords if k in resume_lower]
    missing = [k for k in keywords if k not in resume_lower]
    relevancy = round((len(matched) / len(keywords)) * 100) if keywords else 0

    return {
        "relevancy_score": relevancy,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "keyword_total": len(keywords),
    }


def analyze_resume(text, job_description=None):
    result = score_resume(text)
    jd_match = match_job_description(text, job_description) if job_description else None
    result["jd_match"] = jd_match
    return result
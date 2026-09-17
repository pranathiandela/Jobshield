"""Multi-dimensional JobShield resume screening engine.

Replaces rigid line-by-line checks with holistic ATS evaluation:
  1. Section & Structural Completeness (25 pts)
  2. Content Depth & Vocabulary Volume (25 pts)
  3. Action & Competency Density (25 pts)
  4. Credibility & Measurable Impact (25 pts)
"""
import re

STRONG_ACTION_WORDS = {
    # Engineering / Tech
    "developed", "designed", "implemented", "engineered", "built", "architected",
    "deployed", "configured", "automated", "optimized", "integrated", "programmed",
    # Research / Analysis / Legal
    "researched", "analyzed", "investigated", "evaluated", "formulated", "compiled",
    "drafted", "structured", "examined", "assessed", "reviewed", "documented",
    # Leadership / Management
    "led", "spearheaded", "managed", "coordinated", "supervised", "directed",
    "orchestrated", "guided", "mentored", "oversaw", "planned", "executed",
    # Growth / Results
    "increased", "reduced", "scaled", "delivered", "accelerated", "saved",
    "achieved", "expanded", "negotiated", "resolved", "collaborated"
}

WEAK_WORDS = {
    "helped", "assisted", "tried", "attempted", "participated", "involved"
}

SECTIONS = {
    "contact": [r"\bemail\b", r"\bphone\b", r"\blinkedin\b", r"\bgithub\b", r"@", r"\b\d{10}\b"],
    "summary": [r"summary", r"objective", r"profile", r"about me"],
    "experience_projects": [r"experience", r"project", r"projects", r"internship", r"work history"],
    "education": [r"education", r"university", r"college", r"degree", r"bachelor", r"b\.?tech", r"rgukt"],
    "skills": [r"skills", r"technical skills", r"competencies", r"technologies", r"tools"]
}

METRIC_REGEX = re.compile(r"(\b\d+(\.\d+)?%|\$\d+|\b₹\d+|\b\d+\s*(years?|months?|lpa|cr|k|m)\b|\b\d{1,4}\b)", re.IGNORECASE)


def _evaluate_structure(text):
    text_lower = text.lower()
    points = 0
    found_sections = []
    missing_sections = []

    for section, patterns in SECTIONS.items():
        if any(re.search(pat, text_lower) for pat in patterns):
            points += 5
            found_sections.append(section)
        else:
            missing_sections.append(section)

    return min(25, points), found_sections, missing_sections


def _evaluate_depth(text):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))

    # Well-developed resumes typically have 250-700 words
    if word_count < 80:
        return 5, "Thin resume content"
    if word_count < 160:
        return 12, "Moderate content volume"
    if word_count < 280:
        return 19, "Good content volume"
    return 25, "Substantial professional content"


def _evaluate_action_density(text):
    text_lower = text.lower()
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text_lower))

    action_matches = tokens.intersection(STRONG_ACTION_WORDS)
    weak_matches = tokens.intersection(WEAK_WORDS)

    count = len(action_matches)
    # 8+ distinct action verbs reflects high initiative and clarity
    if count >= 8:
        score = 25
    elif count >= 5:
        score = 20
    elif count >= 3:
        score = 15
    elif count >= 1:
        score = 10
    else:
        score = 5

    if len(weak_matches) > 3:
        score = max(5, score - 5)

    return score, list(action_matches)[:8]


def _evaluate_impact_and_credibility(text):
    metrics = METRIC_REGEX.findall(text)
    metric_count = len(metrics)

    links_present = bool(re.search(r"(linkedin\.com|github\.com|portfolio|http)", text, re.I))

    score = 0
    if metric_count >= 5:
        score += 15
    elif metric_count >= 2:
        score += 10
    elif metric_count >= 1:
        score += 6
    else:
        score += 2

    if links_present:
        score += 5
    if len(text.splitlines()) >= 15:
        score += 5

    return min(25, score), metric_count


def _build_dynamic_summary(score, sections_found, action_words, metric_count):
    points = []
    
    if "experience_projects" in sections_found and "skills" in sections_found:
        points.append("Core experience and technical proficiencies are clearly outlined.")
    
    if len(action_words) >= 4:
        sample_verbs = ", ".join(action_words[:3])
        points.append(f"Employs dynamic action verbs such as '{sample_verbs}'.")
    else:
        points.append("Consider replacing passive phrases with assertive verbs (e.g., spearheaded, engineered, compiled).")

    if metric_count >= 3:
        points.append("Features verifiable data points and quantitative indicators.")
    else:
        points.append("Adding quantifiable outcomes (percentages, project scales, or user numbers) will further improve impact.")

    if score >= 80:
        headline = "Strong, competitive resume profile with clear structural clarity."
    elif score >= 65:
        headline = "Solid resume foundation with good domain articulation."
    elif score >= 50:
        headline = "Moderate resume presentation with scope for stronger results."
    else:
        headline = "Early-stage draft requiring more comprehensive project details and metrics."

    return f"{headline} {' '.join(points)}"


def score_resume(text):
    clean_text = (text or "").strip()
    if not clean_text:
        return {
            "score": 0,
            "bullet_count": 0,
            "line_feedback": [],
            "summary": "No resume text detected. Please paste your text or upload a PDF/DOCX file."
        }

    struct_score, found_secs, missing_secs = _evaluate_structure(clean_text)
    depth_score, depth_label = _evaluate_depth(clean_text)
    action_score, action_words = _evaluate_action_density(clean_text)
    impact_score, metric_count = _evaluate_impact_and_credibility(clean_text)

    raw_score = struct_score + depth_score + action_score + impact_score
    final_score = max(15, min(98, raw_score))

    # Construct line feedback for visual inspector
    raw_lines = [l.strip() for l in clean_text.splitlines() if len(l.strip()) > 15]
    line_feedback = []
    for line in raw_lines[:15]:
        has_verb = any(v in line.lower().split()[:3] for v in STRONG_ACTION_WORDS)
        has_num = bool(METRIC_REGEX.search(line))
        
        flags = []
        if has_verb:
            flags.append("strong verb")
        if has_num:
            flags.append("quantified")
        if not flags:
            flags.append("descriptive")

        status = "strong" if (has_verb and has_num) else ("good" if (has_verb or has_num) else "needs work")
        line_feedback.append({"line": line, "status": status, "flags": flags})

    summary = _build_dynamic_summary(final_score, found_secs, action_words, metric_count)

    return {
        "score": final_score,
        "bullet_count": len(raw_lines),
        "line_feedback": line_feedback,
        "summary": summary
    }


def _keywords_from_jd(jd_text):
    stopwords = {
        "the", "and", "for", "with", "a", "an", "to", "of", "in", "on", "at",
        "is", "are", "be", "as", "or", "will", "we", "you", "your", "our",
        "this", "that", "must", "have", "has", "role", "job", "experience",
        "years", "year", "work", "team", "skills", "candidate", "position"
    }
    words = re.findall(r"[A-Za-z][A-Za-z+\-.#]{1,}", jd_text or "")
    seen, keywords = set(), []
    for w in words:
        wl = w.lower().strip(".")
        if wl in stopwords or len(wl) < 3:
            continue
        if wl not in seen:
            seen.add(wl)
            keywords.append(wl)
    return keywords[:40]


def _is_negated_context(text_before_keyword):
    words = re.findall(r"[a-z']+", text_before_keyword.lower())
    window = words[-5:]
    negation = {"no", "not", "without", "lack", "never", "none", "n't"}
    return any(w in negation or w.endswith("n't") for w in window)


def match_job_description(resume_text, jd_text):
    keywords = _keywords_from_jd(jd_text)
    if not keywords:
        return None

    resume_lower = (resume_text or "").lower()
    matched, missing, negated = [], [], []

    for k in keywords:
        pattern = re.compile(r"\b" + re.escape(k) + r"\b")
        m = pattern.search(resume_lower)
        if m:
            ctx = resume_lower[max(0, m.start() - 50):m.start()]
            if _is_negated_context(ctx):
                negated.append(k)
                missing.append(k)
            else:
                matched.append(k)
        else:
            missing.append(k)

    relevancy = round((len(matched) / len(keywords)) * 100) if keywords else 0

    return {
        "relevancy_score": relevancy,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "negated_keywords": negated,
        "keyword_total": len(keywords)
    }


def analyze_resume(text, job_description=None):
    result = score_resume(text)
    if job_description:
        result["jd_match"] = match_job_description(text, job_description)
    else:
        result["jd_match"] = None
    return result
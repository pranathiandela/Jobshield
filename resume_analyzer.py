"""Rule-based JobShield resume screening engine.

Scores candidate resumes dynamically across:
  - Action verb strength & diversity
  - Quantified impact vs. qualitative accomplishment
  - Brevity, active voice, and weak/passive phrase detection
  - Dynamic, multi-point feedback generation
"""
import re

STRONG_VERBS = {
    # Leadership & Management
    "led", "managed", "directed", "coordinated", "orchestrated", "supervised",
    "steered", "guided", "championed", "headed", "spearheaded", "mentored",
    # Development, Technical & Creation
    "built", "designed", "developed", "engineered", "implemented", "architected",
    "programmed", "coded", "constructed", "created", "authored", "deployed",
    # Research, Analysis & Investigation
    "analyzed", "researched", "investigated", "evaluated", "assessed", "audited",
    "identified", "formulated", "synthesized", "modeled", "mapped", "examined",
    # Optimization & Execution
    "reduced", "increased", "optimized", "automated", "accelerated", "streamlined",
    "delivered", "negotiated", "executed", "transformed", "boosted", "consolidated",
    "resolved", "achieved", "strengthened", "saved", "cut", "scaled", "expanded",
    "standardized", "facilitated", "integrated", "organized", "drafted"
}

WEAK_VERBS = {
    "helped", "assisted", "worked", "tried", "supported", "handled", "did",
    "participated", "contributed", "tasked", "responsible", "involved"
}

VAGUE_PHRASES = [
    "responsible for", "helped with", "worked on", "in charge of",
    "duties included", "general support", "various projects", "various tasks",
    "day to day", "day-to-day", "assigned to", "etc"
]

HEADER_KEYWORDS = {
    "education", "skills", "experience", "projects", "activities", "summary",
    "objective", "certifications", "strengths", "interests", "profile", "achievements"
}

METRIC_PATTERN = re.compile(
    r"(\b\d+(\.\d+)?\s*%|\$\s?\d|₹\s?\d|\b\d+[kKmMbB]?\+?\b|\b(one|two|three|four|five|ten|twenty|hundred|thousand)\b)",
    re.IGNORECASE
)

STOPWORDS = {
    "the", "and", "for", "with", "a", "an", "to", "of", "in", "on", "at",
    "is", "are", "be", "as", "or", "will", "we", "you", "your", "our",
    "this", "that", "must", "have", "has", "role", "job", "experience",
    "years", "year", "work", "working", "team", "including", "etc",
    "strong", "ability", "skills", "using", "who", "their", "they",
    "looking", "seeking", "candidate", "candidates", "plus", "preferred",
    "required", "ideal", "position", "responsibilities", "please",
    "someone", "person", "individual", "applicant"
}


def _is_header_or_metadata(line):
    clean = line.strip().lower()
    if len(clean) < 4:
        return True
    if clean.rstrip(":") in HEADER_KEYWORDS:
        return True
    # Emails, phone numbers, links
    if re.search(r"@|linkedin\.com|github\.com|\+?\d{2,4}[-\s]?\d{3,5}", clean):
        return True
    # Skills list separators (e.g. "Research • Documentation • Analysis")
    if clean.count("•") >= 2 or clean.count("|") >= 2 or clean.count(",") >= 4:
        return True
    return False


def _split_experience_bullets(text):
    raw_lines = (text or "").splitlines()
    bullets = []
    for raw in raw_lines:
        line = raw.strip(" \t-•*–—").strip()
        if len(line) < 15:
            continue
        if _is_header_or_metadata(line):
            continue
        bullets.append(line)
    return bullets


def _find_action_verb(line):
    """Scans the first 3 tokens of a line to support adverbs like 'Successfully led'."""
    tokens = re.findall(r"[a-zA-Z]+", line.lower())[:3]
    for token in tokens:
        if token in STRONG_VERBS:
            return "strong", token
        if token in WEAK_VERBS:
            return "weak", token
    return "none", None


def score_resume(text):
    bullets = _split_experience_bullets(text)
    if not bullets:
        return {
            "score": 40,
            "bullet_count": 0,
            "line_feedback": [],
            "summary": "Could not identify distinct experience bullets. Add clear action-oriented bullet points under your project and experience sections."
        }

    line_feedback = []
    strong_verb_count = 0
    weak_verb_count = 0
    metric_count = 0
    vague_count = 0

    for line in bullets:
        flags = []
        verb_status, matched_verb = _find_action_verb(line)
        has_metric = bool(METRIC_PATTERN.search(line))
        is_vague = any(phrase in line.lower() for phrase in VAGUE_PHRASES)

        line_score = 0

        if verb_status == "strong":
            strong_verb_count += 1
            line_score += 2
            flags.append(f"Strong action verb ({matched_verb})")
        elif verb_status == "weak":
            weak_verb_count += 1
            flags.append("Passive or weak verb")
        else:
            flags.append("Missing clear action verb")

        if has_metric:
            metric_count += 1
            line_score += 2
            flags.append("Quantified impact")
        else:
            line_score += 1  # Standard qualitative bullet still gets base credit

        if is_vague:
            vague_count += 1
            line_score = max(0, line_score - 1)
            flags.append("Vague phrasing")

        if line_score >= 3:
            status = "strong"
        elif line_score >= 2:
            status = "good"
        else:
            status = "needs work"

        line_feedback.append({
            "line": line,
            "status": status,
            "flags": flags
        })

    total_bullets = len(bullets)
    
    # Balanced composite scoring (Base + Verbs + Impact - Vagueness)
    verb_ratio = strong_verb_count / total_bullets
    metric_ratio = metric_count / total_bullets
    vague_ratio = vague_count / total_bullets

    # Calculate final quality score (scale 0-100)
    raw_score = 50 + (verb_ratio * 35) + (metric_ratio * 20) - (vague_ratio * 25)
    final_score = max(35, min(95, round(raw_score)))

    # Dynamic summary generation based on real data signals
    insights = []
    if verb_ratio >= 0.6:
        insights.append(f"{round(verb_ratio * 100)}% of your bullets start with decisive action verbs.")
    elif verb_ratio <= 0.3:
        insights.append("Several bullets lack strong action verbs (e.g., Led, Built, Analyzed, Executed).")

    if metric_ratio >= 0.4:
        insights.append("Solid quantified evidence and measurable outcomes detected across roles.")
    else:
        insights.append("Include more specific metrics, percentages, or measurable scopes to highlight accomplishment.")

    if vague_count > 0:
        insights.append(f"Replace passive phrases like 'responsible for' or 'helped with' in {vague_count} flagged line(s).")
    else:
        insights.append("Clear, direct language throughout with zero ambiguous filler phrases.")

    summary = " ".join(insights)

    return {
        "score": final_score,
        "bullet_count": total_bullets,
        "line_feedback": line_feedback,
        "summary": summary
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
    return keywords[:40]


NEGATION_WORDS = {
    "no", "not", "without", "lack", "lacking", "never", "none",
    "unfamiliar", "excluding", "except", "n't", "nor", "neither"
}


def _is_negated_context(text_before_keyword):
    words = re.findall(r"[a-z']+", text_before_keyword.lower())
    window = words[-5:]
    for w in window:
        if w in NEGATION_WORDS or w.endswith("n't"):
            return True
    return False


def _keyword_status(keyword, resume_lower):
    pattern = re.compile(r"\b" + re.escape(keyword) + r"\b")
    present_unnegated = False
    present_negated = False
    for m in pattern.finditer(resume_lower):
        context_before = resume_lower[max(0, m.start() - 60):m.start()]
        if _is_negated_context(context_before):
            present_negated = True
        else:
            present_unnegated = True
            break
    return present_unnegated, present_negated


def match_job_description(resume_text, jd_text):
    keywords = _keywords_from_jd(jd_text)
    if not keywords:
        return None

    resume_lower = (resume_text or "").lower()
    matched, missing, negated = [], [], []

    for k in keywords:
        present_unnegated, present_negated = _keyword_status(k, resume_lower)
        if present_unnegated:
            matched.append(k)
        elif present_negated:
            missing.append(k)
            negated.append(k)
        else:
            missing.append(k)

    relevancy = round((len(matched) / len(keywords)) * 100) if keywords else 0

    return {
        "relevancy_score": relevancy,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "negated_keywords": negated,
        "keyword_total": len(keywords),
    }


def analyze_resume(text, job_description=None):
    result = score_resume(text)
    jd_match = match_job_description(text, job_description) if job_description else None
    result["jd_match"] = jd_match
    return result
"""JobShield Enterprise ATS Resume Intelligence Engine.

Multi-stage pipeline:
  1. Section Segmentation & Parsing (Summary, Experience, Projects, Education, Skills)
  2. Experience-Scoped Action Leadership (Verb strength, repetitive verb fatigue penalty)
  3. Strict Business Impact Extraction (True metrics vs vague fillers like 'multiple/various')
  4. Dynamic Bullet Rewriter (Synthesizes contextual rewrite examples)
  5. ATS Format & Parsing Compliance Audit (Header detection, length balance, bullet hygiene)
"""
import re
from collections import Counter

STRONG_ACTION_VERBS = {
    "architected", "engineered", "implemented", "developed", "deployed",
    "configured", "automated", "optimized", "refactored", "integrated",
    "programmed", "designed", "constructed", "built", "centralized",
    "spearheaded", "orchestrated", "championed", "directed", "supervised",
    "mentored", "negotiated", "mobilized", "steered", "facilitated",
    "restructured", "guided", "oversaw", "managed", "coordinated",
    "formulated", "investigated", "diagnosed", "audited", "benchmarked",
    "forecasted", "reconciled", "evaluated", "analyzed", "identified",
    "modeled", "synthesized", "reviewed", "documented",
    "accelerated", "maximized", "curtailed", "streamlined", "surpassed",
    "expanded", "generated", "reduced", "increased", "boosted", "delivered"
}

WEAK_PASSIVE_STARTERS = {
    "assisted", "helped", "worked on", "responsible for", "handled",
    "participated", "involved in", "tried", "attempted", "dealt with",
    "tasked with", "duties included", "served as", "contributed to", "supported"
}

VAGUE_FILLERS = {
    "multiple", "various", "several", "a lot of", "many", "numerous",
    "different", "a number of", "some", "few", "hands-on"
}

# Synonyms to cure repetitive verb fatigue
VERB_ALTERNATIVES = {
    "developed": ["Engineered", "Architected", "Constructed", "Formulated"],
    "managed": ["Orchestrated", "Steered", "Spearheaded", "Directed"],
    "led": ["Championed", "Mobilized", "Guided", "Supervised"],
    "optimized": ["Streamlined", "Accelerated", "Enhanced", "Maximized"],
    "implemented": ["Deployed", "Executed", "Configured", "Integrated"],
    "analyzed": ["Audited", "Diagnosed", "Evaluated", "Benchmarked"]
}

# Authentic Metric Regex (Strictly excludes calendar years 1970-2035 and single non-metric digits)
METRIC_PATTERNS = [
    r"\b\d+(\.\d+)?%",                                              # 15%, 4.5%
    r"[\$€£₹]\s?\d+([.,]\d+)?(\s?[kKmMbB]|\s?million|\s?billion)?", # $50K, ₹12L
    r"\b\d+([.,]\d+)?\s?(?:x|fold)\b",                               # 3x, 10-fold
    r"\b\d+\s*(?:k|m|lakh|crore|cr)?\+?\s*(?:users|clients|customers|requests|transactions|records|engineers|members|endpoints|queries|nodes)\b",
    r"\b(?:reduced|increased|improved|saved|cut)\s*(?:by\s*)?\d+([.,]\d+)?",
    r"\b\d+\+?\s*(?:years?|months?)\s+(?:of\s+)?experience\b"
]
COMBINED_METRICS_REGEX = re.compile("|".join(f"(?:{p})" for p in METRIC_PATTERNS), re.IGNORECASE)
YEAR_ONLY_REGEX = re.compile(r"^(19\d{2}|20[0-3]\d)$")

SECTION_HEADER_MAP = {
    "experience": re.compile(r"^(?:work\s+)?(?:experience|employment|work\s+history|professional\s+background)\b", re.I),
    "projects": re.compile(r"^(?:key\s+)?(?:projects|personal\s+projects|academic\s+projects)\b", re.I),
    "education": re.compile(r"^(?:education|academics|academic\s+background|degrees?)\b", re.I),
    "skills": re.compile(r"^(?:technical\s+)?(?:skills|core\s+competencies|technologies|tools)\b", re.I),
    "summary": re.compile(r"^(?:professional\s+)?(?:summary|profile|about\s+me|objective)\b", re.I),
    "certifications": re.compile(r"^(?:certifications?|licenses?|credentials?)\b", re.I)
}


def _segment_resume_sections(text):
    """
    Splits resume text into isolated functional blocks so checks run in their proper scope.
    """
    lines = text.splitlines()
    sections = {
        "header": [],
        "summary": [],
        "experience": [],
        "projects": [],
        "education": [],
        "skills": [],
        "certifications": [],
        "other": []
    }
    
    current_sec = "header"
    
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
            
        # Check if line looks like a header (short, title-case or uppercase)
        is_header = False
        if len(stripped.split()) <= 4 and not stripped.endswith("."):
            clean_hdr = re.sub(r"[^a-zA-Z\s]", "", stripped).strip()
            for sec_name, pattern in SECTION_HEADER_MAP.items():
                if pattern.match(clean_hdr):
                    current_sec = sec_name
                    is_header = True
                    break
        
        if not is_header:
            sections[current_sec].append(stripped)
            
    return sections


def _clean_bullet(line):
    """Strips bullet symbols, numbers, and dashes from start of bullet."""
    return re.sub(r"^[\s\*\-\•\–\—\d\.\)\:]+", "", line).strip()


def _extract_metric_evidence(line):
    """Extracts valid metrics while rejecting standalone dates."""
    matches = COMBINED_METRICS_REGEX.findall(line)
    valid = []
    for m in matches:
        val = (m[0] if isinstance(m, tuple) else m).strip()
        if val and not YEAR_ONLY_REGEX.fullmatch(val):
            valid.append(val)
    return valid


def _find_vague_fillers(line):
    """Detects vague non-quantified filler words."""
    found = []
    lower = line.lower()
    for filler in VAGUE_FILLERS:
        if re.search(r"\b" + re.escape(filler) + r"\b", lower):
            found.append(filler)
    return found


def _evaluate_action_leadership(experience_lines):
    """
    Evaluates action verbs specifically inside Work Experience & Projects.
    Applies penalty for repetitive verb fatigue.
    """
    if not experience_lines:
        return 8, [], 0, 0, []

    starting_verbs = []
    weak_count = 0
    strong_count = 0

    for line in experience_lines:
        cleaned = _clean_bullet(line).lower()
        if not cleaned or len(cleaned) < 12:
            continue

        words = cleaned.split()
        first_word = words[0]
        first_three = " ".join(words[:3])

        if any(cleaned.startswith(w) or w in first_three for w in WEAK_PASSIVE_STARTERS):
            weak_count += 1
            continue

        if first_word in STRONG_ACTION_VERBS:
            strong_count += 1
            starting_verbs.append(first_word)

    # Check for Verb Fatigue (e.g., using "developed" 4 times)
    verb_counts = Counter(starting_verbs)
    fatigued_verbs = [v for v, count in verb_counts.items() if count >= 3]

    ratio = strong_count / max(len(experience_lines), 1)
    
    if ratio >= 0.50 and len(set(starting_verbs)) >= 5:
        score = 25
    elif ratio >= 0.35 and len(set(starting_verbs)) >= 3:
        score = 20
    elif ratio >= 0.20:
        score = 15
    elif strong_count >= 1:
        score = 10
    else:
        score = 5

    # Penalties
    if weak_count >= 3:
        score = max(5, score - 5)
    if len(fatigued_verbs) > 0:
        score = max(5, score - (len(fatigued_verbs) * 3))

    return score, list(set(starting_verbs)), weak_count, len(fatigued_verbs), fatigued_verbs


def _evaluate_impact_and_metrics(experience_lines):
    """
    Strictly checks metrics inside work history. Compares true metrics against vague fillers.
    """
    if not experience_lines:
        return 5, 0, [], 0

    metric_lines_count = 0
    total_metrics = []
    vague_lines_count = 0

    for line in experience_lines:
        cleaned = _clean_bullet(line)
        metrics = _extract_metric_evidence(cleaned)
        vague = _find_vague_fillers(cleaned)

        if metrics:
            metric_lines_count += 1
            total_metrics.extend(metrics)
        elif vague:
            vague_lines_count += 1

    if metric_lines_count >= 5:
        score = 25
    elif metric_lines_count >= 3:
        score = 20
    elif metric_lines_count >= 2:
        score = 15
    elif metric_lines_count >= 1:
        score = 10
    else:
        score = 4

    # Penalty for excessive vague quantifiers without real metrics
    if vague_lines_count >= 3 and metric_lines_count < 2:
        score = max(4, score - 4)

    return score, metric_lines_count, total_metrics[:6], vague_lines_count


def _generate_smart_rewrite(cleaned_line, first_word, is_passive, vague_words):
    """
    Dynamically crafts a concrete rewrite recommendation based on the line's flaw.
    """
    if is_passive:
        # Example: 'Responsible for managing cloud servers' -> 'Architected and maintained cloud servers...'
        core_phrase = re.sub(r"^(?:assisted\s+(?:with|in)?|helped\s+(?:to)?|responsible\s+for|handled|worked\s+on)\s*", "", cleaned_line, flags=re.I)
        core_phrase = core_phrase[0].upper() + core_phrase[1:] if core_phrase else "Key responsibilities"
        return f"Replace passive starter with direct ownership: 'Spearheaded {core_phrase.lower()} resulting in [Insert measurable impact, e.g., 20% efficiency boost]'."

    if vague_words:
        filler = vague_words[0]
        return f"Eliminate vague quantifier '{filler}': state the exact volume, headcount, or dollar amount (e.g., replace '{filler}' with '8+ client accounts' or '1,500 daily requests')."

    if first_word in STRONG_ACTION_VERBS:
        return "Line starts strong with an action verb, but ends without evidence. Complete it using the XYZ formula: 'by doing [Z], resulting in [X% / $Y impact]'."

    return "Restructure to Google XYZ format: 'Accomplished [X], as measured by [Y], by doing [Z]'."


def _analyze_individual_bullets(experience_lines):
    """
    Performs deep diagnostic and rewrite hints for work bullets.
    """
    diagnostics = []

    for line in experience_lines[:20]:
        cleaned = _clean_bullet(line)
        if len(cleaned) < 14:
            continue

        lower = cleaned.lower()
        words = lower.split()
        first_word = words[0] if words else ""
        first_three = " ".join(words[:3])

        is_strong_verb = first_word in STRONG_ACTION_VERBS
        is_passive = any(lower.startswith(w) or w in first_three for w in WEAK_PASSIVE_STARTERS)
        metrics = _extract_metric_evidence(cleaned)
        vague_fillers = _find_vague_fillers(cleaned)

        flags = []
        if is_strong_verb:
            flags.append("action verb")
        if metrics:
            flags.append("quantified")
        if is_passive:
            flags.append("passive phrasing")
        if vague_fillers:
            flags.append("vague quantifier")

        # Determine Status
        if is_strong_verb and metrics and not is_passive:
            status = "strong"
            suggestion = None
        elif (is_strong_verb or metrics) and not is_passive:
            status = "good"
            suggestion = _generate_smart_rewrite(cleaned, first_word, is_passive, vague_fillers)
        else:
            status = "needs work"
            suggestion = _generate_smart_rewrite(cleaned, first_word, is_passive, vague_fillers)

        diagnostics.append({
            "line": line,
            "status": status,
            "flags": flags if flags else ["unsubstantiated"],
            "suggestion": suggestion
        })

    return diagnostics


def _audit_ats_compliance(raw_text, sections):
    """
    Evaluates file hygiene and parsing risks for standard ATS readers.
    """
    warnings = []
    passes = []

    # 1. Contact Info Detection
    has_email = bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", raw_text))
    has_phone = bool(re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", raw_text))
    has_linkedin = bool(re.search(r"linkedin\.com/in/", raw_text, re.I))

    if has_email and (has_phone or has_linkedin):
        passes.append("Contact information (email, phone/LinkedIn) is cleanly formatted.")
    else:
        warnings.append("Missing standard contact points: ensure email and a phone number or LinkedIn URL are present.")

    # 2. Critical Section Check
    missing_core = []
    if not sections["experience"]:
        missing_core.append("Work Experience")
    if not sections["education"]:
        missing_core.append("Education")
    if not sections["skills"]:
        missing_core.append("Skills")

    if not missing_core:
        passes.append("Contains all three essential ATS foundational sections (Experience, Education, Skills).")
    else:
        warnings.append(f"ATS parser warning: Could not identify standard headings for: {', '.join(missing_core)}.")

    # 3. Density & Formatting Check
    word_count = len(re.findall(r"\b[a-zA-Z]{2,}\b", raw_text))
    if 300 <= word_count <= 800:
        passes.append(f"Ideal document length ({word_count} words) for 1-2 page standard format.")
    elif word_count < 250:
        warnings.append(f"Document is brief ({word_count} words). ATS engines may flag this as an incomplete profile.")
    else:
        warnings.append(f"Lengthy document ({word_count} words). High density may cause recruiter fatigue.")

    # 4. Bullet structure check
    bullet_lines = [l for l in (sections["experience"] + sections["projects"]) if len(l) > 15]
    if len(bullet_lines) >= 6:
        passes.append(f"Strong structured bullet count ({len(bullet_lines)} bullets parsed).")
    else:
        warnings.append("Low bullet density in experience sections. Use concise bullet points instead of narrative paragraphs.")

    return passes, warnings


def score_resume(text):
    """
    Main evaluation pipeline returning calibrated ATS score, diagnostics, and compliance audit.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return {
            "score": 0,
            "bullet_count": 0,
            "breakdown": {"structure": 0, "depth": 0, "action": 0, "impact": 0},
            "line_feedback": [],
            "ats_compliance": {"passes": [], "warnings": ["No text detected."]},
            "summary": "No resume text detected. Paste text or upload a document to begin."
        }

    sections = _segment_resume_sections(clean_text)
    exp_and_proj_lines = sections["experience"] + sections["projects"]
    
    # If no explicit experience header was caught, fall back to parsing bullet lines safely
    if not exp_and_proj_lines:
        exp_and_proj_lines = [l.strip() for l in clean_text.splitlines() if len(l.strip()) >= 18 and not l.strip().endswith(":")]

    # 1. Structural Completeness (25 pts)
    detected_secs = [k for k, v in sections.items() if v and k != "other" and k != "header"]
    struct_score = min(25, round(len(detected_secs) * 4.5))

    # 2. Content Substance & Depth (25 pts)
    words = re.findall(r"\b[a-zA-Z]{2,}\b", clean_text)
    word_count = len(words)
    if word_count < 120:
        depth_score = 6
    elif word_count < 250:
        depth_score = 14
    elif word_count <= 850:
        depth_score = 25
    else:
        depth_score = 18

    # 3. Action Leadership (25 pts)
    action_score, verbs_used, weak_count, fatigue_count, fatigued_verbs = _evaluate_action_leadership(exp_and_proj_lines)

    # 4. Measurable Business Impact (25 pts)
    impact_score, metric_count, sample_metrics, vague_count = _evaluate_impact_and_metrics(exp_and_proj_lines)

    total_score = struct_score + depth_score + action_score + impact_score
    final_score = max(15, min(99, total_score))

    # Line diagnostics and ATS compliance
    line_feedback = _analyze_individual_bullets(exp_and_proj_lines)
    passes, warnings = _audit_ats_compliance(clean_text, sections)

    # Contextual Summary Generation
    summary_parts = []
    if final_score >= 85:
        summary_parts.append("Outstanding, highly competitive ATS profile. Work history demonstrates authoritative leadership and quantifiable business metrics.")
    elif final_score >= 70:
        summary_parts.append("Solid profile with established professional substance, but lacks depth in quantifiable results and suffers from minor verb fatigue.")
    else:
        summary_parts.append("Foundational draft requiring structural refinement, stronger executive verbs, and elimination of passive phrasing.")

    if weak_count > 0:
        summary_parts.append(f"Identified {weak_count} bullet(s) using passive phrasing ('responsible for', 'assisted').")
    if fatigue_count > 0:
        summary_parts.append(f"Verb fatigue detected on: {', '.join(fatigued_verbs)}. Diversify your vocabulary.")
    if vague_count >= 2 and metric_count < 3:
        summary_parts.append("Detected vague quantifiers ('multiple', 'various') without measurable numbers.")

    return {
        "score": final_score,
        "bullet_count": len(exp_and_proj_lines),
        "breakdown": {
            "structure": struct_score,
            "depth": depth_score,
            "action": action_score,
            "impact": impact_score
        },
        "detected_sections": detected_secs,
        "ats_compliance": {
            "passes": passes,
            "warnings": warnings
        },
        "line_feedback": line_feedback,
        "summary": " ".join(summary_parts)
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
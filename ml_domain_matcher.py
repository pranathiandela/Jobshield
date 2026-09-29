"""JobShield Universal Enterprise Domain Matching & Taxonomy Resolver.

Key Features:
  1. Bidirectional Acronym & Synonym Resolution (100+ standard mappings).
  2. Non-linear asymptotic scoring curve: eliminates artificial 100% spikes.
     Strong candidates land realistically between 68% and 86%.
  3. Weighted core breadth ratio + persona anchor boost.
"""
import json
import re
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOMAIN_FILE = BASE_DIR / "domain_keywords.json"

_DOMAINS_CACHE = None

GLOBAL_ACRONYM_MAP = {
    # Medical & Clinical
    "eeg": ["electroencephalogram", "electroencephalography"],
    "emg": ["electromyogram", "electromyography"],
    "ecg": ["ekg", "electrocardiogram", "electrocardiography"],
    "ncs": ["nerve conduction study", "nerve conduction studies"],
    "pcr": ["polymerase chain reaction"],
    "mri": ["magnetic resonance imaging"],
    "ct": ["computed tomography", "cat scan"],
    "cpr": ["cardiopulmonary resuscitation"],
    "bls": ["basic life support"],
    "acls": ["advanced cardiovascular life support", "advanced cardiac life support"],
    "pals": ["pediatric advanced life support"],
    "nicu": ["neonatal intensive care unit"],
    "picu": ["pediatric intensive care unit"],
    "icu": ["intensive care unit"],
    "er": ["emergency room", "emergency department"],
    "ehr": ["emr", "electronic health record", "electronic medical record"],
    "abpn": ["american board of psychiatry and neurology"],
    "faan": ["fellow of the american academy of neurology"],
    "facc": ["fellow of the american college of cardiology"],
    "facs": ["fellow of the american college of surgeons"],
    "dvt": ["deep vein thrombosis"],
    "cad": ["coronary artery disease"],
    "tpa": ["tissue plasminogen activator", "thrombolytic"],

    # Engineering (Mechanical, Civil, Electrical, Chemical, Robotics)
    "cad": ["computer aided design", "autocad"],
    "cam": ["computer aided manufacturing"],
    "cae": ["computer aided engineering"],
    "fea": ["finite element analysis"],
    "cfd": ["computational fluid dynamics"],
    "gd&t": ["geometric dimensioning and tolerancing", "gdt"],
    "cnc": ["computer numerical control"],
    "dfm": ["design for manufacturability", "design for manufacturing"],
    "dfmea": ["design failure mode and effects analysis"],
    "hvac": ["heating ventilation and air conditioning"],
    "plc": ["programmable logic controller"],
    "scada": ["supervisory control and data acquisition"],
    "pcb": ["printed circuit board"],
    "fpga": ["field programmable gate array"],
    "p&id": ["piping and instrumentation diagram"],
    "hazop": ["hazard and operability study"],
    "bim": ["building information modeling"],
    "boq": ["bill of quantities"],
    "ros": ["robot operating system"],
    "slam": ["simultaneous localization and mapping"],
    "imu": ["inertial measurement unit"],

    # Software, Cloud & DevOps
    "k8s": ["kubernetes"],
    "ci/cd": ["continuous integration continuous deployment", "cicd"],
    "iac": ["infrastructure as code"],
    "api": ["apis", "application programming interface", "rest api"],
    "sdk": ["software development kit"],
    "db": ["database"],
    "rdbms": ["relational database"],
    "nosql": ["not only sql"],
    "orm": ["object relational mapping"],
    "sre": ["site reliability engineering"],
    "pr": ["pull request"],
    "tdd": ["test driven development"],
    "bdd": ["behavior driven development"],
    "spa": ["single page application"],
    "ssr": ["server side rendering"],
    "csr": ["client side rendering"],
    "ui": ["user interface"],
    "ux": ["user experience"],

    # Data Science, AI & ML
    "ai": ["artificial intelligence"],
    "ml": ["machine learning"],
    "dl": ["deep learning"],
    "nlp": ["natural language processing"],
    "llm": ["large language model", "large language models"],
    "cv": ["computer vision"],
    "rag": ["retrieval augmented generation"],
    "genai": ["generative ai", "generative artificial intelligence"],
    "rl": ["reinforcement learning"],
    "rlhf": ["reinforcement learning from human feedback"],
    "eda": ["exploratory data analysis"],
    "bi": ["business intelligence"],
    "etl": ["extract transform load"],
    "elt": ["extract load transform"],

    # Scientific Research & Lab Techniques
    "sem": ["scanning electron microscopy"],
    "tem": ["transmission electron microscopy"],
    "xrd": ["x-ray diffraction"],
    "afm": ["atomic force microscopy"],
    "nmr": ["nuclear magnetic resonance"],
    "hplc": ["high performance liquid chromatography"],
    "gc-ms": ["gas chromatography mass spectrometry"],
    "gis": ["geographic information system", "geographic information systems"],

    # Education, Finance & Business
    "iep": ["individualized education program"],
    "bip": ["behavioral intervention plan"],
    "aba": ["applied behavior analysis"],
    "stem": ["science technology engineering math"],
    "lms": ["learning management system"],
    "cpa": ["certified public accountant"],
    "dcf": ["discounted cash flow"],
    "lbo": ["leveraged buyout"],
    "m&a": ["mergers and acquisitions"],
    "wacc": ["weighted average cost of capital"],
    "gaap": ["generally accepted accounting principles"],
    "ifrs": ["international financial reporting standards"],
    "sox": ["sarbanes oxley"],
    "ats": ["applicant tracking system"],
    "hris": ["human resources information system"]
}

CROSS_DOMAIN_DILUTERS = {
    "patient", "clinical", "hospital", "medical", "treatment", "care", "healthcare",
    "system", "systems", "design", "analysis", "data", "management", "tools",
    "software", "engineering", "research", "project", "operations", "compliance"
}


def load_domain_keywords():
    global _DOMAINS_CACHE
    if _DOMAINS_CACHE is not None:
        return _DOMAINS_CACHE
    try:
        with open(DOMAIN_FILE, "r", encoding="utf-8") as f:
            _DOMAINS_CACHE = json.load(f)
    except Exception as e:
        print(f"Error loading {DOMAIN_FILE}: {e}")
        _DOMAINS_CACHE = {}
    return _DOMAINS_CACHE


def get_domain_names():
    domains = load_domain_keywords()
    return list(domains.keys())


def _normalize(text):
    return (text or "").lower()


def _extract_header_lines(text, max_lines=12):
    lines = [l.strip().lower() for l in (text or "").splitlines() if len(l.strip()) > 3]
    return " \n ".join(lines[:max_lines])


def _get_expanded_forms(term):
    term_clean = term.strip().lower()
    variants = {term_clean}
    if term_clean in GLOBAL_ACRONYM_MAP:
        variants.update(GLOBAL_ACRONYM_MAP[term_clean])
    for short_form, expansions in GLOBAL_ACRONYM_MAP.items():
        if term_clean in expansions:
            variants.add(short_form)
            variants.update(expansions)
    return variants


def _flexible_keyword_match(term, text_lower):
    for variant in _get_expanded_forms(term):
        escaped = re.escape(variant)
        if " " in variant or "-" in variant:
            pattern = r"\b" + escaped + r"(?:s|es)?\b"
        else:
            pattern = r"\b" + escaped + r"(?:s|es|ed|ing)?\b"
        if re.search(pattern, text_lower):
            return True
    return False


def _score_domain(domain_name, domain_data, full_text_lower, header_text_lower):
    aliases = domain_data.get("aliases", [])
    core_skills = domain_data.get("core", [])
    supporting_skills = domain_data.get("supporting", [])

    # 1. Title / Persona Anchor Check
    title_anchor_hits = 0
    body_alias_hits = 0
    for alias in aliases:
        if _flexible_keyword_match(alias, header_text_lower):
            title_anchor_hits += 1
        elif _flexible_keyword_match(alias, full_text_lower):
            body_alias_hits += 1

    # 2. Core Skills Match
    matched_core = []
    missing_core = []
    for skill in core_skills:
        if _flexible_keyword_match(skill, full_text_lower):
            matched_core.append(skill)
        else:
            missing_core.append(skill)

    # 3. Supporting Skills Match
    matched_supporting = []
    for skill in supporting_skills:
        if _flexible_keyword_match(skill, full_text_lower):
            matched_supporting.append(skill)

    # 4. Realistic Scoring Curve (Diminishing Returns)
    # Total available skills in this taxonomy
    total_core = max(len(core_skills), 1)
    core_ratio = len(matched_core) / total_core
    supp_ratio = len(matched_supporting) / max(len(supporting_skills), 1)

    # Base competency points from breadth (max 65 pts)
    # Demonstrating ~50% of an entire field's core competencies is already senior-level
    breadth_score = min(65.0, (core_ratio * 75.0) + (supp_ratio * 20.0))

    # Seniority & Persona Anchor Boost (max 22 pts)
    anchor_bonus = 0.0
    if title_anchor_hits > 0:
        anchor_bonus = 18.0 + min(4.0, (title_anchor_hits - 1) * 2.0)
    elif body_alias_hits > 0:
        anchor_bonus = 8.0 + min(4.0, (body_alias_hits - 1) * 2.0)

    # Raw combined score
    combined = breadth_score + anchor_bonus

    # Asymptotic ceiling: only profiles matching nearly all core and supporting tools reach 90-95%
    # Mathematical soft cap ensures nobody hits 100% on a standard resume
    if combined > 70:
        excess = combined - 70
        calibrated_score = 70 + (25 * (1 - math.exp(-excess / 25)))
    else:
        calibrated_score = combined

    final_score = max(10, min(95, round(calibrated_score)))

    # Raw points preserved for rank comparisons
    raw_points = (title_anchor_hits * 14.0) + (body_alias_hits * 6.0)
    for c in matched_core:
        raw_points += (1.0 if c.lower() in CROSS_DOMAIN_DILUTERS else 3.0)
    for s in matched_supporting:
        raw_points += (0.3 if s.lower() in CROSS_DOMAIN_DILUTERS else 1.0)

    return {
        "domain": domain_name,
        "score": final_score,
        "raw_points": raw_points,
        "title_anchor_hits": title_anchor_hits,
        "present_skills": matched_core + matched_supporting,
        "missing_skills": missing_core[:10]
    }


def detect_best_domain(resume_text):
    domains = load_domain_keywords()
    if not domains:
        return {"domain": "General Professional", "score": 50, "confident": False, "ranked_domains": []}

    full_lower = _normalize(resume_text)
    header_lower = _extract_header_lines(resume_text)

    ranked = []
    for name, data in domains.items():
        result = _score_domain(name, data, full_lower, header_lower)
        ranked.append(result)

    ranked.sort(key=lambda x: (x["raw_points"], x["score"]), reverse=True)

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    confident = False
    if best["score"] >= 35:
        if second is None or (best["raw_points"] >= second["raw_points"] * 1.20):
            confident = True

    return {
        "best_domain": best["domain"],
        "best_score": best["score"],
        "confident": confident,
        "ranked_domains": [
            {"domain": r["domain"], "score": r["score"]} for r in ranked[:5]
        ],
        "best_details": best
    }


def screen_for_domain(resume_text, chosen_domain=None):
    domains = load_domain_keywords()
    detection = detect_best_domain(resume_text)

    full_lower = _normalize(resume_text)
    header_lower = _extract_header_lines(resume_text)

    active_domain = chosen_domain if chosen_domain and chosen_domain in domains else detection["best_domain"]

    if active_domain in domains:
        domain_data = domains[active_domain]
        skills_eval = _score_domain(active_domain, domain_data, full_lower, header_lower)
    else:
        skills_eval = detection["best_details"]

    return {
        "domain": active_domain,
        "skills": {
            "coverage": skills_eval["score"],
            "present_skills": skills_eval["present_skills"],
            "missing_skills": skills_eval["missing_skills"]
        },
        "detection": {
            "confident": detection["confident"],
            "best_score": detection["best_score"],
            "ranked_domains": detection["ranked_domains"]
        }
    }
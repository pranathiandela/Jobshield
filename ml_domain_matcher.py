"""Calibrated Multi-Domain Hybrid Prediction Engine with Tiered Skill Multipliers, Aliases, and Realistic Benchmark Scoring."""
import json
import math
import os
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_FILE = os.path.join(CURRENT_DIR, "domain_keywords.json")


def load_domain_keywords():
    """Loads domain keywords safely from the JSON file."""
    if not os.path.exists(KEYWORDS_FILE):
        return {}
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_domain_names():
    """Returns a sorted list of all domain titles for dropdown population."""
    data = load_domain_keywords()
    return sorted(list(data.keys()))


def resolve_domain_alias(query, domain_data):
    """Matches a loose query or alias directly to an exact domain name."""
    if not query:
        return None
    q = query.strip().lower()

    # 1. Exact match
    for domain in domain_data:
        if domain.lower() == q:
            return domain

    # 2. Alias match
    for domain, content in domain_data.items():
        aliases = [a.lower() for a in content.get("aliases", [])]
        if q in aliases:
            return domain

    # 3. Substring match
    for domain in domain_data:
        if q in domain.lower() or domain.lower() in q:
            return domain

    return None


def _find_keyword_matches(skill_list, text_lower):
    """Matches words or compound phrases using regex word boundaries."""
    matches = []
    for skill in skill_list:
        clean_skill = skill.lower().strip()
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(clean_skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            matches.append(skill)
    return matches


def predict_domain_hybrid(resume_text):
    """
    Evaluates candidate text across all configured domains.
    Returns:
      top_domain: str
      confidence: float (0.0 to 1.0)
      matched_details: dict of matched core/supporting items and calibrated scores
    """
    domain_data = load_domain_keywords()
    if not domain_data or not (resume_text or "").strip():
        return "General / Unclassified", 0.0, {}

    text_lower = resume_text.lower()
    scores = {}
    matched_details = {}

    for domain, skill_tiers in domain_data.items():
        core_list = skill_tiers.get("core", [])
        supporting_list = skill_tiers.get("supporting", [])

        core_matches = _find_keyword_matches(core_list, text_lower)
        supporting_matches = _find_keyword_matches(supporting_list, text_lower)

        # Core skills count 3x heavier than supporting skills
        raw_core_points = len(core_matches) * 3.0
        raw_supp_points = len(supporting_matches) * 1.0
        raw_score = raw_core_points + raw_supp_points

        # Vocabulary depth normalization
        core_pool_size = max(len(core_list), 1)
        normalization_factor = 1.0 + math.log10(max(core_pool_size, 10) / 10.0)
        calibrated_score = round(raw_score / normalization_factor, 2)

        scores[domain] = calibrated_score
        matched_details[domain] = {
            "core_matches": core_matches,
            "supporting_matches": supporting_matches,
            "raw_core_points": raw_core_points,
            "raw_supporting_points": raw_supp_points,
            "score": calibrated_score,
        }

    # Rank domains by calibrated score descending
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_domain, top_score = ranked[0]

    if top_score == 0:
        return "General / Unclassified", 0.0, matched_details

    # Confidence: Margin relative to top 3 contenders
    top_three_sum = sum(score for _, score in ranked[:3])
    confidence = round(top_score / top_three_sum, 2) if top_three_sum > 0 else 1.0

    return top_domain, confidence, matched_details


def screen_for_domain(resume_text, chosen_domain=None):
    """
    Evaluates domain match using a weighted benchmark scoring model
    rather than dividing by the total dictionary size.
    """
    domain_data = load_domain_keywords()
    if not domain_data:
        return None

    top_domain, confidence, matched_details = predict_domain_hybrid(resume_text)

    # Resolve alias or user selection
    resolved_target = resolve_domain_alias(chosen_domain, domain_data)
    target_domain = resolved_target if resolved_target else top_domain

    if target_domain not in matched_details:
        target_domain = list(domain_data.keys())[0]

    domain_info = matched_details.get(target_domain, {})
    present_core = domain_info.get("core_matches", [])
    present_supporting = domain_info.get("supporting_matches", [])
    present_all = sorted(list(set(present_core + present_supporting)))

    all_domain_skills = domain_data.get(target_domain, {}).get("core", []) + domain_data.get(target_domain, {}).get("supporting", [])
    missing = sorted([s for s in all_domain_skills if s not in present_all])

    # ------------------------------------------------------------------
    # CALIBRATED BENCHMARK SCORING (Fixes denominator inflation)
    # ------------------------------------------------------------------
    # Weighted earned points: Core = 3x, Supporting = 1x
    earned_points = (len(present_core) * 3.0) + (len(present_supporting) * 1.0)

    # A competitive resume is expected to have ~4-5 core skills and ~2-3 supporting skills (Benchmark ~ 15.0 pts)
    BENCHMARK_TARGET = 15.0

    if earned_points == 0:
        coverage = 0
    elif earned_points >= BENCHMARK_TARGET:
        # Scale between 85% and 100% for candidates meeting or exceeding benchmark
        surplus_ratio = min(1.0, (earned_points - BENCHMARK_TARGET) / 10.0)
        coverage = round(85 + (surplus_ratio * 15))
    else:
        # Scale smoothly between 10% and 84% based on progress toward benchmark
        progress = earned_points / BENCHMARK_TARGET
        coverage = round(progress * 84)

    # ------------------------------------------------------------------
    # Build ranked domain list for the UI confidence bars
    # ------------------------------------------------------------------
    ranked_domains = []
    top_scores = sorted(
        [(dom, details.get("score", 0)) for dom, details in matched_details.items()],
        key=lambda x: x[1],
        reverse=True
    )
    max_score = top_scores[0][1] if top_scores and top_scores[0][1] > 0 else 1.0

    for dom, sc in top_scores[:4]:
        rel_pct = int(min(100, round((sc / max_score) * 100))) if max_score > 0 else 0
        ranked_domains.append({
            "domain": dom,
            "score": rel_pct
        })

    return {
        "domain": target_domain,
        "detection": {
            "best_score": int(confidence * 100),
            "confident": confidence >= 0.35,
            "ranked_domains": ranked_domains
        },
        "skills": {
            "coverage": coverage,
            "present_skills": present_all,
            "missing_skills": missing[:18]
        }
    }
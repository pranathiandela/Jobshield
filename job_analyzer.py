"""Rule-based JobShield detector engine.

The engine is intentionally modular so stronger ML/API verification can be added later
without changing the detector UI or authentication flow.
"""
import re
from urllib.parse import urlparse
from detection_ml.ml_detector import (
    MLDetectorError,
    predict_fake_job,
)
FREE_EMAILS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me", "protonmail.com", "icloud.com", "aol.com"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly"}
SUSPICIOUS_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq"}

SCAM_RULES = [
    (["pay a fee", "registration fee", "processing fee", "training fee", "deposit", "pay upfront", "security deposit"], 30, "Requests money, a fee, deposit, or upfront payment."),
    (["whatsapp only", "telegram only", "contact me on whatsapp", "contact via telegram", "message me on whatsapp"], 18, "Pushes communication to an informal channel instead of a normal hiring process."),
    (["guaranteed job", "guaranteed placement", "100% selected", "instant joining", "direct joining"], 16, "Uses unusually strong guarantees about hiring or selection."),
    (["no interview", "without interview", "no interview required"], 18, "Suggests hiring without a normal interview process."),
    (["crypto", "bitcoin", "usdt", "ethereum"], 20, "Mentions cryptocurrency in a hiring or payment context."),
    (["bank account", "otp", "one time password", "credit card", "debit card", "bank details"], 25, "Requests sensitive financial or account information."),
    (["send your aadhaar", "send aadhaar", "send pan", "identity proof before interview"], 18, "Requests sensitive identity information unusually early."),
    (["act immediately", "urgent response", "respond within 1 hour", "limited slots", "offer expires today"], 10, "Uses urgency or pressure to speed up the hiring decision."),
    (["earn ₹", "earn rs", "earn $", "work from home and earn", "easy money"], 8, "Uses potentially unrealistic or promotional earning language."),
]


def _url(value):
    value = (value or "").strip()
    if not value:
        return None
    candidate = value if re.match(r"^https?://", value, re.I) else "https://" + value
    try:
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return parsed
    except Exception:
        return None


def _domain(value):
    parsed = _url(value)
    return parsed.hostname.lower().lstrip("www.") if parsed and parsed.hostname else ""


def _email(value):
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value or "")
    return match.group(0).lower() if match else ""


def _same_or_subdomain(a, b):
    if not a or not b:
        return False
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def _salary_signal(salary, text):
    source = f"{salary or ''} {text or ''}"
    amounts = []
    for value, unit in re.findall(r"(?:₹|rs\.?|inr|\$)\s*([\d,.]+)\s*([kKmMlL]?)(?:\s*(?:per|/)?\s*(?:month|monthly|year|yr|lpa|pa))?", source, re.I):
        try:
            n = float(value.replace(",", ""))
            if unit.lower() == "k": n *= 1000
            if unit.lower() == "m": n *= 1000000
            amounts.append(n)
        except ValueError:
            pass
    if re.search(r"\b(50\s*(?:lpa|lakhs?)|1\s*crore|10000000)\b", source, re.I):
        return "Warning", 12, "Compensation claim may be unusually high and should be independently verified."
    if amounts:
        return "Reviewed", 0, "Salary information was detected and included in the assessment."
    return "Missing", 3, "No clear salary information was supplied; verify compensation independently."


def analyze_job(data=None):
    data = data or {}
    title = (data.get("job_title") or "").strip()
    company = (data.get("company_name") or "").strip()
    text = (data.get("job_text") or "").strip()
    location = (data.get("location") or "").strip()
    salary = (data.get("salary") or "").strip()
    job_url = (data.get("job_url") or "").strip()
    company_website = (data.get("company_website") or "").strip()
    recruiter_name = (data.get("recruiter_name") or "").strip()
    recruiter_contact = (data.get("recruiter_contact") or "").strip()
    recruiter_email = _email(recruiter_contact)
    combined = f"{title} {company} {text}".lower()
    # ML fake-job prediction
    ml_available = False
    ml_prediction = "Unavailable"
    ml_fraud_probability = None
    ml_fraud_percentage = None

    ml_text = f"{title}\n{text}".strip()

    if ml_text:
        try:
            ml_result = predict_fake_job(ml_text)
            ml_available = True
            ml_prediction = ml_result["prediction"]
            ml_fraud_probability = ml_result["fraud_probability"]
            ml_fraud_percentage = ml_result["fraud_percentage"]
        except MLDetectorError:
            pass
    risk = 0
    suspicious = []
    safe = []
    evidence = []

    # 1. Company authentication / domain checks
    company_domain = _domain(company_website)
    job_domain = _domain(job_url)
    company_status = "Not supplied"
    if company_website:
        parsed = _url(company_website)
        if not parsed:
            risk += 8
            company_status = "Invalid URL"
            suspicious.append("Company website does not look like a valid web URL.")
        elif parsed.scheme == "https":
            company_status = "HTTPS present"
            safe.append("Company website uses HTTPS.")
        else:
            risk += 5
            company_status = "HTTP only"
            suspicious.append("Company website does not use HTTPS.")
        if company_domain and any(company_domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            risk += 10
            suspicious.append("Company website uses a domain pattern that deserves extra verification.")
        else:
            evidence.append(f"Company domain detected: {company_domain or 'unresolved'}")
    else:
        risk += 4
        company_status = "Missing"
        suspicious.append("Company website was not supplied for independent verification.")

    # 2. Careers / job URL consistency
    if job_url:
        parsed = _url(job_url)
        if not parsed:
            risk += 8
            careers_status = "Invalid URL"
            suspicious.append("Job URL does not look like a normal web URL.")
        else:
            careers_status = "URL reviewed"
            if parsed.scheme != "https":
                risk += 4
                suspicious.append("Job URL does not use HTTPS.")
            if job_domain in SHORTENERS:
                risk += 12
                suspicious.append("Job URL uses a URL-shortening service.")
            if company_domain and job_domain:
                if _same_or_subdomain(job_domain, company_domain):
                    safe.append("Job URL and company website use matching domains.")
                    careers_status = "Domain matches"
                else:
                    risk += 10
                    suspicious.append("Job URL domain does not match the supplied company website domain.")
                    careers_status = "Domain mismatch"
            evidence.append(f"Job domain detected: {job_domain or 'unresolved'}")
    else:
        risk += 3
        careers_status = "Missing"
        suspicious.append("No job URL was supplied, so the original listing cannot be independently checked.")

    # 3. Recruiter verification
    recruiter_status = "Missing"
    if recruiter_contact:
        if recruiter_email:
            email_domain = recruiter_email.split("@", 1)[1]
            if email_domain in FREE_EMAILS:
                risk += 7
                recruiter_status = "Generic email"
                suspicious.append("Recruiter uses a free email provider; verify the person through the official company website.")
            elif company_domain and _same_or_subdomain(email_domain, company_domain):
                recruiter_status = "Company domain"
                safe.append("Recruiter email domain matches the company website domain.")
            else:
                risk += 8
                recruiter_status = "Domain mismatch"
                suspicious.append("Recruiter email domain does not match the supplied company website.")
            evidence.append(f"Recruiter email domain: {email_domain}")
        elif re.search(r"\d", recruiter_contact):
            recruiter_status = "Phone supplied"
            evidence.append("A recruiter phone/contact number was supplied; ownership is not independently confirmed by this local engine.")
        else:
            risk += 5
            recruiter_status = "Unclear contact"
            suspicious.append("Recruiter contact format could not be confidently validated.")
    else:
        risk += 3
        suspicious.append("No recruiter email or phone was supplied.")
    if recruiter_name:
        safe.append("Recruiter name was supplied for consistency checking.")

    # 4. Salary benchmarking heuristic
    salary_status, salary_points, salary_message = _salary_signal(salary, text)
    risk += salary_points
    (suspicious if salary_points else safe).append(salary_message)

    # 5. Scam pattern detection
    matched_rules = 0
    for words, points, message in SCAM_RULES:
        if any(word in combined for word in words):
            risk += points
            matched_rules += 1
            suspicious.append(message)
    scam_status = f"{matched_rules} pattern(s)" if matched_rules else "No strong patterns"
    if not matched_rules:
        safe.append("No strong rule-based scam phrases were detected in the supplied text.")

    # 6. Completeness / consistency
    missing = [name for name, value in [("job title", title), ("company name", company), ("location", location), ("salary", salary)] if not value]
    if missing:
        risk += min(8, len(missing) * 2)
        evidence.append("Missing fields: " + ", ".join(missing) + ".")
    else:
        safe.append("Core job details were supplied.")

    if title and text and title.lower() not in text.lower():
        evidence.append("Job title was supplied separately; exact title matching inside the description was not assumed.")

    risk = min(100, risk)
    score = max(0, 100 - risk)
    if score >= 70:
        verdict = "Safe"
        summary_title = "No major red flags detected"
        summary = "The supplied listing has relatively few rule-based warning signals. A Safe result is not proof of legitimacy; verify the employer through an independent official source before sharing sensitive information."
    elif score >= 40:
        verdict = "Caution"
        summary_title = "Additional verification recommended"
        summary = "Several signals need attention. Verify the company, job listing and recruiter independently before proceeding."
    else:
        verdict = "Risky"
        summary_title = "Strong warning signals detected"
        summary = "The listing contains multiple risk indicators. Do not send money, OTPs, banking information or sensitive identity documents until the employer is independently verified."

    next_steps = []
    if company_domain and company_status != "Not supplied": next_steps.append("Open the company website yourself and confirm the company and careers section are genuine.")
    else: next_steps.append("Find the company's official website independently rather than relying only on the supplied link.")
    if careers_status != "Domain matches": next_steps.append("Search the official company careers page for the exact job title and location.")
    if recruiter_status != "Company domain": next_steps.append("Verify the recruiter using contact details published on the company's official website or LinkedIn/company profile.")
    if salary_status in {"Warning", "Missing"}: next_steps.append("Compare the salary with reliable market information and the role's experience requirements.")
    if matched_rules: next_steps.append("Do not pay fees or share OTP, bank, card or identity information in response to the listing.")
    next_steps.append("Treat this automated score as a screening aid, not as a guarantee of legitimacy.")

    checks = {
        "company_authentication": {"status": company_status},
        "careers_verification": {"status": careers_status},
        "recruiter_verification": {"status": recruiter_status},
        "salary_analysis": {"status": salary_status},
        "scam_pattern_detection": {"status": scam_status},
        "contact_domain_validation": {"status": "Reviewed" if recruiter_contact or job_url or company_website else "Missing"},
        "evidence_analysis": {"status": "Completed"},
    }

    return {
        "score": score,
        "legitimacy_score": score,
        "risk_points": risk,
        "risk": verdict,
        "verdict": verdict,
        "summary_title": summary_title,
        "summary": summary,
        "checks": checks,
        "safe_signals": list(dict.fromkeys(safe)),
        "suspicious_signals": list(dict.fromkeys(suspicious)),
        "signals": list(dict.fromkeys(suspicious)),
        "evidence": list(dict.fromkeys(evidence)),
        "next_steps": list(dict.fromkeys(next_steps)),
        "detected_keywords": [w for words, _, _ in SCAM_RULES for w in words if w in combined],
   "ml_available": ml_available,
"ml_prediction": ml_prediction,
"ml_fraud_probability": ml_fraud_probability,
"ml_fraud_percentage": ml_fraud_percentage, }

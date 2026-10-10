"""Resume Quality Engine (single file).

Reads a resume the way a recruiter does, then scores it:

  1. Understands the document - finds sections in messy PDF/DOCX/flattened text, rejoins
     wrapped lines, separates job-title/date lines from real content, fixes mis-decoded characters.
  2. Judges every point on what a reader cares about - does it start with a real action, does it
     prove a result (genuine numbers only, not phone numbers or company facts), is it specific,
     or is it generic / templated / padded / buzzword-stuffed?
  3. Scores the whole resume from proportions (so freshers and veterans are judged fairly),
     then applies a learned correction fitted on human-scored real resumes, so scores track
     human judgement.  Weights are monotone: improving a resume can never lower its score.

Everything it learned (phrase list from 2,484 resumes + calibration weights) is embedded at the
bottom of this file.  Python standard library only; no downloads, no training step needed.

Public API (unchanged): analyze_resume(text, job_description=None), score_resume(text),
match_job_description(resume, jd), extract_features(text).
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter

__all__ = ["analyze_resume", "score_resume", "match_job_description",
           "extract_features", "find_metrics"]

# ======================================================================
# 1. Text normalisation & parsing
# ======================================================================

SECTION_NAMES = ["header", "summary", "experience", "projects", "education",
                 "skills", "certifications", "achievements"]
CONTENT_SECTIONS = {"experience", "projects", "achievements"}
_SECTION_WEIGHT = {"experience": 1.0, "projects": 1.0, "achievements": 0.5}

# order matters: more specific sections are tested before "experience"
_SECTION_ORDER = ["projects", "education", "skills", "certifications",
                  "achievements", "summary", "experience"]
_SECTION_PHRASES = {
    "projects": ["projects", "personal projects", "academic projects", "key projects",
                 "selected projects", "project work", "side projects", "open source",
                 "open source contributions", "project experience"],
    "education": ["education", "academics", "academic background", "academic qualifications",
                  "educational qualifications", "education training",
                  "coursework", "relevant coursework"],
    "skills": ["skills", "technical skills", "key skills", "core skills", "core competencies",
               "competencies", "technologies", "tech stack", "highlights", "skill highlights", "core qualifications", "tools", "tools technologies",
               "technical proficiency", "technical expertise", "areas expertise", "expertise",
               "languages", "programming languages", "frameworks", "databases"],
    "certifications": ["certifications", "certification", "certificates", "licenses",
                       "credentials", "courses", "training", "trainings",
                       "professional development", "additional information", "interests", "references",
                       "affiliations", "professional affiliations", "memberships"],
    "achievements": ["achievements", "accomplishments", "awards", "honors", "awards honors",
                     "publications", "leadership", "extracurricular", "extracurricular activities",
                     "activities", "volunteering", "volunteer experience",
                     "positions responsibility"],
    "summary": ["summary", "professional summary", "career summary", "executive summary",
                "profile", "professional profile", "about me", "objective",
                "career objective", "personal statement", "career overview", "qualifications", "overview"],
    "experience": ["experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "career history", "relevant experience",
                   "industry experience", "internship", "internships", "internship experience",
                   "professional background"],
}

_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
           r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_YEAR = r"(?:19|20)\d{2}"
_DATE = rf"(?:{_MONTHS}\.?,?\s+{_YEAR}|\d{{1,2}}/{_YEAR}|{_YEAR})"
_NOW = r"(?:present|current|now|ongoing|till date|to date)"
DATE_RANGE_RE = re.compile(rf"\b{_DATE}\s*(?:-|–|—|to|until)\s*(?:{_NOW}|{_DATE})\b", re.I)
SINGLE_DATE_RE = re.compile(rf"\b(?:{_MONTHS}\.?,?\s+{_YEAR}|\d{{1,2}}/{_YEAR})\b", re.I)

_BULLET_GLYPHS = "\u2022\u25cf\u25aa\u25a0\u25e6\u25cb\u25c6\u25c7\u27a2\u27a4\u25ba\u25b6\u2713\u2714\u00b7\u2023\u2043"
_BULLET_RE = re.compile(
    rf"^\s*(?:[{_BULLET_GLYPHS}]\s*|[\ue000-\uf8ff]\s*|(?:[*\-\u2013\u2014]|\d{{1,2}}[.)]|[a-z][.)])\s+)")


_MOJIBAKE = [("\u00e2\u20ac\u00a2", "\u2022"), ("\u00e2\u20ac\u201c", "-"), ("\u00e2\u20ac\u201d", "-"),
             ("\u00e2\u20ac\u2122", "'"), ("\u00e2\u20ac\u02dc", "'"), ("\u00e2\u20ac\u0153", '"'),
             ("\u00e2\u20ac\u009d", '"'), ("\u00e2\u20ac\u00a6", "..."), ("\u00ef\u00bc\u008d", "-"),
             ("\u00ef\u00bc", "-"), ("\u00c2\u00a0", " "), ("\u00c2", "")]
_PLACEHOLDER_RE = re.compile(r"\bCompany Name\b|\bCity\s*,\s*State\b", re.I)


def _normalize(text: str) -> str:
    for bad, good in _MOJIBAKE:  # UTF-8 text that was mis-decoded upstream (common in PDF exports)
        if bad in (text or ""):
            text = text.replace(bad, good)
    text = _PLACEHOLDER_RE.sub(" ", text or "")
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "  ")
    t = t.replace("\u00a0", " ").replace("\u200b", "")
    t = re.sub(r"([a-z])-\n([a-z])", r"\1\2", t)  # un-hyphenate PDF line wraps
    if any(len(ln) > 300 for ln in t.split("\n")):
        t = "\n".join(_explode_long_line(ln) if len(ln) > 300 else ln for ln in t.split("\n"))
    return t


def _explode_long_line(line: str) -> str:
    """Resumes exported as one flat string: multi-space gaps mark the lost line breaks."""
    line = DATE_RANGE_RE.sub(lambda m: re.sub(r"\s+", " ", m.group(0)), line)
    pieces = [p.strip() for p in re.split(r"\s{3,}|(?<=[.!?])\s{2}(?=\S)", line) if p.strip()]
    pieces = [p for p in pieces if not re.fullmatch(r"[-\u2013\u2014\uff0d]", p)]
    merged = []
    for p in pieces:
        if merged and (merged[-1].endswith((",", "(", "-", "\u2013")) or p[:1] in ",;)" or
                       (p[:1].islower() and not merged[-1].endswith((".", "!", "?")))):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return "\n".join(merged)


def _strip_bullet(line: str):
    m = _BULLET_RE.match(line)
    if m:
        return True, line[m.end():].strip()
    return False, line.strip()


def _section_match(key: str, exact: bool = False):
    for sec in _SECTION_ORDER:
        for p in _SECTION_PHRASES[sec]:
            if key == p or (not exact and (key.startswith(p + " ") or key.endswith(" " + p))):
                return sec
    return None


def _clean_key(s: str) -> str:
    k = re.sub(r"[^a-z ]", " ", s.lower().replace("&", " "))
    return " ".join(w for w in k.split() if w not in {"and", "of", "the"})


def _section_key(line: str):
    """Full-line section header ("WORK EXPERIENCE", "Skills:" ...)."""
    s = line.strip()
    if not s or len(s) > 60 or s.endswith((".", ",", ";")):
        return None
    key = _clean_key(s)
    if not key or len(key.split()) > 5:
        return None
    shaped = s.isupper() or s.istitle() or s.endswith(":") or \
        all(w[:1].isupper() or w.lower() in _LOWER_JOINERS or not w[:1].isalpha() for w in s.split()) or \
        any(key == p for ps in _SECTION_PHRASES.values() for p in ps)
    return _section_match(key) if shaped else None


_INLINE_HDR_RE = re.compile(r"^([A-Za-z &/]{3,40}?)\s*:\s*(\S.*)$")


def _inline_header(line: str):
    """"Skills: Python, SQL" -> ("skills", "Python, SQL")."""
    m = _INLINE_HDR_RE.match(line.strip())
    if not m:
        return None
    rest = m.group(2)
    if re.search(r"@|https?:|www\.|\.com\b", rest, re.I):
        return None
    key = _clean_key(m.group(1))
    if not key or len(key.split()) > 3:
        return None
    sec = _section_match(key, exact=True)  # "System and Product Training: ..." is a bullet, not a header
    return (sec, rest) if sec else None


def _is_meta_line(s: str) -> bool:
    """Job-title / company / date lines: structure, not achievements."""
    words = s.split()
    if len(words) > 16 or s.endswith((".", "!", "?")):
        return False
    if DATE_RANGE_RE.search(s) or SINGLE_DATE_RE.search(s):
        return True
    return "|" in s and len(words) <= 12


_LOWER_JOINERS = {"of", "and", "at", "the", "for", "in", "&", "|", "-", "–", "to"}


def _heading_like(s: str) -> bool:
    words = s.split()
    if not words or len(words) > 6 or s.endswith((".", ",", ";")):
        return False
    return s.isupper() or s.istitle() or all(
        w[:1].isupper() or w.lower() in _LOWER_JOINERS or not w[:1].isalpha() for w in words)


_OPEN_ENDERS = {"and", "or", "with", "using", "for", "to", "of", "in", "on", "by", "a", "an", "the", "via"}


def _bullet_complete(text: str) -> bool:
    words = text.rstrip().split()
    return len(words) >= 6 and not text.rstrip().endswith((",", ";", "-")) \
        and words[-1].lower() not in _OPEN_ENDERS


def _split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def _p90_width(lines):
    lens = sorted(len(l.strip()) for l in lines if len(l.strip()) >= 25)
    return lens[int(0.9 * (len(lens) - 1))] if len(lens) >= 15 else 0


_TERMINAL = (".", "!", "?")


def _parse(text: str):
    raw_text = text or ""
    flat_input = any(len(ln) > 300 for ln in raw_text.replace("\r", "\n").split("\n"))
    lines = _normalize(raw_text).split("\n")
    p90 = _p90_width(lines)
    width_ok = (not flat_input) and p90 >= 40  # real line breaks that reflect page width
    sections = {k: [] for k in SECTION_NAMES}
    seen_headers = set()
    units, loose_bullets = [], []
    stats = {"date_lines": 0, "meta_lines": 0}
    sec = "header"
    cur = None

    def new_unit(body, bullet, section):
        return {"text": body, "bullet": bullet, "section": section, "last_len": len(body)}

    def flush():
        nonlocal cur
        if cur is None:
            return
        if cur["bullet"]:
            units.append(cur)
        else:
            for sent in _split_sentences(cur["text"]):
                if len(sent.split()) >= 4:
                    units.append({"text": sent, "bullet": False, "section": cur["section"]})
        cur = None

    def is_wrap(c, line):
        """Does `line` continue the unit `c` (soft wrap) rather than start a new one?"""
        if line[:1].islower():
            return True
        if c["text"].rstrip().endswith(_TERMINAL):
            return False
        if width_ok:
            return c["last_len"] >= 0.78 * p90
        return True

    for raw in lines:
        s = raw.strip()
        if not s:
            flush()
            continue
        is_b, body = _strip_bullet(s)
        if not is_b:
            key = _section_key(s)
            if key:
                flush()
                sec = key
                seen_headers.add(sec)
                continue
            inline = _inline_header(s)
            if inline:
                flush()
                sec, rest = inline
                seen_headers.add(sec)
                s, body = rest, rest
        if sec not in CONTENT_SECTIONS:
            sections[sec].append(body)
            if is_b and sec in {"header", "summary"}:
                loose_bullets.append(body)
            continue
        sections[sec].append(body)
        if is_b:
            flush()
            cur = new_unit(body, True, sec)
            continue
        if sec == "projects" and len(s.split()) <= 12 and not s.endswith((".", "!", "?")) and \
                (cur is None or _bullet_complete(cur["text"]) or cur["text"].rstrip().endswith(_TERMINAL)):
            stats["project_titles"] = stats.get("project_titles", 0) + 1
        if _is_meta_line(s):
            flush()
            stats["meta_lines"] += 1
            if DATE_RANGE_RE.search(s) or SINGLE_DATE_RE.search(s):
                stats["date_lines"] += 1
            continue
        indented = (len(raw) - len(raw.lstrip(" "))) >= 4
        if cur is None:
            if _heading_like(s):
                stats["meta_lines"] += 1
                continue
            cur = new_unit(s, indented, sec)
            continue
        if _heading_like(s) and (_bullet_complete(cur["text"]) or cur["text"].rstrip().endswith(_TERMINAL)):
            flush()
            stats["meta_lines"] += 1
            continue
        if is_wrap(cur, s):
            cur["text"] += " " + s
            cur["last_len"] = len(s)
        else:
            was_bullet = cur["bullet"]
            flush()
            cur = new_unit(s, was_bullet or indented, sec)
    flush()

    # ---- fallbacks for resumes without recognisable experience/project headers
    if not units and loose_bullets:
        units = [{"text": b, "bullet": True, "section": "experience"} for b in loose_bullets]
    if not units:
        for ln in lines:
            for sent in _split_sentences(ln.strip()):
                if len(sent.split()) >= 7 and not (_CONTACT_LINE_RE.search(sent) and len(sent.split()) < 12):
                    units.append({"text": sent, "bullet": False, "section": "experience"})
    return sections, seen_headers, units, stats


_CONTACT_LINE_RE = re.compile(r"@|linkedin|github|\+?\d[\d\s().-]{8,}\d", re.I)

# ======================================================================
# 2. Linguistic resources (small, generalising: lemma tiers + morphology)
# ======================================================================

_IRREG = {"led": "lead", "built": "build", "ran": "run", "drove": "drive", "wrote": "write",
          "won": "win", "grew": "grow", "taught": "teach", "sold": "sell", "brought": "bring",
          "oversaw": "oversee", "undertook": "undertake", "rebuilt": "rebuild", "spoke": "speak",
          "held": "hold", "began": "begin", "kept": "keep", "chose": "choose", "made": "make",
          "gave": "give", "spent": "spend", "met": "meet", "took": "take", "got": "get",
          "became": "become", "sought": "seek", "fed": "feed", "paid": "pay", "saw": "see",
          "wrote": "write", "shot": "shoot", "bought": "buy", "drew": "draw"}

# Tier 3: ownership / leadership / change-making verbs
_TIER3 = set("""lead spearhead architect pioneer launch drive establish found transform overhaul
orchestrate champion direct head initiate negotiate secure win grow scale accelerate streamline
generate deliver mentor manage supervise oversee own optimize increase reduce boost save improve
cut automate migrate engineer redesign rebuild revamp reengineer consolidate expand surpass exceed
build design develop implement create deploy innovate invent conceive formulate devise transition
turnaround restructure modernize standardize revitalize""".split())
# Tier 2: solid, concrete doing-verbs
_TIER2 = set("""configure integrate refactor analyze analyse evaluate investigate diagnose audit benchmark
model forecast research train test debug document review write coordinate collaborate facilitate
organize organise plan prepare present conduct execute maintain monitor administer resolve
troubleshoot coach teach tutor produce publish propose program code ship operate process onboard
recruit hire sell partner curate compile extract visualize visualise predict classify detect
fine-tune finetune tune query parse scrape deliver schedule budget reconcile draft edit design
craft report translate interview survey advise consult enable apply use leverage utilize utilise
collect clean cleanse validate verify secure harden upgrade install provision containerize
containerise orchestrate simulate prototype wireframe sketch illustrate photograph film curate
cultivate nurture mediate allocate prioritize prioritise track manage serve engage convert
retain acquire close negotiate renegotiate source screen assess evaluate appraise inspect
certify calibrate compose arrange""".split())
_TIER1 = set("assist help support handle participate contribute aid attend observe shadow learn "
             "study explore familiarize familiarise expose involve attempt try deal".split())
_OUTCOME_LEMMAS = set("increase reduce improve boost cut save grow accelerate generate surpass exceed "
                      "streamline expand optimize win secure".split())
_ADJ_ED = {"experienced", "motivated", "dedicated", "skilled", "seasoned", "qualified", "detailed",
           "talented", "accomplished", "dedicated", "committed", "driven", "certified", "advanced",
           "limited", "related", "associated", "based", "named", "called", "unlimited"}

_WEAK_START_RE = re.compile(
    r"^(?:(?:was|am|were|been)\s+)?(?:responsible\s+(?:for|to)|in\s+charge\s+of|"
    r"duties\s+(?:included?|involved?)|tasked\s+with|(?:worked|working|work)\s+(?:on|with|in|for|as)\b|"
    r"involved\s+in|participat(?:ed|ing)\s+in|exposure\s+to|familiar(?:ity)?\s+with|knowledge\s+of|"
    r"part\s+of|member\s+of|helped|helping|assist(?:ed|ing)|handled|handling|dealt\s+with|"
    r"contributed\s+to|served\s+as|supported|supporting)", re.I)

_FLUFF_RE = re.compile(
    r"\b(?:hard[- ]?working|team player|go[- ]?getter|detail[- ]oriented|self[- ]motivated|"
    r"fast learner|quick learner|passionate|dynamic|results[- ]driven|synergy|"
    r"think(?:ing)? outside the box|people person|(?:excellent|strong|good) communication skills|"
    r"proven track record|highly motivated|best[- ]in[- ]class|rockstar|ninja|guru)\b", re.I)
_VAGUE_RE = re.compile(r"\b(?:various|several|numerous|a lot of|a number of|multiple|many|"
                       r"different|some|etc)\b", re.I)

_OUTCOME_RE = re.compile(
    r"\b(?:resulting in|resulted in|leading to|led to|which (?:reduced|increased|improved|saved|"
    r"enabled|allowed|helped|cut|boosted)|enabling|enabled|so that|thereby|thus|"
    r"to (?:reduce|increase|improve|boost|cut|save|enable|achieve|ensure|speed up|accelerate|"
    r"minimi[sz]e|maximi[sz]e|eliminate|prevent|support|streamline)|"
    r"(?:reducing|increasing|improving|boosting|cutting|saving|achieving|ensuring|accelerating|"
    r"minimi[sz]ing|maximi[sz]ing|eliminating|preventing|streamlining)|ensur(?:ed|ing)|"
    r"achiev(?:ed|ing)|improv(?:ed|ing)|won|awarded|ranked|top \d+|#\d+|first place|"
    r"adopted by|used by|serving)\b", re.I)

_WEAK_ALTERNATIVES = {
    "responsible for": "Led / Owned / Managed",
    "worked on": "Built / Developed / Delivered",
    "worked with": "Collaborated with / Partnered with",
    "helped": "Contributed … by doing … / Enabled",
    "assisted": "Supported … by …  (state exactly what you did)",
    "handled": "Managed / Resolved / Processed",
    "participated in": "Contributed to … by …",
    "involved in": "Built / Drove / Contributed … (say what you did)",
}


def _lemma_candidates(w: str):
    yield w
    if w in _IRREG:
        yield _IRREG[w]
    if w.endswith("ied"):
        yield w[:-3] + "y"
    if w.endswith("ed"):
        yield w[:-2]
        yield w[:-1]
        if len(w) > 4 and w[-3] == w[-4]:
            yield w[:-3]
    if w.endswith("ing"):
        yield w[:-3]
        yield w[:-3] + "e"
        if len(w) > 5 and w[-4] == w[-5]:
            yield w[:-4]
    if w.endswith("ies"):
        yield w[:-3] + "y"
    if w.endswith("es"):
        yield w[:-2]
    if w.endswith("s"):
        yield w[:-1]


def _lemma(w: str):
    for cand in _lemma_candidates(w):
        if cand in _TIER3 or cand in _TIER2 or cand in _TIER1:
            return cand
    return None


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.'/-]*")


def _tokens(text: str):
    return [t.strip(".-/'") for t in _WORD_RE.findall(text) if t.strip(".-/'")]


def _classify_word(w: str):
    lem = _lemma(w)
    if lem in _TIER3:
        return "tier3", lem
    if lem in _TIER2:
        return "tier2", lem
    if lem in _TIER1:
        return "weak", lem
    if w in _IRREG or (w.endswith("ed") and len(w) > 3 and w not in _ADJ_ED):
        return "tier2u", _IRREG.get(w, w)
    return None, None


_SKIP_LEAD = {"i", "we", "also", "then", "later", "subsequently", "additionally", "primarily", "mainly"}


def _lead_info(text: str, prose: bool = False):
    """Classify how a bullet (or, for prose, the first verb of a sentence) opens."""
    t = text.strip()
    m = _WEAK_START_RE.match(t)
    if m:
        return {"kind": "weak", "word": m.group(0).lower().strip(), "lemma": None, "first_person": False}
    toks = [x.lower() for x in _tokens(t)[:8]]
    if not toks:
        return {"kind": "noun", "word": "", "lemma": None, "first_person": False}
    first_person = toks[0] in {"i", "my", "we", "our", "me"}
    i = 0
    while i < len(toks) - 1 and (toks[i] in _SKIP_LEAD or (toks[i].endswith("ly") and len(toks[i]) > 4)):
        i += 1
    if toks[i] in {"my", "our", "me"}:
        return {"kind": "first_person", "word": toks[i], "lemma": None, "first_person": True}
    window = range(i, min(len(toks), i + 6)) if prose else [i]
    for j in window:
        kind, lem = _classify_word(toks[j])
        if kind:
            return {"kind": kind, "word": toks[j], "lemma": lem, "first_person": first_person}
    return {"kind": "noun", "word": toks[i], "lemma": None, "first_person": first_person}


# ---------------------------- quantification ------------------------------

_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_PERCENT_RE = re.compile(rf"(?:{_NUM})\s?(?:%|percent\b|per\s?cent\b)", re.I)
_MONEY_RE = re.compile(
    rf"(?:[$€£₹]\s?(?:{_NUM})(?:\s?(?:[kmb]\b|thousand|million|billion|lakhs?|crores?|cr\b))?"
    rf"|(?:{_NUM})\s?(?:lpa|lakhs?|crores?|million|billion|thousand)\b"
    rf"|\b(?:usd|inr|rs\.?|eur)\s?(?:{_NUM})(?:\s?(?:[kmb]\b|thousand|million|billion|lakhs?|crores?|cr\b))?)", re.I)
_MULT_RE = re.compile(rf"(?:(?:{_NUM})\s?(?:x\b|×|fold\b|times\b)|\b(?:doubled|tripled|quadrupled|halved|twice)\b)", re.I)
_RANK_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\s+percentile\b|\b(?:top|ranked)\s+#?\d+\b|\brank(?:ed)?\s+#?\d+\b", re.I)
_FROMTO_RE = re.compile(rf"\bfrom\s+[~$€£₹]?(?:{_NUM})\s?[A-Za-z%]{{0,4}}\s+to\s+[~$€£₹]?(?:{_NUM})", re.I)
_QTY_RE = re.compile(rf"(?<![\w.])(?P<n>{_NUM})(?P<plus>\+?)\s?(?P<unit>[A-Za-z][A-Za-z-]*)?")
_NON_UNIT = {"and", "or", "with", "for", "in", "on", "to", "using", "the", "a", "an", "by", "at",
             "of", "as", "from", "that", "which", "while", "through", "via", "into", "is", "are",
             "was", "were", "be", "it", "its", "than", "then", "also", "but", "so"}
_VERSIONISH_PREV = {"version", "release", "phase", "step", "tier", "level", "sprint", "ver", "v",
                    "section", "chapter", "grade", "class", "semester", "q", "figure", "fig"}
_NUMBER_CONTEXT_PREV = {"by", "to", "from", "of", "over", "under", "than", "about", "around",
                        "approximately", "nearly", "almost", "upto", "~", "above", "below", "exceeding"}
_MONTH_WORDS = re.compile(rf"^{_MONTHS}$", re.I)


_STREET_RE = re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,2}(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Blvd|Boulevard|Lane|Ln|Way|Court|Ct|Highway|Hwy)\b")
_ORD_STREET_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\s+(?:Street|St|Avenue|Ave)\b", re.I)
_ZIP_RE = re.compile(r"(?<![$\u20ac\u00a3\u20b9\d.,])\b\d{5}(?:-\d{4})?\b(?=\s*(?:$|[A-Z]|\d|[,.;]))")
_SALARY_RE = re.compile(r"[$\u20ac\u00a3\u20b9]\s?\d[\d,.]*\s?k?\s?(?:/|per\s|a\s|an\s)\s?(?:yr|year|hr|hour|mo|month|annum)\w*", re.I)
_HOURS_RE = re.compile(r"\b\d+\s?(?:hrs?|hours?)\s?(?:/|per\s|a\s)\s?(?:w|wk|week)\w*", re.I)
_TENURE_RE = re.compile(
    r"\b(?:\d+\+?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:plus\s*)?(?:years?|yrs?)\s+(?:of\s+)?(?:\w+\s+){0,3}?(?:experience|expertise|background)\b"
    r"|\b(?:over|more than|nearly|almost)\s+\d+\+?\s*(?:years?|yrs?)\b", re.I)
_COMPANY_DESC_RE = re.compile(
    r"\b(?:headquartered|global workforce|workforce of|founded in|spread across more than|"
    r"is (?:a|an|the) (?:leading|largest|premier|global)\b|employs (?:over|more than))", re.I)


def _scrub_for_metrics(text: str) -> str:
    """Remove numbers that are identifiers or facts about the employer, not achievements."""
    t = text
    for m in sorted(re.finditer(r"\+?\d[\d\s().-]{7,}\d", t), key=lambda m: -m.start()):
        if len(re.sub(r"\D", "", m.group(0))) >= 9:  # phone-like
            t = t[:m.start()] + " " + t[m.end():]
    for rx in (_SALARY_RE, _HOURS_RE, _STREET_RE, _ORD_STREET_RE, _ZIP_RE, _TENURE_RE):
        t = rx.sub(" ", t)
    return t


def find_metrics(text: str):
    """Return [(kind, matched_text, weight)] for genuine quantification."""
    if _COMPANY_DESC_RE.search(text):
        return []  # sentence describes the employer, not the candidate's results
    text = _scrub_for_metrics(text)
    t = DATE_RANGE_RE.sub(" ", text)
    t = SINGLE_DATE_RE.sub(" ", t)
    found, spans = [], []
    for kind, rx in (("percent", _PERCENT_RE), ("money", _MONEY_RE),
                     ("multiplier", _MULT_RE), ("change", _FROMTO_RE), ("rank", _RANK_RE)):
        for m in rx.finditer(t):
            found.append((kind, m.group(0).strip(), 1.0))
            spans.append(m.span())
    for m in _QTY_RE.finditer(t):
        if any(s <= m.start() < e for s, e in spans):
            continue
        raw, unit = m.group("n"), (m.group("unit") or "").lower()
        digits = raw.replace(",", "")
        intpart = digits.split(".")[0]
        prev_m = re.search(r"([A-Za-z~]+)\W*$", t[:m.start()])
        prev = prev_m.group(1) if prev_m else ""
        prev_l = prev.lower()
        prev_is_first = (prev_m is None) or prev_m.start(1) <= len(t) - len(t.lstrip())
        if prev_l in _VERSIONISH_PREV:
            continue
        has_unit = bool(unit) and unit not in _NON_UNIT and not _MONTH_WORDS.match(unit)
        if re.fullmatch(_YEAR, digits) and (not has_unit or m.group("unit")[:1].isupper()):
            continue
        # "Python 3", "Web 2.0": capitalised word that is not the opening verb + small number
        if (prev[:1].isupper() and not prev_is_first and len(intpart) <= 2 and "," not in raw
                and prev_l not in _NON_UNIT and prev_l not in _NUMBER_CONTEXT_PREV):
            continue
        if has_unit:
            if unit in {"st", "nd", "rd", "th"}:
                continue
            found.append(("quantity", f"{raw}{m.group('plus')} {unit}", 0.8))
        elif "," in raw or len(intpart) >= 3:
            found.append(("number", raw, 0.8))
        elif prev_l in _NUMBER_CONTEXT_PREV and len(intpart) >= 2:
            found.append(("number", raw, 0.8))
    return found


# --------------------------- specificity ----------------------------------

_COMMON_CAPS = {"I", "A", "AN", "THE", "AND", "FOR", "WITH", "OF", "IN", "ON", "TO", "BY", "AT"}


def _count_specifics(text: str) -> int:
    toks = _tokens(text)
    spec = set()
    prev_end = True
    for i, tk in enumerate(toks):
        if i == 0:
            continue
        if tk.isupper() and len(tk) >= 2 and tk not in _COMMON_CAPS:
            spec.add(tk.lower())
        elif re.search(r"[a-z][A-Z]", tk) or re.search(r"[+#]", tk) or re.search(r"\.(?:js|py|net|io)$", tk, re.I):
            spec.add(tk.lower())
        elif tk[0].isupper() and len(tk) > 1:
            spec.add(tk.lower())
    return len(spec)


_FUNCTION_WORDS = set("""a an the of in on at to for from by with without and or but as into onto over under
across through via per using while during after before between within that which who whom whose this these
those it its their his her our my your is are was were be been being has have had do does did not no
than then so thus""".split())
_BUZZ = set("""synergy strategy strategies governance infrastructure enterprise-wide cross-functional end-to-end
scalable lifecycle solutions solution optimization results excellence innovative innovation robust leverage
leveraged holistic paradigm stakeholders deliverables""".split())


def _function_word_ratio(text: str) -> float:
    toks = [t.lower() for t in _tokens(text)]
    return sum(t in _FUNCTION_WORDS for t in toks) / len(toks) if toks else 0.0


def _buzz_density(text: str) -> float:
    toks = [t.lower() for t in _tokens(text)]
    if not toks:
        return 0.0
    hits = sum((t in _BUZZ) or (_lemma(t) in _TIER3 and _lemma(t) not in {"build", "design", "develop", "create"})
               or (_lemma(t) in _TIER2 and t.endswith("ed")) for t in toks)
    return hits / len(toks)


_BOILER = None


def _boiler_set():
    """Phrases (5-word shingles) that recur verbatim across many resumes = templated wording.
    Learned from 2,484 real resumes; stored compactly at the bottom of this file."""
    global _BOILER
    if _BOILER is None:
        try:
            import base64
            import zlib
            data = zlib.decompress(base64.b64decode("".join(_BOILERPLATE_B64)))
            ids, cur, val, shift = set(), 0, 0, 0
            for byte in data:
                val |= (byte & 127) << shift
                if byte & 128:
                    shift += 7
                else:
                    cur += val
                    ids.add(cur)
                    val, shift = 0, 0
            _BOILER = ids
        except Exception:
            _BOILER = False
    return _BOILER or None


def _shingle_ids(text: str):
    import zlib
    w = re.findall(r"[a-z0-9]+", text.lower())
    return [zlib.crc32(" ".join(w[i:i + 5]).encode()) & 0xFFFFFF for i in range(max(0, len(w) - 4))]


def _boiler_fraction(text: str) -> float:
    bs = _boiler_set()
    if not bs:
        return 0.0
    ids = _shingle_ids(text)
    return sum(i in bs for i in ids) / len(ids) if len(ids) >= 3 else 0.0


def _length_quality(n: int) -> float:
    if n < 5:
        return 0.0
    if n < 8:
        return 0.4
    if n < 12:
        return 0.8
    if n <= 32:
        return 1.0
    if n <= 42:
        return 0.85
    if n <= 55:
        return 0.6
    return 0.35


# ======================================================================
# 3. Bullet analysis
# ======================================================================

_ACTION_Q = {"tier3": 1.0, "tier2": 0.85, "tier2u": 0.72, "noun": 0.40, "weak": 0.28, "first_person": 0.30}


def _analyze_bullet(text: str, prose: bool = False) -> dict:
    t = text.strip()
    n = len(t.split())
    lead = _lead_info(t, prose)
    metrics = find_metrics(t)
    outcome = bool(_OUTCOME_RE.search(t)) or lead["lemma"] in _OUTCOME_LEMMAS
    spec = _count_specifics(t)
    fluff = [m.group(0) for m in _FLUFF_RE.finditer(t)]
    vague = bool(_VAGUE_RE.search(t)) and not metrics

    action_q = _ACTION_Q[lead["kind"]] * (0.92 if lead.get("first_person") else 1.0)
    metric_w = max((m[2] for m in metrics), default=0.0)
    if metric_w:
        impact_q = 0.75 + 0.25 * metric_w + (0.0 if not outcome else 0.0)
    elif outcome:
        impact_q = 0.55
    else:
        impact_q = 0.12
    spec_term = min(1.0, (spec + 0.8 * len(metrics) + (0.5 if outcome else 0.0)) / 3.0)
    depth_q = 0.55 * spec_term + 0.45 * _length_quality(n)
    if fluff:
        depth_q *= 0.6
    if vague and spec <= 1:
        depth_q *= 0.8
    boiler = _boiler_fraction(t)
    if boiler >= 0.3:
        depth_q *= (1 - 0.5 * boiler)
        action_q *= (1 - 0.2 * boiler)
    stuffed = False
    if n >= 5 and not metrics:
        fwr, buzz = _function_word_ratio(t), _buzz_density(t)
        if buzz >= 0.45 and spec <= 1:
            stuffed = True               # a pile of impressive words, no real content
        elif fwr < 0.06 and n >= 7 and spec <= 1:
            stuffed = True               # no grammar glue => keyword list, not a sentence
    if stuffed:
        depth_q *= 0.5
        action_q *= 0.6

    score = 100 * (0.30 * action_q + 0.35 * impact_q + 0.35 * depth_q)
    return {
        "text": t, "words": n, "lead": lead, "metrics": metrics, "outcome": outcome,
        "specifics": spec, "fluff": fluff, "vague": vague, "stuffed": stuffed, "boiler": boiler,
        "action_q": action_q, "impact_q": impact_q, "depth_q": depth_q, "score": score,
    }


def _bullet_feedback(b: dict):
    flags, tips = [], []
    lead = b["lead"]
    if lead["kind"] == "weak":
        flags.append("weak opener")
        alt = next((v for k, v in _WEAK_ALTERNATIVES.items() if lead["word"].startswith(k)), "Built / Led / Delivered")
        tips.append(f"Opens with “{lead['word']}”, which hides your role. Start with what YOU did (e.g. {alt}).")
    elif lead["kind"] == "first_person":
        flags.append("first person")
        tips.append("Drop “I/my/we” and start directly with the action verb.")
    elif lead.get("first_person") and b.get("is_bullet", True):
        flags.append("first person")
        tips.append("Drop the leading “I/we” and start directly with the action verb.")
    elif lead["kind"] == "noun":
        flags.append("no action verb")
        tips.append("Start with a past-tense action verb so the reader sees what you did.")
    if not b["metrics"] and not b["outcome"]:
        flags.append("no result")
        tips.append("Say what changed because of your work: add scale (users, records, team size), "
                    "speed/cost/accuracy change, or who used it.")
    elif not b["metrics"]:
        flags.append("no numbers")
        tips.append("Result is stated but not measured; add a number if you have one (%, time saved, volume).")
    if b["words"] < 8:
        flags.append("too short")
        tips.append("Too brief: add the tool/method used and why it mattered.")
    elif b["words"] > 45:
        flags.append("too long")
        tips.append("Too long: split into two bullets (what you did → what it achieved).")
    if b.get("boiler", 0) >= 0.5:
        flags.append("templated wording")
        tips.append("This wording appears almost verbatim in many other resumes. Replace it with your own specifics: tools, scale, and results.")
    if b.get("stuffed"):
        flags.append("buzzword stuffing")
        tips.append("This reads as a list of impressive words, not an achievement. Say what you built or changed, with what, for whom, and the result.")
    if b["fluff"]:
        flags.append("buzzwords")
        tips.append(f"Replace buzzwords ({', '.join(sorted(set(f.lower() for f in b['fluff'])))}) with proof.")
    if b["vague"] and b["specifics"] <= 1:
        flags.append("vague")
        tips.append("Replace vague words (various/several/many) with specifics or counts.")
    return flags, tips


def _status(score: float) -> str:
    return "strong" if score >= 68 else ("good" if score >= 45 else "needs work")


# ======================================================================
# 4. Resume-level evaluation
# ======================================================================

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\d])\+?\d[\d\s().-]{8,}\d")
_DEGREE_RE = re.compile(r"\b(?:b\.?\s?tech|m\.?\s?tech|b\.?\s?e\b|b\.?\s?sc|m\.?\s?sc|b\.?\s?com|m\.?\s?com|bca|mca|mba|bba|phd|ph\.d|bachelor|master'?s?|diploma|b\.?a\b|m\.?a\b)", re.I)
_LINK_RE = re.compile(r"linkedin\.com|github\.com|gitlab\.com|portfolio|behance\.net|kaggle\.com", re.I)


def _has_phone(text: str) -> bool:
    for m in _PHONE_RE.finditer(text):
        d = re.sub(r"\D", "", m.group(0))
        if 10 <= len(d) <= 13 and not re.fullmatch(r"(19|20)\d{2}(19|20)\d{2}\d*", d):
            return True
    return False


def _skills_tokens(lines):
    toks = set()
    for ln in lines:
        ln = re.sub(r"^[^:]{1,30}:\s*", "", ln)  # drop "Languages:" label
        for part in re.split(r"[,;|•·/\u2022]|\s{2,}", ln):
            p = part.strip(" .-–")
            if 1 <= len(p.split()) <= 4 and 1 < len(p) <= 30:
                toks.add(p.lower())
    return toks


def _wmean(pairs):
    tw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / tw if tw else 0.0


_CAL = None


def _load_calibrator():
    """Learned correction (a small monotone linear model, fitted on human-scored resumes).
    The weights live at the bottom of this file, so no other file is needed.
    Set env RESUME_CALIBRATOR=/path/file.json to override with your own retrained weights."""
    global _CAL
    if _CAL is None:
        _CAL = _CALIBRATOR
        override = os.environ.get("RESUME_CALIBRATOR")
        if override:
            try:
                import json
                with open(override) as f:
                    _CAL = json.load(f)
            except Exception:
                pass
    return _CAL or None


FEATURE_NAMES = [
    "n_bullets_log", "frac_quantified", "frac_outcome", "frac_weak", "frac_tier3", "frac_action",
    "mean_bullet_score", "mean_depth", "mean_action", "avg_bullet_words", "spec_per_bullet",
    "fluff_total", "frac_boiler", "dup_frac", "frac_stuffed", "word_count_log", "n_sections", "has_email", "has_phone", "skills_count_log",
    "structure", "depth", "action", "impact", "rule_score",
]


# Features the learned calibrator may use, with the direction in which they must push the
# score.  Constraining signs guarantees sensible behaviour: improving a resume can never
# lower its score because of a quirk in a small training set.  (Length features are left
# out on purpose: their effect is already inside structure/depth and has no fixed sign.)
CALIBRATION_FEATURES = {
    "n_bullets_log": 1, "frac_quantified": 1, "frac_outcome": 1, "frac_weak": -1, "frac_tier3": 1,
    "frac_action": 1, "mean_bullet_score": 1, "mean_depth": 1, "mean_action": 1, "spec_per_bullet": 1,
    "fluff_total": -1, "frac_boiler": -1, "dup_frac": -1, "frac_stuffed": -1, "n_sections": 1, "has_email": 1,
    "has_phone": 1, "skills_count_log": 1, "structure": 1, "depth": 1, "action": 1, "impact": 1,
    "rule_score": 1,
}


def _evaluate(text: str) -> dict:
    clean = (text or "").strip()
    sections, seen, units, pstats = _parse(clean)
    seen_keys, kept_sets, uniq, dups = set(), [], [], 0
    for u in units:
        key = re.sub(r"[^a-z]", "", u["text"].lower())[:120]  # digits ignored: "task 1", "task 2" are the same line
        words = set(re.findall(r"[a-z]{2,}", u["text"].lower()))
        near = len(words) >= 5 and any(len(words & o) / len(words | o) >= 0.6 for o in kept_sets)
        if key in seen_keys or near:  # exact or near-identical sentence already counted
            dups += 1
            continue
        seen_keys.add(key)
        kept_sets.append(words)
        uniq.append(u)
    dup_frac = dups / max(1, dups + len(uniq))
    bullets = [dict(_analyze_bullet(u["text"], prose=not u["bullet"]), section=u["section"], is_bullet=u["bullet"])
               for u in uniq if len(u["text"].split()) >= 3][:60]
    n = len(bullets)
    weights = [_SECTION_WEIGHT.get(b["section"], 1.0) for b in bullets]
    wc = len(re.findall(r"\b[A-Za-z]{2,}\b", clean))

    # ---------------- structure (25) ----------------
    has_email = bool(_EMAIL_RE.search(clean))
    has_phone = _has_phone(clean)
    has_link = bool(_LINK_RE.search(clean))
    has_exp = bool(sections["experience"]) or "experience" in seen
    has_proj = bool(sections["projects"]) or "projects" in seen
    has_edu = bool(sections["education"]) or "education" in seen or bool(_DEGREE_RE.search(clean))
    skills_tokens = _skills_tokens(sections["skills"])
    has_skills = bool(sections["skills"])
    has_summary = bool(sections["summary"])
    content_pts = 0.0
    if n >= 3 and (has_exp or has_proj or n >= 3):
        content_pts = 8.0
    elif n >= 1:
        content_pts = 4.0
    struct = content_pts + (5 if has_edu else 0) + (5 if has_skills else 0) \
        + (2 if has_email else 0) + (2 if has_phone else 0) + (1 if has_link else 0) \
        + (1 if has_summary else 0) + (1 if pstats["date_lines"] >= 1 else 0)
    if wc < 200:
        struct -= min(6.0, (200 - wc) / 200 * 6)
    elif wc > 1200:
        struct -= min(5.0, (wc - 1200) / 300)
    struct = max(0.0, min(25.0, struct))

    if n == 0:
        depth = action = impact = 0.0
        feats = dict.fromkeys(FEATURE_NAMES, 0.0)
        feats.update(structure=struct, word_count_log=math.log1p(wc), has_email=float(has_email),
                     has_phone=float(has_phone))
        return {"sections": sections, "seen": seen, "bullets": [], "stats": pstats,
                "breakdown": {"structure": round(struct), "depth": 0, "action": 0, "impact": 0},
                "raw_total": struct, "features": feats, "wc": wc, "contact": (has_email, has_phone, has_link)}

    mean_action = _wmean(list(zip((b["action_q"] for b in bullets), weights)))
    mean_depth = _wmean(list(zip((b["depth_q"] for b in bullets), weights)))
    mean_score = _wmean(list(zip((b["score"] for b in bullets), weights)))
    kinds = [b["lead"]["kind"] for b in bullets]
    frac_action = sum(k in {"tier3", "tier2", "tier2u"} for k in kinds) / n
    frac_tier3 = sum(k == "tier3" for k in kinds) / n
    frac_weak = sum(k == "weak" for k in kinds) / n
    quantified = [bool(b["metrics"]) for b in bullets]
    frac_q = _wmean(list(zip(map(float, quantified), weights)))
    frac_o = _wmean(list(zip((float(bool(b["outcome"] or b["metrics"])) for b in bullets), weights)))

    # ---------------- action (25) ----------------
    leads = Counter(b["lead"]["lemma"] or b["lead"]["word"] for b in bullets if b["lead"]["kind"] in {"tier3", "tier2", "tier2u"})
    repeat = sum(max(0, c - 2) for c in leads.values()) / n
    action = 25 * (0.75 * mean_action + 0.25 * frac_action) - min(4.0, 8 * repeat)
    action = max(0.0, min(25.0, action))

    # ---------------- impact (25) ----------------
    impact = 25 * (0.70 * min(1.0, frac_q / 0.35) + 0.30 * min(1.0, frac_o / 0.50))

    # ---------------- depth (25) ----------------
    volume = min(1.0, n / 8)
    skills_rich = min(1.0, len(skills_tokens) / 12)
    fluff_total = sum(len(b["fluff"]) for b in bullets) + len(_FLUFF_RE.findall(" ".join(sections["summary"])))
    fluff_ok = 1.0 - min(1.0, fluff_total / 4)
    depth = 25 * (0.55 * mean_depth + 0.20 * volume + 0.15 * skills_rich + 0.10 * fluff_ok)
    depth -= 25 * min(0.16, 0.5 * dup_frac)  # repeated sentences
    depth = max(0.0, min(25.0, depth))
    frac_boiler = _wmean(list(zip((float(b["boiler"] >= 0.5) for b in bullets), weights)))
    frac_stuffed = _wmean(list(zip((float(b["stuffed"]) for b in bullets), weights)))

    raw_total = struct + depth + action + impact
    feats = {
        "n_bullets_log": math.log1p(min(n, 20)), "frac_quantified": frac_q, "frac_outcome": frac_o,
        "frac_weak": frac_weak, "frac_tier3": frac_tier3, "frac_action": frac_action,
        "mean_bullet_score": mean_score, "mean_depth": mean_depth, "mean_action": mean_action,
        "avg_bullet_words": sum(b["words"] for b in bullets) / n,
        "spec_per_bullet": sum(b["specifics"] for b in bullets) / n, "fluff_total": float(fluff_total),
        "frac_boiler": frac_boiler, "dup_frac": dup_frac, "frac_stuffed": frac_stuffed,
        "word_count_log": math.log1p(wc), "n_sections": float(len([s for s in seen if s != "header"])),
        "has_email": float(has_email), "has_phone": float(has_phone),
        "skills_count_log": math.log1p(len(skills_tokens)),
        "structure": struct, "depth": depth, "action": action, "impact": impact, "rule_score": raw_total,
    }
    return {
        "sections": sections, "seen": seen, "bullets": bullets, "stats": pstats,
        "breakdown": {"structure": round(struct), "depth": round(depth), "action": round(action), "impact": round(impact)},
        "raw_total": raw_total, "features": feats, "wc": wc, "contact": (has_email, has_phone, has_link),
        "frac_q": frac_q, "frac_weak": frac_weak, "frac_action": frac_action, "n": n,
        "has_edu": has_edu, "has_skills": has_skills, "has_exp_or_proj": n > 0,
        "quantified_count": sum(quantified), "weak_count": sum(k == "weak" for k in kinds),
    }


def extract_features(text: str) -> dict:
    """Feature dict used by train_calibrator.py."""
    return _evaluate(text)["features"]


def _calibrated(ev: dict, rule: float):
    cal = _load_calibrator()
    if not cal:
        return rule
    try:
        signs = cal.get("signs") or [1] * len(cal["feature_names"])
        x = [ev["features"][k] * sg for k, sg in zip(cal["feature_names"], signs)]
        z = [(v - m) / (sc or 1.0) for v, m, sc in zip(x, cal["mean"], cal["scale"])]
        pred = cal["intercept"] + sum(c * v for c, v in zip(cal["coef"], z))
        blend = float(cal.get("blend", 0.8))
        return (1 - blend) * rule + blend * max(0.0, min(100.0, pred))
    except Exception:
        return rule


def _final_score(ev: dict) -> int:
    n = len(ev["bullets"])
    if not n:
        return int(max(0, min(30, round(ev["raw_total"]))))
    score = _calibrated(ev, ev["raw_total"])
    score *= 1 - 0.55 * ev["features"]["frac_stuffed"]  # lists of impressive words with no content
    score *= 1 - 0.6 * max(0.0, ev["features"]["dup_frac"] - 0.15)  # padding with near-identical lines
    if n < 3:
        score = min(score, 55.0)  # too little content to judge confidently
    return int(max(0, min(96, round(score))))


# ======================================================================
# 5. Summary & public scoring API
# ======================================================================

def _missing_parts(ev):
    miss = []
    has_email, has_phone, _ = ev["contact"]
    if not has_email:
        miss.append("email")
    if not has_phone:
        miss.append("phone number")
    if not ev.get("has_edu"):
        miss.append("education section")
    if not ev.get("has_skills"):
        miss.append("skills section")
    return miss


def _ats_audit(ev):
    passes, warnings = [], []
    has_email, has_phone, has_link = ev["contact"]
    if has_email and has_phone:
        passes.append("Contact information (email and phone) found.")
    else:
        warnings.append("Missing contact details: " + ", ".join(
            x for x, ok in (("email", has_email), ("phone number", has_phone)) if not ok) + ".")
    if has_link:
        passes.append("Professional link (LinkedIn/GitHub/portfolio) found.")
    wc = ev["wc"]
    if 250 <= wc <= 1000:
        passes.append(f"Good length ({wc} words).")
    elif wc < 250:
        warnings.append(f"Resume is short ({wc} words); add detail on what you built and achieved.")
    else:
        warnings.append(f"Resume is long ({wc} words); trim to the most relevant 1-2 pages.")
    if ev.get("has_edu"):
        passes.append("Education section found.")
    else:
        warnings.append("No education section detected.")
    if ev.get("has_skills"):
        passes.append("Skills section found.")
    else:
        warnings.append("No skills section detected; ATS systems look for one.")
    if ev["stats"]["date_lines"] == 0 and ev["bullets"]:
        warnings.append("No dates detected on roles/projects; add them (e.g. Jan 2023 - Present).")
    return passes, warnings


# ======================================================================
# 6. Job-description matching
# ======================================================================

_JD_STOP = set("""the and for with a an to of in on at is are be as or will we you your our this that must have
has had role job experience years year work team skills candidate position ability strong looking join
opportunity company responsibilities requirements qualifications preferred required including etc working
understanding knowledge good excellent proficiency proficient familiarity plus bonus ideal salary benefits
apply location remote full time part able can may should would from by into across within about such other
more most also well using use used new their they them who what which where when how all any each both
per via over under between etc based high level large small day days month months great love passion
passionate looking seeking hire hiring make building build help helping nice exposure desired solid deep
hands proven demonstrated minimum least ensure ensuring responsible such like various etc preferably
willingness eager fast paced environment opportunity impact mission culture equal employer""".split())
_KNOWN_BIGRAMS = {"machine learning", "deep learning", "data science", "data engineering", "natural language",
                  "computer vision", "rest api", "unit testing", "version control", "project management",
                  "product management", "data analysis", "cloud computing", "software development",
                  "web development", "full stack", "front end", "back end", "big data", "supply chain",
                  "customer service", "social media", "digital marketing", "business development",
                  "financial analysis", "content marketing", "generative ai", "data structures",
                  "system design", "test automation", "stakeholder management", "machine learning"}

_PHRASE_ALIASES = [
    (r"\bk8s\b", "kubernetes"), (r"\bjs\b", "javascript"), (r"\bts\b", "typescript"),
    (r"\bml\b", "machine learning"), (r"\bdl\b", "deep learning"),
    (r"\bnlp\b", "natural language processing"), (r"\bllms?\b|large language models?", "llm"),
    (r"node\.?js", "nodejs"), (r"react\.?js", "react"), (r"vue\.?js", "vue"),
    (r"postgres(?:ql)?", "postgresql"), (r"amazon web services", "aws"),
    (r"ci\s*/\s*cd", "cicd"), (r"restful", "rest"), (r"golang", "go"),
    (r"c\+\+", "cpp"), (r"c#", "csharp"), (r"\.net\b", "dotnet"),
    (r"scikit[- ]learn", "sklearn"), (r"tensor flow", "tensorflow"),
    (r"gen(?:erative)?[- ]?ai", "generative ai"),
]
_STEM_SUFFIXES = ("ization", "ations", "ation", "ments", "ment", "ings", "ing", "ers", "er", "ies", "ied", "ed", "es", "s")


def _stem(w: str) -> str:
    for suf in _STEM_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def _alias(text: str) -> str:
    t = text.lower()
    for pat, rep in _PHRASE_ALIASES:
        t = re.sub(pat, rep, t)
    return t


def _jd_terms(jd_text: str, limit: int = 30):
    raw_techy = {m.lower() for m in re.findall(r"\b[A-Za-z]*(?:[A-Z]{2,}|[a-z][A-Z])[A-Za-z0-9+#]*\b", jd_text or "")}
    toks = re.findall(r"[a-z][a-z0-9+#]*", _alias(jd_text or ""))
    uni, bi, display = Counter(), Counter(), {}
    for t in toks:
        if t in _JD_STOP or len(t) < 2:
            continue
        s = _stem(t)
        uni[s] += 1
        display.setdefault(s, t)
    for a, b in zip(toks, toks[1:]):
        if a in _JD_STOP or b in _JD_STOP or len(a) < 2 or len(b) < 2:
            continue
        key = (_stem(a), _stem(b))
        bi[key] += 1
        display.setdefault(key, f"{a} {b}")
    terms = {}
    for s, c in uni.items():
        techy = 1.5 if (display[s] in raw_techy or re.search(r"[0-9+#]", display[s])) else 1.0
        terms[s] = (1 + math.log(c)) * techy
    for k, c in bi.items():
        if c >= 2 or f"{display[k]}" in _KNOWN_BIGRAMS:
            terms[k] = (1 + math.log(c)) * 1.25
    in_bigram = {w for k in terms if isinstance(k, tuple) for w in k}
    terms = {k: w for k, w in terms.items() if isinstance(k, tuple) or k not in in_bigram}
    ranked = sorted(terms.items(), key=lambda kv: -kv[1])[:limit]
    return ranked, display


def _resume_term_set(text: str):
    toks = [t for t in re.findall(r"[a-z][a-z0-9+#]*", _alias(text))]
    stems = [_stem(t) for t in toks]
    return set(stems) | set(zip(stems, stems[1:]))


_EMB = None


def _semantic_similarity(a: str, b: str):
    """Optional: cosine similarity via sentence-transformers if installed."""
    global _EMB
    if _EMB is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMB = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMB = False
    if not _EMB:
        return None
    try:
        import numpy as np
        va, vb = _EMB.encode([a[:4000], b[:4000]], normalize_embeddings=True)
        return float(np.dot(va, vb))
    except Exception:
        return None


def match_job_description(resume_text, jd_text):
    ranked, display = _jd_terms(jd_text)
    if not ranked:
        return None
    have = _resume_term_set(resume_text or "")
    total = sum(w for _, w in ranked)
    matched = [(k, w) for k, w in ranked if k in have]
    missing = [(k, w) for k, w in ranked if k not in have]
    coverage = sum(w for _, w in matched) / total if total else 0.0
    relevancy = coverage * 100
    sem = _semantic_similarity(resume_text or "", jd_text or "")
    if sem is not None:
        relevancy = 0.65 * relevancy + 0.35 * max(0.0, min(1.0, (sem - 0.2) / 0.5)) * 100
    show = lambda k: display[k]
    return {
        "relevancy_score": int(round(relevancy)),
        "matched_keywords": [show(k) for k, _ in matched],
        "missing_keywords": [show(k) for k, _ in missing],
        "keyword_total": len(ranked),
        "semantic_similarity": None if sem is None else round(sem, 3),
    }


# ======================================================================
# 6. Resume Quality v3: document check -> four weighted metrics -> feedback
# ======================================================================
import base64 as _b64
import json as _json
import zlib as _zlib

_DOC_FEATS = ["log_words", "n_sections", "core_section", "email", "phone", "link", "date_ranges", "degree",
              "res_word_density", "jd_phrases", "you_density", "first_person", "short_lines", "long_lines",
              "end_period", "verb_lines", "symbols", "alpha_ratio", "cap_ratio", "unique_ratio",
              "digit_ratio", "salutation", "avg_word_len"]
_RES_WORDS = set("""experience education skills project projects university college degree intern internship
certified certification responsible managed developed designed implemented led built achieved summary objective
profile employment company team worked bachelor master diploma gpa cgpa courses training languages references
coordinated supervised analyzed delivered improved reduced increased""".split())
_JD_PATTERNS = [re.compile(p, re.I) for p in [
    r"we are (?:looking|hiring|seeking)", r"\byou will\b", r"\byou(?:'ll| are| have| should| must)\b",
    r"the ideal candidate", r"responsibilities include", r"job description", r"apply (?:now|today|here)",
    r"what we offer", r"about (?:us|the (?:company|role|team))", r"equal opportunity", r"\bmust have\b",
    r"(?:required|preferred) qualifications", r"(?:join|joining) our team", r"candidates? (?:should|must|will)",
    r"\bwe offer\b", r"benefits include", r"send your (?:cv|resume)", r"years of experience required",
    r"\bour (?:team|company|client)s?\b"]]
_SALUT_RE = re.compile(r"\b(dear (?:sir|madam|hiring|team|mr|ms|dr)|sincerely|yours faithfully|kind regards|"
                       r"best regards|thank you for your (?:time|consideration)|look forward to hearing)\b", re.I)


def _doc_features(text: str):
    t = _normalize(text or "")
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    words = re.findall(r"[A-Za-z][A-Za-z'+#.-]*", t)
    nw = max(1, len(words))
    low = [w.lower().strip(".-'") for w in words]
    try:
        _, seen, _, _ = _parse(text or "")
        ns = len(seen - {"header"})
    except Exception:
        seen, ns = set(), 0
    nl, chars = max(1, len(lines)), max(1, len(t))
    lw = [len(l.split()) for l in lines]
    verb = cnt = 0
    for l in lines:
        if len(l.split()) >= 4:
            cnt += 1
            verb += _lead_info(_strip_bullet(l)[1])["kind"] in ("tier3", "tier2", "tier2u")
    sent_lines = [l for l in lines if len(l.split()) > 6]
    f = dict(
        log_words=min(1, math.log1p(nw) / 8), n_sections=min(1, ns / 6),
        core_section=float(bool(seen & {"experience", "education", "skills", "projects"})),
        email=float(bool(_EMAIL_RE.search(t))), phone=float(_has_phone(t)), link=float(bool(_LINK_RE.search(t))),
        date_ranges=min(8, len(DATE_RANGE_RE.findall(t)) + 0.5 * len(SINGLE_DATE_RE.findall(t))) / 8,
        degree=min(3, len(_DEGREE_RE.findall(t))) / 3,
        res_word_density=min(12, 100 * sum(w in _RES_WORDS for w in low) / nw) / 12,
        jd_phrases=min(6, sum(len(r.findall(t)) for r in _JD_PATTERNS)) / 6,
        you_density=min(4, 100 * sum(w in ("you", "your", "you'll") for w in low) / nw) / 4,
        first_person=min(6, 100 * sum(w in ("i", "my", "me", "we", "our", "i'm", "i've") for w in low) / nw) / 6,
        short_lines=sum(x <= 14 for x in lw) / nl, long_lines=sum(x > 30 for x in lw) / nl,
        end_period=(sum(l.endswith(".") for l in sent_lines) / len(sent_lines)) if sent_lines else 0.0,
        verb_lines=(verb / cnt) if cnt else 0.0,
        symbols=min(6, 100 * len(re.findall(r"[{}\[\]();=<>_#/\\|*]", t)) / chars) / 6,
        alpha_ratio=sum(c.isalpha() for c in t) / chars,
        cap_ratio=sum(w[:1].isupper() for w in words) / nw,
        unique_ratio=min(1, len(set(low[:300])) / max(1, len(low[:300]))),
        digit_ratio=min(0.1, sum(c.isdigit() for c in t) / chars) / 0.1,
        salutation=float(bool(_SALUT_RE.search(t))),
        avg_word_len=min(1, (sum(len(w) for w in low) / nw) / 10))
    return [f[k] for k in _DOC_FEATS], f


def _resume_probability(text: str):
    x, f = _doc_features(text)
    m = _DOC_MODEL
    z = m["intercept"] + sum(c * (v - mu) / (sc or 1.0) for c, v, mu, sc in zip(m["coef"], x, m["mean"], m["scale"]))
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z)), f


def _doc_kind(f: dict, text: str) -> str:
    if f["salutation"]:
        return "letter"
    if f["jd_phrases"] >= 0.3:
        return "job description"
    if f["symbols"] >= 0.35:
        return "code or markup"
    if f["first_person"] >= 0.25 and f["email"] == 0:
        return "personal text"
    if f["unique_ratio"] < 0.35 or f["alpha_ratio"] < 0.6:
        return "unstructured text"
    return "general text"


def _interp(x: float, pts) -> float:
    """Piecewise-linear map through anchor points [(x0, y0), (x1, y1), ...]."""
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / ((x1 - x0) or 1)
    return pts[-1][1]


_NEUTRAL = 68.0


def _pull(score: float, conf: float, neutral: float = _NEUTRAL, strength: float = 0.8) -> float:
    """If we may have missed part of the document, uncertainty alone must never drag a metric far below fair."""
    return score if score >= neutral else score + (neutral - score) * (1 - conf) * strength


def _level(s: float) -> str:
    return "Exceptional" if s >= 85 else "Strong" if s >= 72 else "Solid" if s >= 60 else "Developing" if s >= 50 else "Limited"


# ---------------------------------------------------------------- 1. information
_VOLUME_PTS = [(0, 40), (60, 45), (120, 52), (200, 60), (300, 68), (420, 75), (550, 84), (700, 91), (900, 97),
               (1400, 98), (2200, 88), (3500, 78)]
_DETAIL_PTS = [(0, 40), (25, 45), (38, 53), (48, 62), (56, 70), (64, 78), (72, 86), (84, 94), (96, 100)]


def _information_metric(ev: dict, q: float) -> dict:
    secs = ev["sections"]
    cw = sum(len(l.split()) for k in ("summary", "experience", "projects", "education", "skills",
                                      "certifications", "achievements") for l in secs.get(k, []))
    if cw == 0:
        cw = ev["wc"]
    n = len(ev["bullets"])
    volume = _interp(cw, _VOLUME_PTS)
    detail = _interp(q, _DETAIL_PTS)
    score = 0.35 * volume + 0.65 * detail
    return {"score": score, "words": cw, "points": n, "volume": volume, "detail": detail, "conf": 1.0}


# ---------------------------------------------------------------- 2. skills + courses + projects
_SEED_SKILLS = set("""python java javascript typescript kotlin swift php ruby scala matlab sql mysql postgresql mongodb
redis oracle html css react angular vue nodejs django flask spring fastapi aws azure gcp docker kubernetes terraform
ansible jenkins git github linux bash excel tableau powerbi sap tally quickbooks salesforce hubspot photoshop
illustrator figma autocad solidworks catia ansys primavera jira scrum agile seo analytics pandas numpy tensorflow
pytorch sklearn spark hadoop kafka airflow snowflake dbt etl communication leadership teamwork negotiation budgeting
forecasting auditing gst taxation payroll recruitment onboarding nursing triage teaching curriculum""".split())
_SEED_PHRASES = ["machine learning", "deep learning", "data analysis", "project management", "power bi",
                 "rest api", "unit testing", "version control", "social media", "financial modeling",
                 "customer service", "supply chain", "data science", "natural language processing"]
_COURSE_RE = re.compile(
    r"\b(?:certified|certification|certificate|course|coursera|udemy|edx|nptel|udacity|bootcamp|diploma|workshop|"
    r"nanodegree|specialization|masterclass|linkedin learning|pmp|scrum master|cissp|ccna|aws certified|"
    r"google cloud|oracle certified|microsoft certified)\b", re.I)
_DOMAIN_CACHE = {}


def _domain_profile(text: str):
    """Skills recognised by the site's own domain-keyword engine (ml_domain_matcher), when available."""
    try:
        import ml_domain_matcher as dm
        eng = dm.get_engine()
        if not getattr(eng, "names", None):
            return None
        found = eng.extract(text)
        skills = sorted({eng.display[j] for j in found})
        domain = None
        try:
            det = dm.detect_best_domain(text)
            if det.get("confident"):
                domain = det.get("best_domain")
        except Exception:
            pass
        return {"skills": skills, "domain": domain}
    except Exception:
        return None


def _seed_skills(text: str):
    low = text.lower()
    toks = set(re.findall(r"[a-z][a-z0-9+#]*", re.sub(r"power\s*bi", "powerbi", re.sub(r"node\.?js", "nodejs", low))))
    found = {t for t in toks if t in _SEED_SKILLS}
    found |= {p for p in _SEED_PHRASES if p in low}
    return sorted(found)


_SKILL_PTS = [(0, 35), (0.12, 40), (0.2, 45), (0.35, 60), (0.5, 70), (0.65, 78), (0.78, 86), (0.9, 94), (1.0, 100)]


def _skills_metric(text: str, ev: dict, dom) -> dict:
    secs, seen = ev["sections"], ev["seen"]
    sect_tokens = {t for t in _skills_tokens(secs.get("skills", [])) if len(t) > 1}
    if dom:
        recognised = [s for s in dom["skills"]]
        rec_low = {s.lower() for s in recognised}
    else:
        recognised = _seed_skills(text)
        rec_low = set(recognised)
    skills_all = rec_low | {t for t in sect_tokens if len(t.split()) <= 3}
    S = len(skills_all)
    pool = list(secs.get("certifications", [])) + _normalize(text).split("\n")
    course_lines = {_norm_line(x) for x in pool if 2 <= len(x.split()) <= 25 and _COURSE_RE.search(x)}
    C = min(8, len(course_lines))
    P = ev["stats"].get("project_titles", 0)
    if P == 0:
        pb = sum(1 for b in ev["bullets"] if b["section"] == "projects")
        P = math.ceil(pb / 3) if pb else 0
    if P == 0:
        P = min(2, 0.5 * sum(1 for b in ev["bullets"] if re.search(r"\bproject", b["text"], re.I)))
    E = 0.60 * min(1, S / 14) + 0.15 * min(1, C / 3) + 0.25 * min(1, P / 3)
    raw = _interp(E, _SKILL_PTS)
    sk_conf = 1.0 if ("skills" in seen and len(sect_tokens) >= 3) else (0.7 if len(rec_low) >= 4 else (0.5 if rec_low else 0.25))
    conf = 0.5 * sk_conf + 0.25 * (1.0 if ("projects" in seen or P > 0) else 0.4) + 0.25 * (1.0 if ("certifications" in seen or C > 0) else 0.55)
    shown = [s for s in recognised][:6] or sorted(sect_tokens)[:6]
    return {"score": _pull(raw, conf), "raw": raw, "skills": S, "courses": int(C), "projects": int(math.ceil(P)),
            "conf": conf, "shown": shown, "section_found": "skills" in seen,
            "domain": (dom or {}).get("domain")}


# ---------------------------------------------------------------- 3. English level
_INFORMAL_RE = re.compile(r"\b(?:lol|gonna|wanna|gotta|pls|plz|thx|ur|awesome|cool|stuff|kinda|sorta|etc|bunch of|"
                          r"lots of|a lot of|really|very|nice|good at|guy|guys|okay|yeah|whatever|basically|"
                          r"literally|actually|things?)\b", re.I)
_TRI = None


def _tri():
    global _TRI
    if _TRI is None:
        try:
            _TRI = _json.loads(_zlib.decompress(_b64.b64decode("".join(_TRIGRAM_B64))))
        except Exception:
            _TRI = {}
    return _TRI


def _typo_words(prose: str, allow=frozenset()):
    tri = _tri()
    if not tri:
        return []
    bad = []
    for m in re.finditer(r"\b[a-z]{6,}\b", prose):
        w = m.group(0)
        if w in allow:
            continue
        s = "^^" + w + "$"
        v = [tri.get(s[i:i + 3], -7.5) for i in range(len(s) - 2)]
        if min(v) < -6.5 or sum(v) / len(v) < -3.4:
            bad.append(w)
    return bad


def _syll(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", w)))


def _lang_raw(text: str, ev: dict, allow=frozenset()) -> dict:
    secs = ev["sections"]
    parts = []
    for k in ("summary", "experience", "projects", "achievements"):
        parts += secs.get(k, [])
    prose = " ".join(parts)
    if len(prose.split()) < 25:
        prose = " ".join(l for l in (text or "").split("\n") if not _CONTACT_LINE_RE.search(l))
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", prose)
    n = len(words)
    low = [w.lower() for w in words]
    content = [w for w in low if w not in _FUNCTION_WORDS and len(w) >= 3]
    nc = max(1, len(content))
    win = 40
    ttr = [len(set(content[i:i + win])) / win for i in range(0, max(1, len(content) - win), 10)] or [len(set(content)) / nc]
    leads = Counter((b["lead"]["lemma"] or b["lead"]["word"]) for b in ev["bullets"]
                    if b["lead"]["kind"] in ("tier3", "tier2", "tier2u"))
    typos = _typo_words(prose, allow)
    return {"n": n, "long_share": sum(len(w) >= 8 for w in content) / nc,
            "complex_share": sum(_syll(w) >= 3 for w in content) / nc,
            "mattr": sum(ttr) / len(ttr),
            "verb_var": (len(leads) / max(1, sum(leads.values()))) if leads else 0.5,
            "glue": sum(w in _FUNCTION_WORDS for w in low) / max(1, n),
            "typo_rate": len(typos) / max(1, n), "typos": typos[:5],
            "informal": len(_INFORMAL_RE.findall(prose)) / max(1, n),
            "slips": (len(re.findall(r"\b(\w+) \1\b", prose, re.I)) + len(re.findall(r"(?<=[a-z]),(?=[A-Za-z])", prose))
                      + len(re.findall(r"[!?]{2,}", prose)) + len(re.findall(r"\bi\b", prose))) / max(1, n)}


def _pct(v: float, qs) -> float:
    """Where v falls inside a table of 21 quantiles (0..1)."""
    return _interp(v, [(q, i / (len(qs) - 1)) for i, q in enumerate(qs)]) if qs[-1] > qs[0] else 0.5


def _lang_index(r: dict) -> float:
    Q = _LANG_Q
    soph = (_pct(r["long_share"], Q["long_share"]) + _pct(r["complex_share"], Q["complex_share"])
            + _pct(r["verb_var"], Q["verb_var"])) / 3
    div = _pct(r["mattr"], Q["mattr"])
    err = lambda k: 1 - min(1.0, r[k] / (Q[k][-3] or 1e-9))     # 1 = clean, 0 = as bad as the worst ~10% of resumes
    corr = (0.5 * err("typo_rate") + 0.25 * err("informal") + 0.25 * err("slips"))
    glue = min(1.0, r.get("glue", 0.2) / 0.08)             # real sentences contain connecting words; word salad does not
    return (0.40 * soph + 0.20 * div + 0.40 * corr) * (0.5 + 0.5 * glue)


def _language_metric(text: str, ev: dict, extra_allow=frozenset()) -> dict:
    allow = set(extra_allow)
    for t in _skills_tokens(ev["sections"].get("skills", [])):
        allow |= set(re.findall(r"[a-z]+", t))
    r = _lang_raw(text, ev, allow)
    idx = _lang_index(r)
    score = _interp(idx, _LANG_MAP)
    conf = 1.0 if r["n"] >= 60 else (0.6 if r["n"] >= 30 else 0.35)
    return {"score": _pull(score, conf, 66.0), "raw": score, "index": idx, "conf": conf, **{k: r[k] for k in ("typos", "typo_rate", "informal", "n")}}


# ---------------------------------------------------------------- 4. headings & formatting
_FMT_PTS = [(0, 45), (0.35, 52), (0.5, 58), (0.65, 66), (0.78, 75), (0.88, 86), (1.0, 98)]


def _norm_line(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _format_metric(text: str, ev: dict, layout) -> dict:
    seen = ev["seen"]
    core = sum(bool(x) for x in ((("experience" in seen) or ("projects" in seen)), "education" in seen, "skills" in seen))
    optional = sum(k in seen for k in ("summary", "certifications", "achievements"))
    coverage = min(1.0, core / 3 + 0.08 * optional)
    lines = [l.strip() for l in _normalize(text).split("\n")]
    nonblank = [l for l in lines if l]
    heads = [l for l in nonblank if _section_key(l)]
    notes = {}
    if layout and layout.get("lines"):
        L0 = layout["lines"]
        informative = (any(d.get("bold") or d.get("underline") or (d.get("style") or "").lower().startswith("heading")
                           or d.get("align") == "center" for d in L0)
                       or len({round(d["size"]) for d in L0 if d.get("size")}) > 1)
        if not informative:      # file carries no usable style data (e.g. a flattened export): judge from the text instead
            layout = None
    if layout and layout.get("lines"):
        L = layout["lines"]
        norm_map = {}
        for d in L:
            norm_map.setdefault(_norm_line(d["text"]), d)
        hl = []
        for d in L:
            if _section_key(d["text"]) or (len(d["text"].split()) <= 5 and _norm_line(d["text"]) in
                                           {_norm_line(h) for h in heads}):
                hl.append(d)
        body = [d for d in L if d not in hl and len(d["text"].split()) >= 4]
        sizes = Counter(round(d["size"]) for d in body if d.get("size"))
        body_size = sizes.most_common(1)[0][0] if sizes else None
        if hl:
            def distinct(d):
                st = (d.get("style") or "").lower()
                return bool(d.get("bold") or d.get("underline") or st.startswith("heading") or st == "title"
                            or (body_size and d.get("size") and d["size"] >= body_size + 1.0)
                            or (d.get("caps") and len(d["text"].split()) <= 4) or d.get("align") == "center")
            dist_frac = sum(distinct(d) for d in hl) / len(hl)
            sigs = Counter((bool(d.get("bold")), bool(d.get("underline")), round(d.get("size") or 0),
                            bool(d.get("caps")), d.get("align")) for d in hl)
            consistency = sigs.most_common(1)[0][1] / len(hl)
        else:
            dist_frac, consistency = 0.3, 0.5
        if body_size:
            body_cons = sum(1 for d in body if d.get("size") and abs(round(d["size"]) - body_size) <= 1) / max(1, sum(1 for d in body if d.get("size")))
        else:
            body_cons = 0.7
        left = sum(1 for d in body if d.get("align") in ("left", "justify", None)) / max(1, len(body))
        body_cons = 0.7 * body_cons + 0.3 * left
        conf = 0.9 if layout.get("source") == "docx" else 0.6
        notes.update(source=layout.get("source"), headings=len(hl), styled=int(round(dist_frac * len(hl))),
                     bold=sum(bool(d.get("bold")) for d in hl), underline=sum(bool(d.get("underline")) for d in hl))
    else:
        if heads:
            def cue(h):
                i = lines.index(h)
                return bool(h.isupper() or h.endswith(":") or (i > 0 and not lines[i - 1]) or (i + 1 < len(lines) and re.fullmatch(r"[-=_–—]{3,}", lines[i + 1] or "")))
            dist_frac = sum(cue(h) for h in heads) / len(heads)
            styles = Counter("upper" if h.isupper() else ("colon" if h.endswith(":") else "title") for h in heads)
            consistency = styles.most_common(1)[0][1] / len(heads)
        else:
            dist_frac, consistency = 0.2, 0.4
        body_cons = 0.75
        conf = 0.45
        notes.update(source="text", headings=len(heads))
    paras = [len(l.split()) for l in nonblank]
    chunking = sum(p <= 60 for p in paras) / max(1, len(paras))
    idx = 0.35 * dist_frac + 0.20 * consistency + 0.20 * coverage + 0.15 * body_cons + 0.10 * chunking
    raw = _interp(idx, _FMT_PTS)
    if not (layout and layout.get("lines")):
        raw = min(raw, 82.0)          # wording and spacing alone cannot prove that headings are styled
    return {"score": _pull(raw, conf, 70.0), "raw": raw, "index": idx, "conf": conf, "coverage": coverage,
            "dist_frac": dist_frac, "consistency": consistency, **notes}


# ---------------------------------------------------------------- weighted score + feedback
_W = {"information": 0.40, "skills": 0.28, "language": 0.16, "format": 0.16}
_LABELS = {"information": "Content richness", "skills": "Skills, courses & projects",
           "language": "English & vocabulary", "format": "Headings & formatting"}


def _seed(text: str) -> int:
    return _zlib.crc32((text or "")[:300].encode("utf-8", "ignore"))


def _pick(options, seed: int, salt: int = 0):
    return options[(seed // (salt + 3) + salt) % len(options)]


def _n(n, word, plural=None):
    return f"{n} {word}" if n == 1 else f"{n} {plural or word + 's'}"


_SHORT = {"information": "content richness", "skills": "skills, courses and projects",
          "language": "English and vocabulary", "format": "headings and formatting"}


def _praise(key: str, m: dict):
    if key == "information":
        return f"Substantial, detailed content: {m['words']} words across {_n(m['points'], 'distinct point')}."
    if key == "skills":
        parts = [_n(m["skills"], "skill")] + ([_n(m["courses"], "course or certification")] if m["courses"] else []) + \
                ([_n(m["projects"], "project")] if m["projects"] else [])
        return "A broad capability profile: " + ", ".join(parts) + " identified."
    if key == "language":
        return "Articulate, varied vocabulary with clean spelling and a professional tone."
    return "Clear, consistently styled headings that make the resume easy to scan."


def _verdict_of(score: int) -> str:
    return ("Outstanding" if score >= 88 else "Strong" if score >= 78 else "Solid" if score >= 68 else
            "Promising" if score >= 58 else "Developing" if score >= 48 else "Foundational")


_HEADLINES = {
    "Outstanding": ["A standout resume, ready for competitive applications",
                    "Polished, substantive and highly competitive",
                    "An exceptional profile, presented with confidence"],
    "Strong": ["A strong, well-built resume with clear evidence of capability",
               "Confident and credible, a few refinements away from outstanding",
               "A compelling profile that reads as professional and prepared"],
    "Solid": ["A solid foundation with real strengths to build on",
              "Credible and well organised, with room to sharpen its edge",
              "A dependable resume that can become a distinctive one"],
    "Promising": ["A promising resume with real potential still to unlock",
                  "Good raw material that now needs sharper presentation",
                  "The ingredients are there; the story needs more proof"],
    "Developing": ["A developing resume that needs more substance to compete",
                   "A useful draft: the next step is depth and clarity",
                   "Early-stage content with clear room to grow"],
    "Foundational": ["A starting point: the resume needs building out before it can compete",
                     "Too thin to represent you well yet; there is a lot of upside here",
                     "A first draft that needs substance, structure and polish"],
}


def _insight(key: str, m: dict, seed: int) -> str:
    s = m["score"]
    if key == "information":
        w, n = m["words"], m["points"]
        if s >= 85:
            return _pick([f"Rich, substantive content: about {w} words across {n} distinct points give a reader plenty to evaluate.",
                          f"A genuinely detailed document ({w} words, {n} points) that answers the questions recruiters usually have to ask."], seed)
        if s >= 70:
            return _pick([f"A healthy amount of detail ({w} words, {n} points). A few more specifics on scope and outcomes would turn good coverage into compelling proof.",
                          f"Solid coverage with {n} points. Deepening your two or three strongest roles would lift this noticeably."], seed)
        if s >= 58:
            return _pick([f"The essentials are present but compact ({w} words, {n} points). Expanding key roles with what you built, with what, and what changed would add real weight.",
                          f"Moderate detail so far. Recruiters skim quickly, and each point that shows scale or results works harder than a duty description."], seed)
        return _pick([f"The resume is light on detail ({w} words, {n} points). Thin sections leave questions unanswered; describe what you did, how, and the result.",
                      f"There is not yet enough substance to judge your impact. Aim for specific, evidence-backed points under each role or project."], seed)
    if key == "skills":
        S, C, P = m["skills"], m["courses"], m["projects"]
        if m["conf"] < 0.7 and m["raw"] < 68:
            return ("We could not clearly locate a dedicated skills, courses or projects section, so this area is assessed cautiously rather than penalised. "
                    "Labelled sections would let us credit everything you have done.")
        sample = (" (for example " + ", ".join(m["shown"][:4]) + ")") if m["shown"] else ""
        if s >= 85:
            return f"A rich capability profile: {_n(S, 'identifiable skill')}{sample}, {_n(C, 'course or certification')} and {_n(P, 'project')} show both breadth and applied experience."
        if s >= 68:
            return f"Your capability story is taking shape: {_n(S, 'skill')}{sample}, {_n(C, 'course or certification')} and {_n(P, 'project')} identified. Adding verified depth in the thinner areas would complete it."
        return f"Only a little evidence of skills, courses or projects was identified ({_n(S, 'skill')}, {_n(C, 'course')}, {_n(P, 'project')}). A clearly labelled Skills section plus two or three projects naming the tools you used would transform this area."
    if key == "language":
        if s >= 85:
            return _pick(["Precise, professional English: varied vocabulary, confident verbs and clean spelling.",
                          "Your wording is articulate and varied, and it reads as written by someone comfortable in a professional setting."], seed)
        if s >= 70:
            return _pick(["Clear, professional English overall. Richer verb variety and a few more precise terms would raise the tone further.",
                          "Readable and correct for the most part; swapping generic words for exact professional vocabulary would add polish."], seed)
        extra = (" Possible spelling slips: " + ", ".join(m["typos"][:3]) + ".") if m.get("typos") else ""
        return "The wording is plain or informal in places, and vocabulary is repetitive." + extra + " Precise action verbs and domain terms will make the writing sound more authoritative."
    if key == "format":
        src = m.get("source")
        if s >= 85:
            return "Headings are distinct and consistent, which makes the document easy to scan in the few seconds a recruiter gives it."
        if s >= 68:
            return ("Structure is clear. " + ("Making every section heading visually distinct (bold, a larger size or an underline) and keeping the style uniform would complete the effect."
                                              if src != "text" else "Clear section labels on their own lines will make it easier to scan."))
        if m["conf"] < 0.6:
            return "Headings could only be judged from wording and spacing; clear, consistently styled section labels will help both readers and ATS software."
        return "Section headings are hard to tell apart from body text or are styled inconsistently. Clear, uniform headings (bold or larger, same style throughout) will make the layout feel professional."
    return ""


def _bullet_notes(ev: dict):
    n = len(ev["bullets"])
    good, todo = [], []
    if not n:
        return good, todo
    q, f = ev.get("quantified_count", 0), ev["features"]
    if ev.get("frac_q", 0) >= 0.30:
        good.append(f"{q} of {n} points show measurable results (numbers, percentages, scale).")
    elif ev.get("frac_q", 0) < 0.20:
        todo.append(("None" if q == 0 else f"Only {q}") + f" of {n} points show a measurable result; add scale or outcomes such as users, revenue, time saved or percentage improved.")
    if ev.get("frac_action", 0) >= 0.8 and ev.get("weak_count", 0) == 0:
        good.append("Points open with clear action verbs rather than passive phrases.")
    if ev.get("weak_count", 0):
        todo.append(f"{ev['weak_count']} point(s) open with phrases such as “responsible for” or “worked on”; lead with what you did.")
    if f["frac_boiler"] >= 0.15:
        todo.append(f"About {round(f['frac_boiler'] * 100)}% of points use wording found in many other resumes; replace it with your own specifics.")
    elif f["frac_boiler"] < 0.05 and f["mean_depth"] >= 0.6:
        good.append("The wording is specific to you rather than generic or templated.")
    if f["dup_frac"] > 0:
        todo.append("Some sentences are repeated; remove duplicates so every line adds new information.")
    if f["fluff_total"] >= 2:
        todo.append("Replace buzzwords such as “hard-working” or “team player” with evidence.")
    return good, todo


def _non_resume(text: str, p: float, f: dict, kind=None) -> dict:
    kind = kind or _doc_kind(f, text)
    score = int(round(6 + 24 * p))
    msg = {"job description": "The text reads like a job posting (it addresses a candidate and lists requirements) rather than a person's own resume.",
           "letter": "The text reads like a letter or cover letter rather than a resume.",
           "code or markup": "The text looks like code or markup rather than a resume.",
           "unstructured text": "The text lacks the structure of a resume (sections, dates, roles, skills).",
           "personal text": "The text reads like a personal introduction or message rather than a resume.",
           "general text": "The text reads like general prose rather than a resume.",
           "empty": "No resume text was detected."}[kind]
    return {
        "score": score, "is_resume": False, "resume_confidence": round(p, 3), "document_type": kind,
        "verdict": "Not a resume", "headline": f"This looks like a {kind}, not a resume",
        "summary": msg + " Please upload or paste your own resume with sections such as Experience, Education and Skills, and we will evaluate it in full.",
        "metrics": {k: {"label": _LABELS[k], "score": 0, "level": "Not evaluated", "insight": "Not evaluated because the document is not a resume."} for k in _W},
        "breakdown": {"structure": 0, "depth": 0, "action": 0, "impact": 0},
        "strengths": [], "improvements": ["Upload a resume that includes your experience or projects, education and skills."],
        "notices": [msg], "top_fixes": [], "line_feedback": [], "bullet_count": 0, "detected_sections": [],
        "ats_compliance": {"passes": [], "warnings": [msg]}, "stats": {"words": len(text.split())},
    }


def score_resume(text, layout=None):
    clean = (text or "").strip()
    if not clean:
        r = _non_resume("", 0.0, {}, kind="empty")
        r.update(headline="Nothing to evaluate yet", verdict="No content", score=0,
                 summary="No resume text was detected. Upload a PDF or DOCX, or paste your resume text, to begin.")
        return r
    p, f = _resume_probability(clean)
    if p < 0.40 or len(clean.split()) < 20:
        return _non_resume(clean, p, f)

    ev = _evaluate(clean)
    ev["raw_breakdown"] = {k: float(v) for k, v in ev["breakdown"].items()}
    ev["missing"] = _missing_parts(ev)
    n = len(ev["bullets"])
    q = float(_final_score(ev)) if n else 0.0
    dom = _domain_profile(clean)
    allow = set()
    if dom:
        for s_ in dom["skills"]:
            allow |= set(re.findall(r"[a-z]+", s_.lower()))
    M = {"information": _information_metric(ev, q), "skills": _skills_metric(clean, ev, dom),
         "language": _language_metric(clean, ev, allow), "format": _format_metric(clean, ev, layout)}
    for k in M:
        M[k]["score"] = max(0.0, min(100.0, M[k]["score"]))
    wts = {k: _W[k] * (0.5 + 0.5 * M[k]["conf"]) for k in _W}
    final = sum(wts[k] * M[k]["score"] for k in _W) / sum(wts.values())
    final = min(final, M["information"]["score"] + 14)                 # thin content caps the whole score
    final *= 1 - 0.4 * ev["features"]["frac_stuffed"]                  # impressive-sounding word piles earn nothing
    if p < 0.60:
        final *= 0.88                                                    # part of this document does not look like a resume
    score = int(max(0, min(98, round(final))))

    seed = _seed(clean)
    verdict = _verdict_of(score)
    metrics = {k: {"label": _LABELS[k], "score": int(round(M[k]["score"])), "level": _level(M[k]["score"]),
                   "insight": _insight(k, M[k], seed + i), "confidence": round(M[k]["conf"], 2)}
               for i, k in enumerate(_W)}
    good_b, todo_b = _bullet_notes(ev)
    ranked = sorted(_W, key=lambda k: M[k]["score"])
    weakest, strongest = ranked[0], ranked[-1]
    S, P, C = M["skills"]["skills"], M["skills"]["projects"], M["skills"]["courses"]
    dom_name = M["skills"].get("domain")
    a = _n(S, "identifiable skill")
    openers = ([f"Your resume reads as a {dom_name} profile, backed by {a} and {_n(n, 'detailed point')}.",
                f"This is recognisably a {dom_name} resume: {a} and {_n(n, 'detailed point')} support the story."] if dom_name else
               [f"Your resume presents {a}, {_n(P, 'project')} and {_n(n, 'detailed point')}.",
                f"Across {_n(n, 'detailed point')}, your resume surfaces {a} and {_n(P, 'project')}."])
    opener = _pick(openers, seed, 1)
    lead = _pick([f"It is strongest in {_SHORT[strongest]} ({metrics[strongest]['score']}/100).",
                  f"The standout dimension is {_SHORT[strongest]}, at {metrics[strongest]['score']}/100.",
                  f"{_SHORT[strongest][0].upper() + _SHORT[strongest][1:]} is where it shines ({metrics[strongest]['score']}/100)."], seed, 2)
    if M[weakest]["score"] < 80:
        first = metrics[weakest]["insight"].split(". ")[0].rstrip(".") + "."
        gap = _pick([f"The clearest opportunity is {_SHORT[weakest]} ({metrics[weakest]['score']}/100). {first}",
                     f"To move up, focus on {_SHORT[weakest]} ({metrics[weakest]['score']}/100). {first}",
                     f"Your biggest gain is in {_SHORT[weakest]} ({metrics[weakest]['score']}/100). {first}"], seed, 3)
    else:
        gap = "Every dimension is performing at a high level, so the next step is tailoring this resume to each role you apply for."
    summary = " ".join([opener, lead, gap])

    strengths = [_praise(k, M[k]) for k in _W if M[k]["score"] >= 78 and not (k == "skills" and M[k]["conf"] < 0.7)][:3] + good_b
    improvements = []
    for k in ranked:
        if M[k]["score"] < 72:
            improvements.append(metrics[k]["insight"])
    improvements += todo_b
    if ev["missing"]:
        improvements.append("Missing details: " + ", ".join(ev["missing"]) + ".")
    notices = []
    if M["skills"]["conf"] < 0.7:
        notices.append("Some skills, courses or projects may not have been recognised because the document has no clearly labelled sections for them. Those areas were scored cautiously, never harshly.")
    if M["format"]["conf"] < 0.6:
        notices.append("Bold, underline and alignment can only be read from an uploaded DOCX or a PDF with embedded fonts. Here, headings were judged from wording and spacing, and this metric carries less weight.")
    if p < 0.60:
        notices.append("Parts of this document do not resemble a typical resume, which lowered the score slightly.")

    passes, warnings = _ats_audit(ev)
    line_feedback = []
    for b in ev["bullets"][:30]:
        st = _status(b["score"])
        flags, tips = _bullet_feedback(b)
        line_feedback.append({"line": b["text"], "status": st, "score": round(b["score"]),
                              "flags": flags if st != "strong" else ["clear action", "evidence of impact"] if b["metrics"] else ["clear action"],
                              "suggestion": None if st == "strong" else (" ".join(tips[:2]) if tips else None)})
    top_fixes = []
    for b in sorted(ev["bullets"], key=lambda b: b["score"])[:3]:
        if b["score"] < 68:
            _, tips = _bullet_feedback(b)
            top_fixes.append({"line": b["text"], "suggestion": tips[0] if tips else "Add scale and result."})

    return {
        "score": score, "is_resume": True, "resume_confidence": round(p, 3), "document_type": "resume",
        "verdict": verdict, "headline": _pick(_HEADLINES[verdict], seed), "summary": summary,
        "metrics": metrics,
        "weights": {k: round(wts[k] / sum(wts.values()), 3) for k in _W},
        # kept so older pages keep working: structure<-formatting, depth<-information, action<-language, impact<-skills
        "breakdown": {"structure": int(round(M["format"]["score"] / 4)), "depth": int(round(M["information"]["score"] / 4)),
                      "action": int(round(M["language"]["score"] / 4)), "impact": int(round(M["skills"]["score"] / 4))},
        "strengths": strengths[:5], "improvements": improvements[:6], "notices": notices,
        "skills_found": M["skills"]["shown"], "courses_found": C, "projects_found": P,
        "detected_sections": sorted(s for s in ev["seen"] if s != "header"), "bullet_count": n,
        "ats_compliance": {"passes": passes, "warnings": warnings},
        "line_feedback": line_feedback, "top_fixes": top_fixes,
        "stats": {"quantified_bullets": ev.get("quantified_count", 0), "weak_openers": ev.get("weak_count", 0), "words": ev["wc"]},
    }


def analyze_resume(text, job_description=None, layout=None):
    result = score_resume(text, layout)
    result["jd_match"] = match_job_description(text, job_description) if (job_description and result.get("is_resume")) else None
    return result


# ======================================================================
# Learned data (embedded so the engine is one self-contained file)
# ======================================================================
_CALIBRATOR = {"feature_names": ["n_bullets_log", "frac_quantified", "frac_outcome", "frac_weak", "frac_tier3", "frac_action", "mean_bullet_score", "mean_depth", "mean_action", "spec_per_bullet", "fluff_total", "frac_boiler", "dup_frac", "frac_stuffed", "n_sections", "has_email", "has_phone", "skills_count_log", "structure", "depth", "action", "impact", "rule_score"], "signs": [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "mean": [2.848032, 0.118393, 0.222715, -0.109302, 0.229348, 0.679357, 49.912129, 0.568643, 0.689987, 2.793197, -0.321429, -0.061893, -0.061315, -0.004015, 4.704762, 0.0, 0.021429, 3.006581, 19.362952, 17.460579, 16.735624, 8.48614, 62.045295], "scale": [0.419146, 0.145355, 0.166735, 0.115107, 0.153605, 0.206042, 9.068902, 0.144265, 0.102576, 4.598438, 0.624132, 0.147348, 0.131814, 0.012726, 0.861049, 1.0, 0.144808, 1.160008, 1.686167, 2.787874, 2.956413, 7.316011, 10.205343], "coef": [4.091251, 4.06041, 0.0, 0.0, 4.869506, 0.0, 0.0, 1.298789, 0.0, 0.0, 0.0, 1.849824, 0.1514, 0.0, 0.909881, 0.0, 0.0, 0.605463, 0.0, 0.0, 0.0, 2.970675, 0.0], "intercept": 54.092857, "blend": 1.0}

_BOILERPLATE_B64 = (
    "eNoVt2V4ntl5rn0seBhfBukV25Zlj9ljDyUz6YQmDTRpCmmTpmmSYsp772baNGJmZmZmZrYYLFlkMTNYTJ++v8+vZ6173dd1"
    "nn4oUL1rX4+H4RzRpn7LxHCZcg3f9Z068oT9WSTdZjXMBz15Z5lB55AJVKBNABwUk21cQQfKker5JXoXbqgixQKbJNhAeZKN"
    "rKveh4pnW0CK3GAWqY00C+M7vtdrO6LuYofUiZbrigFFmt0u/Rrl27U8equZRlUaH/0eXSjMkcfoZ46KcmEdDZrlfeRusm4n"
    "OtgesGmZRy+yvxi0CAPfq8PZ5uOyFzq1dKO75Nd3WvVp1mP4EF8a4/WeUhDO5eg5PIjOFOV8u+xjWWRdRYxwfhaV9P+M4DLo"
    "fXuSLKTL/rgMXHPDj/bNG4A/mkUntwbpBTqMiIFjsi859V4Os6sKwYXIjasREnk31RtNF5GonCeyFGl0ObVjcyFmsmXEPson"
    "6yxc9WeqSnbNLNNs6p4ntQAOyBVmiy2Dc9YBdAB9rPR7HAqc7GuNNYbdl+ZBwrU6hFyzi1a7fmPj24tsKL8q1xqGqA6+1fDp"
    "pjb22S6MAzXPMl6G3z0yL32/jFD1Kff5JbLJIsH4+ziLCnBiW0ZW68tQuuDDBhGVKEaxq70iMthWKZxokcL0r5QhfBk8Q4+T"
    "rJbJI1SF/quGLNE0qju4V5bthjX+J2FMG3DnM1hHnEIdaGveX0WLNwPrEgLBDn2sTcChhhTtKDH2UTxuFhrJCHWc4aN8nAwc"
    "SeWINoRt/LwexMJX6llYIW1xq+SuWTE5de9UfGVIRDn6OW0kUw1HHnWRwWQPyFQUk0O45O8uCSKXWuMPiXlyF5eaR1m2y7Gq"
    "rJcNMOcn6dIW6aLaAs3QEa+SLWYVXDW4ZqaMlXiU6pF3H07gEz6G7OcW5UpmT1NtqgTBOAwds2PiEjVMbcMDFIJSqTa8KryC"
    "FcYaq2JLJ7lbX8C5q7rZBqpenDKfYSe5JSkBZsGcRxucuOEQodoVtshU4VTjcmfNKvpJhiIcFgkjqOjDQssqOGJ+JpxAT8UR"
    "6SVVEMs2/XI958l2Yl+HDLJSd6KdM/WAYnDOXaFz82n9qlkx4S+8tYgGE0/TwQzDjihWcB5c/Wbfrz2V+4YDcYuJfBxsfcju"
    "ww5zL3Ukt/piXK6x9mLdpHYil9iSLpR1xNbDKeoKv8xBu9YeZD6ZwbdoA0yJylefJ6IMsIYPhB4Qiw+oCjIcRYFtuEXOgnqz"
    "SsFRXyl4kCnsPD41Bdv7cO2qufv7YokpmgohWuxakPtn3ZYb3MWPFuVOOly0Kbcdx7soGfpzPwkDS8oxuIeHuUGyBUV8fZ4L"
    "U2wJmdjDclreBTlk+qP73SAUWfnKhcRfSHn8n8TjXfjdI3QEfj9xKw6l6rflRs6X7pYSbYKkgtvpsA6WqOZhKLUEa7kA6xSw"
    "RG6SMUSDvpxch0dsiGXq3TZYwmRLoVS4Jo91pi+5UN0hWrP8322HOuUQGSi5oVlF0J+n4lNlnzxk2QWGDRnMa7oLxAEvY5WN"
    "ryGKjrJrBDOSZgZ3wdfC1mNn2YVZsZ0UvKw3XvzgGMzqnmXpf/SbAeZa+H27MR71whOrR6vUBVnGtd7y+ngChHxnje1nY4hr"
    "Uw45A2rIhcdvjSOgBoSDLuljH8GRPNWk4n51tWZSsSgksT5gmx/hI/TVL5oNafQs++WxepyeAIVkBi4gqumiexfEv01S/tCd"
    "TELLZr5UB5kkdHPl+uqbLeojC54MglbbPFRH+RiawagYQa8zgeIrU9gtN2IS0Bt255zXL4O5VehkVkK8A22SB7uJ9nRut/5j"
    "6w/7uCrBRedEnlg/7QHrKE3chi6EK5fIRindUQsTI7wS66RjypVNBWH6GDoR9H09gRp4eq5YVe9bLoN8fhVu0j9P17baOIku"
    "Nl66eDIb+qjygCucklxFnSOMVroZ/moVpOBQ6AMz5WVrT2Ya5FNHaFvzShlqtc10UPyp8S+C5AA2ktj8MIyT0u2ilAsgH2UZ"
    "3qKYe1cghvhVM1f7uIE/pKfNlsRkaoPtNVTTIXSceTqZBas4R1O5NkH05f2pdrAEN+V622HhShUnBBOFmmZwCXyE4YdHKhfl"
    "NZwwS+d92UUHR1xN52ou0Tl1ZeX03rG4qlq5HS6n06VCuxCPGqVF1ADefezHzjM58inqR0PqdT4JzetSyCn2yiJcmcktqYZo"
    "Lz5MPBeenOtec6NizKNuepVaF1PMVxlPlEWHWueAabJOUc3FsOPKWfrM9vomBoJMoeQIbH7ob77CpsM+dQM3yP12CnjrkkGD"
    "6Up9oPtFqriJXsmZ6itDk9hCbVIpcjBuJ399qioU6m3zn3ADUhL+r1GmjU3RfDhlVcEu89EGUzkVRGaZFp/1Wbzi+uCccZ6a"
    "I9/hQuCkd4GrprdEG56m5vDh7SoczXlxy8SCepfLYm5KlO9lOukUGKjae+gN/JAXnMUtzGvtMhssrYNAnM842cxLSSAdJKvL"
    "yCSF0MiWIBSJvayKVDuEu75Es8N18750szgqH7GDbBfnghdgvRCOy1UF7Es/4cq+jrxC0Wwg6jKO4CDVHHHwIg2kfeWjSO0X"
    "pSAP5sJ6/Y+qFB5kNH1tmKbr4Iz+mC3B4x81cjclRqRoevECvf5kis65F3V7Unwt+FO49V/qrcq4eyGmX2Qp3ph2FcnqZrmF"
    "3RRieV+zUVj52YfhMEaTRkYS0dLL+k+jxX0ylQxgxnE+n0NnioM2A7Zj5LJmQZ6QI+HXpxm3R3/XBhtYJ6rMZoxeIjv5MMt4"
    "wVPpyGUIsXjH2KBNx8noDRigI2Gj5Ip7qSGYDlYN7SBcV0DkE5H0GNmFCrUB1BZZgyKJVm3kTXotSqoU4YRoVjYyaeS5pcWa"
    "Ptpm/NYi84r+pxceqmNtuLn/4xVNEptgHAArN3wQSD7wIQvuLN3dI9KFXF07k013fy8QvVNMscXSwJ1pnY+ZCxfI1BmKFNF0"
    "Eu2FW/hCvQvX96BQzNJGw0DrFToW+DGJH24ZP0wkXcH3I0lP5uyDfvilp70/cLOKJSOkwbtJ9nvqDpsNaYwolavsJplu+soY"
    "jRyFAXkCB8vjTzKFvVumDBT39Z8VwSbxleKKacQFzG+77h+Je2if7lf7UFG379ewTeTniXdK+Tbgp30FW3mXOxm41zKHWxD8"
    "hSV+WDgD8UQWrKXdxWU2y+oV3BbqVS2GUHKFTVUM6LaZVNSH3BXBZOKLM8UYsa1eNvqb/UMsqHBIoKpZuoOMJRpwFZsA/j6X"
    "8YJ7UrLdKHlBerERUrLtwcepXPKzKLUrk/T9GIM7E252ADw1NUIoP6ZtNEQxW3aT5Ir1xnfDmD2qycKZ6GK3iDN6msum6oQY"
    "eAGj8BSMJrf1pUTHkyJ1K3PMnlJviGxNOXoxJY1SbpYpSt97qRp/PGPz2jiGatSl1sl0Ndn/STEZhZdQORiDrnwo6rc4Qt7M"
    "3VGczTQwmbpgchVsqKaeNX0lnuu/Nfz+grqSWERv6BKxR71ivJBPBAsXNGlqx19LNLRzF3drQO1jL+BvVaqKReP6fmoG9dCv"
    "yEDdxrM5XATS0MCn71hHU4Fu84NtvoIqsCow20RB5KlFPIrRnSoCmCTen7liVlA6HlYuyZ7SjnrZsEEME05EPDclW35ZQh8y"
    "42QCPWXswO4Ww7b2kcwwDLvnSF7Av98GFVQJkU1SEw7Z9zfERSIThPNXnzobr5TtRL24pAgAn7bYzJBObCUeoYN5XzwG0rlv"
    "NWuDzN3IUZwv1ECFGyrhFlRDkn6d8bZshO3iHngjutk2wkTdApgDMWZuVtE6X/leBPNGiOOuiTKHPvUEswHnP9jg/GCWRTK8"
    "wP6a6JvVWAYz7Dosw3+yKFZr88xyYJA4TFcIPrjAkEu6okOm0digPMdnTKYpGGbjLVwOnehSPprtlv2ex2jyFAPmf/jTE7wO"
    "ZhR1f3/OF0mXRCrjR/XqA+ifz2qzH2UyIaoBeK3eMnNXbUnFyF1T8911MK3Ofd4jOsPEm/Q9R79KYLq0tfSA6jW5xZzfWhei"
    "wRnuUKTajDCvVfvmnVKqHEhUKhrRqSYUD3BV94URmCcFPly6FQBb9CUv58R8Kt8iXKgXUoEreHPrNXZm2oVJIuWzNngOPaii"
    "+/OUE2hDMcoRdhj0oF6rTSlc/Uo3SobQqWDd0h1G6PzJMHholqV6J13r30J/2Ekn66qlTrzEVfJleo/PE9hTopG8pqJuyv0I"
    "j/DV1v1PbGJRCq4TCtkO0t2w+4mHtges2c0qouCiVcatAFWgYeYnDkkqR7xAjcE0VIQTpHrDX+9Qh2D9QZltAh/ONosu8AJ0"
    "Yg+7NKEOZ1IdbI5URGTa+/HaPVtfu0EQKLNe8gZ7V0pSlODBZ376Vn0H0ShVCsWPfGRHdHC3DeyCSnL4oSfmAogFWGfnbXFA"
    "zKuSuBjKQ39qGXozjCr9v1wwrsBHntHHSc3QmZ9mwh4NMd/ztjrS1VtOwgmulhoGx2SQWc2tQl24KcA+zWpFVyysqTf1O8pe"
    "4pBuNOTymWa5qkO0IQ/TzKSmnV1F17j0oQvzRhunbmHG8BWhDWcT/yERhoMDWGxqplZgJh3FTWov8ZFqV3etXVWOUPlUIuX7"
    "tBd4EjfVRZZR/1UgXoJyYzBZCavZV6YzeGGqlsaMzWa/SFIcCS3sGr+hlpaYFercME4/3RZSQITcJIcJzbpk9I6KlM/JMtqP"
    "LP/gRDtiO/d+GKzlF+R3dPx705/VwmVyU+/NTovd4juFl/B+g+Srcvs04mbsm98d1TUb18RfrYND9oi40jZ9x1+Ygb1kGjMn"
    "X7BTaqcPelSJ5IVtlzj6xE97SAyQHSCXaCaOzJOlc5jFd9Ny3u1JsPdwBhzaJTDhVB78US44U7iCanX+j8rIJd1/VvOb1M3G"
    "SK36bG0sXtYMm5/g/zOFJrCrGI9akY/9rWxwqC8xVqA1vAK9mWTWW1qmDj6dF8+ZCyrI+EGKMl/XRU+Lcc+csSd3BfNAPD1i"
    "qHtQi6rZDU06F0S4sKWPRtAEmayq41Z+GCy4gnY5hRnSeMNUtuUGB15xX54iD77s0bQ2RuGqDbTtubtAV9hOGorBBN1kla6a"
    "BbmKSJtAopfNMvbKrYoMOI7csHpGlyi6Scmox6raOGTIdOhVVdw5s+/Gs2Zd9Jp+A/+dCx2Oi79x/M03mgd5VCcawMG3I7lN"
    "6Hs/y1SAEpEfc6+MDMZHwIXO4CJVoXCQbaRvT5maUCVy0R0QA4KfmRPhpqmzPYBz9BrYA5W3k42Rn+Tc8hGLwR7tQc4x71QT"
    "qlni3f088h3l9510KhC+4qMIL6qHCiZ8ZQ/JExWBcTlelObeT5QiGRdAKuQiMk6V/sEk+H2puMDss8XqcMpRFwXadEviNMxV"
    "JYuL0qrmSDdEHIENscr8nRQquImZuNmYZXn9JBdYvhXnhEEzl28/i2BLbMbA7wt0XdiVXaa6+Xpl1Pt/e0D2anvIAFM1WueW"
    "yW2wRfrZXvARZIaqlHtLb1JtaINMV/zWjyiAwbCL1ecp3xC/zmFihRr+Eiwx88wgmmJSyDXCiykxROGrW65iFa6/+1p5JKQj"
    "N+AmdrOfBuqz1c3UmVDJRJjvGMefRiI/Lpm55C/ML6lZbFeDU5hLwxjZcbOta2Se/RI9C8NlZ8tr0zJbCr2EZotexYDQShbz"
    "Od/qvbNJhaErkAGW/u3jIfoErrM7NsWWnsy/Vxlf00vqri+6OH/8iyibHAsvZQ+x/HRA0a4shSNo7EU8uwn/yIkt0/Vg4hWa"
    "4Ev+MMGmmGvhjokZuhYkMr4Peqz7yQa8TTaKo2KUJhEOcf10idEPlsEKdCgMwAJqQOr/+hTXbMrTuIJNMh4e8nnSPu6HTmL3"
    "964tStHmzRijn5R9EgrOiCx63uCEPXV7aFpRQyQRB3AKj2iKrSe5ebLf3AceqY6pTjBEhMBIXbFuS//2k0xTN7kKr8y8uVh5"
    "lXa0jH9pEy0Wo04xHb1DiVqHRfzbZ05MJD8mTt3Z4gftksAg8uPd+Ekiwnz/WY1mTTxjBnCIdlncMMtQVNjncFn6WlSsPqN2"
    "2QQU+nKScCecyQ+dBHfrLthAvlGvvRhkRohh3egP+29geN/y1MwF5JpFKMrxjslbNUW6CdYddtUOLjBNGie3LAJue9g2QQ8Q"
    "SbkTTqpSSz+yl/vn10IZSme9qLLbkcyG1anK3SaJ+YcO6hr7q3rUOTj71oR+iYuhF8Ak9DOsWZSjKhBoCmAfFd5NZ3Of+/EN"
    "cgGcpn2wj0MLctj5PFMZql6GK2AdFzOb/Pr9Gtgle1qncbnU7s8PQDG/otu5e4F83juB4R8nWvSSO2BfXY29lZvEKCzl8x8X"
    "05fmw3SJ2ol1k8bRBSg1OaMR0ts8UfZUnmirle++qNV/EQKzlFOolH311Sa61vqP3fE7roFcE5LF17hdU2Tjcydbu28aB49a"
    "DdtkovUY4XMja4H2yeq3shNfBqdBIg42X4IHylfAUZ8LZsldRZzQZTGKo7XpcJZMpKrsv/7WIZDboPc/KKGDuF34f5MVjqpc"
    "FGG/AiapPPSrFXAlfZjNvKECVO6aRrxH1IBm7i0xzlcqqkAHYRXMrHGLUrbobdtENt4Y0jswQ3RwvtiRH2CXbwo9HKZ8dm1e"
    "gZroFnUddGK2sKMcJGbgRYsqnKXNRBtskZR1x/0PLskfnGorZG/spp1irj9Z/0YDCIEbT0sszsAeOiE8hX7N+EeFigTFIFkG"
    "ykQv5bZtATNn6Xtzbdf3ksAaPwAqb02BSmpYrLGaEN8qMqgNuwzUpmqDM0STVI9ahWh5Q+NncOW3IDMBS6yGmDn9vqYIbxGv"
    "QS2VqwwWqhT5ZuXcLJUp7KAIh+4buBwX/ypF7S4u0v26Q6Yclari/98lWUqdk85cO9soNKjmSW+hDlZzE5plsQstEb7MPg5Q"
    "jpCltpWcH+dGOHLzfKpi83H3g0nzt3YecoxUIveCdKUjqOajuC9SYDPspj0kL8KNvSl/hxFtNJ2BVuk9Xad+nRwg5swncL86"
    "lwlBbZo2Kx8yiyoFLboxeo8avnPNpxMLpFQoFusHYDU5frvV9oRK1c/BwZs7ShdWhGw2h2nku63u3esmAui3hipTgL6A8IdB"
    "3JK5m4Uvxbjddzf/1rL+O8FSwU8KPpm8G889b6O7NfvKvBsIngdzOAqNE1nMkn5MjrXw54ZMTqpwMk3tw32vFkXfW32/U0yw"
    "aBfTQQi9T2QRbfAtUQ3D7HsN/xNgzLRx54bvvcU+D64VP8+G22Q2CAYTYo7q6bZdOIhXbwlx2lO2h/u3ePtMnEKcID3XBjdU"
    "PUItA7xxDDV797/XcK54o6CwDufjFRxxK9Dg+Sjt1jE/imaFY+L9IPtG/sA8SP+tMHLx/TFuBpUoQ5heVffLeH0zWyY1PBqz"
    "L+FL2Q9Czbrpa3EO7HFXuFm3BUZwtTKUDyQrdPvPPtgTG+Cqtur54edxam8Qq41i4gyLhlFTEK4jxtHa82YmAEwYYhkn9kry"
    "102hPDHSuPynx1ym0EC0gHRVAzyBA5ZbqgkxltwTs7U/22dz7qzIh2ZNVg133iovDQlCjvHMmAfcYI5hV/Vv8cQ02KCc7YqA"
    "k67b3t3a0zyVH0bugnYHD96tINqoK/W0WEC6WG1I52jIWGeVwbTCkZvA8Faf2xSzF7CcbuNOiV1pj+vnPJVZVAnK4K/oKa5K"
    "Oc63sX3s//mojtsxnXLRilfS2udJ0glwMfu02lRt3otK+AJujXhr7Ub/rPTT+0tw2uKE2DP3BAdS9a019SJIwBGaIqqXr9T7"
    "kz9a5RI0weaDaFUqZHqZnn95ZbVEnJj709Hi807wLAL2ydnkvsaRDzaLM0vha+DAkzhlpjKJGkUrXNHdZNQMzmC9clrZws7Z"
    "O0kjKBl++Xfz8Fi7SDUbnD7IZHcVp2IXOUn36HocLu75Sr8NgwvQ12JUU/u1EF3s/VzRlxjD/kQj1S0VyJrXygJFIOdo+QrZ"
    "LSvnVYHgipwWx0GdWRnbBR2JeJVtm7hHhH3TjZwg/3CaTn7RTc3fb4cefI/amXwxZFZyPxElAmc65kmK1bk4aONPLbLn8h6M"
    "M4Y7ZBHH2tZnr28oo1KfK6ba+JAJzCaIpaOoFeB/p50YZnaocrMXQdDL8JsQovgGWPLBAD1KzIiBt8fhNlpS+Dq8RhXyIR2o"
    "+f0M54bTwIi9q9pROcelicNsl3Bs3NMckanchmaFLsYRtJ9pXT3B99EfO6NyRcV3B+QrKhOF8GGim9wBc1AUzrRb48qJHioO"
    "lukTqQPsTqeJW3CcOpZ6+GwiAI09dhOvYafZIZcOui2O9H89jYo014pqliuwm0B7phhFJJPjkMXbtWlybTuUPbSjfRQ+vJ0F"
    "ftNpuiIS7kbdabbdsspUx/9hisL9uRv+y3nwGneBLXH7zrm6R1uv+6NIol1yv7eDx9gyw5kpwhRFdhiK7ZNxi2IC5Fp2PHAi"
    "00C/w7hF9aclIJ5oEMLutJCFtmd3Iqh8aYYoIipv3M+Dy/lqsFkfdFfeOJzuUjNmVyCfgnHeJkm9xWXyNZb1eEh3Ct1hA4r8"
    "+NwiWR5g67l++0JQCIKYDCZckYBPNK7SMbGOPLWVKOd5hGKPLbadu50JikGnHGzrYRX37U5m6+VPgpgfv6G9cRvTwr+mBuCF"
    "MRKOP/zhlIN6h2ggXOlULkNbIOeDj/KlEEUyf8Mo8jvmx4l4G/oJb3AhiKPLPvBjjkHe++vMDvYStuH2vRhrizb1Kf+9IOMG"
    "GLdwN+wxszcHSgAhxnS+7daEdp46Zpr06XiHTOIWNd3UFeiRRkA434yz8ZqNL4hQ13Nu9L/m403Jkykwdf2g4HbWjYdOMG70"
    "/2v+8RhVeS9UqHueatVm6yrO0QPmx2abVKQ63L6BacaNwjxof7nNJOJUIknIVA6BNw+HmFbGR/qzFat1vkDbrbjxMnKOdFHs"
    "YId5LlaI5PdhrvkxV0aUi9dUyH+saAPwAdFKpNmdsgdEGtek8+da7c7BL95SJewi0WvTzI6Aym86kvtSMZer2gAuMP2hF/58"
    "gXO/OyCMyM22O1yAkE3pAz4vFT3wGszgBskxsgh0qqvgFLmhyrF8DRvkVjtHIQwvk/3MmS5LujBdK3NNddImqLFM1v71ljwJ"
    "L+4c6iNsrtge5pqNYC5UjoojpuPDVOFKbGV+dyTv4TOzbmqbqsed6hrdpPCbSa6eKcWZbDOaZS7Qgd5JLhU6NG3yFFX6Rb9x"
    "QTpVubKf7zPrdD90/ss6hybzAsMIkQcX+Fj8fzpBMnY4fq/LVIi2UTyxIV1KNTDbrpcP4EptM9kEEAs3jFFMCi623FZmW+RZ"
    "hCnCpH10CTth1Mcl9ttw16roeb1mB5w9O0e7RBOTZfST7gcbQoAXeGf3xqLghf3aZ3E3xH6MEsASkYRzcIPC4+Fns+a5+kj5"
    "WaTsiwdB4FcLqf0HmSZfU66yCS/gK6pYkcK/1YxQ29o0vYucCMce18JKs0E2QgSPy3Q15IzYZrYM6pkTlA9mcbQmnpyCY0Y/"
    "lEsUwQou+0mAso+IpwKtD+8m3RuhUtGxcp35k2X55xv0ulUPfcZnW/RJGzjM/pKj+82dkAe5IYSg4BcDqMXid2HUyRdfumh8"
    "VUkgytBOunJdsrfspOwhMzRj5BtUZvA0VCgPUY15E6jXFumc5HmNBzyDXSiCDSd9RB94hmaoc0Wpdhf4klk2ocLn+/oSVAv/"
    "XorVVTAjlqek+ZlmQ040JVLu6DuFygBUxZWBFKbdbNA4xHhaTfJd/GtjtCLo3hbssk1UXls0kInyyk3yt1Dhlt8dux0Bt6mp"
    "O7mmFSbslqsmUm6Du5pitC5O00u8VcR7PvKSOEV6EId0HXM3BVfw+doKhYf+mnbXBBj6pTwYLjYwbxTzz/KoKc01E6KKxlfK"
    "RBhsMWY7YbcnX4K3eOt7r+Shpz+a4fJER7ClD9SYXHGGQy1ToG2hd+6EctuwwHyemGeH8ZptPRiBffSG9Sl/SB8IbapWlIxm"
    "kKNmX6WLZoN0e6YdRQudb7vCnulX7XxR+Q0dpzAVuJ8IZAPFRugJh1SZmialn75N9U59oE90SCZK6U7wW0fGXwh7GkVd316X"
    "TlA9+uLbeYpShTe9r/XUFeA8dQL/ShNGlsJEsQpc891MDN9IEEHG7//rCoTNxBbVrHp9e+vGDPbgOeg2JFCXqlD2ig5Cnpoi"
    "UgghZvG/fP5+fz6ofhgCgu7+cR2oYsotM8xXQCPXLF4JBbqsBxfffSXkvrelG9V4izYdmgP6EC89WKOLbBu+u+PQRharQxSO"
    "Yi3jrBvUTrGZxsM7jeDrbZbBt510EZYnyiabcbLp5gO8EAY+S+C7URL9tJvx12VpRo0PZ9g++vtOujdEJ+XDDRHljyr4IOKA"
    "PyZ2kRda5JctThTJ5ISmRcgDxygX5qpr6FX7Kn3KD+b4LLkYuNA+cAdncdVkhPCOdmRseug91oMdUjrx3qD0Ezdtr5BG/C5I"
    "N41SiGriv4NAOepHmdKR0EumolBmRfvTHbz8+JqMI0f4Ppgqj8NkatKm0bj1/TErfkBdYujEno8cGUUcdXvyni+oBWfmK8IM"
    "FU9d8MeaE6Ia1dK5xLjFz30kL+CHQoUtNP1eNwgHMzYb7/XITuDy5g3s425lJxgWJhXpvLvoRXkTs7pk0/1N+c/acR4YZ9TX"
    "Oh9ySggWhhWaKyJcjhD677028yIDqXredk41iIrJMuwFB8RC3ABOYLMxlNGdUNXcOd+i2CUWdKf2STfOmWi7RL9Wb94J4Mft"
    "u7WJ5CwfZ++Kmx8ewxmq1DQK3+k86CC6HKSoxk2rwi7dg67gIPa88al3eJ5IYboVfvpKuhfFqJ3uzuNFYoHKkJvQMSxXrBpr"
    "mT+9ok65hUdL5JZVpHW/3IEHyCC8xLSCC2roW03qcNSCPWz6bu9KBSiOu6YryHhqXZUmHMHJPyq5n2bRRaxZThvCmVW2mFrj"
    "5gxdoIJoVfVrRhTeFif8mnGLSHpWpUj+aAFly97sDtyQuuEms6H2u/OO/+gAVpudE7PvXzG/9Fe72yQo+gzz1NV3kj9tNrp9"
    "OwvPgQm1J1vIO5mNCbNwwGoNxsIKplwcxn2KUGUr4WtZTPbz86gKXjCjMBNMKH3Me2w/HVRmIzehl2mAa3zl7ZXnZ8o86E6k"
    "SaPCMsyDJYzojNrh3WEu25xtpqKFcovPHVE7Ec9V42I5xniFpv7cE50/SGBWzJyJTotY6Yg8eS8GbOg3mElm3lhtt3e/HYzS"
    "VTARTnMNWN2sLmbDVEGmmy15FKsqVkagN0yEPCdPEafqU7QD58AovnIIpTzvzusyUL6mR1Vg6c0FQe/7V8oG5EsVMal0l6nj"
    "QYPiVr46UNjR7Guv7/Rb8DEw6+6YzTJI5Hq+dcW0gx1xkDk0uMkDyhFmlBW8LbKYXfkCtPCTTMnjE+3yo23LeLCrXIR/kv2V"
    "RMILh9w+h/mgFLTCVLDBXqjj9Svac+V/ReocmTThkPQQXOGsZZ3l/5/35zAA9uJU/HQdrGv8oDsbhC5Ms2bR9BT/TsgxuMgn"
    "NhmfDVr7gmrtL9uUhffOhFLjyx3bRskRdanC8CnjBfuUzapmphq7qg9NCyiYDRLDDU4wWwy0DLgzaLVvCCZ+1650sssVugzR"
    "1s4vktWBtwu5JDDGzQonhkT1qjKIohetHJ9EwKinvvw0PfANF3yE8tme57PwkNyUOoUo6i29i7qBl3jKuIhOlpt2BXSarotJ"
    "s/Jm/W/N4RCbZSKV2QBh0gTRKKdSHYZp1YfZzzvAGbkhVugK8YUEn9bYrtrMqxd+HMo2K94yKepceVXTzaQp61Ai36k+sw7H"
    "SXjqaYK9CzXF92hWjMF0h8JbHQ1W0PA3hnCwcovNoZJu1P1vU20zqCLqmn3HhDlsg3aHdrQhVt65/2U2fXb/xaxNvkWHkMwl"
    "aY6IPGHSep4MNfncbdGOgDB6BwQZHHGHkMBscalUJ/Cz6XnwMoC6xPu6Yk0D9cCHytMUfLD44TT0s/GlKtT5hmmmVr8i7RAR"
    "cBMVaF/x1YyZvzQG+6R1fZRFMNuqciN+NIGCbA/RKH07/aMU5M5PPnQVYqV9ndxCtlOX9Cr01g4iX1Rh3m/KoGpYR97t7oFZ"
    "DRNBdtsKPboUM28exdFkm8WC8o35iC4ExZKLdBQf/KIJLRIVnDM8xvNclfrEqlqlWTFv4yL4C9xw30m5Bl3ATZwMs0lsAnyL"
    "em9v/bCLDhC776VRjff/YFf53En52qye8WfyuUAumRDHiUW+RtFKv5Wu0BU/apog/vEv0ukZVRVdo3oNRxVL2gtm7qPe97rR"
    "f5Tys2CaDTeLBxlKX+WIcpPqIkPVAfYztCsaIB4fk4v6Y7WPadpU9zjMKgVtkE1MtLBA7eO+G9ffkBOIHlX9vzpTyTapyJ+J"
    "RQVkBxNM57Jnn4fgDtFDP81F4ddAeQTfCmkonLvkM9R1lsEwR/p+o7FVl6o7xzXaXLKMWVMvWLcQxIC6hwpVHBk86DJT9+1O"
    "TbxFIj2uOkWlqBLmksnKXeUls8QW/rxFrFBY5hLlxAo5gN7dKmLO7898saM81ZZRjVKyRbptDNFPHJFRwPfxpqaZbbTqsnJG"
    "eyDmzlspnmrHO8yWdQrnr0qyLzXbpAuEV6S34IoSqSqmCUeR9egtWabt0AcqD/gONEBVMHnEywFLH8tVxaAUw61quvROIIgu"
    "1h+z3UKEOg/MY7BJd2prmTTCj+79ZAiFsoua8O8s6Z3UfjidPOaDLZ3YNaqWGQKrjAsZTLuCZfr0YYh9iDKfdxW2hAtTAmwh"
    "in7qCZxR2LM2ywxFh+21gw8fgs/E2cfrsA1sYV/sCWaoUDjBhisDrI613oo6XY0cwteDNTwmrdIFqgRt3V1P3K3dBjmfttiO"
    "A1f1GVlH5qMR6YL46Q70haV/c36TD10feMNJfhNsfVbyYQKdD86oS8M2yjWugAU+4fmF2bZZhnKH2/ugkJu7FUCk8Bn4r1bJ"
    "LpRLjn20J79lAg3hOM7UZIhXHjOFQhZ1ZdV0wxdb1IbCj0lhihULt4eRl7RCtDyrhRGEE3Ljne0ryTSm5PY4yLaOeLhtPWB7"
    "YesGitCeoZMaF86ocsYfxEtJci/9TuiSZ+CjRnhnGR6LWUSfoolJ4RNNrWj6xaWY9P4sH60sMK/QRZoHyA3Uv87Q+zatRDYz"
    "ZjPHVIt5Zjn42dvbNqXiV1vkJosD0lcOYMPIKNLNvJFM0iWIA4p2epbKJFLRhTGN7pKq6UDymjyy3oexYBJmk5l8oTIEvOOc"
    "mWPlvOCjOgWed1c0r9EpjrGNYWMEuyvCRzrQ9pNJplSiF9azrujjJrqdX/quKxziHEHH9zxgnzxBz1qEsnno4ukUPiWWCRcY"
    "ILfrvMEEtQ8Pb8rBSwq1yLrVo9+Rtq0nqAnbAniKemy9v9p/202M4j3RGply0xOZisznFbBdncQUwM3HHd/JMU7QbihT1UYe"
    "4Ejg8mDfKhxu4GPB4ys9VP4XDfIilWwZRE+YXOQD9D/jOF8RyH27loymarn4r1yaLciZlB9sIueVy9gZhMAbqyVOhFkp+3HS"
    "E1dxnlgzlJmP2Lshb8LXPBXWcdfUorBKeVGvmFnbJEWJvpNw4R+O4hOLLuWecpTdIua5bc2L15bFwB3XsSe4xzRi3LGwD/x0"
    "yWbD8jVZLaZZL5M+fBeaF2OBh9zDzjETOFWKQO38GrOlvLJNB2koRvjdqi5F+e0p7E9Wgze4mVgFX37FBUZQ2WDLIkkdeqsb"
    "loEcOGh5yI8RK7yjzRtcdaedCNT0mef/NgP8skXIoifJH7+TL/SjcEBOuNNCDaB59s3zTYdMPII6qA2cpdnQz4FJi1Wl0000"
    "NeNN9NVzuhS/Y0NRjypc8f0UnChf6ZzIQLnk3gBZTR6jS6GOCUVO1pl8/pNLXEzHcqXaCIsR3THb+UUnarAZlbv/N4j141OF"
    "I9tEccJyC1mfWr7sZGY0J8pKLvtGFf42VdEBD4hwtIOG4ayUp5glR1CmwlecBj76YpwqJOuW5WTFNcpSD1HFOh963bCCs4ge"
    "/Vup7/47mIT7wQpOouzbNH6g5/aIOh2m/PCYPGOy5FrlNrXw7AoNg2shSQ5DgbIL3kcHukzr8A+7QcM3U/CQNIg6mV7jlz5W"
    "hboGsPnQCe8o4/VZ6vMvjnnmmlo2X2VycMUN7i8SKVYedAjzx25Ev3pHmQ8L1H10Fq7BJTZvreOk8H9u0QQQk4qLmxooFtfZ"
    "blTJLwr5H43eXydL+GzrYH4TBj4Ks/Dl/qpPTOKdhXTNtNHrYbHdP0WibikLRIttOn96UAgRW9g3tKucrYtCjXjcMpJ4Z5aC"
    "XcgQuET4E053fGAI988d6hmLVz9w10+DXvUm1cM1PK/XlbBtYqy8TjcT/5PyXpJxiHCGr/i38h6V/cNr0os85tKITY0fbqH2"
    "1Z46n6/2yutSkk2Gw5R6ivODybYz9kHMLDFM/HfufS9yH619uEzOS6lgj95RVbFnVL7Zj0vwEusNFrRuUoXutfTrXmEJ9d8q"
    "Ff3RsupIblddvDjBP66266cjkJdxgv2pm+Yf3zFrTyMN24oVsMjE4iD9qFyhPsBv5IcRxjWiAeVy3po8/aw+4atZrBt1DRNN"
    "A7CerKTGSFcmEzfT4xbbH4bxrbpJdsOiHxbgJpM/WJH8DdcP2zRR1Mi9i7ujuEAc1NUr8xUDZu5W1fSu/QbjY1bJF7FtqAwv"
    "fGeAdSEPhDKzX7jYJVl0EnN6T8WQKpd0VBegfNKN7AIpZI2+23bR2PX9BqstupPs4gqgj7QLlp+NqMrJd8yhZZ2m6n48EQS8"
    "2Uuq68H67UBw7bCM27RXZjVw/lYQiLfp5qaYS6qcb4cF2irqXFEDVukfOcM3f3pCH4J1qZlr5XbFlrtuyhO+VZr0AAN0211/"
    "Oktuhm3CEbEAItkPUx1S+UXUicq+1nJTR/ncpJhGjgib9waEUttIokaMhkFgTWkT+GILtSJneIjPVNuP37y4vFOM6+7c2CM6"
    "wZV8smGIWaXbYaG2C5vibMqfVdo3oY8zNQH89JM/joPRdhVUF5WouDk87a8oI5PMc6hWuAxc5QyQq+uj36L2Ow14+EftqNrU"
    "KbxWBvCO1JXtHjliNmLlq3ez/Ok6aepjOogpsxD1iCLNepvdJquVh/iUvbdOuAhfzvJpFFWNj0zn4Fyshmv0nrEIVFtNooRn"
    "Y9AfJ7FdsPllCDGmq+DdCR8+UIwj41ClbaXiiM1iS61Lb6wokfQkVu9tolNjKO9o/V4CVw3rlZ1iuvWBsZRuE/Osr+j/qPno"
    "FZuv8mRTqV6mUTmky1eOkO1WfuQP56CPxSJ9wSeIZ0aTF3Zn35H/EvTxBd34BxVsheaYHTG/okP4S20X02yxIoXrZ00tRNzt"
    "YugE5jRH8B0dDlqkMzrWao7LkTws+3AkHc5Ug5mbjH1zU1VHTIy++a67+foNnsTK51SAuMa6E/lkuOgm+dmQQ1Sh0k0bxK1o"
    "htA6k6FUl8lzIAZX4Sy1C5GG/cxOmQq0S+3B8DvxugypEv/NvlzEzYIsY7zULAxrTn4QBa/RstxKnnKNmnHVpnwsDIF0Rapx"
    "lWvQ1oExJlvjRvaAQOW58h1oMZ0oh7llh1ijN4q55UVVPHEjIulv1Cjz9KNSk1TEbuk7X/rKwc8yLP+gn/lhoG0nn8SfU4WQ"
    "f02MoXzrKLRPOqM5ek5XRE6SKXDj+436aHSkq4Kt8Fx1AS/5GHYahinH8TVXx2b8cBMU8LnIqgj58L1yPtqTaqVrkK/Kf1Rs"
    "M8QFOgxTiTj628PKE7KKcoPO0t6Nwl0o64kBIoGvR7l2w2ZOsItdo3PYRSLX/L97nxNrvD+/i+Nhqzri2VttPTNu7vHRKFgQ"
    "XIjqe20wnTv+oyyV13Pfp70WKTYb7Ci5gsa5Eb6Vd2IjVb+aYl9TGcqRh2Xk3/jJEbiSiBQvcT6fZIpWHIjBVDUcZ5JuzQIn"
    "OluOFPl51glmgGVxC1+oAj7IUYbemeHKhUtSNf58+aW/tKr8TT8RqN+hc0l/hyYrRfv9HvNUkEQmy9vChPBiGc6wyzca/7KN"
    "Htf069qt0vQN1OqH02ZBmkIu+EE7dGUO7oVqa8mmhyEWvTfc3UgnIm/aD/xnn2ZfyqF96Qm0I1ww51gxbDar3pT3iSWoCoVp"
    "ireSB+FGfL3yo1xy21SDqm2j+TZqFPyFOxz9rv+N1MySVUwBclJuGy9Jf2nn2zXm/14Em0wB1DhxiMbJdjrQFMVXPi7WTjCV"
    "qiB1maGQPSPdRCeqCJYRsew+WESDdJZNgSJPPKV7nil3rY+VuWSMvgFkGsbu7iuX8BuukltT/GeSeagcKHVgV+kY/OuCfSNx"
    "LU2CykfNdxeBO9wjwlUuUp6FI1MLYugrdhKGme0o/LWzutBPrtlo6Ey1ggTkot4ia8DUi2KL57WPE384bFUpR2F37S/7iHr1"
    "hfK1XCMNMBtojw3Wh0jzZIXtqWSWar1tfkk/mOLSifL74YYs+I4rFWfRm/dOmAFiASXII2wOKlN830+a0o6DTn5ZkU364mxj"
    "IHV06wynaypQsMOC1ocsJ9uNRzhUckPpdLV5DeuhumIj7Lb17rDNEGVdTQXo8oQ9hYdZA5HDaHLBbzyf50nh2ittF3VM1iou"
    "hSuYTLQRrmSAyfmvZ6wyVHQk6XAmzojNnB/egu1/vvPQx9oFtxPNVCA6B7nitdAM88R9bg9+JdP4FysgnYhDN8ip8BJK8YQ2"
    "Xh4hJ26qfwm4EPUKF3KPHlZlyb26JLwh9jyce+RhaGVtpoV52IZzpT9rNstizs1duQYHZ9mXasUpLwoehHytTvCzq5Re3d+n"
    "OmEyuQ66VaN0nClOIc3BTQfXr1dY1Ij3Wi2O5AWmnvKivchQlKb0opzpLtUsGuOH0DX7hvc3jIrd6jN9snqI3iU8YAB19CSM"
    "fAN76WmUr/Oza4cphnzK1zjBTcE3pnCNH11nVq6M5jpuxWqqYau2UfKGtaoguyD6jfqIPbf1A5caT8MANaLdEZPElS/SuFmj"
    "H4y4xbcyE0SOMhwlPmxmd2AIn0oPoiI4amp8loIG7i7d/OIKDMXzllvfr6eHpGwi5t4JF0Zc0SNyyk2HNtDReJgfRD2gj3RU"
    "nMNh+3a4AIqlV9QK1QtiKX/yShf3fgZ4RSZreonXZDg5QzfALWXSzW6U4B5lIrHNj+BZu0rdtDB4f14oes9PdCZmvlWC4mAM"
    "4WReYshXbVCXOg85ka7XlOPMB96yGxtvl2i+B/2ZAlwlOfEXOBUu4DOqH33cygZ8PcW2kluk0zQ10jKRR3lpXlTiC+DP1zzM"
    "1iUxvaooolW3Y1hWxJi2UBZTAw7kE+tUJo/6dRlIcKgQmz6cEA+tphQjUgmqNIu33Of0A3jXeKnuxuHC/46ADlxxL5HwwMv8"
    "0e0UsZ8t/uqCZonOlrY1zvw5O8XlK0IVS2QlnLT6aZ+iDk0YXysdzaPZVlD4NWdzL70jUcTEcnnGLDLkQSBcYQLpC20duuWM"
    "/EEx1yK6glzBh+W9YQCKZT3ZzrstP/LCzb904vI1IXSGLskUonBns61D0BrQHAlDn5U+9rgzZJOpS8d98B3cUzlRwe/9x5Li"
    "Sw9qnruA2TCHq9dfmPUoA82273tSy1Sm4jXVBc/EeO7ZmWUyDBHi9RfEG3kUuIp9xg22gnHBHk/fahIMDdaNFm1sgVBA7uI2"
    "dZaul42SXLgJavtr51QUiGFC7qQpTqWT98cexH+wbvuK7ST+d4Ep+eIAldJbdJFYKJ7p0yCKAnN/tnhvjvdBp/azn303nvf7"
    "5jQw+/Mg86y75fQG7UIva8ttY7l5MuKmsk7E8vdC9cNwR/Jgc6X+l6l3B8Vr2M8VgwDyQG5gL6QloYdjmnS17Ag1raWTiHFw"
    "DGZgj3hMv8F9XDJfiDPZMmlEe6QOvn+J9swbbROl0FuFVD7KeBwi1wrtIO9uk3HfphNGkU432trMp7GJOEFYEHuEOaXX77L/"
    "KenDKzwnlf1pCjGqcIV5ljXorYmcxlmfnHxSIPvBfeM8E/hxHFxCk6Di1tt711wXSBWaUBM7pXDC88Zqvd8DT/qE6f7snF2h"
    "942xTBMdAI+175ThwBhj95V5sYN8Y9EFEkzXKPH9KeBv5aRwJQZwsTL+G03ip4v6AfkEu8AJ1Sh7LXeCL0N0jjfRdw7CrVqZ"
    "ZWjubOaj/MsqFMBfwF2pEjVauuoOTVlqohLkQlfpG41sK//7Xbj9uIJ8pXgn7qAZVehN9nZ8FIQdVtQB3JKdr5UYrg2ELSiV"
    "HlPOQz9zN+UCXoLp4j+GK6vIPXFBnn2QIo/IW/KwnP9kwvJ/i61f82fE4mcj9ADpoxvRTiv7hAr+lApA2zAKvQFPe9EkLtdv"
    "W+fIM3Ko+EmjIQoF8XXYyTjAdxp9ZVdTCW62CxK8xXPbCHGAf6PNsdtRLj6eQstaH9yvGYADn4TYFFHLfxCPnMlpdZxpmHQG"
    "/jcA5gwyLR2lRHmCdiL3iFEU/OLGTE0u5psfvQa7VDt/QLrzPtb7N1T0y9l7Z8ADF35tC13r3tk6KgOoD6rJOKZYeUE10JXY"
    "R4h7UsoeSh1UuWEC1jM+UN0p5oATuKpotJ35WacYwa+ALlUG1SuPUK5EgNm2UIurwQpZy9erPMQ60pmqJiK5t8S81onyp+uZ"
    "NnxKn2hayX056Va1Qx05ph+Ul1EljiW9hOQ/LaRrpVbekyu9O8dtqE6Ft8wSu4pyFV18MBX/fBul43Xxe51SGS6HpyjZaoPQ"
    "jPGz2J0qU7ShZocy9AbeC8Sp0h69TAW9WESBrMfPjlDYR+liM9WiTOWc7cKVa8ykbRDvywxbHeNIsk7VyPmwVYoIIcW2/2aW"
    "oaDdotU6j5mnEuDH6xa5dh5sDZvOHT4PMFWCGq0HtH9n/PUqCIBn1J+FWu0Rk6YGYlcc4aoEbypad2S1pm/lK9AJWFQ18dm3"
    "rhQuygmwdrPVHjDPplJYFlY14ar6lw5XhNmwmQdVp1i8ecBLuiDmjXkjOfy0EwTbR2s3bIrVvcBTrOe98BSTAOzX+W0qBfWQ"
    "mYwfU8MdsidsumWscfZ+LZ1uPoxzUREdr04wi3zRQq19O00XZpvPumh2xAEQCv+3Uv0nsRbu7+c+3UJTdC6qUbeS2fiACtP+"
    "eOhjb7ZR+vd/HJRTmXbFifgKNyrNve42kPmm//v60Sko1A3gNEWPsYM60I3CfjaHjMcXbB/tCN+Ib6QHQ/bRplcWh0QGerlN"
    "b8Auoo7ZR18GGxr1v4olZqkT81C7ViZQ3tQ7y/PEjr5CWFMFAEMul8x+WqVrRU2P5uCM3apQzP9/uKGSIA=="
)

_DOC_MODEL = {"intercept": 5.86174, "coef": [2.28736, 1.01673, 2.23137, 0.04374, 1.56851, 0.17923, 2.29702, -0.78612, 2.27534, -1.09143, -0.79869, -0.19739, -0.21585, -0.54448, 0.06338, 0.62606, -1.38473, -0.10194, -0.37684, 0.6173, -0.93388, 0.01407, 0.10935], "mean": [0.73506, 0.37955, 0.54061, 0.30529, 0.24407, 0.15172, 0.58027, 0.42569, 0.25193, 0.05225, 0.07643, 0.09567, 0.58809, 0.21143, 0.41086, 0.25834, 0.11382, 0.76815, 0.28962, 0.60976, 0.16544, 0.01643, 0.59871], "scale": [0.11877, 0.37676, 0.49835, 0.46053, 0.42953, 0.35875, 0.41174, 0.38595, 0.18942, 0.1699, 0.21802, 0.20895, 0.29953, 0.37146, 0.35132, 0.22369, 0.18347, 0.08579, 0.1482, 0.12789, 0.17665, 0.12712, 0.08226]}

_LANG_Q = {"long_share": [0.15789, 0.36601, 0.39464, 0.41322, 0.42511, 0.43846, 0.45171, 0.46218, 0.47217, 0.4831, 0.49118, 0.49812, 0.50487, 0.51125, 0.51772, 0.52695, 0.5382, 0.5534, 0.56826, 0.58311, 0.654], "complex_share": [0.21739, 0.37562, 0.40879, 0.42667, 0.44561, 0.4577, 0.46828, 0.47819, 0.4866, 0.49444, 0.50505, 0.51396, 0.52232, 0.52923, 0.53624, 0.54557, 0.55518, 0.57005, 0.58279, 0.60301, 0.65643], "verb_var": [0.1, 0.52619, 0.58824, 0.63889, 0.66667, 0.7, 0.72222, 0.75, 0.7619, 0.77778, 0.8, 0.81481, 0.83333, 0.85065, 0.875, 0.90476, 0.93417, 1.0, 1.0, 1.0, 1.0], "mattr": [0.725, 0.84347, 0.86037, 0.87037, 0.87978, 0.88639, 0.89129, 0.89476, 0.89869, 0.90246, 0.90448, 0.90766, 0.91125, 0.91492, 0.91856, 0.92267, 0.92761, 0.932, 0.93699, 0.94468, 0.975], "typo_rate": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0015, 0.00185, 0.00214, 0.00255, 0.00313, 0.00351, 0.00385, 0.00424, 0.00516, 0.0059, 0.00664, 0.00775, 0.00888, 0.01088, 0.0139, 0.09211], "informal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00125, 0.00166, 0.00248, 0.00432, 0.03922], "slips": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00135, 0.00202, 0.0177]}

_LANG_MAP = [[0.1311, 42], [0.3437, 47], [0.4867, 56], [0.599, 66], [0.6751, 75], [0.7435, 86], [0.8368, 96], [0.9538, 100]]

_TRIGRAM_B64 = (
    "eNpNvdly7bqxbfsvK/Tou8P29lq2z8cwAgRBEiIIYKIYQ9SN8+9nggJa8m2EgmKBIoveeyb+/z+mqfzxf/6///2fv//tj6mk"
    "3z//8T//+7c/SlK/f/79f/77tz+SWn7//Gf7qRZzX/Dvv/2xmI/7r3/+/rcp3T9//3VK5v63f/3+N7Pd9/19gdnU/W9//e2P"
    "Tf084j+/b5Z+7vv7wWmx919/X7tYP35av903+31fv33cP//Znqbvf2s/dbj/+vtmOpz3v/1+cDjj/fP3BWes9zv8429/xFrG"
    "BbWY8Q7F8MUmfYyf0xT5tmjG0+J97T0kJq2/f/6r3TetYTxtDWl8fEjnuEM6Fa+j/BgH5fX98/fNvDZ8hfl5h3Zfk8dQmxzu"
    "p/1+sxzqeJ1QGcma9PjMpM34TN3v8Fe7w8/w/dW+zY1pmZwZFzij7vu2a/t0/6PNcRpzYVIeP1Pe7zv8fkTe7Riz3cbxOjZ+"
    "jHeYpnzfrA1qVmNIsirjr6rY8fHF5vu+v4fE5p/xbdeuarzOqvQYHaWZTd3v8N92hzB+2uDvn7+vDf7nddoCn+YxUNPMSM59"
    "afy7LQ1Z1TGNp8UUxkJM4WdV/371sLF2tr5b/mq75RzbSZ0fY4FPUx0LZqp+3Kz6ZSwNvzBvKf9syN8X5KJYqffaud9X9X9r"
    "a2f5GK8+FfamSee4QzrtmOOz76y/t52lxkt6VcbEBs8ce1kwZ3TjzaLjJZ35mcLfb2ae2+lj/Jymc1iN6azj2rO6Md3VlTF8"
    "rrCMim13+PPe3TGw9YK7f7ad5exYD87qMVlW81dtzRh123fA35t98GM2g1/vj//9Dn51LK5+3/9tNyvj1XVfD23epp9l3/bQ"
    "zlDvhuVpFC+pXBmrxJV9fHzZ9b0Ifl+wazXuoFUad1DJjDukbl7bxnGBbRo046BZGloxQ8phQCbHoDr788W/R9KuZtx37Y9o"
    "MzSFsWCmEMfrhMh0m4R9SOpjfNtc7Xidat2YLOsWPr6v6tumWrae9cNOWp/HO/iMUch9Pdxb2o2NYx2f6fpnNu9kXuPVzQsv"
    "8rKYbWverNQ3K/VtlrF2zCK+pU/WP9pkqfG+uyrjpyofWM+8jyHJOzO0h/fYWeGdx+i8H/ZXu/E6us/Qf9oMmfFv1vjxbcaX"
    "sVt8Ee/UHUp7Hc9LerWNOVabGU/bzDlstTnN+LfT+GEnp2kZrzN1J/zP5oTjPer/bsacVR1dGDvLhXPY6nCqMYWn+mCjq2vs"
    "Y3Vhf31mk2Wbx/DZXGT7e4xYH77bVvNvs3Xj1Z3LI+xwfXxvW83aSTaw7G/X/PO+ibUzTbikSVWCnLqxqrcPIhulWRr2hd15"
    "2TEkL1v4oL5+/9WuNWMzTPkYSzkfDPVhnUwAM7T0Vd2W54vw4GUci7bPRVs794695zhEDH/s092cTzbj1bPB2Jj0Gm+W+sa5"
    "v0KPL7bayOgQE6jlHJ+5nHasM+vtuJnvE/v3to/Zb6nP0G0J3hibdxjT/e7BU3Osyd1rp5lBR+zpFv5teWPa3nZhoy8MlFow"
    "K0tmd7uQxweFzM+cH9GKH6tk8mH81YcyPigUscrdIraXFBfqtzz28SZrcuq+u13QbXXbF+4aT3NXHtN99c3wnzZ8j/jhc2zI"
    "6bOOO3xWN55WHY4q90e0cXCGVWJkOy11LJjlDn3+1W5WC9YzmTjua7oTbts/JELsVMbNkvzb1Hd3e3WrxyOsPsbN9CFD7RTG"
    "XNUxsar6MW/Vs9+83sc76P0xJDi1acWbKs07aNlZ1rHfnC1j+9vCFBZFCKiK+Kz+FS0Q6IHWne5gqyfzRfrwhWP9MidGrOci"
    "/23Wk0jMruy31bI3bXdJt1t83dPSTGbf/s2pGc+QFBx26RvnP21arjHq6RKndhuFezudshmsx1x5iRxTt3LNKPRkpS0jTxTv"
    "8zqGJK9mfNvaE4V/No+TMK8a26e73bkziTLmrVhudmU73sHmD6bljq7+9RNdEU+qM4877N0itqF2ke0U8S2xf/G/WrTCDqhP"
    "P4S7tR6D55/Zl9izTMKUZTstRo/XMfqDGHHlg1Zxal6xwL3GdWhLLGf75m2rzzA6Zl3HtevKUl6tZkN6wnFf0vj4kgLbNDBQ"
    "wUnc1wOiP1tA5MYjTA+02hwHR3bgKu62sv2nyH2jI9F1snYmQ85r/IL1NMyQyZjXombywplEYXYMlLOYbZsxCvkZjssjFOGX"
    "ynkMVM6SaphlvNmkwQl0JaurKY2NnpIZ35b6LrxznBf+4qWIPXtA30x8sningJULlcVVxfmEdIz7psOO0TkkVp60HU5CWybA"
    "lovsS/a8dryOU3YYBWXPsTxt30PtvmFladzr7Cdo2Mmodktoabedjb5LrrfztF0RjqsXEcjLfMgEnKT2J+HBlK/xZvkiCy2G"
    "cMb0921JcX/E73d4XezuafqJrtrNDrb04cMIUXx4kzu9WbRvB2DlxP6aZRuIy9KD6ZblG2Jw54y4g41wZsNZbuFF0PDiES+x"
    "kyb5MZvJE0z7E2s/RbLm2Ndky770PmJE/Rz1PK5VmbUzrXXslrUSNlfnxgXOXWSWl0QrEW8aFS5fdZffdkAhTCqeb/MPwz+T"
    "gs8G2MJ4ghFv1jG+ZiV+WC1GzBbiqPKIo3py9VebQmDCt8cPRRWH31TRSKT7vh1gM4NvZig9gpFcMfFVj2mpmsw9mzjCJBOV"
    "AJiKGQok/EG8aekff/tYjFixihzyxKGcxtyPaNbeMNSmkCIm84sx+1XHkPyquINqE8BHMoIjYhGTcSSDjld3CijNd2Sv+cLu"
    "Qv/ZPANW2cwEAnMm989Pa8TMTytmcE0y6oEhWQwG2mRsVLYbadTGHG/+A9zzPMd0nyf3PYMXGKt95l93tNIXeINZQmL7pzhG"
    "MsXAv8lsznUZ7qDeluB+2maAhaZMlpQ1kIHew9hveyAGDw7oxD09AxF/SUZmiIjUKNyiKj8BXFtRJ6m9L0QgPvjxvsETnHoD"
    "Nugz2W2ubOnav7ht6W6u2vu+CFxeFpBkEeM4vffx1/fOtbslN7UOu+7EZE6ZENsvPHhRBIZpyRiQDACfI0s5GlIC04OnP1uO"
    "g6Oa+lfc0SsmXvVw5gbuAnh1wNiUvi/afnOss2kzMvOEM6f6GpZAfdnxZl+WiN+ehElnBQquJ8bm7AbkNlcPTPca9/UXYFFR"
    "1zAK6gLNv7pFbB7nDOPN/I1d3bav7CyuvV/7Z9stxDCnYSGazh20xbXXYV73KkFvN1dtQ6o8fiqZzWIBJa3+AAjrIWAzKwuU"
    "w9LdbRsdhaNSgkz7npO1+EwRpHsD/lAkcszdSbSct8bxQTVCRERBkJPk0mcEZY3dQLeg14Mm+azw/jaRURk+0+gyvKkuvE4K"
    "ZCjhDOPNzkC+GQq7JdcTjOk8x2eeJzzZ+eTJLjz6iTs4LTSY1WlMi04AKimAgYQM5OUULlS92W+5E3+3z9rGMjIbMNbWYay2"
    "Czts0QD4y461c1n4llhn8qzZCZkXxlL2z1yE9evChineiEs2iWmVZwrVwkZfnoE3s6kNuNFkoOJcuABnL7z0ZUhTjeERSTxZ"
    "6qh7M47aSIxIQL/UNF6nPhIFmaFpum6X1Nzihcu/nqQbEzDpBPsgbjyqiwjvSsNUXAl7FrqVa+nZBahzCRJpO2Fwc3WJ2ZT3"
    "NRH0VodAoBUw0KETtm3t9EXb9rEwTCUJhGRBF+2LCPpliCrMFyztVzRCrGLXkyAjxjOSyzPhfzBBr0HpmASqNnU302botiU/"
    "gaFloGxfyv9p7uCxWyJxX+QRsQc5N1gkqfLC+y5iBrWEVLov8Pu+uMUsrrnsTMB+//WHJMxvsOI3bubd8ZL26isx19qpgbbO"
    "lsC+CDjLoKE69dPusBCnVRim5zoLY0iuUGHPOll6kwuLQF6aLP+ZS0PprEGPodZPHAYfm+Vm2QCXGk/O6zuaf4eLkUy4CGX2"
    "YhlNK0jOahS8SKljX5QKaJZO3uE0AiZ73KLXDhjLkSi4HiM25y7EiesIRgs75jB2yxySLDnIXVdwEiV4WeBxjHoqpL+T+OMp"
    "ACEF/+s2K+0lfxG45BwITgNxatDE61YxhVPnGdpSfuF8Xk9CsY6N4ypsSX2Qxhf50KWw9q5gabVYe9clIW0KN2ZIaSP5Bbtl"
    "DniyEA6edhBgqARVn7vPatt/O8fS2CRWnhTBqUrA0ekgiT8MqLvVpCVa4NK9Cy/u1F6B5CgiRyVs1NRzvTs2gn+zJ087u5u5"
    "dwtfrLTGfekqgDb2TF9wKMqTHWRD2GFO4CavkhC2vI5zQnj1vKW5A3WKXZ/Hz3OG/Z1FJVMUygzVwaKW4K0Cy78Jct5PqP0R"
    "nMq1B1D7YVgPpu+Ath4iKJXWEDK6z0XzTpUos3qeZrJhEeyQAPuTbt3G1tv2MiLSvRAQLS+sspVE11ueNnWZR5uL0wiXT3Zb"
    "BX94WRAtm8N4sxzAS06hzGwmnqyJ+DfZH89wpzAsxL2eKIBOEdfsJ+j4SbQ99bj6RlmBz1U+xlfkg+k+DHBTVESD6YXS4SX+"
    "benAxz1mQKtWGEdd3BA9lNvd3kZ3UjOg5IznnR/4+q5keZKxngH6MpcKCFWJSxaJU9eVKVwNO8ucH8gQunNviViPNZqBNnow"
    "bUYDWGW1wj6svORqoDJMudB2BAx/CNidKbCzwvyJZ/g0w8J83g++MzWjwJWLAY8yfce2dbb7MXy7R/7k7S9Y+44Q/bMhRASR"
    "VRRhUfQ7xRZsVHdJNwkL1OOu77EIrm/wye/bWd4PXhNh0uIuwubMzXJHZ9p9vR9L2XsIW99T8DbzCjGbe1pPK7tF2AcVsVyR"
    "eDJGFkwKXBBiAl+3BN6vTni1O2js+hmRuMWddH3zQmeHBc59qViNqkceWzXSjckImqR5cO4T23IRz80Wy7K3iqRN30+7o8Eg"
    "OrFpq+ParRoJqdibuQCPlUC8Hk6B6D7xvJ9mFVT4EVeLwdNx+NhYQP5Nh9JavB6TQJWgoeZx7cbm3RK5f7LEiItgxTHA2IRM"
    "tLKI3VFFsUWU8FlWEIFcx/jmakbsme/c9Id6mfF6cxcytI1T0PqUIvGkV8LgYfjDBRJ5ibjGdrSuRSvxQyw4NNik3mMHqDc5"
    "2VshN0wLEekUMMUhoadNGznvpuAWlWdQvf0GSf+GaP9WBYmFZyGqQpj/WSFOqicXqUkgjoyd1DYST0bgvCioZbYkFWpBnBA6"
    "s93y4wywNC1EK0tSYw+lHmu0pG1FIjR5ggZv3qyzNxnV5iWc8Rto6GaGrd4M7vYwpDtGYLcgN5tupe+f9+uscdxhjSIptSc5"
    "Qyf770gXy+UFT50NUmtT3rjFpypN4IUnNEW4mGQhTiIFrn0z3LkpXF0KkNEpfI4lFz6x4F4DLEXDFjEGMGOayVvmJM6989L3"
    "miQ2mgpcc3mgB+4YD3YHDMhhGQdbIPudAaH35iEfQaxyBnSvYSGWW3pS3CKxiqJ8kXS9ljhep0TA2WmDikuKlEtF0Py4A2bs"
    "IjR1QexDRa98SkY1JyWaXisKCjAm649hMv1B5u5tkVffkDxujO9miTUmjUvSinWmHlKTCnumjR0fbyyU5Etc0iLBabIzPO9c"
    "MVfdF/69DQkxeHhOdx7bNGTC/D0wqFlLKpckzP8iKPvSxKm6L8SbBltwHYvhEaneOW+bNwEzfH6PbZrf0M5vyVumF9lXCits"
    "yYpMdDXEO0ofSGsPK5vhkZsSTAcl1OzyIfK9AsX3pN+BDOxCDL6lIEAjHj1URH01Yj3PmEQgvI+dZZ76By8SrBcTIP7YPmMN"
    "eHTbscwGY1WMeVWQxil8DfQ2fAmkeLIQTY/7bn3fr6E3miQM1QZUzUS4gyjifdOXxh3vIPsJIkhSkrHqXaIgBSunCltkMSwY"
    "Y+DGpycyIkZsxVSs5Q0r9wbXMLKqY0BeFp6kxY/Lv/0me953WP4O6MksVWbJ7Z7SkbdwockDcSypItGsQCdTZaPXiNQ6LqiC"
    "y0nkuIijqgKznBJi6yeDR7CXVkKJtQvym/SoPDxvAeLoGs42OoZEzC9EFUtIEvRSd6ASqqlpJpyZlWAgxzYW+LGB1hmP2sJv"
    "aqzJTW3E9gZZoFmlIuKOBm85cq7gUSWx35Jocqa+TVsurZNoIokfqiZn0FJuk3vNSltcKzn69LICJFQuqOi2nYzOr0o0mIJE"
    "Nl/gGl+FlLYKYWCSGwOVOt/daPLMQrSSpk4PurWw5PwCZGv1NaLX8ySxOSvRq7cULoXIxlkcyNP+zFgP4Jsjj9c5BFU7+zJq"
    "1xbeLApgpTpQ858W1hGG+q0MF7oVBEnlqYNm/UYrOHivo2pLWRJz54I494dEyI6V+raUjvhFJAtCVFqHStzMxFyncfisGMcF"
    "qcDEl+oJXBZU+MuCiV8SgeFWmc1qAcp3CWS/ui7ov825n1DUCxDzcuMPN6kZZrLF2XyIxgWkTHk/FkxwMK9O/NsUHCovIR8n"
    "gS3UEyyyiEqconTEPYRDoHVmySAY38QE6wMbXObb+7fRmSV3yuigsyO7NUX0Gr3OpxERDsXH9Ek48xnQuISZ2cyRC2IQ7a1g"
    "eLoEwe3hN00A4Sw7uMbe92bDU5OoTvQ+RlLv1IjtFWxQyZhZ7cboaCemuPCI8BJdhYSApTNit+B2GxvHbszQ1qPXe7oJGtwL"
    "xOVlRc5p3diQziEXcEJP7JbJsp1cuIvemAvXc9Mb/E4wKwmTmSrbv9oitZOk9r4HLndCyjb1Hr2cr6Kn7XBTi7Z9EMHMSwhb"
    "N8Rhs+IrQmJVJ8l8kiV4cgYWZlIrwZ7nK3xNIllAf2beBPS677ebojYjYjKyjKY3mftboXtVGaLHSvZVH4VWahOl5JsN+f4Q"
    "tu8cHv10SoQBlHV5g9E1G2O2SRnE2cehbdOFWMNL6KO2DzL3FOBmVsKvSWDjUBDy7kaAOwPMslo8pE3AWOkEWPKR/CI+y68k"
    "4q9aqkVx+TpAGjsZ32wp7VXLr/HqSw9kb1ieEjAnGgEjhSpnJF13It6vmpWqRXLuerzTgl6BDHI38e2DEtVVyuJ8bFaS8xbg"
    "3T6SbaPvJBW78AxqVoJKcMGra9XuAsQXK9W/RsT0ErmATUGUZj/LqH1m/BDcPmHEEvKnKcFLK4cTdt2mtrzQwzNMphC6F0jj"
    "okVtHCCF9r5ob/wBvK8mL3gJpPEhGqIpUYuYLLSdvcmQnxjx4K+LYcyMIrmy/kOCHHxhWuIAdZJkltECfif1EpYAsDP0l7yl"
    "MaSTJiYJpudhxBbJQs2BvOFQom6SdEctn+R6nxUVsyUXsdmRNYt4P3cYoO23Gemc+ULp8CWQQTg/pK5DSuycOOwnSI0mssrT"
    "TC9hbEGOJdebdjQuu4hKNo8p9pb8bTc/IWtjKrpyq+0hGZJSH1gQC7GKNHEVOMSpZbxOzSgHsqBqp0FDbzpZ2tAZhah66QqK"
    "5iQK1FYVndj0UCQkeJzkjZRZ41iD1NXlakUSXca8eQvbZw1R0C6ccH7w/r4O37IbOwbKWFydTYQ+aiGgL4ooXh1QA99d8NXW"
    "b2KbZvOiYCfrsY/1UxdEkG6jyIY9SK/vWpT214UQZVFecLmLwgTRRFqpOPkM+KH5hi1+6J+vDwpHLUrf6QDJOUQ9lnus0VbJ"
    "0wSR0hYp5rAZwUE+WVF+s2P9boITuEdVZzc2baUeoDNHCVKt70RvRGqkAwFc2LzU9n0x6l+yIU+G71QH/vggrvbhFFE10z2t"
    "ZXi9cybHyVkJFwobVQRIsIUccu4BfbNcGQq1Jj7+ZdA0lB2AbRfy5ktwLqsYyXcno+9oexvOZ0v4+aywv/HeWT8sTK+SvGsf"
    "ePVFRDtaAExdKb+yksKUndfZJSebgySvFxs9ZaL4asD7pqcQB+ZVzewL63Bqq6UpgbrI1C7pPlL6dLeQ6gsfYP2v8cX+F9be"
    "pkvCGdT9yw5dNXXiupmVizT1iqySaGDMjTmQmhzc4ZAqB1N2iuYDhjQsTGwtSIGLw+u5wM56WYxjEmA0d7nALS2AhcnCO6mi"
    "R3Zb7ozq7z+FYfgAG4hIg0ARRjqgTO807vtOdkRtodKyYTUzsLyfBwbtZ8zVHCAX9GEEwyON0n2gWhSUKU2vZR7GvMzkhbHn"
    "Ac1AB/QwITJ8UYxNFuM4bfiAzeHqXM/920aXLT0rWja4CDaoleirzXsYG/N++M0kdieStwhjnqTKzIviblougukLNOnyiM6U"
    "MPxOELjdUilkUxZhQEAp+fRk0ErvPr43KEkEbRTaL3V8CMl9jAmwB/H6IVLgKBh0Dkjyqge28B1xuWte6XugL40yThNiT73+"
    "rVHUJ7jnqUgGN8tm2EIS8pxtukhqZE7SsxBI+GMk0LJSqHJKIKsyAUbqO+sGqUFkfaW8uIYq9CWkRVpIQNbCdtLS08GLSzoj"
    "WHGNSD/Tge8+etlRW56R6DUF9nyYQevCBl7tn5VjAtneD/7zR2KMROirMtRrxzLbZ2oWTLFxBJH63np//qzJN7RHZ1PbZ3or"
    "zPZDYsyK8lKGVgSKUN8AVt/i0atD0+tW2QHuHImYe8jLCrGyOhHXGCfIiOhA9hREsrsxAds+vMi2M8emSwua+9pgz0r/4ruF"
    "hx4L5urm9T9NYUU+f3Wu7g56MYN2AYadDfbB7IyDEsnYS5p1nI+yDUU6+VbEfbov2lsCy5slAYit3se/zd0+NPclBMcZwZii"
    "JaV1BmfpKqn9GQCW3imIAuhBBMfxb6urY51VQ0VaeaIdSghFiIjZYaOqIileXuzuV9BioPVYqUsIgke5USO2Bqp0tCUcTwfr"
    "7FhxM0ZyhkljxM5AdpsCjM3+0OyJFiVsCIRVzy9uzd5CqHZDMj/i55Va5fWiLinfgOC9qr+FBpsO0pJD2m/5h0y/CAVVsPa5"
    "x553rwj4t1zOgZyWkwkIK0ymPtB+HWqDCfrEO332abkhOgDtoJaxW9RGIBsDvjv1TKKlGoK4uErJz6TXYa5WCVHmZ++FXyPW"
    "qE9ZFYUU2Z7Sa0pJA4MHHI1ovWak7Flqf3Okzn3JwNG7oZLws6LGmmIeExszlG8WV6e6XqOFB9VLIweDlTNU02xeCVkqJaJ9"
    "PTRgf2el7rImi/0ehNe3lAqoZ7W+qGQuzOAl2MrUI4W78uZDVsk17lsuhN2XlHrnQyoqFzxD0QDwyj40cJYg3W7SenAj2Nuw"
    "JVuw0lHkF8mKpcrBRirElT9Qn0d6AAXRUtUM/5YqBb/V8W3G72PJ+V0hEDYgGLoQCi9KBNidXmshYMfw7o5kZMLrXRp5P1h9"
    "oWP6EhVHlpZlNoJr2BOs4hS2z25JdEy4mdAz1ubUTnJ0v2KrFyUSlm+86SwKwVlMvLkg/q5A7m8lJ/OFWC4kWjAqhy1xUtX5"
    "JdLas9u+P9vGWbC0DuG8WyHSgsNWezGv/lml8yCm8DhzUdIxK40VpS0xgQ2QjyYyOr7b6pY1V8iF2tPq2+iKtkOAj2nDWW4W"
    "XM4mln1yWRSYdgQu1jI6ZQbve4cTpDexdlIV4bwnynw/iWA4bL85eH9HgPFWTiizY3yQO6B0Zim+/rJk7tsOrb9USMLqpOwo"
    "u7GP14CbCQGlWZAWjMsGoFJnoqtvg8qrOuTp3gkmVkTYYuGSiqpjJFWFJ0sKHYjfiFPNm+k+e1/Lm7C10knukz3/SZPMz3DB"
    "5V9MlolWaoKC9JWgvU6OH9JuVCioGUHzrClLzLGOZV+kBMHe3MFdD5kkmNYHaclxh1Q/Kfhx8NfDiA5kpUdYOGC2DwQSn4FM"
    "4jTUchlN3pJ6dnuLRyU7kKLjzRATGDWPjfMVUQBFm6TJIAH9sToBO+ngo8NPln9X7n4LSCK6NgEHvP0a32Y7atks4r3k/vyR"
    "a4GtLAsIvVYorJQoM6KE4+v1qCQEXVzsQqohYu0iApTwBvq7BM47zTo+yKwoRldR72aFBVea1hW6Q1N3B0uJaSugTo087SsS"
    "qr2sE04Y/mIVxOVUn9QlfRIufkq9iN/SsBr6IAo6pP44bEghZuG79wJsHGYePGcBX8oy4rMiAokaCZDjBnCXNW0VdWRiv24N"
    "508FkqLxWpTCxmnOsPYL91UXkfklfZNMR1xuL7KSZK7Y33Q+qM6v4Tp2wR9Oc4yJDYeQANUMv7mduPHpZFBThAhWT24RLD6I"
    "bmV2VfbFB2bluaJECClV1HOHFG/9L/WFi2QSb9H8f0X2Rcio6EIH7pppu+sA/7ohGQ+dUk5M/BlIq+MOtbV39vc2/OT+6SDQ"
    "ckGU9dLq6KysnZoBapTILo1HHJYsLYaT+O4oO7ZaLR31iHeOOUixF8CdNx59SR/fW9NbhmAxShWqugDuNot0Q3XVSTNtHWZp"
    "W9rCFKtCau8tyL/eGdS9kMpVL+tX+qf6PlANsj1Ria8V81oTetoivWFzTrSSSo9marRssIlUI2lIALXgLFNvrdAg8dsq/3xQ"
    "golPp+C0ved3S5hEWT95aFF7+9gfVPhTItJA+4HSe4fcATIxwZyo9jArJmiTWGNyqN1UJEdPJ2tdzUla1cJsb9ICbA5QRcHX"
    "sQN8xRJMBxL5IxICRstn5hWPvkplqUr72LFpBzfapd+rdeyAGK/hA+KF7dscJQjWIatapSlBlC5uNilpZwWgUqUOMPZSw7u7"
    "ywldNddhf2fpo1UTlSFpQ1EzC325SkutGDGDqxIBiklo1Yofa/0dqlCHxOtpu5AIGSqmrCZG9D1naJlw7+/ZirJEBlYT/WHS"
    "JSI5RWBoDjzk4YBZ5i7luWnnROLoBOe6EAtevYDgDiUQAy0ifAsvaoJiJCkOBftQHNHgLDy6WjYQ2QORfdjoUaM1hZjawsRP"
    "U9st/74FB0Zw8AS2ohWgr00AuX7B2KiE5DHNwglLZYgtiZvtwPK7dJV6S+t4J1nSKwBxzNLJ/iVltV8PvdzGdG+RDRkVEbQS"
    "tVDtTu1u6gkWH7RkHRFzFS+pSj4AiGsMMPGz9ESN2KjYx7ftt4XCXJdZiCkkrNxOtrjXInqYmcxyZt7UDIpydm3dLXaF5/3s"
    "NMI9AbyD67z03fak0OSqfpBy9W5gTayyAcNulU5nppughtl03P4u4wFPfQlWrA8Et8vGlk7SPNamZQyJchCKTrMhp43QZxNL"
    "q3prm1vNglHwCfFo6cH/TVFLQ+45SdEQRU6pguRotQ2kV22UVG0Bc+UDnHu2ZkyWNTOs3Ey93mycqB/teMkgDTXN1z6m8GuH"
    "AN0tLIzr43s3uSIKWtUvxuzXh3TXgjcN+kPQJMLbrI0cKoJnCCc/TwnrFiMVXgn56TR9j+YX07dgr1Z49CJFpj3Pal8steC7"
    "YbcYEckl6Qo8S+npqQjKsmYhHiK2sj6O1ecjwvmYWMpBCqLKvo9YY9+pnvDqQ9BbLIyp1KlNoqyvElLl9MARAVzzerELFaXp"
    "22M2nSftkyaO4Q2d8tZgg8lBKzljEclF0ihnhRuX3o9bx2lvITo8zpI+xECvI2IyFrGgPTmkw7yTkCzwOOdMB59S5OgZZdiF"
    "hW8rXkn1BON7PM5cOJR0DIAhXUR7sK4PQR1GoXaJfNu8IqAqOwZkl3ocbQjgrDTyzR1PvYFcSuzcs83FjoTlUXjXi9NumShq"
    "AOtelKm+FUdsJDCb3coZMhWUtW4ba2ej2jlt0vo1wK4v9gQDkXOCnFtZRqs0LdnQCLxFaqIi8XoSGXlS4DDqDXlTN3qXbhXH"
    "6ns/6IYxVdqQ5WcbSBixIHplE6X3bu8FeuNccmRFRi2UPd40PrVqAB9R6jpMb+bTtqm++LYDdfSxgLKeBsLg6KV7DViSiU19"
    "qFvG2gPvu8mVlgiajbMpRGeqsIeKHEU0K+lspeTUkeqlUwkLxrnCx1fWenlLha1mKe+nRBUlU8a+gapt6ptk5duKpDQNM3g8"
    "II7I1qvzJzSCefTUgZR3cR8DtYr0Pj73EIF3rKJa7fL/u8JWdrdUCvmDdPL70aN8JzxYwjegw7eRs4p+0qhm+Os5xFZLpcp3"
    "lT7Ir0C0skpnzC8pti07Jxvsb47d8tLkSp3sltMJqxFQ91vRaC0SUhUFa1TKhyBlGIWXIvg/ZNRnR/+SKozYNM/D+y/m14i5"
    "qiyu0sVAzTtlpsUY3sEELYfMkAknKf6r8T0+8y2ybHMzbXfGurwFxpJmPurgPJ9j5WZTx5ia36xiJyO2OnZ8/dbQExtNls98"
    "S+ePt6c6MFZRAJ1RNP/wvHpN0gJ3GY9YFyXd/3Hua0VXXDcl6ptHAQxqgJpQ6qSZgodZ8LOgCLR0YvunC8N/BXLesBF+heTR"
    "3ooucwnAC1bN0qcM1zGdfJuxQsVZlCRThY9V0rTP6w8OnLh9Ye9EUId9mC3lK3ZD3a+lVe3qRFGzMepvQeinHkQ2WUrioAWr"
    "FiHzkO8lCZBfXQHUaHIpATu7nOhu8ipdhA4i3TPQd7zYZUCKVtRuqyGzVJF2rrnz6M2ImSxE2oeICADYwsl5KKeUWedMZVP2"
    "vM5iKNSOyklSvI6/LuKlq6OPt93A5TY5wuRtrFCSeNMYIEB3SxcW1XPpNtQrSM6XdiLkfRxQhAXffJDGrfBvR4btmy0s7Rlo"
    "85brOpaRLisTsNKmsJ60Vz89i8BLMXO9QDCMfaDuoj8zFC69LXt+d4Qzsz2AZIStDi+kifpA4XpI+rsEJ2oLulLOGqTh7WEn"
    "i5z8Fy7p677wxUs2w/7OhtB9NW/RMUGO1YMwv3Yq4z5NCpn+qWn3rBdpS9c1022VpDc9gLqyqG3TFxU9r1CkS54wTFIcMZk6"
    "lnJNZM2LpYLZbgRa0zcVEXulrWK6nAhCJbxNmKtUsYh2ppJlno2AfOCIRZacVhCVqiIk29+8QxXiZFZqfLxS1MKY8xwWpig6"
    "jKsIN661MGIORMCJItfmmQrx3u39BovmMQGrAwBylxn2IYp2JvRwvN0sQgdOhdw0yjpTcu1001V//QRwpPauj/rdaABwYLVZ"
    "0K8DGVidhTG/xtNyl9G0hfiS1leRfgq2wOUXKWk9A3FJVlT0KPUhhZiIPN89ArnPXsNk+vAh8C562iDHa07fsODf/ajI1g2s"
    "Z2otckwyQwd4dfUsOS/HJKTlLVlSED0Mrx4ULEyJsODTfcHPGWkqCpxHrJGFivMria73Sro8khQvHeJopsKgyA0LBmSRcvOl"
    "txO8QR385jsQCLwKq+RT0Rn+JajlV0QPHotMrEXBNuVfY3HlX4iffzlGMnZL0DZOfVSWcgcd0DmeAWlMmOUUpYQ7SJnmZJd0"
    "6V8ucM81E9CrjKonXx/ScOyQPqdYmCWI1M+K3tP+GjHiFBd21rOXFyUeUy9Oa/nmhQm65OjFXxWJRQ20onRG4LwEoxsDAoll"
    "kWbYvQv33RGHRaArkNciZ8WZi3ZA6mTedgO1tYlkIUmj+qryWGe+U2YtZ3BQRZcBC4od9blr+L/Hv31/E7p/i4XRVV4nMYVe"
    "Ud/9EkXYHFnVS6UmaHphHJc9yIE0GAV9wrTFk1BiU0h2ryhd2XvX5fbzjfbWhoLeqHDg8Cw5g3MANU7gvNCjlaaacpBjy/Jg"
    "+xAhTmcGnVkEkX2UiJ4Ad6dQGaVIe8lLEnM5jVI9gISsRuijvkRioXDY047t2wtvVopI2R9Nz87X2JuvxF/N04XCB6xaiq81"
    "eXcp1FwtQeREs2Db5jU2ziY412dF7GodSI7rWXOLMmeQvX2heUvpZTxtk+kDneMBp3YIZ1kNNSumItf6DBwGE3bKt9NBbH+c"
    "CsWSfw9kOndQ5zYgoJazIi5RMynBLIcZOdkXzqwjqTjUQyYK81qFtEgLtWdlJ5VTmojULAtGd8U1q9clISCA1ad5SwNFL1DE"
    "Pt73clZO1wWEqg4ReDLXiCrMBVx6eTm5cpHToSMzpJN4kRt7/fEBzkjmQx8MKwdxGwNSthlY+1hJoI3W4+O1lnM9I2nflDH8"
    "00l94bmdnPG3uDHquxWJvBCgeyCACzMh6zxT5TBbQOop5+GHkvlE+SLt+eYZGvcSgu7sLMx9bJEnnfTM2/Ry4xEvJ8cDHFV0"
    "/OAPSrZT0qhs1Rv8wYogNMsxbGom9Zw9j1gCh2atShZX7yPQ7PpLuj+ZxyGhQoYYsYiLHxOwiEr8U1oEZnGs+UIV7EXTq+Qs"
    "viRerzosrZMu0V5puYAt7U5pFaP98PNaSta0pUpnOhi+vdK9oS6cWrZI86F3lnyzh4A3igItGk5aSZ1rkG52FKROt/Llr5/z"
    "N4mCysYiMN2/3Y1xoQ53S4OeXN5jx559TbbPfIPmrycHXqqs5OBadFc2IzzOiQqDVKN0wJb6WI9g/KxQ9dMnGoHPBQ/ptx3D"
    "v0uhtqWY46XgpVf7Ra73Rcmlv94cXPCW48+TNKmQQqtUGB1dXsM+lJecKycxuDmoozIncP+ZuUMUXZAOjG94h3Htl1SR5EP6"
    "+nxLQyFpr74XGLFykkCfTsTE0Yp67BdiNp3HFOpMg8q3tK5QUpf/Ei+SDHGUmTmL73JypPkMh+JyEbgfl1836S93kcIkXeUc"
    "c7pNOP0hJ45+j3f4tlK8qlCHTAcFRjljq63hUNO3yOH8UrAaRXZLgLOcpbz4UxGXTL+gBl7xQ5r+UqvhF0om9iT9QC7c+DU/"
    "jIJ4MgsUoVcltCi64mrphrvKQU1fmmRlztJL3JO3fHZR330AiachlmcPmUxakhfSkhDIAHcplXUBzdOiOHZ2XVEkrAKHuCIl"
    "VQKX2hAFSddIawV+dArN/9mnuyVi38AW3wFLm2Y0vbMghtNO2dGekV2eFjTJhA3vdNLmwknXxFQoJJ62nUhhz6Jd5Iw/K1yS"
    "SaijTXZjS2fp+PaWI6+mnUZb+0XN9ikXJOkiZGbS1LlISxdF61cldXXWUSHuZii+KhJC00+8a9ZoFiG6hSy9pFOJ7c0Obo0L"
    "NsqW77FgkpPmpgXarqQPHNWi8HpqJ/3dEVh6kaKpAxN/LOROQZo+mBew8eS1UNSwBNOWOdfzThF/vJ4IZtSGRGjb0FJNVo0c"
    "3T1ax1fW73xKrzTRFeeLEtFLzpMo0tw/7tJp3dNTsvb2knd19nuYINt3YbNc0gp4miMNx6S/3Bbo5JlWOd5iYweYhcaM8/P0"
    "QXKnT8VpEMpq8QFIl3P6kCqHGa17EuL6Ata8FNUI5izjM085aSh8MQ5RIQyYHeFBkUH1iijo7GuyUXFCPk4XBu+S9nHTQiAw"
    "eVRpSb54fsBNHsB1eqG4WyS8NQ68Ojz7/8rxbm+yRRdAvMMCn+W6cWy8qQfGmiplUlU74dw5MUEL3ncEZBOHFS60kJNVL7rt"
    "N4UUbiF/q708s8EsexbZhDT+kWaAk7QqOHu3lGbMRQxUeoZ9l7FThbqsQXANsEE3Q0yprqJrW1oORDBdG9qIiIA4wczoiuNy"
    "Iok+pR38WyTcB0Z36WByG1RpWTbLGV5T5YQ+ZTCvJZgxkqujWr/siIFmGfVQLjkLilUyeYSxl6Yx+eU1ws0dSjIrkopDapjO"
    "gPdPSxJuZpEtjYB1eQc5NZIELxww8YeDffDbOkx8fABLIQvs9gXvX1Fj1dnIaUf89ZgBodKFnPOagcTDyibzHrRu2lBzT11j"
    "eCNwUuUgEKiRN3Mn8cMZQHKClKmulZes0oR0M+Sxq0LnGArCgFyozPsqQKBVjuqNSSro3kRMiyU8SEJwLJmlUctCL5kFKG36"
    "JNp+acxgerEeVHHQP06JghiIzjtpFVOEyuiH9TV+vqBx2csjaTvJWBWmbQoY0qBgmPLOjt37bLZHHJs0UUCW8jZeFNq0fFqK"
    "1AGukLtJTNASpMmr28Z+2wxS6335kL6L5PPryqp2Fg4lZmZ+k9NBctcYNvP6Cdlke5DTAqIvavDmQGA4zWTj80l5/CTYyvzo"
    "of0Z5YhE6QF0F2jcgNVUGXXbR6ftY9HAGTGv6VrAo4QJelsYGxVBiGZhf4MrIlicsRoSc6Xo5MgggRfMA3slfpgcrXiilxRx"
    "QzpnNk+VbwRbiV6Of6wY6CrWyK9KlL7S/rBb+/v8Quk2PAOoTJ9a+p89FEAcflcdAIXrOEHLnVwWygxlfVwkxA6Q8q7SOnOp"
    "WHszyzltm8RciROU3caW3s+HiSecKdIdOZyaCKToEemWoMayn3p2cGOvTOHUtZZtPQgM8Fac4WW/0BV/Fcp1lScT9nJoS9nw"
    "ASVLm9gSpAkpxVOxClifEWmkAxTlUOQByguWWaQ1qcGsaEN4uzki0tj1iPehF0jv1Q67oy+I4EvcjJe88C1Snmhx47WnRvcx"
    "gkJteUoC5wAjllbUpbqCNteuw7tr7TFMbuPNNqnL15r6gNxzyEbYJkm5pETJduq7MaRirnzgmDC1wlGt2w5Fctee/SQrWg66"
    "yaim8vJoQQ73dVr6Jk2W2YzSTmVSiO/ULyqbfJTD1qXkx1hYueqdiDQQh6kVffXLEFXULji4CwiIXj8rCvhtg/taJRjZLzo9"
    "XEtCOy5l7F6urZ64z0tzSC/QlFrekqOjZjEP61lhVvxFLp2V9KSeQZ7qjI5pyRjz3Iev+cIAFOEumjVf7yAtqpDyvKqcPdGL"
    "T24SViizrjS7tYBYLifKF1OlzeZK/LAX6OHqia6y6KM+pb572kjatotKN1e/xmb4qmQzu4dQXAMx4l5WqclEt7IFxHc6FKgX"
    "m2Stk+OUQKrhFnzhciKoM9ebRg6GUa+n9N7V5KbKUOVgHQPlKmrN1Ur7+itLC2c7NnraIUB3xZlNdgvSbY16XuVwoWYVAEiC"
    "/1w0omrZvEaJ7qpvyBZP9tr1G7M5hUBi+F5Gj/XwK2NWQgYT81IB+u4lojd8Lk375CVNr2BuEyvpQ5JDQqduJ9sjDMWK/sCN"
    "v4QEUCfGPLzpjKkzwbSJaJDnQMJUvZJocGdaRC57KjLLRZo1L1L6FFZk+lGelvsWubtrXZQVBDjsKPZhlbY9amG3lEpKUOQO"
    "1TCSpwKdmVbNwSbCehbR/x6WaoRQCGfylwiodmLlusjxuwKqO5EbTnEd8cO6cIfDQgrNIv83X19jt3x9Qf98BS9njhk5S50t"
    "XS2g2Vw5wMEdMPyHnJVRN+nRmADNoji1kA9UfwJFTE4CIg1ssegPaTAFVJmFy188KUF2qLHqzJKbFxhSLbm/P0BnjsR5VLE3"
    "prk77x8jXrfuEIiDBuJpow/clK7hj72T8/VeVbhm5HDVA4eoiyz0ctKiqq/UNlmVAuVLgNy5roIrQ5G4w0gxMwVRh0GjNYt+"
    "J3eetz1YNCPVwB/Hi75q15v62BjhfMwMqTlr3LiToGyaUVglR+ugqW+nFqd+U/Bb7pzsrx/cE8WznxETO6HXQhVFgtS3VMXh"
    "CadQqGcFJqwLvEjaEDJMwtqnLLqrX5xZWm7f8hNorb/o2iXHN6kT/U7M/NukEYFH6eHqLVLrZT3lhDOqoIJ0F5j0a9zhldnS"
    "Szcg98HsWM+cWCWq/Dy4+c0ee97qR2CWTTavV+jiQ8KAXHLm2Ku64VB2ReqppAnTecqZ8oY67BwoRkqK/nLmzvL/+ulBDFty"
    "BhqZ+VOOCA/EMIv0mV7e0vi990y+da+e0r2uvmnBXpBHaEIUnTn/7RVpoJjk2AylSRGvE5ilaoEBTtioYnCWoRBilyg1gymP"
    "FRW+6ETwJWKgLQqYcdG85eoZ633MEtqZff2Qo4hQou46DtMW5WTxlxNW7qK+O10kbdeJSqZccl7oV5QyND5oivNYGtMCW7J3"
    "IqLhqYVTDbOcRzU56iTc5YnwNF1CZmlH7AMQx6S+AUleNKZ5ZZoH7EFEBLlIa2jOWcmaBgZTyXDuGTuZ1DFe/bAEyDaw3w6p"
    "jYq9QOPujkE4M0tJis7AWNGQF075NZaGX4h/jciy7Up5vF9Eo5Uld3rBPpwCUGz9KJcb5+L4kLBQErhIy5EcyOrsCld3Znqo"
    "fBVqxC7DB53ayqHSHFkRDuLUQ11SKqCJxDT4jn5snCdHRfenmjkiJlY+c/fCzXgumJbXwOVeEtlMyynYKyazZFAq10P35m6l"
    "TthnlnKeob43oQ7Vs/W2HIN5nnJ6EA5FHQCjbzknaEpetBLbMMVbr+VqXlo5wbZZUSZ/jWvzjqW1Pg2p39FbwjUMRAvOJaz9"
    "kllR0wKWOfVGOs3rzZS0hr4Qb06COsvqd/a8B6T2JQpqif6szsKQvq/x832hEdAZXC5ZeiGdM50/1Cfju8y06nJGZJeriK0O"
    "KmFj58naQJWNNGojRMkFUGfa6WeTA6f5+V5J2KzyQtq3CAU1KTrfTV668vQqh/aIhJz+2oV2Pr2gdZIoZGLaTzmf187Ic0rf"
    "ZC3SXXF1Xvp2qBO+sOogJe8evYZ6Uw6tOLYzK9p9TInmF1OXsNzHW3Cq4bVxTuZcaCXlDX5+20Uif6MSf/1Irb0ge8JLh5M8"
    "6y6a/5EpLdRhe5Ex/nKPHrmQsF7UeSEStVlpB784dDbVE8iaS0lZF0xFnSkUnAO67VhoBVFPaTEsJfrHTq1n6rh9s5OeFkpm"
    "Y7+dAeLknKm2yx3pbR+kFSV2ewYx3MiajfnGEkhd0lsqxNd0idyQKL5cZHXFHDB4fS5uFjyySro85w44v+QgIRixw6KZnk4a"
    "KFY5Y+ryHHG0F1DLuZIw6QrD7+ZHKx6pw16QISwL1igGqsyCHPzpErnp7iipmt7Ev3bxwJqZnKwmMjW7Qhp7ZXAoPfNp2KtD"
    "Y7g46Y2VCFzMso+Q1XSPfnMzFFQbT9r3kiUXFE10vQg6dkE4D/mgOlMq4DRHlXk5YfJTiSi190q7tVTk0ibDPiw7iZje8/g2"
    "IyJlLZV5eqePy1vqQp1G/jRZKfZa4ZpXkU+nZcdRRYCwsCO+0/VDCh4wbUEnqaXlEemL89FXOfd3kursrX7C2n8KZGC38fFz"
    "j9ru9aDl+KZIeZulRsHLiSpTJor3QRooHrscUQB1WBdUiuYuPvk5+P7bUXzSz528pZTsgLdapTvyN0/7Bt+ZPLiGF19o84tz"
    "f3upYQvdu76kmasI11zkiA0fqJAJ8pK7qHrUBWB1ydkTPkArGfvJMaMZ42g2kLKtaxpuZSekpsrC2m9IhNSM9NP28xbb1kvI"
    "BaJIl5dTmuVXisjmwM7KFworJSqvdEgJQqU9lJKuGzry8buwBEGiq7IyfJsFI9UHNStHkZNEzONQBtoqmn5w+O06UDrMckE8"
    "pJ5hpb2vkhOJbZUF4zmywu/AsF+auNpfcurItkk7eCqFVI/a7tbFUaA/+E37KJXN9JcLne1r5ELPklrk+KKT0Xt5lPYSZabE"
    "ZO2GyrxzZmLrJVp3aTR7KDTISdHd8NxQL0zzNd7hlLrb9aS2+jwB1dck7utTludFzVV4oyR5n/DS70wdSrT0tjgVKJV6yynD"
    "F/qSNJzP//1/v1ZJzQ=="
)
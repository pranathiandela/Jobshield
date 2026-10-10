"""Text extraction helpers for uploaded files.

Resume Screening continues to use SUPPORTED_EXTENSIONS / extract_text().
Detection has its own extraction entry point so TXT support can be added
without changing Resume Screening behavior.
"""

from pathlib import Path

from pypdf import PdfReader
from docx import Document


# Existing Resume Screening support — DO NOT CHANGE.
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


# Detection supports these three input formats.
DETECTION_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedFileType(Exception):
    pass


def _extract_pdf_text(path):
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_docx_text(path):
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs).strip()


def _extract_txt_text(path):
    """Extract text from a UTF-8/UTF-16/plain-text file."""
    raw = Path(path).read_bytes()

    for encoding in ("utf-8", "utf-16"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue

    # Final fallback for ordinary text files with unusual encoding.
    return raw.decode("utf-8", errors="replace").strip()


def extract_text(path):
    """Existing Resume Screening extractor.

    Supports PDF and DOCX only. Kept unchanged for compatibility.
    """
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf_text(path)

    if ext == ".docx":
        return _extract_docx_text(path)

    raise UnsupportedFileType(f"Unsupported file type: {ext}")


def extract_detection_text(path):
    """Extract job-description text for the Detection feature.

    Detection supports PDF, DOCX and TXT.
    """
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf_text(path)

    if ext == ".docx":
        return _extract_docx_text(path)

    if ext == ".txt":
        return _extract_txt_text(path)

    raise UnsupportedFileType(
        f"Unsupported Detection file type: {ext}"
    )


# ---------------------------------------------------------------------------
# Layout information for Resume Quality (headings, bold, underline, alignment).
# Additive: extract_text() / extract_detection_text() above are unchanged.
# ---------------------------------------------------------------------------

def _docx_layout(path):
    doc = Document(str(path))
    lines = []
    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        style = p.style
        runs = [r for r in p.runs if (r.text or "").strip()]

        def _flag(attr):
            vals = [getattr(r, attr) for r in runs]
            if any(v for v in vals):
                return True
            style_font = getattr(style, "font", None)
            return bool(getattr(style_font, attr, None)) if style_font is not None else False

        size = None
        for r in runs:
            if r.font.size:
                size = r.font.size.pt
                break
        if size is None and style is not None and style.font is not None and style.font.size:
            size = style.font.size.pt
        align = p.alignment
        if align is None and style is not None and getattr(style, "paragraph_format", None) is not None:
            align = style.paragraph_format.alignment
        align = str(align).split(".")[-1].split(" ")[0].lower() if align is not None else "left"
        lines.append({
            "text": text,
            "bold": _flag("bold"),
            "underline": _flag("underline"),
            "size": size,
            "align": {"center": "center", "right": "right", "justify": "justify"}.get(align, "left"),
            "caps": text.isupper() or any(bool(r.font.all_caps) for r in runs),
            "style": getattr(style, "name", None),
        })
    return {"source": "docx", "lines": lines}


def _pdf_layout(path):
    reader = PdfReader(str(path))
    lines = []
    for page in reader.pages:
        width = float(page.mediabox.width) or 612.0
        frags = []

        def visitor(text, cm, tm, font_dict, font_size):
            if not text or not text.strip():
                return
            name = ""
            try:
                name = str((font_dict or {}).get("/BaseFont", "")).lower()
            except Exception:
                pass
            import math
            scale = math.hypot(tm[0], tm[1]) * math.hypot(cm[0], cm[1])
            eff = abs(font_size or 0) * (scale or 1.0)
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]      # position on the page, in points
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            frags.append((round(y, 0), x, text, eff, name))

        page.extract_text(visitor_text=visitor)
        rows = {}
        for y, x, text, size, name in frags:
            key = next((k for k in rows if abs(k - y) <= 2), y)
            rows.setdefault(key, []).append((x, text, size, name))
        for y in sorted(rows, reverse=True):  # top of page first
            parts = sorted(rows[y], key=lambda f: f[0])
            text = "".join(f[1] for f in parts).strip()
            if not text:
                continue
            size = max(f[2] for f in parts)
            x0 = parts[0][0]
            est_w = len(text) * size * 0.5
            mid = (x0 + est_w / 2) / width
            align = "center" if (abs(mid - 0.5) < 0.06 and x0 > 0.12 * width) else \
                ("right" if x0 > 0.55 * width else "left")
            bold_chars = sum(len(f[1]) for f in parts if any(b in f[3] for b in ("bold", "black", "heavy", "semibold", "demi")))
            lines.append({
                "text": text,
                "bold": bold_chars >= 0.6 * max(1, sum(len(f[1]) for f in parts)),
                "underline": None,           # not recoverable from PDF text
                "size": round(size, 1) if size else None,
                "align": align,
                "caps": text.isupper(),
                "style": None,
            })
    return {"source": "pdf", "lines": lines}


def extract_resume_layout(path):
    """Per-line style information for Resume Quality scoring.

    Returns {"source": "pdf"|"docx", "lines": [{text, bold, underline, size, align,
    caps, style}, ...]} or None if styles cannot be read.  Never raises: the resume
    is still scored from its text when layout is unavailable.
    """
    try:
        ext = Path(path).suffix.lower()
        if ext == ".docx":
            return _docx_layout(path)
        if ext == ".pdf":
            return _pdf_layout(path)
    except Exception:
        return None
    return None
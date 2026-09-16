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
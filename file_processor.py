"""Text extraction helpers for uploaded resume files (PDF / DOCX).

Kept separate from resume_analyzer.py so the analyzer only ever deals with
plain text, regardless of whether it came from a paste box or an uploaded file.
"""
from pathlib import Path

from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


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


def extract_text(path):
    """Extract plain text from a resume file. Raises UnsupportedFileType for .doc
    and anything else not yet supported (python-docx cannot read legacy .doc)."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf_text(path)
    if ext == ".docx":
        return _extract_docx_text(path)
    raise UnsupportedFileType(f"Unsupported file type: {ext}")
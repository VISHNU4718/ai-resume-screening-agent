"""Step 1: turn an uploaded file into clean text."""

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

from docx import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.errors import ResumeParseError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

_NAME_STOPWORDS = {"resume", "curriculum", "vitae", "cv", "profile", "summary", "objective"}
_NAME_LINE = re.compile(r"^[A-Za-z][A-Za-z.'\- ]{2,40}$")


@dataclass(frozen=True)
class ParsedResume:
    filename: str
    name: str
    text: str


def normalize_text(text: str) -> str:
    """Strip control characters and collapse runs of whitespace, keeping line structure."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_candidate_name(text: str, filename: str) -> str:
    """Use the first short alphabetic line as the name, else the filename without extension."""
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()][:5]:
        words = line.split()
        if (
            2 <= len(words) <= 4
            and _NAME_LINE.match(line)
            and not _NAME_STOPWORDS.intersection(w.lower() for w in words)
        ):
            return line
    stem = PurePath(filename).stem
    return re.sub(r"[_\-]+", " ", stem).strip() or filename


def _read_txt(content: bytes) -> str:
    if b"\x00" in content[:4096]:
        raise ResumeParseError("File contains binary data and is not a text file.")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _read_pdf(content: bytes) -> str:
    if not content.lstrip().startswith(b"%PDF"):
        raise ResumeParseError("File is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ResumeParseError("PDF is password protected.")
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ResumeParseError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError) as exc:
        raise ResumeParseError("PDF could not be read; it may be corrupted.") from exc


def _read_docx(content: bytes) -> str:
    if not zipfile.is_zipfile(io.BytesIO(content)):
        raise ResumeParseError("File is not a valid DOCX document.")
    try:
        doc = Document(io.BytesIO(content))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ResumeParseError("DOCX could not be read; it may be corrupted.") from exc
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


_READERS = {".txt": _read_txt, ".pdf": _read_pdf, ".docx": _read_docx}


def parse_resume(filename: str, content: bytes) -> ParsedResume:
    """Extract text from a PDF, DOCX or TXT upload.

    Raises ResumeParseError when the type is unsupported or no text can be extracted.
    """
    ext = PurePath(filename).suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        allowed = ", ".join(SUPPORTED_EXTENSIONS)
        raise ResumeParseError(f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}.")
    if not content:
        raise ResumeParseError("File is empty.")

    text = normalize_text(reader(content))
    if not text:
        raise ResumeParseError("No extractable text found (scanned image PDFs are not supported).")
    logger.info("Parsed %s (%d characters)", filename, len(text))
    return ParsedResume(filename=filename, name=guess_candidate_name(text, filename), text=text)

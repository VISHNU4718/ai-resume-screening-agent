import pytest

from app.core.errors import ResumeParseError
from app.services.parser import guess_candidate_name, normalize_text, parse_resume
from tests.conftest import make_docx, make_pdf

RESUME = "Jane Doe\nPython developer\nSkills: Python, SQL"


def test_parses_txt():
    parsed = parse_resume("jane.txt", RESUME.encode())
    assert "Python developer" in parsed.text
    assert parsed.name == "Jane Doe"


def test_txt_falls_back_to_latin1():
    parsed = parse_resume("cafe.txt", "Jos\u00e9 Garc\u00eda\nPython".encode("latin-1"))
    assert "Jos\u00e9" in parsed.text


def test_parses_docx_paragraphs_and_tables():
    import io

    from docx import Document

    doc = Document()
    doc.add_paragraph("Jane Doe")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skills"
    table.rows[0].cells[1].text = "Docker"
    buffer = io.BytesIO()
    doc.save(buffer)

    parsed = parse_resume("jane.docx", buffer.getvalue())
    assert "Docker" in parsed.text


def test_parses_pdf():
    parsed = parse_resume("jane.pdf", make_pdf(RESUME))
    assert "Python developer" in parsed.text


def test_extension_is_case_insensitive():
    assert parse_resume("JANE.TXT", RESUME.encode()).text


@pytest.mark.parametrize("filename", ["resume.png", "resume", "resume.doc"])
def test_rejects_unsupported_type(filename):
    with pytest.raises(ResumeParseError, match="Unsupported"):
        parse_resume(filename, b"data")


def test_rejects_empty_file():
    with pytest.raises(ResumeParseError, match="empty"):
        parse_resume("a.txt", b"")


def test_rejects_whitespace_only_text():
    with pytest.raises(ResumeParseError, match="No extractable text"):
        parse_resume("a.txt", b"  \n\t \n")


def test_rejects_binary_masquerading_as_txt():
    with pytest.raises(ResumeParseError, match="binary"):
        parse_resume("a.txt", b"\x00\x01\x02binary")


def test_rejects_non_pdf_content():
    with pytest.raises(ResumeParseError, match="valid PDF"):
        parse_resume("a.pdf", b"just text")


def test_rejects_truncated_pdf():
    with pytest.raises(ResumeParseError):
        parse_resume("a.pdf", make_pdf(RESUME)[:60])


def test_rejects_pdf_without_text_layer():
    import io

    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.rect(10, 10, 100, 100)  # drawing only, no text
    pdf.save()
    with pytest.raises(ResumeParseError, match="No extractable text"):
        parse_resume("scan.pdf", buffer.getvalue())


def test_rejects_corrupt_docx():
    with pytest.raises(ResumeParseError, match="DOCX"):
        parse_resume("a.docx", b"not a zip file")


def test_rejects_zip_that_is_not_docx():
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(ResumeParseError, match="DOCX"):
        parse_resume("a.docx", buffer.getvalue())


def test_docx_roundtrip_helper():
    assert "Jane" in parse_resume("a.docx", make_docx(RESUME)).text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("a \t b\r\n\r\n\r\n\r\nc\x00") == "a b\n\nc"


@pytest.mark.parametrize(
    ("text", "filename", "expected"),
    [
        ("Jane Doe\nEngineer", "x.txt", "Jane Doe"),
        ("Resume\nJane Doe", "x.txt", "Jane Doe"),
        ("Curriculum Vitae\n12345\njane@x.com", "jane_doe-2024.txt", "jane doe 2024"),
        ("Python, SQL, Docker, Git, AWS\nmore", "cand.txt", "cand"),
    ],
)
def test_guess_candidate_name(text, filename, expected):
    assert guess_candidate_name(text, filename) == expected

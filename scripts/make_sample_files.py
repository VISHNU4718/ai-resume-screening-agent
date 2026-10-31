"""Regenerate the DOCX and PDF sample resumes in samples/ (requires reportlab)."""

from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

SNEHA = """Sneha Reddy
Hyderabad | sneha.reddy@example.com

Objective
Final-year B.Tech student looking for a generative AI internship.

Education
B.Tech in Computer Science and Information Technology, 2022 - 2026

Experience
Machine Learning Intern, Askew Analytics (Jan 2025 - Jul 2025)
- Fine-tuned a Hugging Face Transformers model for text classification using PyTorch.
- Built a small retrieval-augmented generation (RAG) demo with sentence-transformers and FAISS.

Projects
- Study assistant chatbot using LLMs and prompt engineering, served with Flask.
- Image classifier with TensorFlow and Keras.

Skills
Python, PyTorch, TensorFlow, Hugging Face, Transformers, LLMs, RAG, Flask, Git, NumPy, Pandas, SQL
"""

IMRAN = """Imran Khan
Bengaluru | imran.khan@example.com

Summary
Front-end developer with 4 years of experience building responsive web applications.

Experience
Front-end Developer, Pixelcraft (Mar 2022 - Present)
- Built single-page applications in React and TypeScript with Tailwind CSS and Next.js.
- Integrated REST APIs and GraphQL services; wrote unit tests with Jest.
- Maintained CI/CD pipelines on GitHub and used Git daily.

Web Developer, Studio North (Aug 2021 - Feb 2022)
- Delivered marketing sites in HTML, CSS and JavaScript.

Skills
JavaScript, TypeScript, React, Next.js, HTML, CSS, Tailwind CSS, GraphQL, Git, Jest, CI/CD
"""


def write_docx(path: Path, text: str) -> None:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(path)


def write_pdf(path: Path, text: str) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    y = 800
    for line in text.splitlines():
        pdf.drawString(50, y, line)
        y -= 14
    pdf.save()


if __name__ == "__main__":
    write_docx(SAMPLES / "sneha_reddy.docx", SNEHA)
    write_pdf(SAMPLES / "imran_khan.pdf", IMRAN)

"""Capture a screenshot of a finished screening from a running server.

Usage: python scripts/capture_screenshot.py [URL] [OUTPUT] [QUESTION]

Needs playwright and Chrome. Screens the sample resumes, opens the first candidate's details
and asks a question, so the screenshot shows the analysis mode, the evidence and the answer.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "screenshot.png"
QUESTION = (
    sys.argv[3] if len(sys.argv) > 3 else "Which candidate has built a RAG system with LangChain?"
)
RESUMES = ["ananya_rao.txt", "kiran_patel.txt", "sneha_reddy.docx", "imran_khan.pdf"]
SCREEN_TIMEOUT_MS = 600_000

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(URL)
    page.fill("#jd", (ROOT / "samples" / "job_description.txt").read_text())
    page.set_input_files("#files", [str(ROOT / "samples" / n) for n in RESUMES])
    page.click("#submit")
    page.wait_for_selector("#results:not([hidden])", timeout=SCREEN_TIMEOUT_MS)
    page.locator("#results-body tr button").first.click()
    page.fill("#question", QUESTION)
    page.click("#ask-form button[type=submit]")
    page.wait_for_selector("#ask-result:not([hidden])", timeout=SCREEN_TIMEOUT_MS)
    OUTPUT.parent.mkdir(exist_ok=True)
    page.screenshot(path=str(OUTPUT), full_page=True)
    browser.close()

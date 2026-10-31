"""Capture docs/screenshot.png from a running server (requires playwright and Chrome)."""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
RESUMES = ["ananya_rao.txt", "kiran_patel.txt", "sneha_reddy.docx", "imran_khan.pdf"]

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    page.goto(URL)
    page.fill("#jd", (ROOT / "samples" / "job_description.txt").read_text())
    page.set_input_files("#files", [str(ROOT / "samples" / n) for n in RESUMES])
    page.click("#submit")
    page.wait_for_selector("#results:not([hidden])")
    page.locator("#results-body tr button").first.click()
    page.screenshot(path=str(ROOT / "docs" / "screenshot.png"), full_page=True)
    browser.close()

# AI Resume Screening Agent: Project Report

**Author:** Beernelli Vishnu Vardhan, B.Tech CSIT, MVSR Engineering College
**Programme:** Blackbucks Internships 2026, Generative AI Engineer
**GitHub:** VISHNU4718

## 1. Abstract

This project implements a resume screening service that ranks candidates against a job description. A FastAPI backend accepts PDF, DOCX and TXT resumes, extracts text, identifies skills and experience, measures semantic similarity to the job description, and combines these into a 0-100 score with a Shortlist, Maybe or Reject recommendation. An explanation step writes strengths and concerns for each candidate, using a large language model when an API key is configured and a deterministic rule-based writer otherwise, so the system runs offline. A single-page web interface shows the ranked results. The system was tested with an automated suite of 113 tests and demonstrated on four fictional resumes.

## 2. Introduction

Screening resumes is repetitive. A reviewer compares each resume with the same list of requirements, and the quality of that comparison drops as the pile grows. Software can do the mechanical part of this: finding listed skills, estimating experience, and sorting candidates, so a human spends time on the borderline cases.

Large language models can summarise a resume well, but using one alone for scoring makes results hard to reproduce and hard to audit. This project therefore keeps the numeric score deterministic and uses the language model only to produce explanatory text, with a fallback when no model is available.

## 3. Objectives

1. Accept multiple resumes in PDF, DOCX and TXT format together with a job description.
2. Produce a score, matched and missing skills, strengths, concerns and a recommendation for each candidate, and rank them.
3. Structure the work as separate pipeline stages that can be tested and replaced independently.
4. Work without network access or paid services, while allowing an LLM to be plugged in.
5. Validate input and report unreadable files without failing the whole request.
6. Provide a simple web interface, automated tests, and reproducible setup.

## 4. Methodology

The pipeline has six stages, implemented in `app/services/`.

**Parsing.** Text is extracted with pypdf (PDF), python-docx (DOCX, including tables) or direct decoding (TXT). The file content is checked rather than trusting the extension, and files that are empty, corrupt, encrypted, binary, or image-only are rejected with a specific reason. Per-file size and per-request file count limits are enforced.

**Extraction.** Skills are found by matching a taxonomy of 77 skills and their aliases (stored in `skills.json`) using word-boundary rules that avoid false matches such as "java" inside "javascript". The job description is split by headings such as "Nice to have" into required and preferred skills. Years of experience are taken from explicit statements ("3 years of experience") or, if none, from date ranges, merging overlapping periods and ignoring education lines. The job description's requested minimum years is read with a similar pattern.

**Semantic matching.** The job description and each resume are compared by cosine similarity. The preferred backend is sentence embeddings from `all-MiniLM-L6-v2`; if the library or model is unavailable, TF-IDF with unigrams and bigrams is used. The backend in use is reported in every response.

**Scoring.** The final score is a weighted average of skill coverage (weight 0.55), text similarity (0.30) and experience fit (0.15). Preferred skills count half of required skills. Components that do not apply to a job description are dropped and the remaining weights renormalised. Recommendation thresholds are 70 (Shortlist) and 45 (Maybe). All weights and thresholds are configuration values.

**Explanation.** With `LLM_API_KEY` set, the facts from the previous stages and an excerpt of the resume are sent to an OpenAI-compatible chat completions endpoint, which returns strengths, concerns and a summary as JSON. The output is validated and length-limited. On any error, or without a key, a rule-based writer generates the text from the same facts. The score and recommendation are not influenced by this stage.

**Ranking.** Candidates are sorted by score (ties broken by name) and numbered.

## 5. Modules

| Module | Responsibility |
|--------|----------------|
| `app/main.py` | Application factory, startup, static UI |
| `app/api/routes.py` | `GET /health`, `POST /screen`, request validation |
| `app/core/` | Settings from environment, logging, error types and handlers |
| `app/models/schemas.py` | Typed response models |
| `app/services/parser.py` | File parsing and name guessing |
| `app/services/skills.py` | Skill extraction and job description sections |
| `app/services/experience.py` | Years of experience estimation |
| `app/services/similarity.py` | Embedding and TF-IDF backends |
| `app/services/scoring.py` | Score combination and recommendation |
| `app/services/reasoning.py` | LLM and rule-based explanation writers |
| `app/services/pipeline.py` | Orchestration and ranking |
| `static/` | Single-page interface |
| `tests/` | Unit and API tests |

## 6. Results

**Tests.** The pytest suite has 113 tests covering the parser (all three formats and failure cases such as truncated PDFs, scanned PDFs, corrupt DOCX and binary data), skill and experience extraction (word boundaries, aliases, overlapping date ranges, education lines), scoring (weights, renormalisation, threshold boundaries, configuration validation), similarity backends (including fallback behaviour), the LLM step (request shape, malformed responses, HTTP errors, timeouts, all falling back to rule-based text), and the API (ranking, partial failures, validation errors, file limits). All pass, and `ruff` reports no issues.

**Demonstration.** Four fictional resumes were screened against a junior LLM application engineer job description (`samples/`).

| Rank | Candidate | Profile | Score (TF-IDF) | Recommendation |
|------|-----------|---------|----------------|----------------|
| 1 | Ananya Rao | Backend engineer with LLM, RAG and FastAPI work | 79.6 | Shortlist |
| 2 | Sneha Reddy | Student with a six-month ML internship | 48.4 | Maybe |
| 3 | Kiran Patel | Data analyst, some ML | 38.1 | Reject |
| 4 | Imran Khan | Front-end developer | 29.3 | Reject |

With sentence embeddings the order is unchanged and the scores are 95.1, 59.8, 50.3 and 40.8. The ranking matches what a human would expect for these constructed profiles. This is a demonstration on a very small hand-written set; no accuracy, precision or recall has been measured, and the results should not be read as evidence of performance on real resumes.

## 7. Limitations

- Keyword matching misses skills described in other words.
- Experience estimation relies on common formats and can be wrong.
- No OCR, so scanned PDFs are rejected.
- Weights and normalisation constants were chosen by hand and not tuned on labelled data.
- Embedding models read only the start of long documents.
- A scoring system built on resume text can reproduce biases present in job descriptions and resumes; it should assist, not replace, human review.

## 8. Conclusion

The project delivers a working, tested screening pipeline with a clear separation between deterministic scoring and optional language model explanation. It runs offline by default, reports problems with individual files without failing, and exposes its reasoning (score breakdown, matched and missing skills) so that a reviewer can check the result. The main next step is evaluation against labelled data, which would allow the weights and thresholds to be tuned rather than assumed.

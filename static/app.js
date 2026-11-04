(function () {
  "use strict";

  const form = document.getElementById("screen-form");
  const filesInput = document.getElementById("files");
  const fileList = document.getElementById("file-list");
  const submit = document.getElementById("submit");
  const formError = document.getElementById("form-error");
  const results = document.getElementById("results");
  const body = document.getElementById("results-body");
  let screeningId = null;
  let sampleFiles = null;
  const sampleButton = document.getElementById("try-sample");
  const SAMPLE_JOB = "Junior LLM Application Engineer\n\nAbout the role\nWe are building internal tools that use large language models to summarise and search company documents. You will work with a small team to design, build and ship these services.\n\nRequirements\n- 1+ years of experience building software, internships included\n- Strong Python and experience building REST APIs with FastAPI\n- Hands-on experience with LLMs, prompt engineering and retrieval-augmented generation (RAG)\n- Familiarity with LangChain or the Hugging Face ecosystem\n- Working knowledge of Git and SQL\n- Docker for packaging services\n\nNice to have\n- AWS or another cloud platform\n- Vector databases such as FAISS or Pinecone\n- Flask, scikit-learn, Pandas and NumPy\n- CI/CD pipelines\n";
  const SAMPLE_RESUMES = [{"name":"ananya_rao.txt","text":"Ananya Rao\nHyderabad | ananya.rao@example.com\n\nSummary\nSoftware engineer focused on LLM-backed backend services. Roughly 2 years of experience in Python development.\n\nExperience\nSoftware Engineer, Lumen Labs (Jun 2024 - Present)\n- Built a document question-answering service with FastAPI, LangChain and a FAISS vector database (RAG).\n- Wrote prompt engineering templates and an evaluation script to compare Hugging Face and hosted LLMs.\n- Packaged services with Docker and deployed them on AWS using GitHub Actions CI/CD.\n- Designed REST APIs backed by PostgreSQL and SQL migrations.\n\nPython Developer Intern, Northwind Data (Jan 2024 - May 2024)\n- Cleaned and analysed datasets with Pandas and NumPy; trained baseline models with scikit-learn.\n\nSkills\nPython, FastAPI, LangChain, Hugging Face, RAG, LLMs, Docker, AWS, Git, SQL, PostgreSQL, Pandas, NumPy, scikit-learn\n\nEducation\nB.Tech in Computer Science, 2020 - 2024\n"},{"name":"kiran_patel.txt","text":"Kiran Patel\nPune | kiran.patel@example.com\n\nProfile\nData analyst with 3 years of experience moving into machine learning.\n\nExperience\nData Analyst, Brightline Retail (Jul 2023 - Present)\n- Built dashboards in Power BI and Tableau and wrote SQL against MySQL for weekly reporting.\n- Used Python with Pandas, NumPy and Matplotlib for data analysis and data visualization.\n- Trained churn prediction models with scikit-learn and tracked experiments in Jupyter notebooks.\n\nBusiness Analyst, Corvid Services (Jun 2022 - Jun 2023)\n- Maintained Excel models and wrote SQL queries for finance teams.\n\nProjects\n- Sentiment classifier using NLP techniques and a simple Flask app for demo.\n\nSkills\nPython, SQL, MySQL, Pandas, NumPy, scikit-learn, Flask, Power BI, Tableau, Git, NLP, Machine Learning\n"}];
  sampleButton.addEventListener("click", () => {
    document.getElementById("jd").value = SAMPLE_JOB;
    sampleFiles = SAMPLE_RESUMES.map(s => new File([s.text], s.name, {type:"text/plain"}));
    fileList.replaceChildren(...sampleFiles.map(f => el("li", {text:f.name + " · fictional sample"})));
    form.requestSubmit();
  });

  // Build elements with textContent only, so resume-derived text is never parsed as HTML.
  function el(tag, opts, children) {
    const node = document.createElement(tag);
    opts = opts || {};
    if (opts.className) node.className = opts.className;
    if (opts.text !== undefined) node.textContent = opts.text;
    if (opts.attrs) Object.entries(opts.attrs).forEach(([k, v]) => node.setAttribute(k, v));
    (children || []).forEach((c) => node.appendChild(c));
    return node;
  }

  function formatSize(bytes) {
    return bytes < 1024 * 1024
      ? Math.max(1, Math.round(bytes / 1024)) + " KB"
      : (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = !message;
  }

  function chips(items, kind) {
    if (!items.length) return el("p", { text: "None", className: "meta" });
    return el("div", { className: "chips" }, items.map((s) => el("span", { className: "chip " + kind, text: s })));
  }

  function list(items) {
    return el("ul", {}, items.map((t) => el("li", { text: t })));
  }

  function section(title, content, extraClass) {
    return el("div", { className: extraClass || "" }, [el("h3", { text: title }), content]);
  }

  function pct(value) {
    return value === null || value === undefined ? "n/a" : Math.round(value) + "%";
  }

  const STATUS_LABEL = { met: "Met", partial: "Partial", not_found: "Not found", unverified: "Unverified" };

  function evidenceTable(items) {
    if (!items.length) return el("p", { text: "No requirements to check.", className: "meta" });
    const rows = items.map((e) => {
      const quotes = e.quotes.length
        ? el("div", {}, e.quotes.map((q) => el("q", { text: q })))
        : el("span", { className: "meta", text: "no supporting line" });
      return el("tr", {}, [
        el("td", { text: e.requirement }),
        el("td", { className: "status " + e.status, text: STATUS_LABEL[e.status] || e.status }),
        el("td", {}, [quotes]),
      ]);
    });
    return el("table", { className: "evidence" }, [el("tbody", {}, rows)]);
  }

  function modeTag(mode) {
    return el("span", { className: "tag " + mode, text: mode === "llm" ? "LLM" : "Fallback" });
  }

  function detailsRow(c) {
    const b = c.breakdown;
    const years = c.years_experience === null ? "not determined" : c.years_experience + " years";
    const flagged = c.unsupported_claims.concat(
      c.unsupported_skills.map((s) => s + " (listed by the extractor but not found in the resume text)")
    );
    const grid = el("div", { className: "details-grid" }, [
      section("Summary", el("p", { text: c.summary }), "span-2"),
      section("Matched skills", chips(c.matched_skills, "matched")),
      section("Missing required skills", chips(c.missing_skills, "missing")),
      section("Strengths", list(c.strengths)),
      section("Concerns", list(c.concerns)),
      section("Evidence from the resume", evidenceTable(c.evidence), "span-2"),
      flagged.length ? section("Unsupported claims (excluded from the score)", list(flagged), "span-2") : null,
      section("Suggested interview questions", el("ol", {}, c.interview_questions.map((t) => el("li", { text: t }))), "span-2"),
      section(
        "Score breakdown",
        el("p", {
          text: "Skill coverage " + pct(b.skill_coverage) + " \u00b7 Text similarity " +
            pct(b.semantic_similarity) + " \u00b7 Experience fit " + pct(b.experience_fit) +
            " \u00b7 Experience " + years,
        }),
        "span-2"
      ),
      section("File", el("p", { text: c.filename }), "span-2"),
    ].filter(Boolean));
    const td = el("td", { attrs: { colspan: "6" } }, [grid]);
    return el("tr", { className: "details" }, [td]);
  }

  function candidateRow(c) {
    const bar = el("div", { className: "bar" }, [el("span", { attrs: { style: "width:" + c.score + "%" } })]);
    const toggle = el("button", { className: "link-button", text: "Show details", attrs: { type: "button", "aria-expanded": "false" } });
    let open = null;
    toggle.addEventListener("click", () => {
      if (open) {
        open.remove();
        open = null;
        toggle.textContent = "Show details";
        toggle.setAttribute("aria-expanded", "false");
      } else {
        open = detailsRow(c);
        row.after(open);
        toggle.textContent = "Hide details";
        toggle.setAttribute("aria-expanded", "true");
      }
    });
    const row = el("tr", {}, [
      el("td", { text: String(c.rank) }),
      el("td", { text: c.name }),
      el("td", {}, [el("div", { className: "score" }, [bar, el("span", { className: "value", text: c.score.toFixed(1) })])]),
      el("td", {}, [el("span", { className: "badge " + c.recommendation.toLowerCase(), text: c.recommendation })]),
      el("td", {}, [modeTag(c.analysis_mode)]),
      el("td", {}, [toggle]),
    ]);
    return row;
  }

  function render(data) {
    body.replaceChildren(...data.candidates.map(candidateRow));

    const banner = document.getElementById("mode-banner");
    banner.className = "mode-banner " + data.analysis_mode;
    banner.textContent = data.analysis_mode === "llm"
      ? "Analysis mode: LLM (" + data.llm_model + "). Extraction, evidence and the written report came from the model; the score is computed from that data by fixed rules."
      : "Analysis mode: fallback (rule-based, no LLM). " +
        (data.llm_enabled ? "The LLM was configured but could not be used for some or all steps." : "No LLM is configured.");

    const job = data.job;
    const parts = [];
    parts.push("Must have: " + (job.must_have.join(", ") || "none detected"));
    if (job.nice_to_have.length) parts.push("Nice to have: " + job.nice_to_have.join(", "));
    if (job.min_years !== null) parts.push("Experience asked: " + job.min_years + "+ years");
    document.getElementById("job-profile").textContent = parts.join(" \u00b7 ");

    const notices = document.getElementById("notices");
    const messages = data.warnings.slice();
    data.rejected_files.forEach((r) => messages.push("Skipped " + r.filename + ": " + r.reason));
    notices.replaceChildren(...messages.map((m) => el("li", { text: m })));

    document.getElementById("results-meta").textContent =
      data.candidates.length + " screened \u00b7 similarity: " + data.similarity_backend +
      " \u00b7 " + data.processing_ms + " ms";
    screeningId = data.screening_id;
    document.getElementById("ask-result").hidden = true;
    results.hidden = false;
  }

  filesInput.addEventListener("change", () => {
    sampleFiles = null;
    fileList.replaceChildren(
      ...Array.from(filesInput.files).map((f) =>
        el("li", {}, [el("span", { text: f.name }), el("span", { className: "size", text: formatSize(f.size) })])
      )
    );
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");
    const jd = document.getElementById("jd").value.trim();
    if (!jd) return showError("Enter a job description.");
    const selectedFiles = sampleFiles || Array.from(filesInput.files);
    if (!selectedFiles.length) return showError("Choose at least one resume.");

    const data = new FormData();
    data.append("job_description", jd);
    selectedFiles.forEach((f) => data.append("resumes", f));

    submit.disabled = true;
    sampleButton.disabled = true;
    submit.textContent = "Screening\u2026";
    try {
      const response = await fetch("/screen", { method: "POST", body: data });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        let message = typeof payload.detail === "string" ? payload.detail : "Request failed (" + response.status + ").";
        if (payload.rejected_files) {
          message += " " + payload.rejected_files.map((r) => r.filename + ": " + r.reason).join(" ");
        }
        throw new Error(message);
      }
      render(payload);
    } catch (err) {
      results.hidden = true;
      showError(err.message || "Could not reach the server.");
    } finally {
      submit.disabled = false;
      sampleButton.disabled = false;
      submit.textContent = "Screen resumes";
    }
  });

  const askForm = document.getElementById("ask-form");
  const askError = document.getElementById("ask-error");
  const askSubmit = document.getElementById("ask-submit");

  askForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    askError.hidden = true;
    const question = document.getElementById("question").value.trim();
    if (!question || !screeningId) return;
    askSubmit.disabled = true;
    askSubmit.textContent = "Asking\u2026";
    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ screening_id: screeningId, question: question }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : "Request failed (" + response.status + ").");
      }
      renderAnswer(payload);
    } catch (err) {
      document.getElementById("ask-result").hidden = true;
      askError.textContent = err.message || "Could not reach the server.";
      askError.hidden = false;
    } finally {
      askSubmit.disabled = false;
      askSubmit.textContent = "Ask";
    }
  });

  function renderAnswer(data) {
    const note = data.analysis_mode === "llm"
      ? (data.grounded ? "LLM answer with citations; check that the excerpts support it" : "LLM answer, but it cited no excerpt: treat it as unverified")
      : "Fallback: no LLM answer, showing the most relevant resume excerpts";
    document.getElementById("ask-mode").textContent = note;
    document.getElementById("ask-answer").textContent = data.answer;
    document.getElementById("ask-citations").replaceChildren(
      ...data.citations.map((c) =>
        el("li", { attrs: { value: String(c.ref) } }, [
          el("strong", { text: c.candidate }),
          el("span", { text: " (" + c.filename + "): " + c.excerpt }),
        ])
      )
    );
    document.getElementById("ask-result").hidden = false;
  }

  fetch("/health")
    .then((r) => r.json())
    .then((h) => {
      document.getElementById("status").textContent =
        "Service ok \u00b7 v" + h.version + " \u00b7 " + (h.llm_enabled ? "LLM: " + h.llm_model : "no LLM configured (fallback mode)");
    })
    .catch(() => {
      document.getElementById("status").textContent = "Service unreachable";
    });
})();

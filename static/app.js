(function () {
  "use strict";

  const form = document.getElementById("screen-form");
  const filesInput = document.getElementById("files");
  const fileList = document.getElementById("file-list");
  const submit = document.getElementById("submit");
  const formError = document.getElementById("form-error");
  const results = document.getElementById("results");
  const body = document.getElementById("results-body");

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

  function detailsRow(c) {
    const b = c.breakdown;
    const years = c.years_experience === null ? "not determined" : c.years_experience + " years (estimated)";
    const grid = el("div", { className: "details-grid" }, [
      section("Summary", el("p", { text: c.summary }), "span-2"),
      section("Matched skills", chips(c.matched_skills, "matched")),
      section("Missing required skills", chips(c.missing_skills, "missing")),
      section("Strengths", list(c.strengths)),
      section("Concerns", list(c.concerns)),
      section(
        "Score breakdown",
        el("p", {
          text: "Skill coverage " + pct(b.skill_coverage) + " \u00b7 Text similarity " +
            pct(b.semantic_similarity) + " \u00b7 Experience fit " + pct(b.experience_fit) +
            " \u00b7 Experience " + years,
        }),
        "span-2"
      ),
      section("File", el("p", { text: c.filename + " \u00b7 explanation: " + (c.reasoner === "llm" ? "LLM" : "rule-based") }), "span-2"),
    ]);
    const td = el("td", { attrs: { colspan: "5" } }, [grid]);
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
      el("td", {}, [toggle]),
    ]);
    return row;
  }

  function render(data) {
    body.replaceChildren(...data.candidates.map(candidateRow));

    const job = data.job;
    const parts = [];
    parts.push("Required skills detected: " + (job.required_skills.join(", ") || "none"));
    if (job.preferred_skills.length) parts.push("Preferred: " + job.preferred_skills.join(", "));
    if (job.required_years !== null) parts.push("Experience asked: " + job.required_years + "+ years");
    document.getElementById("job-profile").textContent = parts.join(" \u00b7 ");

    const notices = document.getElementById("notices");
    const messages = data.warnings.slice();
    data.rejected_files.forEach((r) => messages.push("Skipped " + r.filename + ": " + r.reason));
    notices.replaceChildren(...messages.map((m) => el("li", { text: m })));

    document.getElementById("results-meta").textContent =
      data.candidates.length + " screened \u00b7 similarity: " + data.similarity_backend +
      " \u00b7 explanations: " + (data.llm_enabled ? "LLM with rule-based fallback" : "rule-based") +
      " \u00b7 " + data.processing_ms + " ms";
    results.hidden = false;
  }

  filesInput.addEventListener("change", () => {
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
    if (!filesInput.files.length) return showError("Choose at least one resume.");

    const data = new FormData();
    data.append("job_description", jd);
    Array.from(filesInput.files).forEach((f) => data.append("resumes", f));

    submit.disabled = true;
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
      submit.textContent = "Screen resumes";
    }
  });

  fetch("/health")
    .then((r) => r.json())
    .then((h) => {
      document.getElementById("status").textContent = "Service ok \u00b7 v" + h.version;
    })
    .catch(() => {
      document.getElementById("status").textContent = "Service unreachable";
    });
})();

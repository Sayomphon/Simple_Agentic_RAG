/**
 * UI controller: one state object, one render pass per change.
 *
 * All pipeline data comes from RagApi.runWorkflow — this file never knows
 * whether it is talking to fixtures or the Python backend.
 */
(function (global, document) {
  "use strict";

  var api = global.RagApi;
  var mock = global.RAG_MOCK;

  var SAMPLE_QUERIES = [
    "What is the policy on international travel?",
    "Can I work remotely?",
    "What is the refund policy?"
  ];

  var BADGE_LABELS = {
    waiting: "Waiting",
    running: "Running",
    done: "Completed",
    empty: "No evidence found",
    error: "Failed"
  };

  var BADGE_OVERRIDES = {
    query: { done: "Received" },
    answer: { empty: "Not found" }
  };

  var el = {
    form: document.getElementById("query-form"),
    input: document.getElementById("query-input"),
    runButton: document.getElementById("run-button"),
    chips: document.getElementById("sample-chips"),
    sourceToggle: document.getElementById("source-toggle"),
    sourceLabel: document.getElementById("source-label"),
    liveRegion: document.getElementById("live-region"),
    emptyState: document.getElementById("empty-state"),
    errorState: document.getElementById("error-state"),
    errorMessage: document.getElementById("error-message"),
    retryButton: document.getElementById("retry-button"),
    pipeline: document.getElementById("pipeline"),
    stageQuery: document.getElementById("stage-query"),
    toolCall: document.querySelector("#stage-tool-call code"),
    retrieverMeta: document.getElementById("retriever-meta"),
    evidenceList: document.getElementById("evidence-list"),
    generatorMeta: document.getElementById("generator-meta"),
    answerBody: document.getElementById("answer-body"),
    answerMeta: document.getElementById("answer-meta"),
    copyAnswer: document.getElementById("copy-answer")
  };

  var state = {
    status: "idle", // idle | running | done | error
    query: "",
    result: null,
    error: null,
    firstRun: true
  };

  /* ── Helpers ──────────────────────────────────────────────────────────── */

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function announce(message) {
    el.liveRegion.textContent = message;
  }

  function sectionTitle(chunk) {
    return mock && mock.sectionTitle
      ? mock.sectionTitle(chunk)
      : (/^---\s*(.+?)\s*---/.exec(chunk) || [null, "Untitled section"])[1];
  }

  function setBadge(stage, status) {
    var badge = document.querySelector('[data-badge="' + stage + '"]');
    if (!badge) return;
    var override = BADGE_OVERRIDES[stage] || {};
    badge.textContent = override[status] || BADGE_LABELS[status];
    badge.setAttribute("data-state", status);
  }

  function setStage(stage, status) {
    var step = document.querySelector('.step[data-step="' + stage + '"]');
    if (step) step.setAttribute("data-state", status);
    setBadge(stage, status);

    // Placeholders keep the layout stable while an agent is still working.
    if (stage === "evidence" && status === "running") {
      el.evidenceList.innerHTML = skeleton(3);
    }
    if (stage === "answer" && status === "running") {
      el.answerBody.innerHTML = skeleton(3);
      el.answerBody.classList.remove("is-notfound");
    }
  }

  /** The answer cannot be in flight before the generator that produces it. */
  function handleStage(stage, status) {
    setStage(stage, status);
    if (stage === "generator" && status === "running") setStage("answer", "running");
  }

  function resetStages(status) {
    ["query", "retriever", "evidence", "generator", "answer"].forEach(function (stage) {
      setStage(stage, status);
    });
  }

  function skeleton(lines) {
    var html = "";
    for (var i = 0; i < lines; i += 1) {
      html += '<div class="skeleton' + (i === lines - 1 ? " short" : "") + '"></div>';
    }
    return html;
  }

  function metaHtml(items) {
    return items
      .filter(Boolean)
      .map(function (item) {
        return (
          "<div><dt>" +
          escapeHtml(item[0]) +
          "</dt><dd>" +
          escapeHtml(item[1]) +
          "</dd></div>"
        );
      })
      .join("");
  }

  /* ── Answer rendering ─────────────────────────────────────────────────── */

  /** Minimal, escape-first formatting: paragraphs, bullets, and citations. */
  function formatAnswer(text) {
    return text
      .trim()
      .split(/\n\s*\n/)
      .map(function (block) {
        var lines = block.split("\n").filter(function (line) {
          return line.trim() !== "";
        });
        var isList =
          lines.length > 0 &&
          lines.every(function (line) {
            return /^\s*([-*•]|\d+[.)])\s+/.test(line);
          });

        if (isList) {
          return (
            "<ul>" +
            lines
              .map(function (line) {
                return "<li>" + inline(line.replace(/^\s*([-*•]|\d+[.)])\s+/, "")) + "</li>";
              })
              .join("") +
            "</ul>"
          );
        }
        return "<p>" + inline(lines.join(" ")) + "</p>";
      })
      .join("");
  }

  function inline(text) {
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[([^\[\]\n]{2,60})\]/g, '<span class="cite">$1</span>');
  }

  /* ── Stage renderers ──────────────────────────────────────────────────── */

  function renderEvidence(snippets) {
    if (!snippets.length) {
      el.evidenceList.innerHTML =
        '<div class="evidence-none">' +
        "<span>The retrieval tool returned no sections above the relevance threshold.</span>" +
        "</div>";
      return;
    }

    el.evidenceList.innerHTML = snippets
      .map(function (chunk, index) {
        return (
          '<figure class="snippet">' +
          '<figcaption class="snippet-head">' +
          '<span class="snippet-num">' +
          (index + 1) +
          "</span>" +
          '<span class="snippet-title">' +
          escapeHtml(sectionTitle(chunk)) +
          "</span>" +
          '<span class="snippet-tag">raw</span>' +
          '<button type="button" class="btn-icon" data-copy="' +
          index +
          '">Copy</button>' +
          "</figcaption>" +
          "<pre>" +
          escapeHtml(chunk) +
          "</pre>" +
          "</figure>"
        );
      })
      .join("");
  }

  function renderToolCall(query) {
    el.toolCall.innerHTML =
      '<span class="fn">search_knowledge_base</span>({ query: "' +
      escapeHtml(query) +
      '" })';
  }

  function renderResult(result) {
    var count = result.snippets.length;

    renderToolCall(result.query);
    el.retrieverMeta.innerHTML = metaHtml([
      ["Tool calls", "1"],
      ["Snippets returned", String(count)],
      ["Query forwarded", result.query === state.query ? "Unchanged" : "Modified"]
    ]);

    renderEvidence(result.snippets);

    el.generatorMeta.innerHTML = metaHtml([
      ["Input", count + (count === 1 ? " snippet" : " snippets")],
      [
        "Mode",
        count ? "Grounded synthesis" : "Deterministic fallback — no LLM call"
      ],
      count && result.source === "live" ? ["Model", api.CONFIG.model] : null
    ]);

    el.answerBody.innerHTML = formatAnswer(result.report);
    el.answerBody.classList.toggle("is-notfound", result.notFound);
    el.answerMeta.innerHTML = metaHtml([
      ["Grounded in", count + (count === 1 ? " section" : " sections")],
      ["Elapsed", (result.durationMs / 1000).toFixed(1) + "s"],
      ["Source", result.source === "live" ? "Live backend" : "Mock data"]
    ]);
    el.copyAnswer.hidden = false;
  }

  /* ── Run ──────────────────────────────────────────────────────────────── */

  function setBusy(isBusy) {
    el.runButton.disabled = isBusy;
    el.runButton.setAttribute("data-busy", String(isBusy));
    el.runButton.querySelector(".btn-label").textContent = isBusy ? "Running" : "Run";
    Array.prototype.forEach.call(el.chips.querySelectorAll(".chip"), function (chip) {
      chip.disabled = isBusy;
    });
  }

  function run(query) {
    if (state.status === "running") return;

    state.status = "running";
    state.query = query;
    state.result = null;
    state.error = null;

    setBusy(true);
    el.emptyState.hidden = true;
    el.errorState.hidden = true;
    el.pipeline.hidden = false;
    el.copyAnswer.hidden = true;

    el.stageQuery.textContent = query;
    renderToolCall(query);
    el.retrieverMeta.innerHTML = "";
    el.generatorMeta.innerHTML = "";
    el.answerMeta.innerHTML = "";

    resetStages("waiting");
    setStage("query", "done");
    setStage("evidence", "running");
    announce("Running the workflow.");

    if (state.firstRun) {
      state.firstRun = false;
      el.pipeline.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    api
      .runWorkflow(query, { onStage: handleStage })
      .then(function (result) {
        state.status = "done";
        state.result = result;
        renderResult(result);
        announce(
          result.notFound
            ? "Workflow completed. No evidence found in the knowledge base."
            : "Workflow completed with " + result.snippets.length + " snippets retrieved."
        );
      })
      .catch(function (error) {
        state.status = "error";
        state.error = error;
        showError(error);
      })
      .then(function () {
        setBusy(false);
      });
  }

  function showError(error) {
    var message =
      (error && error.message) || "The workflow could not be completed. Please try again.";

    el.errorMessage.textContent = message;
    el.errorState.hidden = false;

    // Keep the pipeline visible so the failing stage stays identifiable.
    ["retriever", "evidence", "generator", "answer"].forEach(function (stage) {
      var step = document.querySelector('.step[data-step="' + stage + '"]');
      var current = step && step.getAttribute("data-state");
      if (current === "waiting" || current === "running") setStage(stage, "error");
    });

    el.evidenceList.innerHTML = "";
    el.answerBody.innerHTML = "";
    el.answerBody.classList.remove("is-notfound");
    announce("Workflow failed. " + message);
  }

  /* ── Wiring ───────────────────────────────────────────────────────────── */

  function renderChips() {
    el.chips.innerHTML = SAMPLE_QUERIES.map(function (query) {
      return '<button type="button" class="chip">' + escapeHtml(query) + "</button>";
    }).join("");
  }

  function renderSourceToggle() {
    var mode = api.getMode();
    el.sourceToggle.setAttribute("data-mode", mode);
    el.sourceLabel.textContent = mode === "live" ? "Live backend" : "Mock data";
    el.sourceToggle.title =
      mode === "live"
        ? "Querying " + api.CONFIG.endpoint + " — click to use bundled mock data"
        : "Using bundled mock data — click to query " + api.CONFIG.endpoint;
  }

  function copyToClipboard(text, button) {
    var done = function () {
      var original = button.textContent;
      button.textContent = "Copied";
      global.setTimeout(function () {
        button.textContent = original;
      }, 1400);
    };

    if (global.navigator.clipboard && global.navigator.clipboard.writeText) {
      global.navigator.clipboard.writeText(text).then(done, function () {
        /* Clipboard blocked (e.g. insecure origin) — leave the label untouched. */
      });
    }
  }

  el.form.addEventListener("submit", function (event) {
    event.preventDefault();
    var query = el.input.value.trim();
    if (!query) {
      el.input.focus();
      announce("Enter a question before running the workflow.");
      return;
    }
    run(query);
  });

  el.chips.addEventListener("click", function (event) {
    var chip = event.target.closest(".chip");
    if (!chip || chip.disabled) return;
    el.input.value = chip.textContent;
    run(chip.textContent);
  });

  el.retryButton.addEventListener("click", function () {
    if (state.query) run(state.query);
  });

  el.evidenceList.addEventListener("click", function (event) {
    var button = event.target.closest("[data-copy]");
    if (!button || !state.result) return;
    copyToClipboard(state.result.snippets[Number(button.dataset.copy)], button);
  });

  el.copyAnswer.addEventListener("click", function () {
    if (state.result) copyToClipboard(state.result.report, el.copyAnswer);
  });

  el.sourceToggle.addEventListener("click", function () {
    api.setMode(api.getMode() === "live" ? "mock" : "live");
    renderSourceToggle();
    announce("Data source set to " + el.sourceLabel.textContent + ".");
  });

  document.addEventListener("keydown", function (event) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      if (typeof el.form.requestSubmit === "function") {
        el.form.requestSubmit();
      } else {
        el.form.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    }
  });

  renderChips();
  renderSourceToggle();
  resetStages("waiting");
})(window, document);

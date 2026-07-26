/**
 * UI controller: one state object, one render pass per change.
 *
 * All pipeline data comes from RagApi.runWorkflow — this file never knows
 * whether it is talking to fixtures or the Python backend. Everything in here
 * is presentation: the business logic (retrieval, reporting, error contract)
 * lives behind api.js and is not redefined here.
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

  var STAGE_ORDER = ["query", "retriever", "evidence", "generator", "answer"];

  /** Short step names for the run headline ("Step 2 of 5 · Data retriever"). */
  var STAGE_NAMES = {
    query: "User query",
    retriever: "Data retriever",
    evidence: "Evidence handoff",
    generator: "Report generator",
    answer: "Final answer"
  };

  var BADGE_LABELS = {
    waiting: "Waiting",
    running: "Running",
    done: "Completed",
    received: "Received",
    empty: "No evidence found",
    error: "Failed",
    notrun: "Not run"
  };

  var BADGE_OVERRIDES = {
    answer: { empty: "Not found" }
  };

  /**
   * Status is never conveyed by color alone: every badge pairs an icon with
   * its label (check=completed, spinner=running, dot=pending, triangle=warning,
   * x-circle=failed, arrow-in=received, dash-circle=not run). 12x12 viewBox.
   * The spinner carries a second, static shape that CSS swaps in under
   * prefers-reduced-motion.
   */
  var BADGE_ICONS = {
    waiting:
      '<circle cx="6" cy="6" r="2.5" fill="none" stroke="currentColor" stroke-width="1.5"/>',
    running:
      '<g class="spin"><path d="M6 1.5A4.5 4.5 0 1 1 1.5 6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></g>' +
      '<circle class="spin-static" cx="6" cy="6" r="2.5" fill="currentColor" stroke="none"/>',
    done:
      '<path d="M2.5 6.5l2.4 2.4L9.5 3.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>',
    received:
      '<path d="M6 1.8v5.4M3.6 5 6 7.4 8.4 5M2.4 10h7.2" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>',
    empty:
      '<path d="M6 1.9 10.6 9.9H1.4z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/>' +
      '<path d="M6 4.9v2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>' +
      '<circle cx="6" cy="8.5" r="0.6" fill="currentColor" stroke="none"/>',
    error:
      '<circle cx="6" cy="6" r="4.4" fill="none" stroke="currentColor" stroke-width="1.3"/>' +
      '<path d="M4.5 4.5l3 3M7.5 4.5l-3 3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>',
    notrun:
      '<circle cx="6" cy="6" r="4.4" fill="none" stroke="currentColor" stroke-width="1.3"/>' +
      '<path d="M4 6h4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
  };

  var FLASH_MS = 1600; // citation-jump highlight lifetime (a state, not motion)
  var COPY_FEEDBACK_MS = 1400;
  var BACKOFF_MAX_S = 16;

  var el = {
    form: document.getElementById("query-form"),
    input: document.getElementById("query-input"),
    runButton: document.getElementById("run-button"),
    runTimer: document.getElementById("run-timer"),
    chips: document.getElementById("sample-chips"),
    sourceToggle: document.getElementById("source-toggle"),
    sourceLabel: document.getElementById("source-label"),
    liveRegion: document.getElementById("live-region"),
    queryCard: document.getElementById("query-card"),
    emptyState: document.getElementById("empty-state"),
    errorState: document.getElementById("error-state"),
    errorCode: document.getElementById("error-code"),
    errorMessage: document.getElementById("error-message"),
    errorDetails: document.getElementById("error-details"),
    errorRequestId: document.getElementById("error-request-id"),
    errorRaw: document.getElementById("error-raw"),
    retryButton: document.getElementById("retry-button"),
    retryStatus: document.getElementById("retry-status"),
    pipeline: document.getElementById("pipeline"),
    stageQuery: document.getElementById("stage-query"),
    toolCall: document.querySelector("#stage-tool-call code"),
    toolCallScroll: document.querySelector("#stage-tool-call"),
    retrieverMeta: document.getElementById("retriever-meta"),
    evidenceClamp: document.getElementById("evidence-clamp"),
    evidenceExpand: document.getElementById("evidence-expand"),
    evidenceList: document.getElementById("evidence-list"),
    generatorMeta: document.getElementById("generator-meta"),
    answerBody: document.getElementById("answer-body"),
    answerMeta: document.getElementById("answer-meta"),
    copyAnswer: document.getElementById("copy-answer"),
    compactBar: document.getElementById("compact-bar"),
    compactStatus: document.getElementById("compact-status"),
    compactQuery: document.getElementById("compact-query"),
    compactTimer: document.getElementById("compact-timer"),
    compactAction: document.getElementById("compact-action"),
    runStatus: document.getElementById("run-status"),
    runStatusBadge: document.getElementById("run-status-badge"),
    runStatusDetail: document.getElementById("run-status-detail")
  };

  /** Per-stage element cache — resolved once so render passes never re-query. */
  var stageEls = {};
  STAGE_ORDER.forEach(function (stage) {
    var step = document.querySelector('.step[data-step="' + stage + '"]');
    stageEls[stage] = {
      step: step,
      badge: step.querySelector("[data-badge]"),
      toggle: step.querySelector(".step-toggle"),
      card: step.querySelector(".step-card")
    };
  });

  var state = {
    status: "idle", // idle | running | done | error
    query: "",
    result: null,
    error: null,
    firstRun: true,
    runId: 0,
    traceId: "",
    controller: null,
    startedAt: 0,
    timerHandle: null,
    retryCount: 0,
    backoffHandle: null
  };

  var motionQuery =
    typeof global.matchMedia === "function"
      ? global.matchMedia("(prefers-reduced-motion: reduce)")
      : null;

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

  function scrollBehavior() {
    return motionQuery && motionQuery.matches ? "auto" : "smooth";
  }

  /** Single source for the collapse threshold: the token, not a JS constant. */
  function collapseMax() {
    var raw = global
      .getComputedStyle(document.documentElement)
      .getPropertyValue("--size-collapse-max");
    var parsed = parseFloat(raw);
    return isNaN(parsed) ? 240 : parsed;
  }

  /* ── Elapsed-time ticker (run button + compact bar) ─────────────────────── */

  function startTimer() {
    state.startedAt = Date.now();
    tickTimer();
    state.timerHandle = global.setInterval(tickTimer, 100);
  }

  function stopTimer() {
    if (state.timerHandle) {
      global.clearInterval(state.timerHandle);
      state.timerHandle = null;
    }
  }

  function tickTimer() {
    var elapsed = ((Date.now() - state.startedAt) / 1000).toFixed(1) + "s";
    el.runTimer.textContent = elapsed;
    el.compactTimer.textContent = elapsed;
  }

  /* ── A · StatusBadge ──────────────────────────────────────────────────── */

  /**
   * "Received" is a presentation alias: the query stage's terminal state maps
   * to the info token so it can never read as the same green as "Completed".
   */
  function visualState(stage, status) {
    if (stage === "query" && status === "done") return "received";
    return status;
  }

  function renderBadge(node, vstate, label, count) {
    // Skip identical re-renders: headline updates fire on every stage event.
    var key = vstate + "|" + label + "|" + (count == null ? "" : count);
    if (node.dataset.badgeKey === key) return;
    node.dataset.badgeKey = key;

    node.setAttribute("data-state", vstate);
    node.innerHTML =
      '<svg class="badge-icon" viewBox="0 0 12 12" aria-hidden="true">' +
      (BADGE_ICONS[vstate] || BADGE_ICONS.waiting) +
      "</svg>" +
      '<span class="badge-label">' +
      escapeHtml(label) +
      "</span>" +
      (count == null
        ? ""
        : '<span class="badge-count">' + escapeHtml(String(count)) + "</span>");
  }

  function setBadge(stage, status, count) {
    var vstate = visualState(stage, status);
    var override = BADGE_OVERRIDES[stage] || {};
    renderBadge(stageEls[stage].badge, vstate, override[status] || BADGE_LABELS[vstate], count);
  }

  /** Collapse state tracks the step lifecycle: idle steps fold to their
      header so all five statuses fit one viewport; a step unfolds the moment
      work reaches it. Manual toggles persist until the next state change. */
  function setCardCollapsed(stage, collapsed) {
    var els = stageEls[stage];
    els.toggle.setAttribute("aria-expanded", String(!collapsed));
    els.card.classList.toggle("is-collapsed", collapsed);
  }

  function setStage(stage, status, count) {
    var els = stageEls[stage];
    var next = visualState(stage, status);
    if (els.step.getAttribute("data-state") !== next) {
      els.step.setAttribute("data-state", next);
      setCardCollapsed(stage, next === "waiting" || next === "notrun");
    }
    setBadge(stage, status, count);
    updateHeadline();

    // Placeholders keep the layout stable while an agent is still working;
    // the .clamp / .answer-body height reservations live in CSS.
    if (stage === "evidence" && status === "running") {
      el.evidenceList.innerHTML = skeleton(4);
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
    STAGE_ORDER.forEach(function (stage) {
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

  /** items: [term, value, emphasis?] — emphasis "high" enlarges the value. */
  function metaHtml(items) {
    return items
      .filter(Boolean)
      .map(function (item) {
        return (
          "<div" +
          (item[2] ? ' data-emphasis="' + escapeHtml(item[2]) + '"' : "") +
          "><dt>" +
          escapeHtml(item[0]) +
          "</dt><dd>" +
          escapeHtml(item[1]) +
          "</dd></div>"
        );
      })
      .join("");
  }

  /* ── Compact bar (H) ──────────────────────────────────────────────────── */

  function overallState() {
    if (state.status === "running") return "running";
    if (state.status === "error") return "error";
    if (state.status === "done") {
      return state.result && state.result.notFound ? "empty" : "done";
    }
    return "waiting";
  }

  /**
   * The one-line answer to "where is this run?": progress fraction plus the
   * step currently in flight (or the outcome once the run has settled).
   */
  function headlineDetail() {
    if (state.status === "done") {
      var count = state.result ? state.result.snippets.length : 0;
      return state.result && state.result.notFound
        ? "5/5 · no evidence found"
        : "5/5 · grounded in " + count + (count === 1 ? " section" : " sections");
    }
    for (var i = 0; i < STAGE_ORDER.length; i += 1) {
      var stage = STAGE_ORDER[i];
      var vstate = stageEls[stage].step.getAttribute("data-state");
      if (vstate === "running" || vstate === "error") {
        var position = i + 1 + "/" + STAGE_ORDER.length;
        return (
          (vstate === "error" ? "failed at " : "") + position + " · " + STAGE_NAMES[stage]
        );
      }
    }
    return "";
  }

  /**
   * Single writer for both run headlines: the row inside the query card and
   * the sticky compact bar. Idempotent and cheap (renderBadge dedupes), so
   * every stage transition can call it unconditionally.
   */
  function updateHeadline() {
    var vstate = overallState();
    var detail = headlineDetail();
    renderBadge(el.compactStatus, vstate, BADGE_LABELS[vstate]);
    el.compactQuery.textContent = detail ? state.query + " — " + detail : state.query;
    el.runStatus.hidden = state.status === "idle";
    renderBadge(el.runStatusBadge, vstate, BADGE_LABELS[vstate]);
    el.runStatusDetail.textContent = detail;
  }

  function sectionTitle(chunk) {
    return mock && mock.sectionTitle
      ? mock.sectionTitle(chunk)
      : (/^---\s*(.+?)\s*---/.exec(chunk) || [null, "Untitled section"])[1];
  }

  /* ── Answer rendering (E) ─────────────────────────────────────────────── */

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

  /** Citations render as buttons that jump to their evidence snippet. */
  function inline(text) {
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(
        /\[([^\[\]\n]{2,60})\]/g,
        '<button type="button" class="cite" data-cite="$1">$1</button>'
      );
  }

  /* ── D/F · Evidence rendering ─────────────────────────────────────────── */

  function formatScore(value) {
    return (Math.round(value * 10) / 10).toFixed(1);
  }

  function renderEvidenceEmpty(retrieval) {
    var reason;
    if (retrieval && retrieval.bestScore > 0 && retrieval.bestScore < retrieval.cutoff) {
      reason =
        "Best match scored " +
        formatScore(retrieval.bestScore) +
        " — below the " +
        formatScore(retrieval.cutoff) +
        " relevance threshold.";
    } else if (retrieval && retrieval.bestScore > 0) {
      // Score cleared the numeric cutoff but failed the structural gate
      // (a single body-term hit with no title match does not count as evidence).
      reason =
        "Best match scored " +
        formatScore(retrieval.bestScore) +
        " from a single body term — too weak to qualify (needs a title match or two matching terms).";
    } else if (retrieval) {
      reason = "No section matched any term in the query.";
    } else {
      reason = "The retrieval tool returned no sections above the relevance threshold.";
    }

    var topics = mock && mock.topics ? mock.topics : [];

    el.evidenceList.innerHTML =
      '<div class="evidence-none">' +
      '<svg class="evidence-none-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">' +
      '<path d="M12 3.8 21.4 20H2.6z" stroke-linejoin="round"/>' +
      '<path d="M12 9.6v4.2M12 17h.01" stroke-linecap="round"/>' +
      "</svg>" +
      '<div class="evidence-none-main">' +
      '<p class="evidence-none-title">No evidence found above the threshold.</p>' +
      '<p class="evidence-none-reason">' +
      escapeHtml(reason) +
      "</p>" +
      '<div class="evidence-none-actions">' +
      '<button type="button" class="btn-secondary" data-action="rewrite">Rewrite the query</button>' +
      (topics.length
        ? '<button type="button" class="btn-secondary" data-action="topics" aria-expanded="false" aria-controls="kb-topics">Show covered topics</button>'
        : "") +
      "</div>" +
      (topics.length
        ? '<ul class="kb-topics" id="kb-topics" hidden>' +
          topics
            .map(function (topic) {
              return "<li>" + escapeHtml(topic) + "</li>";
            })
            .join("") +
          "</ul>"
        : "") +
      "</div>" +
      "</div>";
  }

  function markMonoOverflow(root) {
    Array.prototype.forEach.call(root.querySelectorAll(".mono-scroll"), function (wrap) {
      var scroller = wrap.querySelector("pre");
      if (!scroller) return;
      wrap.classList.toggle("has-overflow", scroller.scrollWidth > scroller.clientWidth + 1);
    });
  }

  function setClamped(isClamped, hiddenCount) {
    el.evidenceClamp.classList.toggle("is-clamped", isClamped);
    el.evidenceExpand.setAttribute("aria-expanded", String(!isClamped));
    if (isClamped) {
      el.evidenceExpand.dataset.hidden = String(hiddenCount);
      el.evidenceExpand.textContent =
        "Show " + hiddenCount + " more " + (hiddenCount === 1 ? "section" : "sections");
    } else {
      el.evidenceExpand.textContent = "Collapse sections";
    }
  }

  function renderEvidence(result) {
    var snippets = result.snippets;
    var retrieval = result.retrieval;

    el.evidenceClamp.classList.remove("is-clamped");
    el.evidenceExpand.hidden = true;
    el.evidenceExpand.setAttribute("aria-expanded", "false");

    if (!snippets.length) {
      renderEvidenceEmpty(retrieval);
      return;
    }

    el.evidenceList.innerHTML = snippets
      .map(function (chunk, index) {
        var title = sectionTitle(chunk);
        var score =
          retrieval && retrieval.scores && retrieval.scores.length > index
            ? formatScore(retrieval.scores[index])
            : null;
        return (
          '<figure class="snippet" id="snippet-' +
          index +
          '" data-section="' +
          escapeHtml(title) +
          '">' +
          '<figcaption class="snippet-head">' +
          '<span class="snippet-num">' +
          (index + 1) +
          "</span>" +
          '<span class="snippet-title">' +
          escapeHtml(title) +
          "</span>" +
          (score !== null
            ? '<span class="snippet-score" title="Retrieval score (title-weighted term match)">score ' +
              score +
              "</span>"
            : "") +
          '<span class="snippet-tag">raw</span>' +
          '<button type="button" class="btn-icon" data-copy="' +
          index +
          '">Copy</button>' +
          "</figcaption>" +
          '<div class="mono-scroll"><pre>' +
          escapeHtml(chunk) +
          "</pre></div>" +
          "</figure>"
        );
      })
      .join("");

    markMonoOverflow(el.evidenceList);

    // Long bodies default to collapsed at the token height; the toggle
    // reports how many sections are fully hidden (spec C).
    var max = collapseMax();
    if (el.evidenceList.scrollHeight > max) {
      var hiddenCount = Array.prototype.filter.call(
        el.evidenceList.querySelectorAll(".snippet"),
        function (snippet) {
          return snippet.offsetTop >= max;
        }
      ).length;
      el.evidenceExpand.hidden = false;
      setClamped(true, Math.max(1, hiddenCount));
    }
  }

  function renderToolCall(query) {
    el.toolCall.innerHTML =
      '<span class="fn">search_knowledge_base</span>({ query: "' +
      escapeHtml(query) +
      '" })';
    markMonoOverflow(document);
  }

  /* ── E · Result + metrics ─────────────────────────────────────────────── */

  /** Rough size estimate for display only (~4 chars/token heuristic). */
  function estTokens(text) {
    return Math.max(1, Math.round(String(text).length / 4));
  }

  function formatCost(usd) {
    if (usd <= 0) return "$0.00";
    return "≈$" + (usd < 0.01 ? usd.toFixed(4) : usd.toFixed(2));
  }

  function renderResult(result) {
    var count = result.snippets.length;

    renderToolCall(result.query);
    el.retrieverMeta.innerHTML = metaHtml([
      ["Tool calls", "1"],
      ["Snippets returned", String(count)],
      ["Query forwarded", result.query === state.query ? "Unchanged" : "Modified"],
      result.retrieval
        ? ["Score cutoff", formatScore(result.retrieval.cutoff)]
        : null
    ]);

    renderEvidence(result);
    setBadge("evidence", result.notFound ? "empty" : "done", count || null);

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

    // Footer metrics as a definition list (spec E). Token/cost figures are
    // estimates (~4 chars/token, CONFIG.pricing) and are labeled as such;
    // the deterministic not-found path makes no LLM call, so they show "—".
    var pricing = api.CONFIG.pricing;
    var tokensIn = estTokens(result.query + "\n" + result.snippets.join("\n"));
    var tokensOut = estTokens(result.report);
    var cost =
      (tokensIn * pricing.inputPer1M + tokensOut * pricing.outputPer1M) / 1e6;

    el.answerMeta.innerHTML =
      metaHtml([
        ["Grounded in", count + (count === 1 ? " section" : " sections"), "high"],
        ["Elapsed", (result.durationMs / 1000).toFixed(1) + "s", "high"],
        [
          "Model",
          result.notFound
            ? "— (no LLM call)"
            : result.source === "live"
              ? api.CONFIG.model
              : api.CONFIG.model + " (mocked)"
        ],
        ["Tokens in / out", result.notFound ? "—" : "≈" + tokensIn + " / ≈" + tokensOut],
        ["Est. cost", result.notFound ? "$0.00" : formatCost(cost)]
      ]) +
      '<div><dt>Trace id</dt><dd><button type="button" class="btn-icon" id="trace-copy" title="Copy trace id">' +
      escapeHtml(state.traceId) +
      "</button></dd></div>";

    el.copyAnswer.hidden = false;
  }

  /* ── Run ──────────────────────────────────────────────────────────────── */

  function setBusy(isBusy) {
    el.runButton.setAttribute("data-busy", String(isBusy));
    el.runButton.querySelector(".btn-label").textContent = isBusy ? "Cancel" : "Run";
    el.runTimer.hidden = !isBusy;
    el.compactTimer.hidden = !isBusy;
    el.compactAction.hidden = !isBusy;
    Array.prototype.forEach.call(el.chips.querySelectorAll(".chip"), function (chip) {
      chip.disabled = isBusy;
    });
  }

  /**
   * api.js normalizes every failure into a plain Error with a descriptive
   * message before it reaches the UI (see runLive's catch), so classify by
   * message rather than by constructor/name.
   */
  function classifyError(error) {
    var message = (error && error.message) || "";
    if (!message) return "UNKNOWN";
    if (/could not reach the backend/i.test(message)) return "NETWORK";
    if (/timed out/i.test(message)) return "TIMEOUT";
    if (/^Backend responded with \d/.test(message)) return "BACKEND";
    if (/malformed response/i.test(message)) return "BACKEND";
    return "RUNTIME";
  }

  function clearBackoff() {
    if (state.backoffHandle) {
      global.clearInterval(state.backoffHandle);
      state.backoffHandle = null;
    }
    el.retryButton.disabled = false;
  }

  function run(query) {
    if (state.status === "running") return;

    clearBackoff();
    state.status = "running";
    state.query = query;
    state.result = null;
    state.error = null;
    state.runId += 1;
    state.traceId = "run-" + state.runId + "-" + Date.now().toString(36);
    var runId = state.runId;
    state.controller = typeof AbortController === "function" ? new AbortController() : null;

    updateHeadline();
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
    startTimer();

    if (state.firstRun) {
      state.firstRun = false;
      el.pipeline.scrollIntoView({ behavior: scrollBehavior(), block: "start" });
    }

    // Promise.resolve() wrapper: a synchronous throw inside runWorkflow must
    // land in .catch as a normal failure, never leave the UI stuck "running".
    Promise.resolve()
      .then(function () {
        return api.runWorkflow(query, {
          onStage: handleStage,
          signal: state.controller ? state.controller.signal : undefined
        });
      })
      .then(function (result) {
        if (runId !== state.runId) return;
        state.status = "done";
        state.result = result;
        state.retryCount = 0;
        el.retryStatus.textContent = "";
        updateHeadline();
        renderResult(result);
        announce(
          result.notFound
            ? "Workflow completed. No evidence found in the knowledge base."
            : "Workflow completed with " + result.snippets.length + " snippets retrieved."
        );
      })
      .catch(function (error) {
        if (runId !== state.runId) return;
        if (error && error.name === "CancelledError") {
          state.status = "idle";
          resetStages("waiting");
          el.pipeline.hidden = true;
          el.emptyState.hidden = false;
          updateHeadline();
          announce("Workflow cancelled.");
          return;
        }
        state.status = "error";
        state.error = error;
        showError(error);
      })
      .then(function () {
        if (runId !== state.runId) return;
        stopTimer();
        setBusy(false);
      });
  }

  /* ── G · ErrorBanner ──────────────────────────────────────────────────── */

  function updateRetryStatus(secondsLeft) {
    var attempts =
      state.retryCount === 0
        ? ""
        : state.retryCount === 1
          ? "Retried once"
          : "Retried " + state.retryCount + " times";
    if (secondsLeft > 0) {
      el.retryStatus.textContent =
        (attempts ? attempts + " · " : "") + "next retry in " + secondsLeft + "s";
    } else {
      el.retryStatus.textContent = attempts;
    }
  }

  /** Exponential backoff between manual retries: 2s, 4s, 8s, capped at 16s. */
  function startBackoff() {
    if (state.retryCount === 0) {
      updateRetryStatus(0);
      return;
    }
    var delay = Math.min(Math.pow(2, state.retryCount), BACKOFF_MAX_S);
    var readyAt = Date.now() + delay * 1000;
    el.retryButton.disabled = true;
    updateRetryStatus(delay);
    state.backoffHandle = global.setInterval(function () {
      var left = Math.ceil((readyAt - Date.now()) / 1000);
      if (left <= 0) {
        clearBackoff();
        updateRetryStatus(0);
      } else {
        updateRetryStatus(left);
      }
    }, 250);
  }

  /**
   * Failure semantics (spec G): red marks ONLY the step that actually failed —
   * the first one still in flight. Every step behind it reads "Not run" in a
   * neutral tone, because "the retriever failed" must not render as five
   * failures.
   */
  function showError(error) {
    var message =
      (error && error.message) || "The workflow could not be completed. Please try again.";

    updateHeadline();
    el.errorCode.textContent = classifyError(error);
    el.errorMessage.textContent = message;
    el.errorRequestId.textContent = state.traceId;
    el.errorRaw.textContent = (error && (error.stack || error.message)) || String(error);
    el.errorDetails.open = false;
    el.errorState.hidden = false;

    var failing = null;
    STAGE_ORDER.forEach(function (stage) {
      var current = stageEls[stage].step.getAttribute("data-state");
      if (current !== "running" && current !== "waiting") return; // keep finished states
      if (!failing && current === "running") {
        failing = stage;
        setStage(stage, "error");
      } else {
        setStage(stage, "notrun");
      }
    });
    // Defensive: if nothing was mid-flight, pin the failure on the retriever.
    if (!failing) setStage("retriever", "error");

    el.evidenceList.innerHTML = "";
    el.answerBody.innerHTML = "";
    el.answerBody.classList.remove("is-notfound");
    startBackoff();
    announce("Workflow failed. " + message);
  }

  /* ── Citations → evidence cross-links (E) ─────────────────────────────── */

  function findSnippet(title) {
    var snippets = el.evidenceList.querySelectorAll(".snippet");
    for (var i = 0; i < snippets.length; i += 1) {
      if (snippets[i].getAttribute("data-section") === title) return snippets[i];
    }
    return null;
  }

  function expandEvidenceCard() {
    var toggle = document.querySelector(
      '.step[data-step="evidence"] .step-toggle[aria-expanded="false"]'
    );
    if (toggle) toggle.click();
    if (el.evidenceClamp.classList.contains("is-clamped")) {
      setClamped(false, 0);
    }
  }

  var flashHandle = null;

  function jumpToSnippet(title) {
    var target = findSnippet(title);
    if (!target) return;
    expandEvidenceCard();
    target.scrollIntoView({ behavior: scrollBehavior(), block: "center" });
    Array.prototype.forEach.call(
      el.evidenceList.querySelectorAll(".snippet.is-flash"),
      function (other) {
        other.classList.remove("is-flash");
      }
    );
    target.classList.add("is-flash");
    if (flashHandle) global.clearTimeout(flashHandle);
    flashHandle = global.setTimeout(function () {
      target.classList.remove("is-flash");
    }, FLASH_MS);
  }

  function setCiteRef(title, on) {
    var target = findSnippet(title);
    if (target) target.classList.toggle("is-ref", on);
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
      }, COPY_FEEDBACK_MS);
    };

    if (global.navigator.clipboard && global.navigator.clipboard.writeText) {
      global.navigator.clipboard.writeText(text).then(done, function () {
        /* Clipboard blocked (e.g. insecure origin) — leave the label untouched. */
      });
    }
  }

  el.form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (state.status === "running") {
      if (state.controller) state.controller.abort();
      return;
    }
    var query = el.input.value.trim();
    if (!query) {
      el.input.focus();
      announce("Enter a question before running the workflow.");
      return;
    }
    state.retryCount = 0;
    run(query);
  });

  el.chips.addEventListener("click", function (event) {
    var chip = event.target.closest(".chip");
    if (!chip || chip.disabled) return;
    el.input.value = chip.textContent;
    state.retryCount = 0;
    run(chip.textContent);
  });

  el.retryButton.addEventListener("click", function () {
    if (el.retryButton.disabled || !state.query) return;
    state.retryCount += 1;
    run(state.query);
  });

  el.evidenceList.addEventListener("click", function (event) {
    var copyButton = event.target.closest("[data-copy]");
    if (copyButton && state.result) {
      copyToClipboard(state.result.snippets[Number(copyButton.dataset.copy)], copyButton);
      return;
    }
    var action = event.target.closest("[data-action]");
    if (!action) return;
    if (action.dataset.action === "rewrite") {
      el.queryCard.scrollIntoView({ behavior: scrollBehavior(), block: "start" });
      el.input.focus();
      el.input.select();
    }
    if (action.dataset.action === "topics") {
      var list = document.getElementById("kb-topics");
      if (!list) return;
      var isOpen = !list.hidden;
      list.hidden = isOpen;
      action.setAttribute("aria-expanded", String(!isOpen));
      action.textContent = isOpen ? "Show covered topics" : "Hide covered topics";
    }
  });

  el.evidenceExpand.addEventListener("click", function () {
    var isClamped = el.evidenceClamp.classList.contains("is-clamped");
    if (isClamped) {
      setClamped(false, 0);
    } else {
      setClamped(true, Number(el.evidenceExpand.dataset.hidden || 1));
      el.evidenceClamp.scrollIntoView({ behavior: scrollBehavior(), block: "nearest" });
    }
  });

  el.copyAnswer.addEventListener("click", function () {
    if (state.result) copyToClipboard(state.result.report, el.copyAnswer);
  });

  el.answerMeta.addEventListener("click", function (event) {
    var trace = event.target.closest("#trace-copy");
    if (trace) copyToClipboard(state.traceId, trace);
  });

  // Citation chips: click jumps to the snippet; hover/focus cross-highlights it.
  el.answerBody.addEventListener("click", function (event) {
    var cite = event.target.closest(".cite");
    if (cite) jumpToSnippet(cite.dataset.cite);
  });
  el.answerBody.addEventListener("mouseover", function (event) {
    var cite = event.target.closest(".cite");
    if (cite) setCiteRef(cite.dataset.cite, true);
  });
  el.answerBody.addEventListener("mouseout", function (event) {
    var cite = event.target.closest(".cite");
    if (cite) setCiteRef(cite.dataset.cite, false);
  });
  el.answerBody.addEventListener("focusin", function (event) {
    var cite = event.target.closest(".cite");
    if (cite) setCiteRef(cite.dataset.cite, true);
  });
  el.answerBody.addEventListener("focusout", function (event) {
    var cite = event.target.closest(".cite");
    if (cite) setCiteRef(cite.dataset.cite, false);
  });

  el.sourceToggle.addEventListener("click", function () {
    api.setMode(api.getMode() === "live" ? "mock" : "live");
    renderSourceToggle();
    announce("Data source set to " + el.sourceLabel.textContent + ".");
  });

  el.compactAction.addEventListener("click", function () {
    if (state.controller) state.controller.abort();
  });

  // The whole card header is a toggle target; the button keeps the
  // aria-expanded state, so route header clicks through it (spec C).
  el.pipeline.addEventListener("click", function (event) {
    var toggle = event.target.closest(".step-toggle");
    if (!toggle) {
      var head = event.target.closest(".step-head");
      if (head && !event.target.closest("button")) {
        toggle = head.querySelector(".step-toggle");
      }
    }
    if (!toggle) return;
    var wasExpanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!wasExpanded));
    var card = toggle.closest(".step-card");
    if (card) card.classList.toggle("is-collapsed", wasExpanded);
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

  // Sticky summary bar: visible only while a run has happened and the full
  // query card has scrolled out of view.
  if (typeof IntersectionObserver === "function") {
    var queryCardVisibility = new IntersectionObserver(
      function (entries) {
        var entry = entries[entries.length - 1];
        el.compactBar.hidden = entry.isIntersecting || el.pipeline.hidden;
      },
      { threshold: 0 }
    );
    queryCardVisibility.observe(el.queryCard);
  }

  // Re-measure mono overflow fades when the viewport changes.
  global.addEventListener("resize", function () {
    markMonoOverflow(document);
  });

  renderChips();
  renderSourceToggle();
  resetStages("waiting");
  // Idle steps start folded to their headers; setStage unfolds each one the
  // moment work reaches it (resetStages can't do this — the markup is
  // already "waiting", so no state transition fires on boot).
  STAGE_ORDER.forEach(function (stage) {
    setCardCollapsed(stage, true);
  });
  updateHeadline();
})(window, document);

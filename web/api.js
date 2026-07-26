/**
 * The single seam between the UI and the two-agent pipeline.
 *
 * The whole app talks to `RagApi.runWorkflow(query, { onStage })` and nothing
 * else, so swapping mock fixtures for the Python backend touches only this file.
 *
 * Expected backend contract — one POST that mirrors `PipelineState`:
 *
 *   POST /api/query   { "query": "Can I work remotely?" }
 *   200  application/json
 *   {
 *     "query":    "Can I work remotely?",
 *     "snippets": ["--- Remote Work Policy ---\nEmployees may ...", ...],
 *     "report":   "Yes. You may work remotely up to 3 days ...",
 *     "retrieval_telemetry": [{
 *       "mode": "lexical",
 *       "query": "Can I work remotely?",
 *       "latency_ms": 0.08,
 *       "empty_reason": null,
 *       "snippets": [{
 *         "title": "Remote Work Policy",
 *         "score": 4.1,
 *         "method": "lexical",
 *         "detail": "matched_terms=remote, work"
 *       }]
 *     }]
 *   }
 *
 * Snippets must stay byte-identical to what `search_knowledge_base` returned —
 * the evidence panel exists to prove the Retriever did not rewrite them.
 * An empty `snippets` array is a valid result, not an error: the Report
 * Generator short-circuits it to NOT_FOUND_SENTENCE with no LLM call.
 */
(function (global) {
  "use strict";

  var CONFIG = {
    endpoint: "/api/query",
    /** Shown as metadata only; the backend decides the real model. */
    model: "gpt-5-mini",
    /** Display-only cost estimate basis (USD per 1M tokens). Not billing data. */
    pricing: { inputPer1M: 0.25, outputPer1M: 2.0, currency: "USD" },
    timeoutMs: 60000
  };

  // Must stay byte-identical to NOT_FOUND_SENTENCE in src/agents/reporter.py.
  var NOT_FOUND_SENTENCE = "I could not find this information in the knowledge base.";

  var MODE_KEY = "rag.mode";
  var STAGES = ["retriever", "evidence", "generator", "answer"];

  function getMode() {
    try {
      return global.localStorage.getItem(MODE_KEY) === "live" ? "live" : "mock";
    } catch (error) {
      return "mock"; // Private browsing / file:// with storage disabled.
    }
  }

  function setMode(mode) {
    try {
      global.localStorage.setItem(MODE_KEY, mode === "live" ? "live" : "mock");
    } catch (error) {
      /* Not persisting the toggle is acceptable. */
    }
  }

  function wait(ms) {
    return new Promise(function (resolve) {
      global.setTimeout(resolve, ms);
    });
  }

  function cancelledError() {
    var error = new Error("Cancelled by user.");
    error.name = "CancelledError";
    return error;
  }

  function finiteNonNegative(value) {
    return typeof value === "number" && isFinite(value) && value >= 0;
  }

  function normalizeTrace(raw) {
    if (!raw || typeof raw !== "object") return null;
    if (typeof raw.title !== "string" || !raw.title.trim()) return null;
    if (!finiteNonNegative(raw.score)) return null;

    var methods = ["lexical", "semantic", "both"];
    var method = methods.indexOf(raw.method) === -1 ? "unknown" : raw.method;
    return {
      title: raw.title.trim().slice(0, 200),
      score: raw.score,
      method: method,
      detail: typeof raw.detail === "string" ? raw.detail.slice(0, 500) : ""
    };
  }

  function normalizeTelemetry(payload) {
    if (!Array.isArray(payload)) return [];

    return payload.slice(0, 16).map(function (raw) {
      if (!raw || typeof raw !== "object") return null;

      var modes = ["lexical", "semantic", "hybrid"];
      var mode = modes.indexOf(raw.mode) === -1 ? "unknown" : raw.mode;
      var emptyReasons = ["no_query_terms", "gated_out"];
      var emptyReason =
        emptyReasons.indexOf(raw.empty_reason) === -1 ? null : raw.empty_reason;
      var traces = Array.isArray(raw.snippets)
        ? raw.snippets
            .slice(0, 1000)
            .map(normalizeTrace)
            .filter(function (trace) {
              return trace !== null;
            })
        : [];

      return {
        mode: mode,
        query: typeof raw.query === "string" ? raw.query.slice(0, 2000) : "",
        latencyMs: finiteNonNegative(raw.latency_ms) ? raw.latency_ms : 0,
        emptyReason: traces.length ? null : emptyReason,
        snippets: traces
      };
    }).filter(function (attempt) {
      return attempt !== null;
    });
  }

  function normalize(payload, query) {
    var snippets = Array.isArray(payload && payload.snippets) ? payload.snippets : [];
    var report = payload && typeof payload.report === "string" ? payload.report.trim() : "";
    return {
      query:
        payload && typeof payload.query === "string" && payload.query.trim()
          ? payload.query
          : query,
      snippets: snippets.filter(function (snippet) {
        return typeof snippet === "string" && snippet.trim() !== "";
      }),
      report: report || NOT_FOUND_SENTENCE,
      retrievalTelemetry: normalizeTelemetry(
        payload && payload.retrieval_telemetry
      )
    };
  }

  /** Fixture run with realistic pacing so each stage transition is visible. */
  function runMock(query, emit, signal) {
    var detail;
    var snippets;
    var retrievalLatencyMs = 0;

    function throwIfCancelled() {
      if (signal && signal.aborted) throw cancelledError();
    }

    emit("retriever", "running");
    return wait(650)
      .then(function () {
        throwIfCancelled();
        var startedAt =
          global.performance && typeof global.performance.now === "function"
            ? global.performance.now()
            : Date.now();
        detail = global.RAG_MOCK.retrieveDetailed(query);
        var finishedAt =
          global.performance && typeof global.performance.now === "function"
            ? global.performance.now()
            : Date.now();
        retrievalLatencyMs = Math.max(0, finishedAt - startedAt);
        snippets = detail.snippets;
        emit("retriever", "done");
        emit("evidence", snippets.length ? "done" : "empty");
        if (!snippets.length) return null;

        emit("generator", "running");
        return wait(900);
      })
      .then(function () {
        throwIfCancelled();
        var report = global.RAG_MOCK.report(query, snippets);
        var result = normalize({ query: query, snippets: snippets, report: report }, query);
        result.retrievalTelemetry = normalizeTelemetry([
          {
            mode: "lexical",
            query: query,
            latency_ms: retrievalLatencyMs,
            empty_reason: detail.emptyReason,
            snippets: detail.traces
          }
        ]);
        // Keep the earlier mock-only cutoff diagnostic for empty-state detail.
        result.retrieval = {
          scores: detail.scores,
          bestScore: detail.bestScore,
          cutoff: detail.cutoff
        };
        return result;
      });
  }

  /**
   * Live run against the Python backend.
   *
   * The pipeline answers in one round trip, so stage transitions are derived
   * from that single response rather than observed per node. To report true
   * per-node progress, expose a streaming endpoint (SSE over LangGraph's
   * `graph.stream`) and emit here as each node event arrives.
   */
  function runLive(query, emit, signal) {
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timer = global.setTimeout(function () {
      if (controller) controller.abort();
    }, CONFIG.timeoutMs);

    // A caller-supplied signal (Cancel button) aborts the same controller as
    // the timeout, so both paths land in the single AbortError branch below.
    var cancelledByCaller = false;
    if (signal && controller) {
      if (signal.aborted) {
        cancelledByCaller = true;
        controller.abort();
      } else {
        signal.addEventListener(
          "abort",
          function () {
            cancelledByCaller = true;
            controller.abort();
          },
          { once: true }
        );
      }
    }

    emit("retriever", "running");

    return fetch(CONFIG.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query }),
      signal: controller ? controller.signal : undefined
    })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return null;
          })
          .then(function (payload) {
            if (!response.ok) {
              throw new Error(
                (payload && (payload.error || payload.detail)) ||
                  "Backend responded with " + response.status + " " + response.statusText
              );
            }
            if (!payload || typeof payload !== "object") {
              throw new Error("Backend returned a malformed response.");
            }
            return normalize(payload, query);
          });
      })
      .then(function (result) {
        emit("retriever", "done");
        emit("evidence", result.snippets.length ? "done" : "empty");
        if (result.snippets.length) emit("generator", "running");
        return result;
      })
      .catch(function (error) {
        if (error && error.name === "AbortError") {
          if (cancelledByCaller) throw cancelledError();
          throw new Error("The workflow timed out before the backend responded.");
        }
        if (error instanceof TypeError) {
          throw new Error(
            "Could not reach the backend at " +
              CONFIG.endpoint +
              ". Start the Python service, or switch back to mock data."
          );
        }
        throw error;
      })
      .finally(function () {
        global.clearTimeout(timer);
      });
  }

  /**
   * Run the full workflow for one query.
   *
   * @param {string} query
   * @param {{ onStage?: (stage: string, status: string) => void,
   *           signal?: AbortSignal }} [options] - `signal` cancels the run;
   *           the returned promise then rejects with `name: "CancelledError"`.
   * @returns {Promise<{query: string, snippets: string[], report: string,
   *                    source: string, durationMs: number, notFound: boolean,
   *                    retrievalTelemetry: object[]}>}
   */
  function runWorkflow(query, options) {
    var opts = options || {};
    var emit =
      typeof opts.onStage === "function"
        ? opts.onStage
        : function () {
            /* no-op */
          };
    var mode = getMode();
    var startedAt = Date.now();

    var run = mode === "live" ? runLive : runMock;

    return run(query, emit, opts.signal).then(function (result) {
      var finished =
        result ||
        normalize({ query: query, snippets: [], report: NOT_FOUND_SENTENCE }, query);
      var notFound = finished.snippets.length === 0;

      emit("generator", "done");
      emit("answer", notFound ? "empty" : "done");

      return {
        query: finished.query,
        snippets: finished.snippets,
        report: notFound ? NOT_FOUND_SENTENCE : finished.report,
        notFound: notFound,
        source: mode,
        durationMs: Date.now() - startedAt,
        retrievalTelemetry: finished.retrievalTelemetry || [],
        /* Optional scoring evidence — present on mock runs, null on live. */
        retrieval: finished.retrieval || null
      };
    });
  }

  global.RagApi = {
    CONFIG: CONFIG,
    NOT_FOUND_SENTENCE: NOT_FOUND_SENTENCE,
    STAGES: STAGES,
    getMode: getMode,
    setMode: setMode,
    runWorkflow: runWorkflow
  };
})(window);

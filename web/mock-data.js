/**
 * Offline fixtures so the UI is fully demonstrable without a backend.
 *
 * SECTIONS are copied verbatim from `knowledge_base.txt`, and the matcher below
 * mirrors the real tool's gate (title weight 1.5, body 1.0, keep >= 60% of the
 * best score, plus the two-term sibling rule) closely enough that the demo
 * retrieves the same sections the Python pipeline does. It is a fixture, not a
 * reimplementation — delete this file once the live backend is wired up.
 */
(function (global) {
  "use strict";

  var SECTIONS = [
    "--- Remote Work Policy ---\n" +
      "Employees may work remotely up to 3 days per week with manager approval.\n" +
      "Requests must be submitted through the FlexWork portal by Thursday of the\n" +
      "preceding week. Remote employees must be reachable on Microsoft Teams during\n" +
      "agreed working hours and use the TigerLink VPN for internal systems. New hires\n" +
      "must work fully on-site during their first 30 calendar days.",

    "--- Hybrid Work Guidelines ---\n" +
      "Siam Innovate uses a hybrid working model. Core collaboration hours are 10:00\n" +
      "to 16:00 Bangkok time, during which employees must be available for meetings\n" +
      "regardless of location. Tuesday is the company-wide office anchor day, and\n" +
      "teams may define one additional anchor day. Office desks must be reserved in\n" +
      "the SpaceLy app at least one day in advance.",

    "--- Annual Leave ---\n" +
      "Full-time employees receive 15 days of paid annual leave per calendar year,\n" +
      "accrued at 1.25 days per month. The entitlement increases to 20 days after\n" +
      "five full years of service. Up to 5 unused days may be carried into the first\n" +
      "quarter of the following year and expire on 31 March. Leave requests must be\n" +
      "submitted through LeaveDesk at least 3 business days in advance for absences\n" +
      "of 3 days or fewer and 14 days in advance for longer absences.",

    "--- International Travel Approval Process ---\n" +
      "Employees traveling internationally for business must obtain written approval\n" +
      "from their department head at least 14 days before departure. Requests are\n" +
      'submitted through TravelHub under "Overseas Trip Request" with a trip purpose,\n' +
      "itinerary, and cost estimate. Trips longer than 10 business days also require\n" +
      "Managing Director approval. Once approved, the Travel Desk books economy-class\n" +
      "airfare through the contracted travel agency.",

    "--- International Travel Daily Allowance ---\n" +
      "Approved international business travel receives a daily allowance of 2,400 THB\n" +
      "for meals and incidental expenses. Departure and return days receive half the\n" +
      "daily rate, or 1,200 THB. Hotel caps are 5,500 THB per night in Southeast Asia\n" +
      "and 8,000 THB elsewhere. Hotel and ground-transport receipts must be uploaded\n" +
      "to ExpenseFlow within 45 days after returning.",

    "--- International Travel Insurance ---\n" +
      "International business trips are automatically covered by the SafeJourney\n" +
      "Plan B group travel insurance. Coverage includes emergency medical treatment\n" +
      "up to 3,000,000 THB, emergency evacuation, and lost-luggage compensation up to\n" +
      "40,000 THB per trip. Claims must be filed through the SafeJourney portal within\n" +
      "30 days of returning, with original receipts and the TravelHub approval.",

    "--- Expense Reimbursement ---\n" +
      "Business expenses must be claimed through ExpenseFlow with scanned receipts\n" +
      "within 45 days of the expense date. Claims below 20,000 THB require line-manager\n" +
      "approval; claims of 20,000 THB or more also require Finance approval. Approved\n" +
      "amounts are paid with the next payroll. Alcohol, traffic fines, minibar charges,\n" +
      "and personal subscriptions are not reimbursable.",

    "--- PaySiam Gateway Product Overview ---\n" +
      "PaySiam Gateway is an online payment platform for Thai small and medium\n" +
      "merchants. It supports PromptPay QR, domestic and international cards, mobile\n" +
      "banking transfers, and installment plans. Standard pricing is 1.85% per\n" +
      "successful domestic card or QR transaction and 2.95% for international cards,\n" +
      "with no setup fee or monthly minimum. Settlement uses a T+2 business-day cycle.",

    "--- Customer Support Service Levels ---\n" +
      "Standard product support operates from 09:00 to 18:00 Bangkok time on working\n" +
      "days. Platinum support operates around the clock. For a P1 production outage,\n" +
      "the first-response target is 15 minutes for Platinum and 1 hour for Standard.\n" +
      "For a P2 major degradation, the targets are 1 hour and 4 hours respectively.\n" +
      "Cases are created in the customer portal and receive a CS-XXXXXX tracking\n" +
      "number.",

    "--- Support Escalation Process ---\n" +
      "Customers or account managers may escalate a support case through the customer\n" +
      "portal or escalations@siaminnovate.example using the CS case number. Escalated\n" +
      "cases are owned by the duty manager. A P1 case open for more than 4 hours\n" +
      "automatically starts a bridge call and status updates every 30 minutes until\n" +
      "service is restored. A cause summary is completed within 5 business days after\n" +
      "an escalated case closes."
  ];

  // Terms that describe sentence structure or appear in most sections, so they
  // never identify which section the user needs.
  var IGNORED = [
    "a", "about", "an", "and", "are", "as", "at", "available", "be", "business",
    "can", "company", "could", "detail", "details", "did", "do", "does",
    "employee", "employees", "each", "for", "from", "has", "have", "how", "i",
    "in", "information", "is", "it", "its", "many", "may", "me", "much", "my",
    "of", "on", "or", "our", "please", "policies", "policy", "process",
    "quickly", "request", "requests", "rule", "rules", "should", "tell", "that",
    "the", "their", "them", "there", "this", "to", "us", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "allow", "allowed"
  ];

  var ALIASES = {
    lodging: "hotel",
    overseas: "international",
    remotely: "remote",
    staff: "employee",
    submitted: "submit",
    vacation: "leave",
    vacations: "leave"
  };

  var PHRASES = [
    ["work from home", "remote work"],
    ["per diem", "daily allowance"],
    ["response time", "first response"]
  ];

  var TITLE_WEIGHT = 1.5;
  var BODY_WEIGHT = 1.0;
  var RELATIVE_CUTOFF = 0.6;
  var ABSOLUTE_CUTOFF = 1.0;

  /** Hand-written grounded answers for the sample chips. */
  var CANNED_ANSWERS = {
    "what is the policy on international travel?":
      "International business travel requires written approval from your department " +
      "head at least 14 days before departure, requested through TravelHub under " +
      '"Overseas Trip Request" with a trip purpose, itinerary, and cost estimate. ' +
      "Trips longer than 10 business days also need Managing Director approval, and " +
      "the Travel Desk books economy-class airfare once approved. " +
      "[International Travel Approval Process]\n\n" +
      "Approved trips receive a daily allowance of 2,400 THB for meals and incidental " +
      "expenses, halved to 1,200 THB on departure and return days. Hotels are capped " +
      "at 5,500 THB per night in Southeast Asia and 8,000 THB elsewhere, with hotel " +
      "and ground-transport receipts uploaded to ExpenseFlow within 45 days of " +
      "returning. [International Travel Daily Allowance]\n\n" +
      "Every trip is automatically covered by the SafeJourney Plan B group travel " +
      "insurance, which includes emergency medical treatment up to 3,000,000 THB, " +
      "emergency evacuation, and lost-luggage compensation up to 40,000 THB per trip. " +
      "Claims go through the SafeJourney portal within 30 days of returning, with " +
      "original receipts and the TravelHub approval. [International Travel Insurance]",

    "can i work remotely?":
      "Yes. You may work remotely up to 3 days per week with manager approval, " +
      "requested through the FlexWork portal by Thursday of the preceding week. " +
      "While remote you must stay reachable on Microsoft Teams during agreed working " +
      "hours and use the TigerLink VPN for internal systems. New hires work fully " +
      "on-site for their first 30 calendar days. [Remote Work Policy]\n\n" +
      "Two constraints come from the hybrid model:\n\n" +
      "- Core collaboration hours are 10:00–16:00 Bangkok time, when you must be " +
      "available for meetings regardless of location.\n" +
      "- Tuesday is the company-wide office anchor day, and your team may define one " +
      "additional anchor day. Office desks are reserved in the SpaceLy app at least " +
      "one day in advance. [Hybrid Work Guidelines]"
  };

  function tokenize(text) {
    var normalized = text.toLowerCase();
    PHRASES.forEach(function (pair) {
      normalized = normalized.split(pair[0]).join(pair[1]);
    });
    return (normalized.match(/[a-z0-9]+/g) || []).map(function (token) {
      return ALIASES[token] || token;
    });
  }

  function uniqueTerms(text, isQuery) {
    var seen = Object.create(null);
    tokenize(text).forEach(function (token) {
      if (!isQuery || IGNORED.indexOf(token) === -1) seen[token] = true;
    });
    return Object.keys(seen);
  }

  function intersect(terms, lookup) {
    return terms.filter(function (term) {
      return lookup.indexOf(term) !== -1;
    });
  }

  /**
   * Deterministic keyword retrieval over the bundled sections.
   * Returns snippets plus the scoring evidence the UI can surface
   * (per-snippet score, best score, applied cutoff). retrieve() below
   * keeps the original snippets-only contract.
   */
  function retrieveDetailed(query) {
    var empty = {
      snippets: [],
      scores: [],
      bestScore: 0,
      cutoff: ABSOLUTE_CUTOFF,
      emptyReason: "gated_out",
      traces: []
    };

    var queryTerms = uniqueTerms(query, true);
    if (!queryTerms.length) {
      empty.emptyReason = "no_query_terms";
      return empty;
    }

    var scored = SECTIONS.map(function (chunk, index) {
      var split = chunk.indexOf("\n");
      var titleTerms = uniqueTerms(chunk.slice(0, split), false);
      var bodyTerms = uniqueTerms(chunk.slice(split + 1), false);

      var titleMatches = intersect(queryTerms, titleTerms);
      var bodyMatches = intersect(queryTerms, bodyTerms).filter(function (term) {
        return titleMatches.indexOf(term) === -1;
      });

      return {
        index: index,
        chunk: chunk,
        titleMatches: titleMatches,
        matchedTerms: titleMatches.concat(bodyMatches),
        matchCount: titleMatches.length + bodyMatches.length,
        score: titleMatches.length * TITLE_WEIGHT + bodyMatches.length * BODY_WEIGHT
      };
    });

    var candidates = scored.filter(function (candidate) {
      return candidate.titleMatches.length > 0 || candidate.matchCount >= 2;
    });

    if (!candidates.length) {
      // Nothing passed the structural filter — report the best raw score so
      // the UI can say how close the nearest section came to the cutoff.
      empty.bestScore = Math.max.apply(
        null,
        scored.map(function (candidate) {
          return candidate.score;
        }).concat([0])
      );
      return empty;
    }

    var best = Math.max.apply(
      null,
      candidates.map(function (candidate) {
        return candidate.score;
      })
    );
    var cutoff = Math.max(ABSOLUTE_CUTOFF, best * RELATIVE_CUTOFF);
    var selected = candidates.filter(function (candidate) {
      return candidate.score >= cutoff;
    });

    // A focused two-term topic keeps a weaker sibling that shares a title anchor
    // with a full-coverage match — this is what pairs Hybrid Work with Remote Work.
    if (queryTerms.length === 2) {
      var anchors = [];
      selected.forEach(function (candidate) {
        if (candidate.matchCount === queryTerms.length) {
          anchors = anchors.concat(candidate.titleMatches);
        }
      });
      candidates.forEach(function (candidate) {
        var alreadyPicked = selected.indexOf(candidate) !== -1;
        if (!alreadyPicked && intersect(candidate.titleMatches, anchors).length) {
          selected.push(candidate);
        }
      });
    }

    var ordered = selected.sort(function (a, b) {
      return b.score - a.score || a.index - b.index;
    });

    return {
      snippets: ordered.map(function (candidate) {
        return candidate.chunk;
      }),
      scores: ordered.map(function (candidate) {
        return candidate.score;
      }),
      bestScore: best,
      cutoff: cutoff,
      emptyReason: null,
      traces: ordered.map(function (candidate) {
        return {
          title: sectionTitle(candidate.chunk),
          score: candidate.score,
          method: "lexical",
          detail: "matched_terms=" + candidate.matchedTerms.slice().sort().join(", ")
        };
      })
    };
  }

  /** Original snippets-only contract, unchanged for existing callers. */
  function retrieve(query) {
    return retrieveDetailed(query).snippets;
  }

  function sectionTitle(chunk) {
    var match = /^---\s*(.+?)\s*---/.exec(chunk);
    return match ? match[1] : "Untitled section";
  }

  /** Stand-in for the Report Generator when a query has no canned answer. */
  function synthesize(snippets) {
    return snippets
      .map(function (chunk) {
        var body = chunk.slice(chunk.indexOf("\n") + 1).replace(/\s+/g, " ").trim();
        var sentences = body.split(". ").slice(0, 2).join(". ");
        if (sentences.charAt(sentences.length - 1) !== ".") sentences += ".";
        return sentences + " [" + sectionTitle(chunk) + "]";
      })
      .join("\n\n");
  }

  function report(query, snippets) {
    if (!snippets.length) return null; // Caller applies the not-found sentence.
    var canned = CANNED_ANSWERS[query.trim().toLowerCase()];
    return canned || synthesize(snippets);
  }

  global.RAG_MOCK = {
    sections: SECTIONS,
    topics: SECTIONS.map(sectionTitle),
    retrieve: retrieve,
    retrieveDetailed: retrieveDetailed,
    report: report,
    sectionTitle: sectionTitle
  };
})(window);

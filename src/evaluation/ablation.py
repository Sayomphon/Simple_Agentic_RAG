"""Ablation variants: the scoring pipeline with layers removed one by one.

Each variant adds one design layer on top of the previous, so comparing
adjacent rows in the report shows what that layer buys. The final variant
must equal the production defaults — a test enforces this so the ablation
can never drift from the shipped behavior.
"""

from __future__ import annotations

from src.tools.retrieval import (
    BODY_MATCH_WEIGHT,
    K_TF,
    TITLE_MATCH_WEIGHT,
    RetrievalSettings,
)

# (variant name, settings this layer turns on), applied cumulatively.
# TF saturation was evaluated as a seventh rung and dropped: it matched
# V5 on every calibration metric (docs/DESIGN_NOTES.md), so the ladder
# ends at the stemming layer, which is the shipped configuration.
_LAYERS: tuple[tuple[str, dict[str, object]], ...] = (
    ("V0_raw_overlap", {}),
    ("V1_+query_filters", {"use_query_filters": True}),
    ("V2_+aliases", {"use_aliases": True}),
    (
        "V3_+idf_title_weight",
        {"use_idf": True, "title_weight": TITLE_MATCH_WEIGHT},
    ),
    (
        "V4_+gate_sibling",
        {"use_relevance_gate": True, "use_sibling_expansion": True},
    ),
    ("V5_current_+stemming", {"use_stemming": True}),
)

_EVERYTHING_OFF: dict[str, object] = {
    "use_query_filters": False,
    "use_aliases": False,
    "use_idf": False,
    "title_weight": BODY_MATCH_WEIGHT,
    "use_relevance_gate": False,
    "use_sibling_expansion": False,
    "use_stemming": False,
    "use_tf_saturation": False,
    "k_tf": K_TF,
}


def build_variants() -> dict[str, RetrievalSettings]:
    """Return the ordered {name: settings} ablation ladder."""
    variants: dict[str, RetrievalSettings] = {}
    accumulated = dict(_EVERYTHING_OFF)
    for name, layer in _LAYERS:
        accumulated.update(layer)
        variants[name] = RetrievalSettings(**accumulated)
    return variants


__all__ = ["build_variants"]

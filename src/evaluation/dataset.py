"""Single source of truth for loading labeled retrieval fixtures.

Both the unit tests and the evaluation runners load cases through this
module, so a schema change or a new dataset is made in exactly one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

DATASET_FILES: dict[str, str] = {
    "calibration": "retrieval_cases.json",
    "heldout": "retrieval_heldout.json",
    "negatives": "retrieval_negatives.json",
    "thai": "retrieval_thai.json",
}


class DatasetValidationError(ValueError):
    """Raised when a fixture file violates the evaluation-case schema."""


@dataclass(frozen=True)
class EvalCase:
    """One labeled retrieval query with expected and forbidden titles."""

    id: str
    category: str
    query: str
    expected_titles: tuple[str, ...]
    forbidden_titles: tuple[str, ...]

    @property
    def is_negative(self) -> bool:
        """True when no knowledge-base section should be returned."""
        return not self.expected_titles


def _require_str(raw_case: dict[str, object], key: str, source: str) -> str:
    value = raw_case.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(
            f"{source}: case field {key!r} must be a non-empty string"
        )
    return value


def _require_str_list(
    raw_case: dict[str, object],
    key: str,
    source: str,
) -> tuple[str, ...]:
    value = raw_case.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise DatasetValidationError(
            f"{source}: case field {key!r} must be a list of non-empty strings"
        )
    return tuple(value)


def _parse_case(raw_case: object, source: str) -> EvalCase:
    if not isinstance(raw_case, dict):
        raise DatasetValidationError(f"{source}: each case must be an object")

    case = EvalCase(
        id=_require_str(raw_case, "id", source),
        category=_require_str(raw_case, "category", source),
        query=_require_str(raw_case, "query", source),
        expected_titles=_require_str_list(raw_case, "expected_titles", source),
        forbidden_titles=_require_str_list(raw_case, "forbidden_titles", source),
    )
    overlap = set(case.expected_titles) & set(case.forbidden_titles)
    if overlap:
        raise DatasetValidationError(
            f"{source}: case {case.id!r} lists titles as both expected "
            f"and forbidden: {sorted(overlap)}"
        )
    return case


def load_cases(name: str) -> list[EvalCase]:
    """Load and validate one named retrieval dataset."""
    try:
        fixture_path = FIXTURES_DIR / DATASET_FILES[name]
    except KeyError:
        known = ", ".join(sorted(DATASET_FILES))
        raise DatasetValidationError(
            f"Unknown dataset {name!r}; expected one of: {known}"
        ) from None

    raw_cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DatasetValidationError(
            f"{fixture_path.name}: fixture must be a non-empty JSON array"
        )

    cases = [_parse_case(raw_case, fixture_path.name) for raw_case in raw_cases]

    seen_ids = {case.id for case in cases}
    if len(seen_ids) != len(cases):
        raise DatasetValidationError(
            f"{fixture_path.name}: case ids must be unique"
        )
    return cases


def load_all() -> dict[str, list[EvalCase]]:
    """Load every dataset whose fixture file exists on disk."""
    return {
        name: load_cases(name)
        for name, filename in DATASET_FILES.items()
        if (FIXTURES_DIR / filename).exists()
    }


__all__ = [
    "DATASET_FILES",
    "DatasetValidationError",
    "EvalCase",
    "FIXTURES_DIR",
    "load_all",
    "load_cases",
]

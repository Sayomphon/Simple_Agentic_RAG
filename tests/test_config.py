"""Offline tests for environment parsing and validation in src.config."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src import config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EnvHelperTests(unittest.TestCase):
    def test_unset_and_blank_values_fall_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"BLANK": "", "SPACES": "   "}):
            self.assertEqual(config._env_str("BLANK", "fallback"), "fallback")
            self.assertEqual(config._env_str("SPACES", "fallback"), "fallback")
            self.assertEqual(
                config._env_str("NEVER_SET_VALUE", "fallback"), "fallback"
            )

    def test_set_values_are_stripped_not_replaced(self) -> None:
        with patch.dict(os.environ, {"NAME": "  value  "}):
            self.assertEqual(config._env_str("NAME", "fallback"), "value")

    def test_numeric_helpers_parse_valid_input(self) -> None:
        with patch.dict(os.environ, {"RATIO": "0.25", "COUNT": "3"}):
            self.assertEqual(config._env_float("RATIO", "9"), 0.25)
            self.assertEqual(config._env_int("COUNT", "9"), 3)

    def test_blank_numeric_values_use_the_default(self) -> None:
        with patch.dict(os.environ, {"RATIO": "", "COUNT": ""}):
            self.assertEqual(config._env_float("RATIO", "0.5"), 0.5)
            self.assertEqual(config._env_int("COUNT", "60"), 60)

    def test_malformed_float_error_names_the_variable_and_value(self) -> None:
        with patch.dict(os.environ, {"TEMPERATURE": "warm"}):
            with self.assertRaises(ValueError) as context:
                config._env_float("TEMPERATURE", "0")

        self.assertIn("TEMPERATURE", str(context.exception))
        self.assertIn("'warm'", str(context.exception))

    def test_malformed_int_error_names_the_variable_and_value(self) -> None:
        with patch.dict(os.environ, {"RRF_K": "sixty"}):
            with self.assertRaises(ValueError) as context:
                config._env_int("RRF_K", "60")

        self.assertIn("RRF_K", str(context.exception))
        self.assertIn("'sixty'", str(context.exception))

    def test_float_helper_rejects_only_the_parsed_value(self) -> None:
        # A default is trusted source code; only environment input may fail.
        with patch.dict(os.environ, {"RATIO": ""}):
            self.assertEqual(config._env_float("RATIO", "1.5"), 1.5)


class ConfigImportTests(unittest.TestCase):
    """End-to-end import checks in a subprocess, isolating module state."""

    def _import_config(self, overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", "import src.config"],
            cwd=str(_PROJECT_ROOT),
            env={**os.environ, **overrides},
            capture_output=True,
            text=True,
            check=False,
        )

    def test_blank_env_lines_do_not_break_import(self) -> None:
        completed = self._import_config(
            {
                "MODEL_NAME": "",
                "TEMPERATURE": "",
                "LLM_TIMEOUT_SECONDS": "",
                "LLM_MAX_RETRIES": "",
                "KB_PATH": "",
                "SEARCH_MODE": "",
                "MIN_COSINE": "",
                "RRF_K": "",
                "EMBED_CACHE_DIR": "",
            }
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_malformed_numeric_env_fails_loudly_with_the_variable(self) -> None:
        completed = self._import_config({"TEMPERATURE": "warm"})

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("TEMPERATURE", completed.stderr)
        self.assertIn("'warm'", completed.stderr)


if __name__ == "__main__":
    unittest.main()

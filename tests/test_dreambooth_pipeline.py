"""Tests for src/image/dreambooth.py

Covers _inject_subject_token, the word-boundary regex logic, without
loading any model weights.
"""

from __future__ import annotations

import pytest

from src.image.dreambooth import DreamBoothPipeline


def _make_pipeline(subject_token: str) -> DreamBoothPipeline:
    """Create a bare DreamBoothPipeline instance (bypasses __init__)."""
    obj = object.__new__(DreamBoothPipeline)
    obj.subject_token = subject_token
    return obj


class TestInjectSubjectToken:
    def test_injects_when_absent(self):
        pipe = _make_pipeline("sks dog")
        result = pipe._inject_subject_token("a photo in the park")
        assert result.startswith("sks dog,")

    def test_does_not_inject_when_present(self):
        pipe = _make_pipeline("sks dog")
        prompt = "sks dog sitting in the park"
        assert pipe._inject_subject_token(prompt) == prompt

    def test_no_false_positive_on_substring(self):
        """'sks dog' should NOT match inside 'sks doghouse'."""
        pipe = _make_pipeline("sks dog")
        result = pipe._inject_subject_token("a photo of sks doghouse in a forest")
        assert result.startswith("sks dog,")

    def test_no_false_positive_on_prefix_match(self):
        """'sks' should NOT match inside 'sksk character'."""
        pipe = _make_pipeline("sks")
        result = pipe._inject_subject_token("a photo of sksk character")
        assert result.startswith("sks,")

    def test_case_sensitive_injection(self):
        """Token 'sks dog' should NOT match 'SKS dog' (case-sensitive)."""
        pipe = _make_pipeline("sks dog")
        result = pipe._inject_subject_token("a photo of SKS dog in the park")
        assert result.startswith("sks dog,")

    def test_injects_at_front(self):
        pipe = _make_pipeline("sks wizard")
        result = pipe._inject_subject_token("casting a spell")
        assert result == "sks wizard, casting a spell"

    def test_token_with_special_regex_chars(self):
        """Token containing regex special chars (e.g. '.') is matched literally."""
        pipe = _make_pipeline("sks.token")
        # The dot in "sks.token" should only match literally
        result = pipe._inject_subject_token("a photo of sks.token here")
        assert "sks.token" in result
        # Should NOT inject because the literal token is present
        assert not result.startswith("sks.token,")

    def test_token_present_mid_sentence(self):
        pipe = _make_pipeline("sks character")
        prompt = "an epic render of sks character in battle"
        assert pipe._inject_subject_token(prompt) == prompt

    def test_empty_prompt(self):
        pipe = _make_pipeline("sks hero")
        result = pipe._inject_subject_token("")
        assert result.startswith("sks hero,")

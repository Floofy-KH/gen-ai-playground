"""Tests for src/utils/prompt_helpers.py

All tests cover pure-Python logic and require no GPU or model weights.
"""

from __future__ import annotations

import pytest

from src.utils.prompt_helpers import (
    PromptTemplate,
    build_anchored_prompt,
    extract_slots,
    validate_prompt,
)


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_basic_build(self):
        t = PromptTemplate(
            anchor="sks character",
            template="{anchor}, {action}",
        )
        result = t.build(action="flying")
        assert result == "sks character, flying"

    def test_quality_tags_appended(self):
        t = PromptTemplate(
            anchor="hero",
            template="{anchor}, {action}",
            quality_tags=["masterpiece", "best quality"],
        )
        result = t.build(action="standing")
        assert "masterpiece" in result
        assert "best quality" in result

    def test_anchor_auto_filled(self):
        """The {anchor} slot is automatically populated from self.anchor."""
        t = PromptTemplate(anchor="test anchor", template="{anchor}, {scene}")
        result = t.build(scene="forest")
        assert "test anchor" in result

    def test_missing_slot_raises(self):
        t = PromptTemplate(anchor="a", template="{anchor}, {action}, {setting}")
        with pytest.raises(KeyError):
            t.build(action="flying")  # missing "setting"

    def test_build_batch_returns_correct_count(self):
        t = PromptTemplate(anchor="x", template="{anchor}, {v}")
        results = t.build_batch([{"v": "1"}, {"v": "2"}, {"v": "3"}])
        assert len(results) == 3

    def test_build_batch_fills_each_variation(self):
        t = PromptTemplate(anchor="hero", template="{anchor}, {action}")
        results = t.build_batch([{"action": "run"}, {"action": "jump"}])
        assert "run" in results[0]
        assert "jump" in results[1]

    def test_empty_quality_tags(self):
        t = PromptTemplate(anchor="a", template="{anchor}", quality_tags=[])
        result = t.build()
        assert result == "a"

    def test_negative_prompt_stored(self):
        t = PromptTemplate(negative_prompt="blurry, deformed")
        assert t.negative_prompt == "blurry, deformed"


# ---------------------------------------------------------------------------
# build_anchored_prompt
# ---------------------------------------------------------------------------


class TestBuildAnchoredPrompt:
    def test_basic(self):
        result = build_anchored_prompt("sks hero", "flying")
        assert "sks hero" in result
        assert "flying" in result

    def test_all_parts_included(self):
        result = build_anchored_prompt(
            anchor="anchor",
            action="action",
            setting="setting",
            style="style",
            quality_tags=["tag1", "tag2"],
        )
        for part in ["anchor", "action", "setting", "style", "tag1", "tag2"]:
            assert part in result

    def test_optional_parts_omitted_when_empty(self):
        result = build_anchored_prompt("anchor", "action", setting="", style="")
        assert result == "anchor, action"

    def test_custom_separator(self):
        result = build_anchored_prompt("a", "b", separator=" | ")
        assert " | " in result

    def test_whitespace_stripped(self):
        result = build_anchored_prompt("  anchor  ", "  action  ")
        assert result == "anchor, action"


# ---------------------------------------------------------------------------
# extract_slots
# ---------------------------------------------------------------------------


class TestExtractSlots:
    def test_simple(self):
        assert extract_slots("{anchor}, {action}") == ["action", "anchor"]

    def test_sorted_unique(self):
        slots = extract_slots("{z}, {a}, {z}, {m}")
        assert slots == ["a", "m", "z"]
        assert len(slots) == len(set(slots))

    def test_no_slots(self):
        assert extract_slots("no placeholders here") == []

    def test_single_slot(self):
        assert extract_slots("{only}") == ["only"]

    def test_complex_template(self):
        slots = extract_slots("{anchor}, {action} in {setting}, {style}")
        assert "anchor" in slots
        assert "action" in slots
        assert "setting" in slots
        assert "style" in slots


# ---------------------------------------------------------------------------
# validate_prompt
# ---------------------------------------------------------------------------


class TestValidatePrompt:
    def test_short_prompt_valid(self):
        assert validate_prompt("a red fox") is True

    def test_exactly_at_limit(self):
        prompt = " ".join(["word"] * 75)
        assert validate_prompt(prompt, max_tokens=75) is True

    def test_one_over_limit(self):
        prompt = " ".join(["word"] * 76)
        assert validate_prompt(prompt, max_tokens=75) is False

    def test_empty_prompt_valid(self):
        assert validate_prompt("") is True

    def test_custom_max_tokens(self):
        prompt = " ".join(["w"] * 10)
        assert validate_prompt(prompt, max_tokens=10) is True
        assert validate_prompt(prompt, max_tokens=9) is False

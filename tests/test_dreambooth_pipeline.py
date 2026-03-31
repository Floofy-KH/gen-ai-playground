"""Tests for src/image/dreambooth.py

Covers _inject_subject_token, the word-boundary regex logic, and generate()
without loading any model weights.
"""

from __future__ import annotations

from PIL import Image

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


# ---------------------------------------------------------------------------
# Helpers for generate() tests
# ---------------------------------------------------------------------------


def _make_fake_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(128, 0, 200))


def _make_loaded_dreambooth_pipeline(subject_token: str = "sks dog") -> DreamBoothPipeline:
    """Return a DreamBoothPipeline with _pipe already mocked (no GPU needed)."""
    from unittest.mock import MagicMock

    pipe = DreamBoothPipeline(model_id="fake/model", subject_token=subject_token, device="cpu")
    fake_output = MagicMock()
    fake_output.images = [_make_fake_image()]
    pipe._pipe = MagicMock(return_value=fake_output)
    return pipe


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestDreamBoothPipelineGenerate:
    def test_returns_pil_image(self):
        pipe = _make_loaded_dreambooth_pipeline()
        result = pipe.generate("sks dog in the park")
        assert isinstance(result, Image.Image)

    def test_subject_token_injected_by_default(self):
        pipe = _make_loaded_dreambooth_pipeline(subject_token="sks wizard")
        pipe.generate("casting a spell")
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["prompt"].startswith("sks wizard,")

    def test_inject_subject_token_false_skips_injection(self):
        pipe = _make_loaded_dreambooth_pipeline(subject_token="sks wizard")
        pipe.generate("casting a spell", inject_subject_token=False)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["prompt"] == "casting a spell"

    def test_no_injection_when_token_already_present(self):
        pipe = _make_loaded_dreambooth_pipeline(subject_token="sks dog")
        pipe.generate("sks dog on a mountain")
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["prompt"] == "sks dog on a mountain"

    def test_negative_prompt_forwarded(self):
        pipe = _make_loaded_dreambooth_pipeline()
        pipe.generate("sks dog sitting", negative_prompt="blurry")
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["negative_prompt"] == "blurry"

    def test_inference_steps_forwarded(self):
        pipe = _make_loaded_dreambooth_pipeline()
        pipe.generate("sks dog", num_inference_steps=50)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["num_inference_steps"] == 50

    def test_guidance_scale_forwarded(self):
        pipe = _make_loaded_dreambooth_pipeline()
        pipe.generate("sks dog", guidance_scale=8.0)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["guidance_scale"] == 8.0

    def test_width_and_height_forwarded(self):
        pipe = _make_loaded_dreambooth_pipeline()
        pipe.generate("sks dog", width=512, height=512)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["width"] == 512
        assert call_kwargs["height"] == 512

    def test_no_generator_when_seed_is_none(self):
        pipe = _make_loaded_dreambooth_pipeline()
        pipe.generate("sks dog", seed=None)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["generator"] is None

    def test_generator_created_when_seed_provided(self):
        import pytest

        torch = pytest.importorskip("torch")
        pipe = _make_loaded_dreambooth_pipeline()
        pipe.generate("sks dog", seed=7)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert isinstance(call_kwargs["generator"], torch.Generator)

    def test_returns_first_image_from_output(self):
        from unittest.mock import MagicMock

        pipe = DreamBoothPipeline(model_id="fake/model", subject_token="sks", device="cpu")
        expected = _make_fake_image()
        fake_output = MagicMock()
        fake_output.images = [expected, _make_fake_image()]
        pipe._pipe = MagicMock(return_value=fake_output)
        result = pipe.generate("sks subject")
        assert result is expected

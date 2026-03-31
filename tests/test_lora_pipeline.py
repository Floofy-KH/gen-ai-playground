"""Tests for src/image/lora.py

Covers the LoRAPipeline constructor validation logic, generate(), and
unload_lora() without loading any model weights (no GPU or internet
access required).  The diffusers pipeline call and torch.Generator are
mocked so that every test is pure-Python and completes in milliseconds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.image.lora import LoRAPipeline

# ---------------------------------------------------------------------------
# Helpers (mirrors test_image_pipeline.py)
# ---------------------------------------------------------------------------


def _make_fake_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(0, 128, 255))


def _make_loaded_lora_pipeline(
    lora_weights="lora.safetensors",
    lora_scale=0.8,
) -> LoRAPipeline:
    """Return a LoRAPipeline with _pipe already set to a MagicMock.

    This bypasses _load_pipeline() entirely so tests never need diffusers or a GPU.
    """
    pipe = LoRAPipeline("base", lora_weights=lora_weights, lora_scale=lora_scale, device="cpu")
    fake_output = MagicMock()
    fake_output.images = [_make_fake_image()]
    pipe._pipe = MagicMock(return_value=fake_output)
    return pipe


class TestLoRAPipelineInit:
    def test_single_weight_single_float_scale(self):
        p = LoRAPipeline("base", lora_weights="a.safetensors", lora_scale=0.7)
        assert p.lora_weights == ["a.safetensors"]
        assert p.lora_scale == [0.7]

    def test_multiple_weights_single_float_broadcasts(self):
        p = LoRAPipeline("base", lora_weights=["a.safetensors", "b.safetensors"], lora_scale=0.8)
        assert p.lora_scale == [0.8, 0.8]

    def test_multiple_weights_matching_list_scale(self):
        p = LoRAPipeline(
            "base",
            lora_weights=["a.safetensors", "b.safetensors"],
            lora_scale=[0.5, 1.0],
        )
        assert p.lora_scale == [0.5, 1.0]

    def test_mismatched_scale_raises_value_error(self):
        with pytest.raises(ValueError, match="Length of lora_scale"):
            LoRAPipeline(
                "base",
                lora_weights=["a.safetensors", "b.safetensors"],
                lora_scale=[0.5, 0.7, 1.0],  # 3 scales, 2 weights
            )

    def test_mismatched_scale_too_short_raises(self):
        with pytest.raises(ValueError, match="Length of lora_scale"):
            LoRAPipeline(
                "base",
                lora_weights=["a.safetensors", "b.safetensors", "c.safetensors"],
                lora_scale=[0.5],  # 1 scale, 3 weights
            )

    def test_error_message_mentions_pass_float(self):
        with pytest.raises(ValueError, match="single float"):
            LoRAPipeline("base", lora_weights=["a", "b"], lora_scale=[1.0, 2.0, 3.0])

    def test_default_scale_is_one(self):
        p = LoRAPipeline("base", lora_weights="single.safetensors")
        assert p.lora_scale == [1.0]

    def test_model_id_stored(self):
        p = LoRAPipeline("my-model-id", lora_weights="lora.safetensors")
        assert p.model_id == "my-model-id"

    def test_single_weight_normalised_to_list(self):
        p = LoRAPipeline("base", lora_weights="only.safetensors")
        assert isinstance(p.lora_weights, list)
        assert p.lora_weights == ["only.safetensors"]


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestLoRAPipelineGenerate:
    def test_returns_pil_image(self):
        pipe = _make_loaded_lora_pipeline()
        result = pipe.generate("a painting in impressionist style")
        assert isinstance(result, Image.Image)

    def test_default_cross_attention_uses_lora_scale(self):
        """When cross_attention_kwargs is None, default scale is lora_scale[0]."""
        pipe = _make_loaded_lora_pipeline(lora_scale=0.75)
        pipe.generate("a fox")
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["cross_attention_kwargs"] == {"scale": 0.75}

    def test_explicit_cross_attention_kwargs_forwarded(self):
        pipe = _make_loaded_lora_pipeline()
        ca = {"scale": 0.5}
        pipe.generate("a fox", cross_attention_kwargs=ca)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["cross_attention_kwargs"] == {"scale": 0.5}

    def test_prompt_forwarded_to_pipe(self):
        pipe = _make_loaded_lora_pipeline()
        pipe.generate("sks dog on a mountain")
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["prompt"] == "sks dog on a mountain"

    def test_negative_prompt_forwarded(self):
        pipe = _make_loaded_lora_pipeline()
        pipe.generate("a fox", negative_prompt="blurry, low quality")
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["negative_prompt"] == "blurry, low quality"

    def test_inference_steps_forwarded(self):
        pipe = _make_loaded_lora_pipeline()
        pipe.generate("a fox", num_inference_steps=20)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["num_inference_steps"] == 20

    def test_guidance_scale_forwarded(self):
        pipe = _make_loaded_lora_pipeline()
        pipe.generate("a fox", guidance_scale=9.0)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["guidance_scale"] == 9.0

    def test_width_and_height_forwarded(self):
        pipe = _make_loaded_lora_pipeline()
        pipe.generate("a fox", width=768, height=512)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["width"] == 768
        assert call_kwargs["height"] == 512

    def test_no_generator_when_seed_is_none(self):
        pipe = _make_loaded_lora_pipeline()
        pipe.generate("a fox", seed=None)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["generator"] is None

    def test_generator_created_when_seed_provided(self):
        torch = pytest.importorskip("torch")
        pipe = _make_loaded_lora_pipeline()
        pipe.generate("a fox", seed=42)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert isinstance(call_kwargs["generator"], torch.Generator)

    def test_returns_first_image_from_output(self):
        pipe = LoRAPipeline("base", lora_weights="lora.safetensors", device="cpu")
        expected = _make_fake_image()
        fake_output = MagicMock()
        fake_output.images = [expected, _make_fake_image()]
        pipe._pipe = MagicMock(return_value=fake_output)
        result = pipe.generate("a fox")
        assert result is expected


# ---------------------------------------------------------------------------
# unload_lora()
# ---------------------------------------------------------------------------


class TestLoRAPipelineUnloadLora:
    def test_calls_unload_lora_weights(self):
        pipe = _make_loaded_lora_pipeline()
        pipe.unload_lora()
        pipe._pipe.unload_lora_weights.assert_called_once()

    def test_no_error_when_pipe_is_none(self):
        """unload_lora() should be a no-op when the pipeline has not been loaded."""
        pipe = LoRAPipeline("base", lora_weights="lora.safetensors")
        # _pipe is None at construction time; this should not raise
        pipe.unload_lora()

    def test_unload_called_once(self):
        pipe = _make_loaded_lora_pipeline()
        pipe.unload_lora()
        pipe.unload_lora()
        assert pipe._pipe.unload_lora_weights.call_count == 2


# ---------------------------------------------------------------------------
# _load_pipeline(): LoRA weight loading (mocked, no GPU)
# ---------------------------------------------------------------------------


class TestLoRAPipelineLoad:
    """Tests for _load_pipeline() LoRA weight loading (mocked, no GPU)."""

    @staticmethod
    def _patch_base_load(pipe):
        """Return a context manager that replaces ImageGenerationPipeline._load_pipeline
        with a side-effect that assigns a MagicMock _pipe."""

        def _side_effect(self=pipe):
            pipe._pipe = MagicMock()

        from src.image.pipeline import ImageGenerationPipeline

        return patch.object(ImageGenerationPipeline, "_load_pipeline", side_effect=_side_effect)

    def test_single_lora_loaded_with_adapter_name(self):
        pipe = LoRAPipeline("base", lora_weights="my_lora.safetensors", device="cpu")
        with self._patch_base_load(pipe):
            pipe._load_pipeline()
        pipe._pipe.load_lora_weights.assert_called_once_with(
            "my_lora.safetensors", adapter_name="adapter_0"
        )

    def test_multiple_loras_each_loaded(self):
        pipe = LoRAPipeline(
            "base",
            lora_weights=["lora_a.safetensors", "lora_b.safetensors"],
            lora_scale=[0.5, 1.0],
            device="cpu",
        )
        with self._patch_base_load(pipe):
            pipe._load_pipeline()
        calls = pipe._pipe.load_lora_weights.call_args_list
        assert len(calls) == 2
        assert calls[0].args == ("lora_a.safetensors",)
        assert calls[0].kwargs == {"adapter_name": "adapter_0"}
        assert calls[1].args == ("lora_b.safetensors",)
        assert calls[1].kwargs == {"adapter_name": "adapter_1"}

    def test_multiple_loras_set_adapters_called(self):
        pipe = LoRAPipeline(
            "base",
            lora_weights=["lora_a.safetensors", "lora_b.safetensors"],
            lora_scale=[0.5, 1.0],
            device="cpu",
        )
        with self._patch_base_load(pipe):
            pipe._load_pipeline()
        pipe._pipe.set_adapters.assert_called_once_with(
            ["adapter_0", "adapter_1"], adapter_weights=[0.5, 1.0]
        )

    def test_single_lora_does_not_call_set_adapters(self):
        pipe = LoRAPipeline("base", lora_weights="lora.safetensors", device="cpu")
        with self._patch_base_load(pipe):
            pipe._load_pipeline()
        pipe._pipe.set_adapters.assert_not_called()

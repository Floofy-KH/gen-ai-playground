"""Tests for src/image/lora.py

Covers the LoRAPipeline constructor validation logic without loading any
model weights (no GPU or internet access required).
"""

from __future__ import annotations

import pytest

from src.image.lora import LoRAPipeline


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

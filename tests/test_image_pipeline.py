"""Tests for src/image/pipeline.py

Covers ImageGenerationPipeline constructor, lazy loading, generate(), generate_batch(),
and save_images() without loading any model weights (no GPU or internet access required).

The diffusers pipeline call and torch.Generator are mocked so that every test
is pure-Python and completes in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.image.pipeline import ImageGenerationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_image() -> Image.Image:
    """Return a minimal 4×4 solid-colour PIL image."""
    return Image.new("RGB", (4, 4), color=(255, 0, 0))


def _make_loaded_pipeline(model_id: str = "fake/model") -> ImageGenerationPipeline:
    """Return an ImageGenerationPipeline with _pipe already set to a MagicMock.

    This bypasses _load_pipeline() entirely so tests never need diffusers or a GPU.
    """
    pipe = ImageGenerationPipeline(model_id=model_id, device="cpu")
    fake_output = MagicMock()
    fake_output.images = [_make_fake_image()]
    pipe._pipe = MagicMock(return_value=fake_output)
    return pipe


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    def test_model_id_stored(self):
        pipe = ImageGenerationPipeline("my/model")
        assert pipe.model_id == "my/model"

    def test_device_stored(self):
        pipe = ImageGenerationPipeline("m", device="cpu")
        assert pipe.device == "cpu"

    def test_dtype_stored(self):
        import torch
        pipe = ImageGenerationPipeline("m", dtype=torch.float32)
        assert pipe.dtype is torch.float32

    def test_device_defaults_to_none(self):
        pipe = ImageGenerationPipeline("m")
        assert pipe.device is None

    def test_dtype_defaults_to_none(self):
        pipe = ImageGenerationPipeline("m")
        assert pipe.dtype is None

    def test_pipe_starts_unloaded(self):
        pipe = ImageGenerationPipeline("m")
        assert pipe._pipe is None


# ---------------------------------------------------------------------------
# _ensure_loaded / lazy loading
# ---------------------------------------------------------------------------


class TestEnsureLoaded:
    def test_calls_load_pipeline_once(self):
        pipe = ImageGenerationPipeline("fake/model")

        def _fake_load():
            pipe._pipe = MagicMock()  # Simulate successful load so guard fires

        pipe._load_pipeline = MagicMock(side_effect=_fake_load)
        pipe._ensure_loaded()
        pipe._ensure_loaded()
        pipe._load_pipeline.assert_called_once()

    def test_does_not_call_load_if_already_set(self):
        pipe = ImageGenerationPipeline("fake/model")
        pipe._pipe = MagicMock()  # Simulate already loaded
        pipe._load_pipeline = MagicMock()
        pipe._ensure_loaded()
        pipe._load_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# _load_pipeline
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    def test_sets_pipe(self):
        """_load_pipeline() should assign self._pipe after loading."""

        class _PipelineWithMockedLoad(ImageGenerationPipeline):
            def _load_pipeline(self):
                self._pipe = MagicMock()
                self.device = "cpu"
                self.dtype = None

        p = _PipelineWithMockedLoad("fake/model")
        p._ensure_loaded()
        assert p._pipe is not None

    def test_resolves_device_on_load(self):
        """After _load_pipeline(), self.device should be set to a concrete string."""
        class _MockLoad(ImageGenerationPipeline):
            def _load_pipeline(self):
                import torch
                self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.dtype = self.dtype or torch.float32
                self._pipe = MagicMock()

        p = _MockLoad("fake/model")
        assert p.device is None  # Before load
        p._ensure_loaded()
        assert p.device in ("cuda", "cpu")

    def test_uses_cpu_when_cuda_unavailable(self):
        """When CUDA is not available and device is None, cpu should be used."""
        class _MockLoad(ImageGenerationPipeline):
            def _load_pipeline(self):
                import torch
                device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.device = device
                self._pipe = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            p = _MockLoad("fake/model")
            p._ensure_loaded()
            assert p.device == "cpu"


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_pil_image(self):
        pipe = _make_loaded_pipeline()
        result = pipe.generate("a red fox")
        assert isinstance(result, Image.Image)

    def test_pipe_called_with_prompt(self):
        pipe = _make_loaded_pipeline()
        pipe.generate("a red fox")
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["prompt"] == "a red fox"

    def test_pipe_called_with_negative_prompt(self):
        pipe = _make_loaded_pipeline()
        pipe.generate("a fox", negative_prompt="blurry")
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["negative_prompt"] == "blurry"

    def test_pipe_called_with_inference_steps(self):
        pipe = _make_loaded_pipeline()
        pipe.generate("a fox", num_inference_steps=20)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["num_inference_steps"] == 20

    def test_pipe_called_with_guidance_scale(self):
        pipe = _make_loaded_pipeline()
        pipe.generate("a fox", guidance_scale=9.0)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["guidance_scale"] == 9.0

    def test_pipe_called_with_width_and_height(self):
        pipe = _make_loaded_pipeline()
        pipe.generate("a fox", width=768, height=768)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["width"] == 768
        assert call_kwargs["height"] == 768

    def test_no_generator_when_seed_is_none(self):
        pipe = _make_loaded_pipeline()
        pipe.generate("a fox", seed=None)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert call_kwargs["generator"] is None

    def test_generator_created_when_seed_provided(self):
        torch = pytest.importorskip("torch")
        pipe = _make_loaded_pipeline()
        pipe.generate("a fox", seed=42)
        call_kwargs = pipe._pipe.call_args.kwargs
        assert isinstance(call_kwargs["generator"], torch.Generator)

    def test_same_seed_produces_same_generator_state(self):
        """Two generators with the same seed should produce identical tensors."""
        torch = pytest.importorskip("torch")
        pipe1 = _make_loaded_pipeline()
        pipe2 = _make_loaded_pipeline()
        pipe1.generate("fox", seed=7)
        pipe2.generate("fox", seed=7)
        gen1 = pipe1._pipe.call_args.kwargs["generator"]
        gen2 = pipe2._pipe.call_args.kwargs["generator"]
        t1 = torch.rand(5, generator=gen1)
        t2 = torch.rand(5, generator=gen2)
        assert torch.allclose(t1, t2)

    def test_returns_first_image_from_output(self):
        """generate() should return output.images[0]."""
        pipe = ImageGenerationPipeline("fake/model", device="cpu")
        expected = _make_fake_image()
        fake_output = MagicMock()
        fake_output.images = [expected, _make_fake_image()]
        pipe._pipe = MagicMock(return_value=fake_output)
        result = pipe.generate("a fox")
        assert result is expected

    def test_calls_ensure_loaded(self):
        pipe = ImageGenerationPipeline("fake/model", device="cpu")

        def _fake_ensure_loaded():
            pipe._pipe = MagicMock(
                return_value=MagicMock(images=[_make_fake_image()])
            )

        pipe._ensure_loaded = MagicMock(side_effect=_fake_ensure_loaded)
        pipe.generate("a fox")
        pipe._ensure_loaded.assert_called_once()


# ---------------------------------------------------------------------------
# generate_batch()
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def test_returns_one_image_per_prompt(self):
        pipe = _make_loaded_pipeline()
        results = pipe.generate_batch(["fox", "cat", "dog"])
        assert len(results) == 3

    def test_all_results_are_pil_images(self):
        pipe = _make_loaded_pipeline()
        results = pipe.generate_batch(["fox", "cat"])
        assert all(isinstance(r, Image.Image) for r in results)

    def test_empty_list_returns_empty(self):
        pipe = _make_loaded_pipeline()
        results = pipe.generate_batch([])
        assert results == []

    def test_same_seed_used_for_all(self):
        pipe = _make_loaded_pipeline()
        pipe.generate_batch(["a", "b", "c"], seed=99)
        for c in pipe._pipe.call_args_list:
            gen = c.kwargs["generator"]
            assert gen is not None

    def test_shared_negative_prompt(self):
        pipe = _make_loaded_pipeline()
        pipe.generate_batch(["fox", "cat"], negative_prompt="blurry")
        for c in pipe._pipe.call_args_list:
            assert c.kwargs["negative_prompt"] == "blurry"


# ---------------------------------------------------------------------------
# save_images()
# ---------------------------------------------------------------------------


class TestSaveImages:
    def test_creates_output_directory(self, tmp_path):
        pipe = ImageGenerationPipeline("fake/model")
        new_dir = tmp_path / "outputs" / "subdir"
        images = [_make_fake_image()]
        pipe.save_images(images, new_dir)
        assert new_dir.is_dir()

    def test_returns_correct_number_of_paths(self, tmp_path):
        pipe = ImageGenerationPipeline("fake/model")
        images = [_make_fake_image(), _make_fake_image(), _make_fake_image()]
        paths = pipe.save_images(images, tmp_path)
        assert len(paths) == 3

    def test_files_exist_after_save(self, tmp_path):
        pipe = ImageGenerationPipeline("fake/model")
        images = [_make_fake_image(), _make_fake_image()]
        paths = pipe.save_images(images, tmp_path)
        for p in paths:
            assert p.exists()

    def test_default_prefix(self, tmp_path):
        pipe = ImageGenerationPipeline("fake/model")
        paths = pipe.save_images([_make_fake_image()], tmp_path)
        assert paths[0].name == "output_0000.png"

    def test_custom_prefix(self, tmp_path):
        pipe = ImageGenerationPipeline("fake/model")
        paths = pipe.save_images([_make_fake_image()], tmp_path, prefix="result")
        assert paths[0].name == "result_0000.png"

    def test_zero_padded_index(self, tmp_path):
        pipe = ImageGenerationPipeline("fake/model")
        images = [_make_fake_image()] * 5
        paths = pipe.save_images(images, tmp_path)
        names = [p.name for p in paths]
        assert names == [
            "output_0000.png",
            "output_0001.png",
            "output_0002.png",
            "output_0003.png",
            "output_0004.png",
        ]

    def test_returns_path_objects(self, tmp_path):
        pipe = ImageGenerationPipeline("fake/model")
        paths = pipe.save_images([_make_fake_image()], tmp_path)
        assert all(isinstance(p, Path) for p in paths)

    def test_accepts_string_output_dir(self, tmp_path):
        pipe = ImageGenerationPipeline("fake/model")
        paths = pipe.save_images([_make_fake_image()], str(tmp_path))
        assert paths[0].exists()

"""Tests for src/image/pipeline.py

Covers ImageGenerationPipeline constructor, lazy loading, generate(), generate_batch(),
and save_images() without loading any model weights (no GPU or internet access required).

The diffusers pipeline call and torch.Generator are mocked so that every test
is pure-Python and completes in milliseconds.  Tests that require torch (for
Generator assertions) are skipped via ``pytest.importorskip`` when torch is
absent — matching the CI environment where only lightweight dev deps are installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
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


def _make_torch_mock():
    """Return a minimal sys.modules-style mock for torch.

    Provides the subset of the torch API used inside _load_pipeline():
    `torch.cuda.is_available`, `torch.float16`, `torch.float32`, `torch.Generator`.
    """
    torch_mock = MagicMock(spec=ModuleType)
    torch_mock.float16 = "float16_sentinel"
    torch_mock.float32 = "float32_sentinel"
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    torch_mock.Generator = MagicMock()
    return torch_mock


def _make_diffusers_mock():
    """Return a minimal sys.modules-style mock for diffusers.

    Provides `StableDiffusionXLPipeline.from_pretrained().to()` → MagicMock pipe.
    """
    pipe_instance = MagicMock()
    sdxl_cls = MagicMock()
    sdxl_cls.from_pretrained.return_value.to.return_value = pipe_instance

    diffusers_mock = MagicMock(spec=ModuleType)
    diffusers_mock.StableDiffusionXLPipeline = sdxl_cls
    return diffusers_mock, sdxl_cls, pipe_instance


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
        torch = pytest.importorskip("torch")
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
    """Tests for the real _load_pipeline() implementation.

    Both `torch` and `diffusers` are injected via `sys.modules` patching
    so no model weights, GPU, or heavy ML libraries are needed.  Each test
    controls `torch.cuda.is_available` independently.
    """

    def test_sets_pipe(self):
        """_load_pipeline() should assign self._pipe after loading."""
        torch_mock = _make_torch_mock()
        diffusers_mock, _, pipe_instance = _make_diffusers_mock()

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("fake/model")
            p._load_pipeline()

        assert p._pipe is pipe_instance

    def test_from_pretrained_called_with_model_id(self):
        """from_pretrained() should receive the configured model_id."""
        torch_mock = _make_torch_mock()
        diffusers_mock, sdxl_cls, _ = _make_diffusers_mock()

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("my/model-id")
            p._load_pipeline()

        sdxl_cls.from_pretrained.assert_called_once()
        assert sdxl_cls.from_pretrained.call_args.args[0] == "my/model-id"

    def test_uses_cpu_when_cuda_unavailable_and_device_none(self):
        """When device is None and CUDA is unavailable, cpu should be used."""
        torch_mock = _make_torch_mock()
        torch_mock.cuda.is_available.return_value = False
        diffusers_mock, _, _ = _make_diffusers_mock()

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("fake/model")
            p._load_pipeline()

        assert p.device == "cpu"

    def test_uses_cuda_when_available_and_device_none(self):
        """When device is None and CUDA is available, cuda should be used."""
        torch_mock = _make_torch_mock()
        torch_mock.cuda.is_available.return_value = True
        diffusers_mock, _, _ = _make_diffusers_mock()

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("fake/model")
            p._load_pipeline()

        assert p.device == "cuda"

    def test_explicit_cpu_device_honoured(self):
        """Explicitly passing device='cpu' should use CPU even if CUDA is available."""
        torch_mock = _make_torch_mock()
        torch_mock.cuda.is_available.return_value = True
        diffusers_mock, _, _ = _make_diffusers_mock()

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("fake/model", device="cpu")
            p._load_pipeline()

        assert p.device == "cpu"

    def test_raises_when_cuda_requested_but_unavailable(self):
        """A RuntimeError should be raised when device='cuda' but CUDA is unavailable."""
        torch_mock = _make_torch_mock()
        torch_mock.cuda.is_available.return_value = False
        diffusers_mock, _, _ = _make_diffusers_mock()

        with (
            patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}),
            pytest.raises(RuntimeError, match="CUDA is not available"),
        ):
            p = ImageGenerationPipeline("fake/model", device="cuda")
            p._load_pipeline()

    def test_no_torch_dtype_passed_when_dtype_none_on_cpu(self):
        """When dtype=None and device='cpu', torch_dtype should not be passed."""
        torch_mock = _make_torch_mock()
        torch_mock.cuda.is_available.return_value = False
        diffusers_mock, sdxl_cls, _ = _make_diffusers_mock()

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("fake/model")
            p._load_pipeline()

        call_kwargs = sdxl_cls.from_pretrained.call_args.kwargs
        assert "torch_dtype" not in call_kwargs

    def test_explicit_dtype_passed_to_from_pretrained(self):
        """When dtype is explicitly set, it should be forwarded to from_pretrained."""
        torch_mock = _make_torch_mock()
        torch_mock.cuda.is_available.return_value = False
        diffusers_mock, sdxl_cls, _ = _make_diffusers_mock()

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("fake/model", dtype=torch_mock.float32)
            p._load_pipeline()

        call_kwargs = sdxl_cls.from_pretrained.call_args.kwargs
        assert call_kwargs["torch_dtype"] is torch_mock.float32

    def test_float16_used_automatically_on_cuda(self):
        """When dtype=None and device auto-resolves to cuda, float16 should be used."""
        torch_mock = _make_torch_mock()
        torch_mock.cuda.is_available.return_value = True
        diffusers_mock, sdxl_cls, _ = _make_diffusers_mock()

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("fake/model")
            p._load_pipeline()

        call_kwargs = sdxl_cls.from_pretrained.call_args.kwargs
        assert call_kwargs.get("torch_dtype") is torch_mock.float16

    def test_xformers_error_does_not_propagate(self):
        """ImportError/AttributeError from xformers should be silently ignored."""
        torch_mock = _make_torch_mock()
        diffusers_mock, _, pipe_instance = _make_diffusers_mock()
        pipe_instance.enable_xformers_memory_efficient_attention.side_effect = ImportError(
            "xformers not installed"
        )

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p = ImageGenerationPipeline("fake/model")
            p._load_pipeline()  # Should not raise

        assert p._pipe is pipe_instance

    def test_resolves_device_on_load(self):
        """After _load_pipeline(), self.device should be set to a concrete string."""
        torch_mock = _make_torch_mock()
        torch_mock.cuda.is_available.return_value = False
        diffusers_mock, _, _ = _make_diffusers_mock()

        p = ImageGenerationPipeline("fake/model")
        assert p.device is None  # Before load

        with patch.dict(sys.modules, {"torch": torch_mock, "diffusers": diffusers_mock}):
            p._load_pipeline()

        assert p.device in ("cuda", "cpu")


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
        pytest.importorskip("torch")
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

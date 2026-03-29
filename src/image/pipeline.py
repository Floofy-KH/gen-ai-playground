"""
src/image/pipeline.py

Base image generation pipeline wrapper around Hugging Face Diffusers.

This module provides a simple, unified interface for text-to-image generation
using Stable Diffusion (or compatible) models loaded via the ``diffusers``
library.  More specialised pipelines (IP-Adapter, ControlNet, etc.) inherit
from or compose with this base class.

References:
    - Diffusers StableDiffusionPipeline:
      https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
    - Stable Diffusion paper (LDM): https://arxiv.org/abs/2112.10752
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from PIL import Image


class ImageGenerationPipeline:
    """Thin wrapper around a Diffusers ``StableDiffusionPipeline`` (or SDXL
    equivalent) that adds convenience helpers for seed locking, batching, and
    saving outputs.

    Args:
        model_id: Hugging Face Hub model ID or local path to a diffusers
            model directory (e.g. ``"runwayml/stable-diffusion-v1-5"``).
        device: PyTorch device string.  Defaults to ``"cuda"`` if available,
            otherwise ``"cpu"``.
        dtype: Optional torch dtype for the model weights (e.g.
            ``torch.float16``).  Defaults to ``None`` (use model default).

    Example::

        pipeline = ImageGenerationPipeline("runwayml/stable-diffusion-v1-5")
        image = pipeline.generate("a red fox in a snowy forest", seed=42)
        image.save("fox.png")
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype=None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._pipe = None  # Lazy-loaded on first call to `generate`

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Load the underlying Diffusers pipeline from the Hub or disk.

        Called automatically on the first call to :meth:`generate`.
        Override this method in subclasses to load different pipeline types.
        """
        import torch
        from diffusers import StableDiffusionXLPipeline

        if self.device is not None and self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Device '{self.device}' was requested but CUDA is not available on this machine. "
                    "Pass device='cpu' or omit the device argument to use CPU automatically."
                )
            device = self.device
        else:
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {}
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype
        elif device.startswith("cuda"):
            kwargs["torch_dtype"] = torch.float16

        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            **kwargs,
        ).to(device)

        # Enable memory-efficient attention when xformers is available
        # Cast VAE to float32 for numerical stability with float16 pipelines
        # (replaces the deprecated upcast_vae behaviour in SDXL)
        if self.dtype == torch.float16:
            self._pipe.vae.to(torch.float32)

        # Enable memory-efficient attention when xformers is available
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except (ImportError, AttributeError, ValueError):
            pass

        # Persist resolved device and dtype so subclasses and generate() can
        # reference them without re-computing the defaults.
        self.device = device
        self.dtype = kwargs.get("torch_dtype")

    def _ensure_loaded(self) -> None:
        """Ensure the pipeline is loaded before inference."""
        if self._pipe is None:
            self._load_pipeline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """Generate a single image from a text prompt.

        Args:
            prompt: Positive text prompt describing the desired image.
            negative_prompt: Optional negative text prompt to suppress
                unwanted content.
            seed: Random seed for reproducibility.  Pass the same seed with
                the same prompt to get identical outputs.
            num_inference_steps: Number of denoising steps.  More steps
                generally yield higher quality but take longer.
            guidance_scale: Classifier-free guidance scale.  Higher values
                follow the prompt more closely at the cost of diversity.
            width: Output image width in pixels.
            height: Output image height in pixels.

        Returns:
            A ``PIL.Image.Image`` containing the generated output.
        """
        self._ensure_loaded()

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
        return output.images[0]

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> List[Image.Image]:
        """Generate a batch of images, one per prompt.

        When ``seed`` is provided every prompt is generated with the *same*
        seed so that outputs share a common visual "fingerprint" (useful for
        cross-prompt consistency experiments).

        Args:
            prompts: List of text prompts.
            negative_prompt: Shared negative prompt applied to all images.
            seed: Fixed seed for all generations.  See :meth:`generate`.
            num_inference_steps: Denoising steps per image.
            guidance_scale: Classifier-free guidance scale.
            width: Output width in pixels.
            height: Output height in pixels.

        Returns:
            List of ``PIL.Image.Image`` objects, one per prompt.
        """
        return [
            self.generate(
                prompt=p,
                negative_prompt=negative_prompt,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            for p in prompts
        ]

    def save_images(
        self,
        images: List[Image.Image],
        output_dir: Union[str, Path],
        prefix: str = "output",
    ) -> List[Path]:
        """Save a list of images to ``output_dir``.

        Args:
            images: List of PIL images to save.
            output_dir: Directory path where images will be written.
            prefix: Filename prefix.  Files are named
                ``{prefix}_{index:04d}.png``.

        Returns:
            List of :class:`~pathlib.Path` objects pointing to saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            path = output_dir / f"{prefix}_{i:04d}.png"
            img.save(path)
            paths.append(path)
        return paths

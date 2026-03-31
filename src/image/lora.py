"""
src/image/lora.py

LoRA (Low-Rank Adaptation) adapter loading and inference helpers.

LoRA adapters are small, efficient fine-tuned weight deltas that can be
loaded on top of a frozen base model at inference time.  Multiple LoRA
adapters can be fused or applied simultaneously for style and subject
consistency.

References:
    - LoRA paper: https://arxiv.org/abs/2106.09685
    - Hugging Face PEFT (LoRA): https://huggingface.co/docs/peft/conceptual_guides/lora
    - Diffusers LoRA inference guide:
      https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from PIL import Image

from src.image.pipeline import ImageGenerationPipeline


class LoRAPipeline(ImageGenerationPipeline):
    """Image generation pipeline with one or more LoRA adapters loaded.

    LoRA adapters are applied on top of a frozen base model.  This class
    supports loading a single adapter or fusing multiple adapters with
    configurable per-adapter weights.

    Args:
        base_model_id: Hub ID of the base diffusion model.
        lora_weights: Hub ID, local path, or list of Hub IDs / paths for
            the LoRA adapter(s) to load.
        lora_scale: Blending weight for the LoRA adapter (0 = no effect,
            1 = full strength).  When multiple adapters are provided, pass a
            list of per-adapter scales.
        device: PyTorch device string.
        dtype: Optional torch dtype.

    Example::

        from src.image.lora import LoRAPipeline

        pipe = LoRAPipeline(
            base_model_id="runwayml/stable-diffusion-v1-5",
            lora_weights="path/to/my_lora.safetensors",
            lora_scale=0.8,
        )
        image = pipe.generate("sks person hiking in the mountains", seed=0)
        image.save("out.png")
    """

    def __init__(
        self,
        base_model_id: str,
        lora_weights: Union[str, List[str]],
        lora_scale: Union[float, List[float]] = 1.0,
        device: Optional[str] = None,
        dtype=None,
    ) -> None:
        super().__init__(model_id=base_model_id, device=device, dtype=dtype)
        self.lora_weights = lora_weights if isinstance(lora_weights, list) else [lora_weights]
        # Broadcast a single scale value to all adapters, or validate that
        # per-adapter scales match the number of adapter weight paths.
        if isinstance(lora_scale, list):
            if len(lora_scale) != len(self.lora_weights):
                raise ValueError(
                    f"Length of lora_scale ({len(lora_scale)}) must match "
                    f"length of lora_weights ({len(self.lora_weights)}) when "
                    "providing per-adapter scales.  Pass a single float to "
                    "apply the same scale to all adapters."
                )
            self.lora_scale = lora_scale
        else:
            # Broadcast the single scale to every adapter
            self.lora_scale = [lora_scale] * len(self.lora_weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Load the base model and apply LoRA adapter(s).

        Implementation outline::

            from diffusers import StableDiffusionPipeline
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = self.dtype or torch.float16

            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(device)

            for weights in self.lora_weights:
                self._pipe.load_lora_weights(weights)

            # Optionally fuse all loaded LoRA weights into the model for
            # faster inference:
            # self._pipe.fuse_lora(lora_scale=self.lora_scale[0])
        """
        super()._load_pipeline()

        for i, weights in enumerate(self.lora_weights):
            adapter_name = f"adapter_{i}"
            self._pipe.load_lora_weights(weights, adapter_name=adapter_name)

        if len(self.lora_weights) > 1:
            adapter_names = [f"adapter_{i}" for i in range(len(self.lora_weights))]
            self._pipe.set_adapters(adapter_names, adapter_weights=self.lora_scale)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(  # type: ignore[override]
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        cross_attention_kwargs: Optional[Dict] = None,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """Generate an image using the LoRA-augmented pipeline.

        Args:
            prompt: Text prompt.
            negative_prompt: Optional negative prompt.
            cross_attention_kwargs: Extra kwargs forwarded to the cross-
                attention layers (e.g. ``{"scale": 0.8}`` to adjust LoRA
                scale at runtime without fusing).
            seed: Random seed for reproducibility.
            num_inference_steps: Denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Output width in pixels.
            height: Output height in pixels.

        Returns:
            Generated ``PIL.Image.Image``.
        """
        self._ensure_loaded()

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=self.device).manual_seed(seed)

        ca_kwargs = cross_attention_kwargs
        if ca_kwargs is None:
            ca_kwargs = {"scale": self.lora_scale[0]}

        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cross_attention_kwargs=ca_kwargs,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
        return output.images[0]

    def unload_lora(self) -> None:
        """Remove all loaded LoRA adapters, restoring the original base model.

        Useful when you want to switch between LoRA adapters without
        re-instantiating the full pipeline.
        """
        if self._pipe is not None:
            self._pipe.unload_lora_weights()

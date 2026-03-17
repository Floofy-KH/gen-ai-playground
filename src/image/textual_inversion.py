"""
src/image/textual_inversion.py

Textual Inversion inference utilities.

Textual Inversion trains a new *pseudo-word* token whose embedding captures a
visual concept (character, style, object) from a small set of reference images.
At inference time the learned embedding is loaded into the text encoder and the
pseudo-word is used in the prompt like any regular word.

References:
    - Textual Inversion paper: https://arxiv.org/abs/2208.01618
    - Diffusers Textual Inversion guide:
      https://huggingface.co/docs/diffusers/training/text_inversion
    - Diffusers inference with Textual Inversion:
      https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters#textual-inversion
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

from src.image.pipeline import ImageGenerationPipeline


class TextualInversionPipeline(ImageGenerationPipeline):
    """Image generation pipeline with one or more Textual Inversion embeddings
    loaded into the text encoder.

    Args:
        base_model_id: Hub ID or local path to the base diffusion model.
        embedding_paths: Path(s) to ``.bin`` or ``.pt`` embedding files
            containing the learned pseudo-word token vectors.
        tokens: The placeholder token string(s) corresponding to each
            embedding file (e.g. ``"<my-style>"``).  Must match the token(s)
            used during training.
        device: PyTorch device string.
        dtype: Optional torch dtype.

    Example::

        from src.image.textual_inversion import TextualInversionPipeline

        pipe = TextualInversionPipeline(
            base_model_id="runwayml/stable-diffusion-v1-5",
            embedding_paths=["embeddings/my_style.bin"],
            tokens=["<my-style>"],
        )
        image = pipe.generate(
            "a <my-style> painting of a sunset over the ocean", seed=7
        )
        image.save("out.png")
    """

    def __init__(
        self,
        base_model_id: str,
        embedding_paths: Union[str, Path, List[Union[str, Path]]],
        tokens: Union[str, List[str]],
        device: Optional[str] = None,
        dtype=None,
    ) -> None:
        super().__init__(model_id=base_model_id, device=device, dtype=dtype)
        self.embedding_paths = (
            embedding_paths if isinstance(embedding_paths, list) else [embedding_paths]
        )
        self.tokens = tokens if isinstance(tokens, list) else [tokens]

        if len(self.embedding_paths) != len(self.tokens):
            raise ValueError(
                f"Length of embedding_paths ({len(self.embedding_paths)}) must "
                f"match length of tokens ({len(self.tokens)})."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Load the base model and inject Textual Inversion embedding(s).

        Implementation outline::

            from diffusers import StableDiffusionPipeline
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = self.dtype or torch.float16

            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(device)

            for path, token in zip(self.embedding_paths, self.tokens):
                self._pipe.load_textual_inversion(str(path), token=token)
        """
        # TODO: Implement as described in the docstring above.
        raise NotImplementedError("Stub: implement _load_pipeline() for Textual Inversion.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(  # type: ignore[override]
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """Generate an image using the Textual Inversion-augmented pipeline.

        Include the placeholder token(s) in the prompt to activate the learned
        concept (e.g. ``"a <my-style> landscape"``).

        Args:
            prompt: Text prompt, typically containing one or more of the
                loaded placeholder tokens.
            negative_prompt: Optional negative prompt.
            seed: Random seed for reproducibility.
            num_inference_steps: Denoising steps.
            guidance_scale: Classifier-free guidance scale.
            width: Output width in pixels.
            height: Output height in pixels.

        Returns:
            Generated ``PIL.Image.Image``.
        """
        self._ensure_loaded()
        # TODO: Build generator from seed, call self._pipe, return images[0].
        raise NotImplementedError("Stub: implement generate() for Textual Inversion.")

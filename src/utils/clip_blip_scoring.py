"""
src/utils/clip_blip_scoring.py

CLIP and BLIP embedding similarity scoring for generated image evaluation.

This module provides two scorers:

- :class:`CLIPScorer`: Measures **image-image** and **image-text** cosine
  similarity using OpenAI CLIP embeddings.  Higher values indicate that two
  images (or an image and a text caption) are semantically similar.

- :class:`BLIPScorer`: Uses Salesforce BLIP to generate a natural-language
  caption for an image and then optionally compares that caption to a
  reference text using string or embedding similarity.

These scores are the backbone of automated consistency filtering: generate a
batch of images, score each against a reference, and keep only those above a
threshold.

References:
    - CLIP paper: https://arxiv.org/abs/2103.00020
    - BLIP paper: https://arxiv.org/abs/2201.12086
    - Hugging Face CLIP model: https://huggingface.co/openai/clip-vit-large-patch14
    - Hugging Face BLIP model: https://huggingface.co/Salesforce/blip-image-captioning-large
    - OpenAI CLIP GitHub: https://github.com/openai/CLIP
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image


class CLIPScorer:
    """Compute CLIP-based cosine similarity between images and/or text.

    Uses the ``transformers`` library's CLIP implementation (no dependency on
    the separate ``clip`` package, though that also works).

    Args:
        model_id: Hugging Face Hub ID for the CLIP model.
        device: PyTorch device (``"cuda"`` or ``"cpu"``).

    Example::

        from PIL import Image
        from src.utils.clip_blip_scoring import CLIPScorer

        scorer = CLIPScorer()
        img_a = Image.open("reference.png")
        img_b = Image.open("generated.png")
        sim = scorer.image_similarity(img_a, img_b)
        print(f"CLIP image similarity: {sim:.4f}")

        text_score = scorer.image_text_similarity(img_b, "a dog on the moon")
        print(f"CLIP image-text similarity: {text_score:.4f}")
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load CLIP model and processor from the Hub (lazy init).

        Implementation outline::

            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id).to(device)
            self._model.eval()
            self.device = device
        """
        # TODO: Implement as described above.
        raise NotImplementedError("Stub: implement _load_model() for CLIPScorer.")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load_model()

    def _get_image_embedding(self, image: Image.Image):
        """Extract normalised CLIP image embedding.

        Args:
            image: PIL image.

        Returns:
            Normalised embedding tensor of shape ``(1, embedding_dim)``.
        """
        self._ensure_loaded()
        # TODO: Process image, run through CLIP vision encoder,
        #       L2-normalise, return tensor.
        raise NotImplementedError("Stub: implement _get_image_embedding().")

    def _get_text_embedding(self, text: str):
        """Extract normalised CLIP text embedding.

        Args:
            text: Input string.

        Returns:
            Normalised embedding tensor of shape ``(1, embedding_dim)``.
        """
        self._ensure_loaded()
        # TODO: Tokenise text, run through CLIP text encoder,
        #       L2-normalise, return tensor.
        raise NotImplementedError("Stub: implement _get_text_embedding().")

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two embedding tensors.

        Args:
            a: First embedding tensor (normalised).
            b: Second embedding tensor (normalised).

        Returns:
            Scalar cosine similarity in ``[-1, 1]``.
        """
        # For normalised vectors: cosine_sim = dot product
        return float((a * b).sum().item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def image_similarity(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
    ) -> float:
        """Compute CLIP cosine similarity between two images.

        Args:
            image_a: First PIL image (typically the reference).
            image_b: Second PIL image (typically the generated output).

        Returns:
            Cosine similarity in ``[-1, 1]`` (higher = more similar).
        """
        emb_a = self._get_image_embedding(image_a)
        emb_b = self._get_image_embedding(image_b)
        return self._cosine_similarity(emb_a, emb_b)

    def image_text_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> float:
        """Compute CLIP cosine similarity between an image and a text string.

        Args:
            image: PIL image to evaluate.
            text: Reference text description.

        Returns:
            Cosine similarity in ``[-1, 1]``.
        """
        img_emb = self._get_image_embedding(image)
        txt_emb = self._get_text_embedding(text)
        return self._cosine_similarity(img_emb, txt_emb)

    def text_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """Compute CLIP cosine similarity between two text strings.

        Args:
            text_a: First text string.
            text_b: Second text string.

        Returns:
            Cosine similarity in ``[-1, 1]`` (higher = more similar).
        """
        emb_a = self._get_text_embedding(text_a)
        emb_b = self._get_text_embedding(text_b)
        return self._cosine_similarity(emb_a, emb_b)

    def batch_image_similarity(
        self,
        reference: Image.Image,
        images: List[Image.Image],
    ) -> List[float]:
        """Compute CLIP similarity of each image in ``images`` against
        ``reference``.

        Args:
            reference: Reference PIL image.
            images: List of generated images to score.

        Returns:
            List of cosine similarity scores.
        """
        ref_emb = self._get_image_embedding(reference)
        return [self._cosine_similarity(ref_emb, self._get_image_embedding(img)) for img in images]


class BLIPScorer:
    """Generate image captions using BLIP and compare them to reference text.

    Args:
        model_id: Hugging Face Hub ID for the BLIP model.
        device: PyTorch device.

    Example::

        from PIL import Image
        from src.utils.clip_blip_scoring import BLIPScorer

        scorer = BLIPScorer()
        image = Image.open("generated.png")
        caption = scorer.caption(image)
        print(f"Generated caption: {caption}")

        similarity = scorer.caption_similarity(image, "an astronaut on the moon")
        print(f"Caption similarity: {similarity:.3f}")
    """

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-large",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load BLIP model and processor (lazy init).

        Implementation outline::

            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            ).to(device)
            self._model.eval()
            self.device = device
        """
        # TODO: Implement as described above.
        raise NotImplementedError("Stub: implement _load_model() for BLIPScorer.")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def caption(
        self,
        image: Image.Image,
        max_new_tokens: int = 50,
    ) -> str:
        """Generate a natural-language caption for an image using BLIP.

        Args:
            image: PIL image to caption.
            max_new_tokens: Maximum length of the generated caption (in tokens).

        Returns:
            Caption string.
        """
        self._ensure_loaded()
        # TODO: Process image through BLIP, decode generated tokens, return string.
        raise NotImplementedError("Stub: implement caption().")

    def caption_similarity(
        self,
        image: Image.Image,
        reference_text: str,
        use_clip_for_comparison: bool = True,
    ) -> float:
        """Score an image's BLIP caption against a reference text description.

        If ``use_clip_for_comparison`` is ``True``, the comparison is performed
        in CLIP embedding space for semantic robustness.  Otherwise, a simple
        token overlap (Jaccard) coefficient is used.

        Args:
            image: PIL image to evaluate.
            reference_text: Target description to compare against.
            use_clip_for_comparison: Use CLIP embeddings for comparison when
                ``True``; use lexical Jaccard similarity when ``False``.

        Returns:
            Similarity score in ``[0, 1]``.
        """
        generated_caption = self.caption(image)

        if use_clip_for_comparison:
            clip_scorer = CLIPScorer(device=self.device)
            raw_score = clip_scorer.text_similarity(generated_caption, reference_text)
            # Normalise from [-1, 1] to [0, 1]
            return (raw_score + 1.0) / 2.0

        # Fallback: Jaccard token overlap
        a_tokens = set(generated_caption.lower().split())
        b_tokens = set(reference_text.lower().split())
        if not a_tokens and not b_tokens:
            return 1.0
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

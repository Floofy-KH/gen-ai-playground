"""
src/utils/consistency_scoring.py

Consistency metrics and scoring helpers for generative image outputs.

This module provides a unified :class:`ConsistencyScorer` that aggregates
multiple similarity signals (CLIP embedding cosine similarity, pixel-level
metrics, etc.) to give an overall consistency score between generated images
and a reference.

Typical usage::

    from PIL import Image
    from src.utils.consistency_scoring import ConsistencyScorer

    scorer = ConsistencyScorer()
    reference = Image.open("reference.png")
    generated = Image.open("output.png")

    score = scorer.score(reference, generated)
    print(f"Consistency score: {score:.4f}")  # higher = more consistent
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image


class ConsistencyScorer:
    """Aggregate consistency scorer that combines multiple similarity signals.

    Currently supported signals:
    - **CLIP cosine similarity**: Semantic similarity between image embeddings.
    - **Pixel MSE**: Mean squared error in pixel space (after resizing to a
      common resolution).

    Additional signals (e.g. LPIPS, face embedding distance) can be added by
    extending :meth:`score`.

    Args:
        use_clip: Whether to include CLIP embedding similarity.
        use_pixel_mse: Whether to include pixel-level MSE.
        clip_weight: Relative weight for the CLIP component.
        pixel_weight: Relative weight for the pixel MSE component.
        device: PyTorch device for CLIP inference.

    Example::

        scorer = ConsistencyScorer(use_pixel_mse=True)
        s = scorer.score(reference_image, generated_image)
        print(f"Similarity: {s:.3f}")

        # Enable CLIP once CLIPScorer is implemented:
        scorer = ConsistencyScorer(use_clip=True, clip_weight=1.0, use_pixel_mse=False)
    """

    def __init__(
        self,
        use_clip: bool = False,
        use_pixel_mse: bool = True,
        clip_weight: float = 1.0,
        pixel_weight: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.use_clip = use_clip
        self.use_pixel_mse = use_pixel_mse
        self.clip_weight = clip_weight
        self.pixel_weight = pixel_weight
        self.device = device
        self._clip_scorer = None  # Lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_clip_scorer(self):
        """Lazily initialise the CLIP scorer."""
        if self._clip_scorer is None:
            from src.utils.clip_blip_scoring import CLIPScorer
            self._clip_scorer = CLIPScorer(device=self.device)
        return self._clip_scorer

    @staticmethod
    def _pixel_mse(
        image_a: Image.Image,
        image_b: Image.Image,
        size: int = 256,
    ) -> float:
        """Compute mean squared error between two images in pixel space.

        Both images are resized to ``(size, size)`` before comparison.

        Args:
            image_a: First PIL image.
            image_b: Second PIL image.
            size: Common resolution for comparison.

        Returns:
            MSE value (lower = more similar).  Range is ``[0, 255**2]``.
        """
        import numpy as np

        a = np.array(image_a.convert("RGB").resize((size, size), Image.LANCZOS), dtype=float)
        b = np.array(image_b.convert("RGB").resize((size, size), Image.LANCZOS), dtype=float)
        return float(np.mean((a - b) ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        reference: Image.Image,
        generated: Image.Image,
    ) -> float:
        """Compute a scalar consistency score between ``reference`` and
        ``generated``.

        Scores are normalised so that 1.0 means "maximally consistent" and
        0.0 means "maximally different" (for CLIP similarity).  Pixel MSE is
        normalised to ``[0, 1]`` and inverted (1 = identical).

        Args:
            reference: Reference PIL image.
            generated: Generated PIL image to evaluate.

        Returns:
            Weighted aggregate consistency score in ``[0, 1]``.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        if self.use_clip and self.clip_weight > 0:
            clip_sim = self._get_clip_scorer().image_similarity(reference, generated)
            # CLIP cosine similarity is in [-1, 1]; normalise to [0, 1]
            clip_sim_norm = (clip_sim + 1.0) / 2.0
            weighted_sum += self.clip_weight * clip_sim_norm
            total_weight += self.clip_weight

        if self.use_pixel_mse and self.pixel_weight > 0:
            mse = self._pixel_mse(reference, generated)
            # Normalise MSE: invert and map to [0, 1] (255^2 = 65025 max)
            mse_score = 1.0 - min(mse / 65025.0, 1.0)
            weighted_sum += self.pixel_weight * mse_score
            total_weight += self.pixel_weight

        if total_weight == 0:
            raise ValueError("At least one scoring component must be enabled.")

        return weighted_sum / total_weight

    def score_batch(
        self,
        reference: Image.Image,
        generated_images: List[Image.Image],
    ) -> List[float]:
        """Score a list of generated images against a single reference.

        Args:
            reference: Reference PIL image.
            generated_images: List of generated images to evaluate.

        Returns:
            List of consistency scores, one per generated image.
        """
        return [self.score(reference, img) for img in generated_images]

    def rank_by_consistency(
        self,
        reference: Image.Image,
        generated_images: List[Image.Image],
    ) -> List[tuple]:
        """Rank generated images by consistency score (highest first).

        Args:
            reference: Reference PIL image.
            generated_images: List of generated images to rank.

        Returns:
            List of ``(score, image)`` tuples sorted by score descending.
        """
        scored = [
            (self.score(reference, img), img) for img in generated_images
        ]
        return sorted(scored, key=lambda x: x[0], reverse=True)

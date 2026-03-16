"""Tests for src/utils/consistency_scoring.py

Tests cover the pure-Python pixel-MSE path and scorer configuration.
CLIP-dependent paths are tested only for proper error handling since
CLIPScorer is still a stub.
"""

from __future__ import annotations

import pytest
from PIL import Image

from src.utils.consistency_scoring import ConsistencyScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_image(color: tuple[int, int, int], size: int = 64) -> Image.Image:
    """Return a solid-colour RGB PIL image."""
    img = Image.new("RGB", (size, size), color)
    return img


# ---------------------------------------------------------------------------
# ConsistencyScorer initialisation
# ---------------------------------------------------------------------------


class TestConsistencyScorerInit:
    def test_defaults(self):
        scorer = ConsistencyScorer()
        assert scorer.use_clip is False
        assert scorer.use_pixel_mse is True
        assert scorer.clip_weight == 1.0
        assert scorer.pixel_weight == 1.0
        assert scorer.device is None
        assert scorer._clip_scorer is None

    def test_custom_weights(self):
        scorer = ConsistencyScorer(clip_weight=0.5, pixel_weight=2.0)
        assert scorer.clip_weight == 0.5
        assert scorer.pixel_weight == 2.0

    def test_no_signal_enabled_raises_on_score(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=False)
        ref = _solid_image((255, 0, 0))
        gen = _solid_image((0, 255, 0))
        with pytest.raises(ValueError, match="At least one scoring component"):
            scorer.score(ref, gen)


# ---------------------------------------------------------------------------
# Pixel-MSE path
# ---------------------------------------------------------------------------


class TestPixelMSE:
    def test_identical_images_score_one(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        img = _solid_image((128, 64, 32))
        score = scorer.score(img, img)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_black_vs_white_score_near_zero(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        black = _solid_image((0, 0, 0))
        white = _solid_image((255, 255, 255))
        score = scorer.score(black, white)
        assert score < 0.1

    def test_score_range(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        a = _solid_image((100, 150, 200))
        b = _solid_image((50, 100, 150))
        score = scorer.score(a, b)
        assert 0.0 <= score <= 1.0

    def test_score_symmetry(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        a = _solid_image((200, 100, 50))
        b = _solid_image((10, 20, 30))
        assert scorer.score(a, b) == pytest.approx(scorer.score(b, a))


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def test_returns_correct_count(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        ref = _solid_image((0, 0, 0))
        images = [_solid_image((i * 10, 0, 0)) for i in range(5)]
        scores = scorer.score_batch(ref, images)
        assert len(scores) == 5

    def test_higher_score_for_closer_image(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        ref = _solid_image((100, 100, 100))
        close = _solid_image((105, 105, 105))
        far = _solid_image((0, 200, 50))
        scores = scorer.score_batch(ref, [close, far])
        assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# rank_by_consistency
# ---------------------------------------------------------------------------


class TestRankByConsistency:
    def test_returns_tuples(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        ref = _solid_image((128, 128, 128))
        images = [_solid_image((i, i, i)) for i in range(0, 256, 64)]
        ranked = scorer.rank_by_consistency(ref, images)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in ranked)

    def test_descending_order(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        ref = _solid_image((128, 128, 128))
        images = [_solid_image((i, i, i)) for i in range(0, 256, 32)]
        ranked = scorer.rank_by_consistency(ref, images)
        scores = [s for s, _ in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_identical_image_ranks_first(self):
        scorer = ConsistencyScorer(use_clip=False, use_pixel_mse=True)
        ref = _solid_image((200, 100, 50))
        identical = _solid_image((200, 100, 50))
        different = _solid_image((0, 0, 0))
        ranked = scorer.rank_by_consistency(ref, [different, identical])
        top_score, _ = ranked[0]
        assert top_score == pytest.approx(1.0, abs=1e-6)

"""Tests for src/utils/latent_utils.py

Covers _lerp, _slerp, add_noise_to_latent, and interpolate_latents without
any GPU or model weights (no internet access required).  All tests that
require torch are skipped via pytest.importorskip when torch is absent —
matching the CI environment where only lightweight dev deps are installed.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# _lerp
# ---------------------------------------------------------------------------


class TestLerp:
    def test_alpha_zero_returns_a(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _lerp

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = _lerp(a, b, 0.0)
        assert torch.allclose(result, a)

    def test_alpha_one_returns_b(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _lerp

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = _lerp(a, b, 1.0)
        assert torch.allclose(result, b)

    def test_alpha_half_returns_midpoint(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _lerp

        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([4.0, 8.0])
        result = _lerp(a, b, 0.5)
        assert torch.allclose(result, torch.tensor([2.0, 4.0]))

    def test_preserves_shape(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _lerp

        a = torch.ones(2, 4, 8, 8)
        b = torch.zeros(2, 4, 8, 8)
        result = _lerp(a, b, 0.3)
        assert result.shape == (2, 4, 8, 8)

    def test_interpolates_linearly(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _lerp

        a = torch.tensor([0.0])
        b = torch.tensor([10.0])
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _lerp(a, b, alpha)
            expected = torch.tensor([10.0 * alpha])
            assert torch.allclose(result, expected), f"Failed at alpha={alpha}"


# ---------------------------------------------------------------------------
# _slerp
# ---------------------------------------------------------------------------


class TestSlerp:
    def test_alpha_zero_near_a(self):
        """At alpha=0, slerp should return a value close to a."""
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _slerp

        a = torch.tensor([[3.0, 4.0]])
        b = torch.tensor([[0.0, 5.0]])
        result = _slerp(a, b, 0.0)
        # cos(0) * a + sin(0) * ... = a
        assert torch.allclose(result, a, atol=1e-5)

    def test_alpha_one_near_b_direction(self):
        """At alpha=1, result should point in the direction of b."""
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _slerp

        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 2.0]])
        result = _slerp(a, b, 1.0)
        # Normalise both and check they point the same way
        b_dir = b / torch.norm(b)
        result_dir = result / (torch.norm(result) + 1e-8)
        assert torch.allclose(result_dir, b_dir, atol=1e-4)

    def test_output_shape_preserved(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _slerp

        a = torch.randn(1, 4, 8, 8)
        b = torch.randn(1, 4, 8, 8)
        result = _slerp(a, b, 0.5)
        assert result.shape == a.shape

    def test_collinear_vectors_do_not_raise(self):
        """Identical (collinear) vectors should not cause a NaN or exception."""
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _slerp

        a = torch.tensor([[1.0, 2.0, 3.0]])
        result = _slerp(a, a, 0.5)
        assert not torch.isnan(result).any(), "SLERP produced NaN for collinear vectors"

    def test_interpolates_between_endpoints(self):
        """Midpoint slerp should differ from both a and b (for non-parallel vectors)."""
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _slerp

        a = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 1.0, 0.0]])
        mid = _slerp(a, b, 0.5)
        assert not torch.allclose(mid, a, atol=1e-3)
        assert not torch.allclose(mid, b, atol=1e-3)


# ---------------------------------------------------------------------------
# add_noise_to_latent
# ---------------------------------------------------------------------------


class TestAddNoiseToLatent:
    def test_output_shape_preserved(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import add_noise_to_latent

        latent = torch.zeros(1, 4, 8, 8)
        result = add_noise_to_latent(latent, noise_level=0.1)
        assert result.shape == latent.shape

    def test_zero_noise_level_returns_unchanged(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import add_noise_to_latent

        latent = torch.ones(1, 4, 8, 8) * 3.14
        result = add_noise_to_latent(latent, noise_level=0.0)
        assert torch.allclose(result, latent)

    def test_noise_added_makes_tensor_different(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import add_noise_to_latent

        latent = torch.zeros(1, 4, 8, 8)
        result = add_noise_to_latent(latent, noise_level=1.0)
        assert not torch.allclose(result, latent)

    def test_same_seed_produces_same_result(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import add_noise_to_latent

        latent = torch.zeros(1, 4, 8, 8)
        result_a = add_noise_to_latent(latent, noise_level=0.5, seed=42)
        result_b = add_noise_to_latent(latent, noise_level=0.5, seed=42)
        assert torch.allclose(result_a, result_b)

    def test_different_seeds_produce_different_results(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import add_noise_to_latent

        latent = torch.zeros(1, 4, 8, 8)
        result_a = add_noise_to_latent(latent, noise_level=0.5, seed=1)
        result_b = add_noise_to_latent(latent, noise_level=0.5, seed=2)
        assert not torch.allclose(result_a, result_b)

    def test_noise_scales_with_noise_level(self):
        """Higher noise_level should produce larger deviations from the input."""
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import add_noise_to_latent

        latent = torch.zeros(1, 4, 32, 32)
        low_noise = add_noise_to_latent(latent, noise_level=0.01, seed=0)
        high_noise = add_noise_to_latent(latent, noise_level=1.0, seed=0)
        low_std = low_noise.std().item()
        high_std = high_noise.std().item()
        assert high_std > low_std


# ---------------------------------------------------------------------------
# interpolate_latents (integration)
# ---------------------------------------------------------------------------


class TestInterpolateLatents:
    def test_lerp_dispatch(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _lerp, interpolate_latents

        a = torch.randn(1, 4, 8, 8)
        b = torch.randn(1, 4, 8, 8)
        result = interpolate_latents(a, b, 0.5, method="lerp")
        expected = _lerp(a, b, 0.5)
        assert torch.allclose(result, expected)

    def test_slerp_dispatch(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import _slerp, interpolate_latents

        a = torch.randn(1, 4, 8, 8)
        b = torch.randn(1, 4, 8, 8)
        result = interpolate_latents(a, b, 0.5, method="slerp")
        expected = _slerp(a, b, 0.5)
        assert torch.allclose(result, expected)

    def test_unknown_method_raises(self):
        torch = pytest.importorskip("torch")
        from src.utils.latent_utils import interpolate_latents

        a = torch.randn(1, 4, 8, 8)
        b = torch.randn(1, 4, 8, 8)
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            interpolate_latents(a, b, 0.5, method="cubic")

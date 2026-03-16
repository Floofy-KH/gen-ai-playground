"""Tests for src/utils/seed_utils.py

These tests cover only the pure-Python functions that require no GPU or
model weights.  Torch-dependent helpers (make_generator) are tested against
a graceful ImportError path.
"""

from __future__ import annotations

import sys

import pytest

from src.utils.seed_utils import generate_with_locked_seed, lock_seed, make_generator, seed_range


# ---------------------------------------------------------------------------
# lock_seed
# ---------------------------------------------------------------------------


class TestLockSeed:
    def test_sets_python_random(self):
        import random

        lock_seed(0)
        v1 = random.random()
        lock_seed(0)
        v2 = random.random()
        assert v1 == v2, "lock_seed should produce reproducible Python random state"

    def test_different_seeds_produce_different_values(self):
        import random

        lock_seed(1)
        v1 = random.random()
        lock_seed(2)
        v2 = random.random()
        assert v1 != v2

    def test_sets_numpy_random(self):
        np = pytest.importorskip("numpy")
        lock_seed(99)
        a = np.random.rand(5)
        lock_seed(99)
        b = np.random.rand(5)
        assert (a == b).all()

    def test_does_not_raise_without_numpy(self, monkeypatch):
        """lock_seed should not raise if numpy is not installed."""
        monkeypatch.setitem(sys.modules, "numpy", None)
        # Should complete without exception
        lock_seed(42)

    def test_does_not_raise_without_torch(self, monkeypatch):
        """lock_seed should not raise if torch is not installed."""
        monkeypatch.setitem(sys.modules, "torch", None)
        lock_seed(42)


# ---------------------------------------------------------------------------
# make_generator
# ---------------------------------------------------------------------------


class TestMakeGenerator:
    def test_none_seed_returns_none(self):
        assert make_generator(None) is None

    def test_none_seed_returns_none_with_device(self):
        assert make_generator(None, device="cpu") is None

    def test_raises_import_error_without_torch(self, monkeypatch):
        """make_generator should raise ImportError when torch is not installed."""
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match="PyTorch is required"):
            make_generator(42)

    def test_torch_generator_when_available(self):
        """When torch is installed, returns a torch.Generator."""
        torch = pytest.importorskip("torch")
        gen = make_generator(42, device="cpu")
        assert isinstance(gen, torch.Generator)

    def test_torch_generator_deterministic(self):
        """Same seed produces the same random state."""
        torch = pytest.importorskip("torch")
        gen1 = make_generator(7, device="cpu")
        gen2 = make_generator(7, device="cpu")
        # Both generators should produce identical random tensors
        t1 = torch.rand(5, generator=gen1)
        t2 = torch.rand(5, generator=gen2)
        assert torch.allclose(t1, t2)


# ---------------------------------------------------------------------------
# seed_range
# ---------------------------------------------------------------------------


class TestSeedRange:
    def test_returns_correct_count(self):
        seeds = seed_range(0, 5)
        assert len(seeds) == 5

    def test_all_integers(self):
        seeds = seed_range(0, 10)
        assert all(isinstance(s, int) for s in seeds)

    def test_all_within_valid_range(self):
        seeds = seed_range(0, 50)
        assert all(0 <= s <= 2**31 - 1 for s in seeds)

    def test_reproducible(self):
        assert seed_range(42, 8) == seed_range(42, 8)

    def test_different_starts_differ(self):
        # Very unlikely to collide across 10 seeds
        assert seed_range(0, 10) != seed_range(1, 10)

    def test_zero_count(self):
        assert seed_range(0, 0) == []

    def test_count_one(self):
        result = seed_range(7, 1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# generate_with_locked_seed
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Minimal pipeline stub that records calls."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, prompt: str, seed: int, **kwargs):
        self.calls.append({"prompt": prompt, "seed": seed, **kwargs})
        return f"image:{prompt}:{seed}"


class TestGenerateWithLockedSeed:
    def test_produces_one_output_per_prompt(self):
        pipe = _FakePipeline()
        prompts = ["a", "b", "c"]
        results = generate_with_locked_seed(pipe, prompts, seed=7)
        assert len(results) == 3

    def test_same_seed_for_all(self):
        pipe = _FakePipeline()
        generate_with_locked_seed(pipe, ["x", "y"], seed=99)
        seeds = [c["seed"] for c in pipe.calls]
        assert seeds == [99, 99]

    def test_prompts_passed_correctly(self):
        pipe = _FakePipeline()
        generate_with_locked_seed(pipe, ["hello", "world"], seed=1)
        prompts = [c["prompt"] for c in pipe.calls]
        assert prompts == ["hello", "world"]

    def test_extra_kwargs_forwarded(self):
        pipe = _FakePipeline()
        generate_with_locked_seed(pipe, ["p"], seed=5, generate_kwargs={"steps": 10})
        assert pipe.calls[0]["steps"] == 10

    def test_empty_prompt_list(self):
        pipe = _FakePipeline()
        results = generate_with_locked_seed(pipe, [], seed=0)
        assert results == []

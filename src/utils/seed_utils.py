"""
src/utils/seed_utils.py

Seed locking and reproducibility helpers.

Fixing the random seed in PyTorch and NumPy is the simplest and cheapest
consistency technique: identical seeds + identical prompts = identical outputs.
This module provides utilities to lock seeds across all relevant random number
generators and to run batch generations with a shared seed.

References:
    - PyTorch reproducibility guide:
      https://pytorch.org/docs/stable/notes/randomness.html
    - Diffusers generator usage:
      https://huggingface.co/docs/diffusers/using-diffusers/reproducibility
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


def lock_seed(seed: int) -> None:
    """Set a fixed random seed across Python, NumPy, and PyTorch.

    Calling this before any generation or training call ensures full
    reproducibility (on deterministic hardware).

    Args:
        seed: Integer seed value.

    Example::

        from src.utils.seed_utils import lock_seed

        lock_seed(42)
        # All subsequent random operations are seeded from 42
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_generator(seed: Optional[int], device: str = "cpu"):
    """Create a ``torch.Generator`` initialised with the given seed.

    Args:
        seed: Random seed.  If ``None``, returns ``None`` (Diffusers will use
            a random seed internally).
        device: The device on which to place the generator.  Should match the
            device of the pipeline (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        A ``torch.Generator`` if ``seed`` is not ``None``, else ``None``.

    Example::

        generator = make_generator(42, device="cuda")
        image = pipe(prompt="...", generator=generator).images[0]
    """
    if seed is None:
        return None
    try:
        import torch
        return torch.Generator(device=device).manual_seed(seed)
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for make_generator(). "
            "Install it with: pip install torch"
        ) from exc


def generate_with_locked_seed(
    pipeline: Any,
    prompts: List[str],
    seed: int = 42,
    generate_kwargs: Optional[Dict] = None,
) -> List[Any]:
    """Generate one image per prompt, all using the same seed.

    This is the simplest cross-prompt consistency technique: by using the same
    seed for every prompt the underlying noise pattern is shared, which often
    results in visually coherent outputs across prompts that share the same
    base anchor.

    Args:
        pipeline: Any object with a ``generate(prompt, seed=..., **kwargs)``
            method (e.g. :class:`~src.image.pipeline.ImageGenerationPipeline`).
        prompts: List of text prompts.
        seed: Shared random seed for all generations.
        generate_kwargs: Additional keyword arguments forwarded to
            ``pipeline.generate()``.

    Returns:
        List of generated images (the return type of ``pipeline.generate()``),
        one per prompt.

    Example::

        from src.image.pipeline import ImageGenerationPipeline
        from src.utils.seed_utils import generate_with_locked_seed

        pipe = ImageGenerationPipeline("runwayml/stable-diffusion-v1-5")
        images = generate_with_locked_seed(
            pipe,
            prompts=["sks person in a forest", "sks person on the moon"],
            seed=7,
        )
    """
    kwargs = generate_kwargs or {}
    return [pipeline.generate(prompt=p, seed=seed, **kwargs) for p in prompts]


def seed_range(start: int, count: int) -> List[int]:
    """Generate a deterministic list of seeds starting from ``start``.

    Useful when you want diversity across multiple runs but still need
    reproducible results.

    Args:
        start: Seed for the internal RNG used to generate the seed list.
        count: How many seeds to generate.

    Returns:
        List of ``count`` integer seeds.

    Example::

        seeds = seed_range(0, 5)
        # [1684, 8321, 2945, 7132, 394]  (example — actual values are fixed)
    """
    rng = random.Random(start)
    return [rng.randint(0, 2**31 - 1) for _ in range(count)]

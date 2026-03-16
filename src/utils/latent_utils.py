"""
src/utils/latent_utils.py

Shared latent-space utilities for diffusion model pipelines.

Working directly in the VAE latent space allows techniques such as:

- **Latent interpolation**: Smoothly blend between two image latents to
  generate visual transitions or style mixing.
- **Latent initialisation**: Encode a reference image as the starting latent
  for ``img2img`` generation, keeping the output "close" to the reference.
- **Noise injection**: Add controlled noise to a latent to explore the space
  around a reference image.

References:
    - Diffusers img2img pipeline:
      https://huggingface.co/docs/diffusers/using-diffusers/img2img
    - VAE encoding/decoding in Diffusers:
      https://huggingface.co/docs/diffusers/api/models/autoencoderkl
"""

from __future__ import annotations

from typing import Optional

from PIL import Image


def encode_image_to_latent(
    image: Image.Image,
    vae,
    device: Optional[str] = None,
    dtype=None,
):
    """Encode a PIL image into the VAE latent space.

    The latent vector can be used as the starting point for ``img2img``
    generation or as the basis for latent interpolation.

    Args:
        image: Input PIL image.  Should be resized to the model's expected
            resolution (e.g. 512×512 for SD 1.5) before calling this function.
        vae: A Diffusers ``AutoencoderKL`` instance.
        device: Target device.  Defaults to the VAE's current device.
        dtype: Optional dtype for the latent tensor.

    Returns:
        Latent tensor of shape ``(1, 4, H/8, W/8)``.

    Example::

        from diffusers import AutoencoderKL
        from src.utils.latent_utils import encode_image_to_latent

        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",
                                             subfolder="vae")
        latent = encode_image_to_latent(my_image, vae)
    """
    # TODO: Convert image to tensor, encode through vae.encode(), sample from
    #       the latent distribution, scale by vae.config.scaling_factor.
    #
    # Implementation outline:
    #   import torch
    #   from torchvision import transforms
    #
    #   transform = transforms.Compose([
    #       transforms.ToTensor(),
    #       transforms.Normalize([0.5], [0.5]),
    #   ])
    #   img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device, dtype)
    #
    #   with torch.no_grad():
    #       latent_dist = vae.encode(img_tensor).latent_dist
    #       latent = latent_dist.sample() * vae.config.scaling_factor
    #   return latent
    raise NotImplementedError("Stub: implement encode_image_to_latent().")


def decode_latent_to_image(
    latent,
    vae,
) -> Image.Image:
    """Decode a latent tensor back into a PIL image.

    Args:
        latent: Latent tensor of shape ``(1, 4, H/8, W/8)``.  Should already
            be un-scaled (divide by ``vae.config.scaling_factor`` before
            passing if the latent is still in "diffusion scale").
        vae: A Diffusers ``AutoencoderKL`` instance.

    Returns:
        Decoded PIL image.

    Example::

        image = decode_latent_to_image(latent / vae.config.scaling_factor, vae)
        image.save("decoded.png")
    """
    # TODO: Run latent through vae.decode(), convert output tensor to PIL Image.
    #
    # Implementation outline:
    #   import torch
    #   from PIL import Image
    #   import numpy as np
    #
    #   with torch.no_grad():
    #       decoded = vae.decode(latent).sample  # (1, 3, H, W) in [-1, 1]
    #   decoded = (decoded / 2 + 0.5).clamp(0, 1)
    #   decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #   return Image.fromarray((decoded * 255).astype(np.uint8))
    raise NotImplementedError("Stub: implement decode_latent_to_image().")


def interpolate_latents(
    latent_a,
    latent_b,
    alpha: float,
    method: str = "lerp",
):
    """Interpolate between two latent tensors.

    Args:
        latent_a: Starting latent tensor.
        latent_b: Ending latent tensor.
        alpha: Interpolation weight in ``[0, 1]``.  ``0`` returns ``latent_a``,
            ``1`` returns ``latent_b``.
        method: Interpolation method — ``"lerp"`` (linear) or ``"slerp"``
            (spherical linear, preserves magnitude, better for semantic
            interpolation).

    Returns:
        Interpolated latent tensor with the same shape as inputs.

    Raises:
        ValueError: If ``method`` is not ``"lerp"`` or ``"slerp"``.

    Example::

        mid_latent = interpolate_latents(latent_a, latent_b, alpha=0.5, method="slerp")
        mid_image = decode_latent_to_image(mid_latent / vae.config.scaling_factor, vae)
    """
    if method == "lerp":
        return _lerp(latent_a, latent_b, alpha)
    elif method == "slerp":
        return _slerp(latent_a, latent_b, alpha)
    else:
        raise ValueError(f"Unknown interpolation method: {method!r}. Use 'lerp' or 'slerp'.")


def _lerp(a, b, alpha: float):
    """Linear interpolation: ``a * (1 - alpha) + b * alpha``."""
    # TODO: Implement using torch arithmetic.
    raise NotImplementedError("Stub: implement _lerp().")


def _slerp(a, b, alpha: float, eps: float = 1e-6):
    """Spherical linear interpolation (SLERP) between two tensors.

    SLERP travels along the surface of a hypersphere, which tends to produce
    smoother and more semantically meaningful interpolations than linear lerp
    in high-dimensional embedding spaces.

    Args:
        a: Starting tensor.
        b: Ending tensor.
        alpha: Interpolation weight in ``[0, 1]``.
        eps: Small epsilon to avoid division by zero.

    Returns:
        Interpolated tensor.
    """
    # TODO: Implement SLERP:
    #   import torch
    #   a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + eps)
    #   b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + eps)
    #   dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
    #   theta = torch.acos(dot) * alpha
    #   relative = b - dot * a_norm
    #   relative = relative / (torch.norm(relative, dim=-1, keepdim=True) + eps)
    #   return torch.cos(theta) * a + torch.sin(theta) * relative
    raise NotImplementedError("Stub: implement _slerp().")


def add_noise_to_latent(
    latent,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
):
    """Add Gaussian noise to a latent tensor.

    Useful for creating local variations around a reference image latent
    without fully randomising the starting point.

    Args:
        latent: Input latent tensor.
        noise_level: Standard deviation of the Gaussian noise to add.
            Larger values produce more divergent outputs.
        seed: Optional seed for reproducible noise.

    Returns:
        Noised latent tensor with the same shape as input.
    """
    # TODO: Create a Generator from `seed` if provided, sample Gaussian noise,
    #       add to latent: latent + noise_level * torch.randn_like(latent, generator=g)
    raise NotImplementedError("Stub: implement add_noise_to_latent().")

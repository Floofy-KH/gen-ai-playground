"""
src/training/train_textual_inversion.py

Textual Inversion training script for Stable Diffusion models.

Textual Inversion optimises a new token embedding vector in the text encoder's
vocabulary to represent a visual concept from a small set of reference images.
Only the new embedding is trained; the rest of the model remains frozen.

Training approach:
    1. Add a new placeholder token (e.g. ``<my-concept>``) to the tokenizer and
       initialise its embedding from a related existing token (e.g. ``"sculpture"``).
    2. Freeze the entire model (UNet, VAE, all other text encoder parameters).
    3. Optimise *only* the new embedding vector using the standard diffusion
       denoising loss on the reference images.
    4. Save the learned embedding as a ``.bin`` / ``.pt`` file.

References:
    - Textual Inversion paper: https://arxiv.org/abs/2208.01618
    - Diffusers Textual Inversion training:
      https://huggingface.co/docs/diffusers/training/text_inversion
    - Original Textual Inversion implementation:
      https://github.com/rinongal/textual_inversion
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TextualInversionTrainingConfig:
    """Hyperparameters and settings for Textual Inversion training.

    Attributes:
        base_model_id: Hub ID or local path of the base Stable Diffusion model.
        train_data_dir: Directory containing 3–10 reference images of the
            target concept.
        output_dir: Directory where the learned embedding will be saved.
        placeholder_token: The new token string to add to the vocabulary
            (e.g. ``"<my-concept>"``).  Use angle brackets to avoid collisions
            with existing vocabulary.
        initializer_token: An existing token whose embedding is used to
            initialise the new placeholder (e.g. ``"sculpture"`` or
            ``"painting"``).  Choose a token whose meaning is semantically
            close to your concept.
        learnable_property: What property to learn — ``"object"`` for a
            specific object/subject, or ``"style"`` for an artistic style.
        num_vectors: Number of token vectors to use for the concept.  Using
            more than 1 can increase expressivity (multi-vector embeddings).
        resolution: Training image resolution.
        train_batch_size: Batch size per GPU.
        max_train_steps: Maximum number of gradient update steps.
        learning_rate: Embedding learning rate.
        mixed_precision: Mixed precision mode.
        seed: Random seed for reproducibility.
        save_steps: Save the embedding every N steps.
    """

    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    train_data_dir: str = "data/concept_images"
    output_dir: str = "outputs/textual_inversion"
    placeholder_token: str = "<my-concept>"
    initializer_token: str = "object"
    learnable_property: str = "object"  # "object" or "style"
    num_vectors: int = 1
    resolution: int = 1024   # SDXL native resolution; use 512 for SD 1.5
    train_batch_size: int = 1
    max_train_steps: int = 3000
    learning_rate: float = 5e-4
    mixed_precision: str = "fp16"
    seed: int = 42
    save_steps: int = 500


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train(config: TextualInversionTrainingConfig) -> Path:
    """Run Textual Inversion training according to ``config``.

    Args:
        config: A populated :class:`TextualInversionTrainingConfig` instance.

    Returns:
        Path to the directory containing the saved embedding file(s).

    Implementation outline::

        from accelerate import Accelerator
        from diffusers import StableDiffusionPipeline
        from transformers import CLIPTextModel, CLIPTokenizer
        import torch

        accelerator = Accelerator(mixed_precision=config.mixed_precision)

        # 1. Load tokenizer and add new placeholder token(s)
        #      tokenizer.add_tokens([config.placeholder_token])
        #      token_id = tokenizer.convert_tokens_to_ids(config.placeholder_token)

        # 2. Resize token embeddings and initialise from initializer_token
        #      text_encoder.resize_token_embeddings(len(tokenizer))
        #      init_id = tokenizer.convert_tokens_to_ids(config.initializer_token)
        #      with torch.no_grad():
        #          text_encoder.get_input_embeddings().weight[token_id] = (
        #              text_encoder.get_input_embeddings().weight[init_id].clone()
        #          )

        # 3. Freeze all model parameters EXCEPT the new embedding row
        #      text_encoder.text_model.embeddings.token_embedding.weight.requires_grad_(False)
        #      # Re-enable grad only for the new token(s):
        #      text_encoder.get_input_embeddings().weight[token_id].requires_grad_(True)

        # 4. Build dataset and dataloader (pair each image with a prompt
        #    like "a photo of <my-concept>")

        # 5. Training loop (optimise only the embedding, clip gradients):
        #      for step in range(config.max_train_steps):
        #          for batch in dataloader:
        #              loss = compute_diffusion_loss(batch, unet, ...)
        #              accelerator.backward(loss)
        #              # Zero gradients for all embeddings except the new token
        #              optimizer.step(); optimizer.zero_grad()

        # 6. Save: torch.save(
        #        {"string_to_param": {config.placeholder_token: embedding}},
        #        output_dir / "learned_embeds.bin"
        #    )
    """
    # TODO: Implement the training loop as outlined above.
    raise NotImplementedError(
        "Stub: implement train() for Textual Inversion."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> TextualInversionTrainingConfig:
    """Parse command-line arguments and return a
    :class:`TextualInversionTrainingConfig`."""
    parser = argparse.ArgumentParser(
        description="Train a Textual Inversion embedding on Stable Diffusion."
    )
    parser.add_argument("--base_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/textual_inversion")
    parser.add_argument("--placeholder_token", type=str, required=True)
    parser.add_argument("--initializer_token", type=str, default="object")
    parser.add_argument("--learnable_property", type=str, default="object")
    parser.add_argument("--num_vectors", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=500)

    args = parser.parse_args()
    return TextualInversionTrainingConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    output_path = train(cfg)
    print(f"Training complete. Embedding saved to: {output_path}")

"""
src/training/train_dreambooth.py

DreamBooth fine-tuning script for Stable Diffusion models.

DreamBooth fine-tunes the entire (or most of the) base model to bind a unique
identifier token (e.g. ``sks``) to a specific subject.  It uses a
**prior-preservation loss** to prevent the model from forgetting the broader
concept class (e.g. "person" or "dog") while learning the specific subject.

Training approach:
    1. Collect 3–30 images of the target subject.
    2. Generate *class images* from the base model using a class prompt
       (e.g. ``"a photo of a dog"``) to compute the prior-preservation loss.
    3. Fine-tune the UNet (and optionally the text encoder) jointly on the
       instance images (subject) and class images (prior).
    4. Save the full fine-tuned model checkpoint.

References:
    - DreamBooth paper: https://arxiv.org/abs/2208.12242
    - Diffusers DreamBooth training guide:
      https://huggingface.co/docs/diffusers/training/dreambooth
    - TheLastBen's fast-DreamBooth (Colab-friendly):
      https://github.com/TheLastBen/fast-stable-diffusion
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DreamBoothTrainingConfig:
    """Hyperparameters and settings for DreamBooth fine-tuning.

    Attributes:
        base_model_id: Hub ID or local path of the base Stable Diffusion model.
        instance_data_dir: Directory containing the subject reference images.
        class_data_dir: Directory for class (prior-preservation) images.
            Created and populated automatically if ``with_prior_preservation``
            is ``True`` and the directory is empty.
        output_dir: Directory where the fine-tuned model is saved.
        instance_prompt: Training prompt for instance images (subject), e.g.
            ``"a photo of sks dog"``.
        class_prompt: Training prompt for class images (prior), e.g.
            ``"a photo of a dog"``.
        with_prior_preservation: Whether to use prior-preservation loss.
            Strongly recommended to prevent catastrophic forgetting.
        prior_loss_weight: Weight of the prior-preservation loss term.
        num_class_images: Number of class images to generate if the
            ``class_data_dir`` is empty.
        resolution: Training image resolution.
        train_batch_size: Batch size per GPU.
        num_train_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        train_text_encoder: Whether to fine-tune the text encoder as well.
        mixed_precision: Mixed precision mode (``"no"``, ``"fp16"``, ``"bf16"``).
        seed: Random seed for reproducibility.
        checkpointing_steps: Save an intermediate checkpoint every N steps.
    """

    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    instance_data_dir: str = "data/instance_images"
    class_data_dir: str = "data/class_images"
    output_dir: str = "outputs/dreambooth"
    instance_prompt: str = "a photo of sks subject"
    class_prompt: str = "a photo of a subject"
    with_prior_preservation: bool = True
    prior_loss_weight: float = 1.0
    num_class_images: int = 200
    resolution: int = 1024  # SDXL native resolution; use 512 for SD 1.5
    train_batch_size: int = 1
    num_train_epochs: int = 4
    learning_rate: float = 5e-6
    train_text_encoder: bool = False
    mixed_precision: str = "fp16"
    seed: int = 42
    checkpointing_steps: int = 500


# ---------------------------------------------------------------------------
# Class image generation helper
# ---------------------------------------------------------------------------


def generate_class_images(config: DreamBoothTrainingConfig) -> None:
    """Generate prior-preservation class images using the base model.

    Checks whether ``config.class_data_dir`` already contains enough images;
    if not, generates the remainder using the base model and
    ``config.class_prompt``.

    Args:
        config: Populated :class:`DreamBoothTrainingConfig`.

    Implementation outline::

        from diffusers import StableDiffusionPipeline
        import torch

        class_dir = Path(config.class_data_dir)
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        num_to_generate = config.num_class_images - len(existing)
        if num_to_generate <= 0:
            return

        pipe = StableDiffusionPipeline.from_pretrained(
            config.base_model_id, torch_dtype=torch.float16
        ).to("cuda")

        for i in range(num_to_generate):
            img = pipe(config.class_prompt).images[0]
            img.save(class_dir / f"class_{len(existing) + i:04d}.png")
    """
    # TODO: Implement as described in the docstring above.
    raise NotImplementedError("Stub: implement generate_class_images().")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train(config: DreamBoothTrainingConfig) -> Path:
    """Run DreamBooth fine-tuning according to ``config``.

    Args:
        config: A populated :class:`DreamBoothTrainingConfig` instance.

    Returns:
        Path to the directory containing the saved fine-tuned model.

    Implementation outline::

        from accelerate import Accelerator
        from diffusers import StableDiffusionPipeline
        import torch

        if config.with_prior_preservation:
            generate_class_images(config)

        accelerator = Accelerator(mixed_precision=config.mixed_precision)

        # 1. Load UNet, VAE, text encoder, tokenizer, noise scheduler
        # 2. Optionally freeze/unfreeze text encoder based on config
        # 3. Build combined dataset (instance + class images) and dataloader
        # 4. Set up AdamW optimizer on UNet (and optionally text encoder) params
        # 5. Training loop:
        #      for epoch in range(config.num_train_epochs):
        #          for batch in dataloader:
        #              instance_loss = compute_diffusion_loss(instance_batch)
        #              prior_loss = compute_diffusion_loss(class_batch)
        #              loss = instance_loss + config.prior_loss_weight * prior_loss
        #              accelerator.backward(loss)
        #              optimizer.step(); optimizer.zero_grad()
        # 6. Save full pipeline: pipeline.save_pretrained(config.output_dir)
    """
    # TODO: Implement the training loop as outlined above.
    raise NotImplementedError("Stub: implement train() for DreamBooth.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> DreamBoothTrainingConfig:
    """Parse command-line arguments and return a :class:`DreamBoothTrainingConfig`."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a Stable Diffusion model with DreamBooth."
    )
    parser.add_argument(
        "--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--class_data_dir", type=str, default="data/class_images")
    parser.add_argument("--output_dir", type=str, default="outputs/dreambooth")
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--class_prompt", type=str, required=True)
    parser.add_argument(
        "--with_prior_preservation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use prior-preservation loss during training (recommended). "
            "Pass --no-with_prior_preservation to disable."
        ),
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    args = parser.parse_args()
    return DreamBoothTrainingConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    output_path = train(cfg)
    print(f"Training complete. Model saved to: {output_path}")

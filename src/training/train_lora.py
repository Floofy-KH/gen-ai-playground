"""
src/training/train_lora.py

LoRA fine-tuning script for Stable Diffusion models.

This script fine-tunes a low-rank adapter (LoRA) on top of a frozen Stable
Diffusion base model using a small dataset of reference images.  The resulting
``.safetensors`` file can be loaded at inference time with minimal overhead and
is compatible with Diffusers, AUTOMATIC1111, and ComfyUI.

Training approach:
    1. Load a pre-trained Stable Diffusion model.
    2. Inject trainable LoRA matrices into the UNet (and optionally the text
       encoder) attention layers using the PEFT library.
    3. Fine-tune only the LoRA parameters on the reference dataset.
    4. Save the adapter weights (not the full model) for efficient storage and
       sharing.

References:
    - LoRA paper: https://arxiv.org/abs/2106.09685
    - PEFT library: https://huggingface.co/docs/peft
    - Diffusers LoRA training guide:
      https://huggingface.co/docs/diffusers/training/lora
    - Kohya-ss training scripts (advanced):
      https://github.com/kohya-ss/sd-scripts
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainingConfig:
    """Hyperparameters and settings for LoRA fine-tuning.

    Attributes:
        base_model_id: Hub ID or local path of the base Stable Diffusion model.
        instance_data_dir: Directory containing the reference images.
        output_dir: Directory where the trained adapter will be saved.
        instance_prompt: Prompt used during training, typically containing the
            unique subject token (e.g. ``"a photo of sks dog"``).
        resolution: Training image resolution.
        train_batch_size: Batch size per GPU.
        num_train_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        lora_rank: LoRA decomposition rank.  Higher = more capacity but slower.
        lora_alpha: LoRA scaling factor.  Commonly set equal to ``lora_rank``.
        train_text_encoder: Whether to also fine-tune LoRA layers in the text
            encoder (increases consistency for rare tokens).
        mixed_precision: Mixed precision training mode
            (``"no"``, ``"fp16"``, or ``"bf16"``).
        seed: Random seed for reproducibility.
        checkpointing_steps: Save an intermediate checkpoint every N steps.
        report_to: Logging backend (e.g. ``"tensorboard"`` or ``"wandb"``).
        resume_from_checkpoint: Path to resume training from a checkpoint.
    """

    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    instance_data_dir: str = "data/instance_images"
    output_dir: str = "outputs/lora"
    instance_prompt: str = "a photo of sks subject"
    resolution: int = 512
    train_batch_size: int = 1
    num_train_epochs: int = 100
    learning_rate: float = 1e-4
    lora_rank: int = 4
    lora_alpha: int = 4
    train_text_encoder: bool = False
    mixed_precision: str = "fp16"
    seed: int = 42
    checkpointing_steps: int = 500
    report_to: str = "tensorboard"
    resume_from_checkpoint: Optional[str] = None
    target_modules: List[str] = field(
        default_factory=lambda: ["to_q", "to_v", "to_k", "to_out.0"]
    )


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train(config: LoRATrainingConfig) -> Path:
    """Run LoRA fine-tuning according to ``config``.

    Args:
        config: A populated :class:`LoRATrainingConfig` instance.

    Returns:
        Path to the directory containing the saved adapter weights.

    Implementation outline::

        from accelerate import Accelerator
        from diffusers import StableDiffusionPipeline
        from peft import LoraConfig, get_peft_model
        import torch
        from torch.utils.data import DataLoader

        accelerator = Accelerator(mixed_precision=config.mixed_precision)

        # 1. Load base model components (VAE, UNet, text encoder, tokenizer)
        # 2. Freeze all base model parameters
        # 3. Apply LoRA config to the UNet via PEFT:
        #      lora_config = LoraConfig(r=config.lora_rank, ...)
        #      unet = get_peft_model(unet, lora_config)
        # 4. Build dataset & dataloader from config.instance_data_dir
        # 5. Set up AdamW optimizer on LoRA parameters only
        # 6. Training loop:
        #      for epoch in range(config.num_train_epochs):
        #          for batch in dataloader:
        #              loss = compute_diffusion_loss(batch, unet, ...)
        #              accelerator.backward(loss)
        #              optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
        # 7. Save adapter: unet.save_pretrained(config.output_dir)
    """
    # TODO: Implement the training loop as outlined above.
    raise NotImplementedError("Stub: implement train() for LoRA fine-tuning.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> LoRATrainingConfig:
    """Parse command-line arguments and return a :class:`LoRATrainingConfig`."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a LoRA adapter on Stable Diffusion."
    )
    parser.add_argument("--base_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/lora")
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    args = parser.parse_args()
    return LoRATrainingConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    output_path = train(cfg)
    print(f"Training complete. Adapter saved to: {output_path}")

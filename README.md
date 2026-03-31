# 🎨 Generative AI Playground — Consistent Image Generation

> **Research & Experimental Playground**
> This repository is a hands-on, experimental space for exploring state-of-the-art techniques in **consistent generative AI image generation**. Everything here is intended as a starting point for research and practical experimentation — expect stubs, evolving implementations, and rough edges.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Consistency Techniques](#consistency-techniques)
3. [Repository Structure](#repository-structure)
4. [Getting Started](#getting-started)
5. [Running Tests](#running-tests)
6. [Usage Examples](#usage-examples)
7. [Further Reading & Resources](#further-reading--resources)
8. [Contributing](#contributing)

---

## Overview

The goal of this playground is to explore and experiment with techniques that produce **consistent** image outputs from generative AI models — meaning the same character, style, scene, or aesthetic is preserved across multiple generations.

The focus is on **illustrated and anime styles** (concept art, fantasy illustration, manga-inspired visuals). Photorealistic generation is out of scope; prompts, templates, and model defaults throughout this repo reflect the illustrated-first approach.

This matters for:
- Storyboarding and concept art (same character across scenes)
- Illustrated character sheets and multi-panel comics
- Brand/style consistency in AI-assisted design
- Research into controllability of diffusion models

> **Default model:** `stabilityai/stable-diffusion-xl-base-1.0` (SDXL). Community fine-tunes
> such as [Animagine XL](https://huggingface.co/cagliostrolab/animagine-xl-3.1) can be swapped in
> via `configs/model_config.yaml` for stronger anime-style results.
>
> **Note:** Video generation support is planned for a future phase. All current tooling is focused exclusively on image generation.

---

## Consistency Techniques

### IP-Adapter
Conditions generation on a **reference image embedding** (via a CLIP image encoder), keeping the subject or style consistent without any fine-tuning of the base model.

- 🔗 [IP-Adapter Paper](https://arxiv.org/abs/2308.06721)
- 🔗 [IP-Adapter GitHub](https://github.com/tencent-ailab/IP-Adapter)
- 🔗 [Hugging Face Diffusers Integration](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)

### LoRA (Low-Rank Adaptation)
Trains a small, efficient adapter on a handful of reference images. Fast to train, easy to swap, and composable with other techniques.

- 🔗 [LoRA Paper](https://arxiv.org/abs/2106.09685)
- 🔗 [PEFT Library (Hugging Face)](https://huggingface.co/docs/peft/conceptual_guides/lora)
- 🔗 [Diffusers LoRA Training Guide](https://huggingface.co/docs/diffusers/training/lora)

### DreamBooth
Fine-tunes the base model to bind a unique token (e.g., `sks person`) to a specific subject from a small set of images. Higher fidelity but more compute-intensive than LoRA.

- �� [DreamBooth Paper](https://arxiv.org/abs/2208.12242)
- 🔗 [Diffusers DreamBooth Training Guide](https://huggingface.co/docs/diffusers/training/dreambooth)

### Textual Inversion
Learns a new "word" (embedding) that represents a concept from a small set of images. Lightweight, but less expressive than LoRA/DreamBooth.

- 🔗 [Textual Inversion Paper](https://arxiv.org/abs/2208.01618)
- 🔗 [Diffusers Textual Inversion Guide](https://huggingface.co/docs/diffusers/training/text_inversion)

### ControlNet
Adds spatial conditioning (pose, depth, edge maps) to the generation process, keeping composition and structure consistent across outputs.

- 🔗 [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- 🔗 [Diffusers ControlNet Guide](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)

### Reference-Only / Self-Attention Injection
Injects attention features from a reference image directly into the denoising process at inference time — **no training required**. Also known as "reference-only" conditioning in Diffusers.

- 🔗 [Diffusers Reference-Only Pipeline](https://huggingface.co/docs/diffusers/using-diffusers/reference_only)

### Seed Locking
Using the same random seed across generations ensures reproducibility. Combined with a fixed base prompt, it produces visually similar outputs.

### Prompt Anchors
Define a **base prompt** with fixed descriptors (lighting, style, character tags) and vary only the action or scene. This forms the simplest consistency technique.

### CLIP / BLIP Similarity Scoring
Automatically score generated outputs against a reference image or text description using [CLIP](https://github.com/openai/CLIP) or [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) embeddings. Used to filter and rank outputs for consistency.

### Shared Latent Space
Encode a reference image into the model's latent space and use it as the starting point (or noise initializer) for variations — keeping the generation "close" to the reference.

---

## Repository Structure

```
gen-ai-playground/
├── notebooks/                               # Jupyter notebooks for interactive experimentation
│   ├── 01_image_gen.ipynb                   # Basic image generation with Stable Diffusion
│   ├── 02_lora_ip_adapter_dreambooth.ipynb  # LoRA, IP-Adapter, DreamBooth exploration
│   └── 03_consistency_experiments.ipynb     # Prompt consistency and scoring experiments
│
├── src/
│   ├── image/                          # Image generation pipeline modules
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # Base image generation pipeline wrappers
│   │   ├── ip_adapter.py               # IP-Adapter consistency technique
│   │   ├── lora.py                     # LoRA adapter loading and inference
│   │   ├── dreambooth.py               # DreamBooth inference utilities
│   │   ├── textual_inversion.py        # Textual Inversion inference utilities
│   │   ├── controlnet.py               # ControlNet-conditioned generation
│   │   └── reference_only.py           # Reference-only / attention injection
│   │
│   ├── training/                       # Training scripts for adapters
│   │   ├── __init__.py
│   │   ├── train_lora.py               # LoRA fine-tuning script
│   │   ├── train_dreambooth.py         # DreamBooth fine-tuning script
│   │   └── train_textual_inversion.py  # Textual Inversion training script
│   │
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       ├── consistency_scoring.py      # Consistency metrics and scoring helpers
│       ├── seed_utils.py               # Seed locking and reproducibility helpers
│       ├── prompt_helpers.py           # Prompt building and anchor utilities
│       ├── clip_blip_scoring.py        # CLIP/BLIP embedding similarity scoring
│       └── latent_utils.py             # Shared latent space utilities
│
├── tests/                              # Pytest test suite (no GPU required)
│   ├── test_seed_utils.py
│   ├── test_prompt_helpers.py
│   ├── test_consistency_scoring.py
│   ├── test_lora_pipeline.py
│   └── test_dreambooth_pipeline.py
│
├── configs/
│   ├── model_config.yaml               # Example model and pipeline configurations
│   └── prompt_templates.yaml           # Reusable prompt templates and anchor patterns
│
├── assets/
│   └── README.md                       # Instructions for adding reference/output images
│
├── pyproject.toml                      # Project metadata + uv / pytest / ruff config
├── requirements.txt                    # Core Python dependencies
└── README.md                           # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A CUDA-capable GPU is strongly recommended (≥8 GB VRAM for SDXL, ≥4 GB for SD 1.5)
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/) (**recommended** — fast Python package manager)

### Installation with `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a modern, ultra-fast Python package and environment manager.
Install it once, then use it everywhere instead of pip/venv.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository
git clone https://github.com/Floofy-KH/gen-ai-playground.git
cd gen-ai-playground

# 2. Install all dependencies and activate the environment
uv sync --extra dev --extra ml
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

> **GPU / CUDA users:** add the `cuda` extra so `torch`/`torchvision` are included,
> then replace them with CUDA-enabled wheels for your driver version:
>
> ```bash
> uv sync --extra dev --extra ml --extra cuda
>
> # Replace with the CUDA-enabled wheels that match your driver:
> uv pip install torch torchvision \
>     --index-url https://download.pytorch.org/whl/cu128   # adjust cu128 to your CUDA version
> ```

### Installation with pip (alternative)

```bash
# 1. Clone the repository
git clone https://github.com/Floofy-KH/gen-ai-playground.git
cd gen-ai-playground

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
# Launch Jupyter Lab
jupyter lab

# Or classic Jupyter Notebook
jupyter notebook
```

Open any notebook from the `notebooks/` directory to get started.

---

## Running Tests

The test suite covers pure-Python utilities (no GPU or model weights required).

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing
```

---

## Usage Examples

### Basic Image Generation (Python)

```python
from src.image.pipeline import ImageGenerationPipeline

# SDXL — native 1024 px, great for illustrated and anime styles
pipeline = ImageGenerationPipeline(model_id="stabilityai/stable-diffusion-xl-base-1.0")
image = pipeline.generate(
    prompt="sks character, 1girl, blue eyes, long silver hair, standing in a fantasy forest, anime style, masterpiece, best quality",
    negative_prompt="photorealistic, realistic, blurry, deformed, extra limbs",
    seed=42,
    num_inference_steps=30,
    width=1024,
    height=1024,
)
image.save("output.png")
```

### IP-Adapter Consistency

```python
from PIL import Image
from src.image.ip_adapter import IPAdapterPipeline

pipeline = IPAdapterPipeline(
    base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
    ip_adapter_model_id="h94/IP-Adapter",
)
reference_image = Image.open("assets/reference.png")
image = pipeline.generate(
    prompt="sks character in a cyberpunk city, anime style, neon lights, masterpiece",
    reference_image=reference_image,
    ip_adapter_scale=0.6,
    seed=42,
)
image.save("output_ip_adapter.png")
```

### CLIP Similarity Scoring

```python
from src.utils.clip_blip_scoring import CLIPScorer

scorer = CLIPScorer()
score = scorer.image_similarity(image_a, image_b)
print(f"CLIP similarity: {score:.4f}")  # 1.0 = identical embeddings
```

### Locked Seed Batch Generation

```python
from src.utils.seed_utils import generate_with_locked_seed
from src.image.pipeline import ImageGenerationPipeline

pipeline = ImageGenerationPipeline(model_id="stabilityai/stable-diffusion-xl-base-1.0")
prompts = [
    "sks warrior standing on a mountain peak, fantasy illustration, masterpiece",
    "sks warrior in a dark dungeon, dramatic lighting, anime style, best quality",
    "sks warrior at the edge of a glowing portal, concept art, artstation",
]
images = generate_with_locked_seed(pipeline, prompts, seed=42)
```

---

## Further Reading & Resources

### Papers

| Paper | Link |
|---|---|
| Stable Diffusion (LDM) | [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752) |
| SDXL | [arxiv.org/abs/2307.01952](https://arxiv.org/abs/2307.01952) |
| IP-Adapter | [arxiv.org/abs/2308.06721](https://arxiv.org/abs/2308.06721) |
| LoRA | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| DreamBooth | [arxiv.org/abs/2208.12242](https://arxiv.org/abs/2208.12242) |
| Textual Inversion | [arxiv.org/abs/2208.01618](https://arxiv.org/abs/2208.01618) |
| ControlNet | [arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543) |
| CLIP | [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020) |

### Libraries & Tools

| Resource | Link |
|---|---|
| Hugging Face Diffusers | [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers) |
| Hugging Face PEFT (LoRA) | [huggingface.co/docs/peft](https://huggingface.co/docs/peft) |
| Hugging Face Accelerate | [huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate) |
| Hugging Face Transformers | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) |
| IP-Adapter GitHub | [github.com/tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) |
| Hugging Face Hub | [huggingface.co](https://huggingface.co) |
| OpenAI CLIP | [github.com/openai/CLIP](https://github.com/openai/CLIP) |
| Stable Diffusion WebUI | [github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) |

---

## Contributing

Contributions are very welcome! This is an experimental playground, so:

- Feel free to open issues with ideas, bug reports, or questions
- PRs adding new techniques, notebooks, or utility functions are encouraged
- Keep implementations in the appropriate `src/` subdirectory
- Add a docstring to every new function/class (see existing stubs for the style)
- If you add a new dependency, update `requirements.txt`

---

> ⚠️ **Disclaimer:** This is a research playground. Model weights, adapters, and outputs may be subject to the licenses of their respective base models. Always check licensing before using generated content commercially.

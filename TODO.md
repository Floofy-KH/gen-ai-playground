# TODO

Outstanding tasks for the gen-ai-playground project, grouped by area. Items marked
**[requires GPU]** need model weights and a CUDA-capable device. Pure-Python items
can be completed and tested locally.

---

## Image pipelines (`src/image/`)

All pipeline classes currently raise `NotImplementedError`. Each has a detailed
`# Implementation outline:` comment in its docstring to guide the work.

- [ ] **`pipeline.py`** — Implement `ImageGenerationPipeline._load_pipeline()`:
  load a `diffusers` `StableDiffusionXLPipeline` (or equivalent) from
  `self.model_id`, move to device, enable memory-efficient attention.
- [ ] **`pipeline.py`** — Implement `ImageGenerationPipeline.generate()`:
  build a `torch.Generator` from `seed`, call `self._pipe`, return `PIL.Image`.
- [ ] **`lora.py`** — Implement `LoRAPipeline._load_pipeline()`: load base
  pipeline, then load each LoRA weight via `load_lora_weights()` and fuse.
- [ ] **`lora.py`** — Implement `LoRAPipeline.generate()`: pass
  `cross_attention_kwargs={"scale": lora_scale}` to the pipeline call.
- [ ] **`lora.py`** — Implement `LoRAPipeline.unload_lora()`: call
  `self._pipe.unload_lora_weights()`.
- [ ] **`dreambooth.py`** — Implement `DreamBoothPipeline._load_pipeline()`:
  load the DreamBooth fine-tuned checkpoint from `self.model_id`.
- [ ] **`dreambooth.py`** — Implement `DreamBoothPipeline.generate()`:
  prepend `subject_token` to the prompt (already handled by `_inject_subject_token`),
  build generator, call `self._pipe`, return image.
- [ ] **`ip_adapter.py`** — Implement `IPAdapterPipeline._load_pipeline()`:
  load base pipeline, then call `load_ip_adapter()` with the configured subfolder
  and weight name (SDXL defaults: `sdxl_models` / `ip-adapter_sdxl.bin`).
- [ ] **`ip_adapter.py`** — Implement `IPAdapterPipeline.generate()`:
  set `ip_adapter_scale` on the pipe, pass `ip_adapter_image`, call the pipeline.
- [ ] **`textual_inversion.py`** — Implement `TextualInversionPipeline._load_pipeline()`:
  load base pipeline and call `load_textual_inversion()` for each embedding.
- [ ] **`textual_inversion.py`** — Implement `TextualInversionPipeline.generate()`.
- [ ] **`controlnet.py`** — Implement `ControlNetPipeline._load_pipeline()`:
  instantiate `ControlNetModel` from `controlnet_model_id`, then
  `StableDiffusionXLControlNetPipeline`.
- [ ] **`controlnet.py`** — Implement `ControlNetPipeline.generate()`:
  pass `image=control_image` and `controlnet_conditioning_scale` to the pipeline.
- [ ] **`controlnet.py`** — Implement `ControlNetPipeline.prepare_depth()`:
  run a depth-estimation model (e.g. `Intel/dpt-large`) on the input image.
- [ ] **`reference_only.py`** — Implement `ReferenceOnlyPipeline._load_pipeline()`:
  load the pipeline with reference-attention support.
- [ ] **`reference_only.py`** — Implement `ReferenceOnlyPipeline.generate()`:
  pass `ref_image` and `reference_attn_weight` / `reference_adain_weight`.

---

## Training scripts (`src/training/`)

All training loops raise `NotImplementedError`. Each script includes a full
`# Implementation outline:` in its docstring.

- [ ] **`train_lora.py`** — Implement `train()`: set up `accelerate`, freeze
  base weights, inject LoRA layers into UNet attention modules, run the
  diffusion training loop, save checkpoints.
- [ ] **`train_dreambooth.py`** — Implement `generate_class_images()`:
  generate prior-preservation class images with the base model.
- [ ] **`train_dreambooth.py`** — Implement `train()`: set up `accelerate`,
  combine instance + class datasets, run DreamBooth training loop, save
  fine-tuned checkpoint.
- [ ] **`train_textual_inversion.py`** — Implement the training loop:
  register a new token embedding, freeze all other weights, optimise the
  embedding on the provided images, save the learned embedding.

---

## Scoring utilities (`src/utils/`)

- [ ] **`clip_blip_scoring.py — CLIPScorer`** — Implement `_load_model()`:
  load `openai/clip-vit-large-patch14` (or configurable) via `transformers`.
- [ ] **`clip_blip_scoring.py — CLIPScorer`** — Implement `_get_image_embedding()`:
  preprocess image with `CLIPProcessor`, run through vision encoder, L2-normalise.
- [ ] **`clip_blip_scoring.py — CLIPScorer`** — Implement `_get_text_embedding()`:
  tokenise text, run through text encoder, L2-normalise.
- [ ] **`clip_blip_scoring.py — BLIPScorer`** — Implement `_load_model()`:
  load `Salesforce/blip-image-captioning-large` via `transformers`.
- [ ] **`clip_blip_scoring.py — BLIPScorer`** — Implement `caption()`:
  run image through BLIP, decode output tokens, return caption string.
- [ ] **`clip_blip_scoring.py — BLIPScorer`** — Implement `similarity()`:
  caption the candidate image and compare against the reference caption using
  BLIP's image-text matching score or token overlap.
- [ ] **`consistency_scoring.py`** — Wire up `CLIPScorer` in `ConsistencyScorer`
  once `CLIPScorer` is implemented; enable `use_clip=True` as the new default.

---

## Latent utilities (`src/utils/latent_utils.py`)

- [ ] Implement `encode_image_to_latent()`: convert `PIL.Image` → tensor,
  encode through `vae.encode()`, sample from the posterior distribution,
  scale by `vae.config.scaling_factor`.
- [ ] Implement `decode_latent_to_image()`: unscale latent, run through
  `vae.decode()`, convert output tensor to `PIL.Image`.
- [ ] Implement `_lerp()`: linear interpolation between two tensors.
- [ ] Implement `_slerp()`: spherical linear interpolation (SLERP) between
  two tensors; handle the collinear edge case gracefully.
- [ ] Implement `add_noise_to_latent()`: create a `torch.Generator` from
  `seed` if provided, sample Gaussian noise scaled by `noise_strength`, add
  to the input latent.

---

## Notebooks (`notebooks/`)

- [ ] Run `01_image_gen.ipynb` end-to-end once `ImageGenerationPipeline` is
  implemented; replace `# TODO` stub cells with live outputs.
- [ ] Run `02_lora_ip_adapter_dreambooth.ipynb` once IP-Adapter, LoRA, and
  DreamBooth pipelines are implemented; fill in real model paths.
- [ ] Run `03_consistency_experiments.ipynb` once `CLIPScorer` and
  `latent_utils` are implemented; replace `# TODO` cells with live outputs.

---

## Testing

- [ ] Add tests for `latent_utils` once the pure-Python helpers (`_lerp`,
  `_slerp`) are implemented (no GPU required for those two).
- [ ] Add integration / smoke tests for pipeline classes using a tiny
  test-only checkpoint once the stubs are filled in.

---

## Infrastructure / DX

- [x] Add a GitHub Actions CI workflow (`.github/workflows/ci.yml`) to run
  `pytest` on every push / PR (pure-Python tests pass without a GPU).
- [x] Consider adding `ruff` and `black` formatting checks to CI.
- [x] Evaluate adding a `uv.lock` file to pin exact dependency versions for
  reproducible environments.

# assets/

This directory holds **reference images** and **example outputs** used for
demonstration, testing, and consistency benchmarking.

## Structure

```
assets/
├── reference/          # Reference images for IP-Adapter, DreamBooth, etc.
│   └── (place your reference images here, e.g. my_character.png)
├── outputs/            # Generated output images for comparison / showcase
│   └── (generated images land here by default)
└── README.md           # This file
```

## Adding Reference Images

Place your reference images in `assets/reference/`.  Recommended guidelines:

- **Format**: PNG or JPEG
- **Resolution**: At least 512×512 pixels (crop/resize as needed)
- **Quantity**: 3–30 images per subject for LoRA/DreamBooth; a single image
  works for IP-Adapter and reference-only sampling
- **Variety**: Include different angles, lighting conditions, and expressions
  for best adapter quality

## Notes

- Reference images are **not committed** to the repository by default.
  Add your own images locally and update `.gitignore` if you want to keep
  them private.
- Example synthetic outputs generated during testing may be committed for
  documentation purposes, subject to the license of the model used.

> ⚠️ Only use images you have the right to use.  Check model licenses before
> generating or distributing outputs.

"""
src/utils/prompt_helpers.py

Prompt building, anchor patterns, and template utilities.

A common and lightweight consistency technique is **prompt anchoring**: define
a fixed "base prompt" that describes invariant properties (character identity,
style, lighting) and vary only the action or setting.  This module provides
helpers for building, validating, and expanding anchored prompts.

References:
    - Prompt engineering for Stable Diffusion:
      https://stable-diffusion-art.com/prompt-guide/
    - Hugging Face prompt techniques:
      https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptTemplate:
    """A reusable prompt template with fixed anchors and variable slots.

    Anchors are fixed descriptor strings that are concatenated to every prompt
    to ensure consistent style, character, or quality.  Variable slots are
    placeholders (``{slot_name}``) that are filled in per-generation.

    Attributes:
        anchor: Fixed base descriptors that appear in every generated prompt
            (e.g. character description, art style, quality tags).
        template: Template string with ``{slot}`` placeholders for the
            variable part of the prompt (e.g. ``"{subject} {action} in
            {setting}"``).
        negative_prompt: Shared negative prompt to suppress unwanted content.
        quality_tags: Optional list of quality booster tokens appended to
            every prompt (e.g. ``["masterpiece", "best quality"]``).

    Example::

        from src.utils.prompt_helpers import PromptTemplate

        t = PromptTemplate(
            anchor="sks person, blue eyes, long dark hair, 1girl",
            template="{anchor}, {action}, {setting}, dramatic lighting",
            quality_tags=["masterpiece", "best quality", "8k"],
            negative_prompt="blurry, deformed, extra limbs",
        )
        prompt = t.build(action="sitting by a campfire", setting="forest at night")
        # "sks person, blue eyes, long dark hair, 1girl, sitting by a campfire,
        #  forest at night, dramatic lighting, masterpiece, best quality, 8k"
    """

    anchor: str = ""
    template: str = "{anchor}, {action}, {setting}"
    negative_prompt: str = ""
    quality_tags: List[str] = field(default_factory=list)

    def build(self, **slot_values: str) -> str:
        """Fill template slots and append quality tags.

        Args:
            **slot_values: Keyword arguments matching the ``{slot}``
                placeholders in ``self.template``.  The ``{anchor}``
                placeholder is always populated from ``self.anchor``.

        Returns:
            Fully expanded prompt string.

        Raises:
            KeyError: If a required slot in the template is not provided.
        """
        slot_values.setdefault("anchor", self.anchor)
        prompt = self.template.format(**slot_values)
        if self.quality_tags:
            prompt = prompt.rstrip(", ") + ", " + ", ".join(self.quality_tags)
        return prompt

    def build_batch(self, variations: List[Dict[str, str]]) -> List[str]:
        """Build a batch of prompts from a list of slot-value dicts.

        Args:
            variations: List of dicts, each mapping slot names to values.

        Returns:
            List of expanded prompt strings.
        """
        return [self.build(**v) for v in variations]


def build_anchored_prompt(
    anchor: str,
    action: str,
    setting: str = "",
    style: str = "",
    quality_tags: Optional[List[str]] = None,
    separator: str = ", ",
) -> str:
    """Build a simple anchored prompt by concatenating fixed and variable parts.

    This is a functional alternative to :class:`PromptTemplate` for quick
    one-off prompt construction.

    Args:
        anchor: Fixed character/style descriptor (e.g. ``"sks person, blue
            hair"``).
        action: Variable action or pose (e.g. ``"walking on the beach"``).
        setting: Optional scene/environment descriptor.
        style: Optional art style descriptor (e.g. ``"oil painting, baroque"``).
        quality_tags: Optional list of quality booster tokens.
        separator: Token used to join prompt parts.

    Returns:
        Assembled prompt string.

    Example::

        p = build_anchored_prompt(
            anchor="sks astronaut",
            action="floating in zero gravity",
            setting="aboard a space station",
            style="cinematic, volumetric lighting",
            quality_tags=["4k", "photorealistic"],
        )
    """
    parts = [anchor, action]
    if setting:
        parts.append(setting)
    if style:
        parts.append(style)
    if quality_tags:
        parts.extend(quality_tags)
    # Remove empty strings and strip whitespace
    parts = [p.strip() for p in parts if p.strip()]
    return separator.join(parts)


def extract_slots(template: str) -> List[str]:
    """Extract the names of all ``{slot}`` placeholders in a template string.

    Args:
        template: Template string (e.g. ``"{anchor}, {action} in {setting}"``).

    Returns:
        Sorted list of unique slot names.

    Example::

        slots = extract_slots("{anchor}, {action} in {setting}")
        # ["action", "anchor", "setting"]
    """
    return sorted(set(re.findall(r"\{(\w+)\}", template)))


def validate_prompt(prompt: str, max_tokens: int = 75) -> bool:
    """Check whether a prompt is within the CLIP tokenizer token limit.

    Standard CLIP text encoders have a hard limit of 77 tokens (including
    the start and end tokens), leaving 75 tokens for the prompt content.
    Prompts exceeding this are silently truncated by the tokenizer.

    Args:
        prompt: Prompt string to validate.
        max_tokens: Maximum number of whitespace-delimited words to allow
            as a rough proxy for token count.  The actual token count depends
            on the tokenizer vocabulary.

    Returns:
        ``True`` if the prompt is likely within the limit, ``False`` otherwise.

    Note:
        This is a word-count heuristic.  For exact token counts, use the
        actual CLIP tokenizer from the ``transformers`` library::

            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            tokens = tokenizer(prompt, truncation=False)["input_ids"]
            is_valid = len(tokens) <= 77
    """
    word_count = len(prompt.split())
    return word_count <= max_tokens

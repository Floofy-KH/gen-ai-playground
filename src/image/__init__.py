"""
src/image/__init__.py

Image generation sub-package.

Exposes the primary pipeline classes and consistency technique helpers
so callers can do, e.g.::

    from src.image import ImageGenerationPipeline, IPAdapterPipeline
"""

from src.image.controlnet import ControlNetPipeline
from src.image.dreambooth import DreamBoothPipeline
from src.image.ip_adapter import IPAdapterPipeline
from src.image.lora import LoRAPipeline
from src.image.pipeline import ImageGenerationPipeline
from src.image.reference_only import ReferenceOnlyPipeline
from src.image.textual_inversion import TextualInversionPipeline

__all__ = [
    "ImageGenerationPipeline",
    "IPAdapterPipeline",
    "LoRAPipeline",
    "DreamBoothPipeline",
    "TextualInversionPipeline",
    "ControlNetPipeline",
    "ReferenceOnlyPipeline",
]

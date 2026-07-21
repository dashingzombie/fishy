"""Model factories and the shared multi-task classification head."""

from .factory import build_model
from .multitask import MultiTaskClassifier, build_multitask_model
from .long_tail import CosineClassifier, ProjectionHead, PrototypeClassifier

__all__ = [
    "CosineClassifier", "MultiTaskClassifier", "ProjectionHead",
    "PrototypeClassifier", "build_model", "build_multitask_model",
]

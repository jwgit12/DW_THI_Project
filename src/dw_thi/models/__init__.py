"""Model registry for DWI -> DTI architectures."""

from __future__ import annotations

from typing import Type

import torch.nn as nn

from ..model import QSpaceUNet
from .mrd_cnn import MRDCNN
from .qspace_attention_net import QSpaceAttentionNet

MODEL_REGISTRY: dict[str, Type[nn.Module]] = {
    "qspace_unet": QSpaceUNet,
    "qspace_attention": QSpaceAttentionNet,
    "mrd_cnn": MRDCNN,
}


def model_names() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def build_model(name: str, **kwargs) -> nn.Module:
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        options = ", ".join(model_names())
        raise ValueError(f"Unknown model {name!r}; expected one of: {options}") from exc
    return model_cls(**kwargs)


__all__ = [
    "MODEL_REGISTRY",
    "MRDCNN",
    "QSpaceAttentionNet",
    "QSpaceUNet",
    "build_model",
    "model_names",
]

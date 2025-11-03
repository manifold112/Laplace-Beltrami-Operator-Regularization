from typing import Dict, Any

import torch.nn as nn

from .hma import HMABackbone
from .signbert import SignBERTBackbone
from .skim import SKIMBackbone


def build_model_from_config(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    """
    Build a backbone + classifier model according to config.

    Config fields (under `model` in YAML):
        name:        one of ["hma", "signbert", "skim"]
        input_dim:   input feature dimension (e.g., T * J * C)
        feat_dim:    embedding dimension for LBOR
        hidden_dim:  hidden dimension inside MLP (optional)

    Example in YAML:

        model:
          name: "hma"
          input_dim: 3900         # e.g. 52 frames * 25 joints * 3 coords
          feat_dim: 512
          hidden_dim: 1024
    """
    name = str(cfg.get("name", "hma")).lower()
    input_dim = int(cfg.get("input_dim"))
    feat_dim = int(cfg.get("feat_dim", 512))
    hidden_dim = int(cfg.get("hidden_dim", 1024))

    if input_dim <= 0:
        raise ValueError("`model.input_dim` must be a positive integer in the config.")

    if name == "hma":
        model = HMABackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
        )
    elif name == "signbert":
        model = SignBERTBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
        )
    elif name == "skim":
        model = SKIMBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model

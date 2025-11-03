import torch
import torch.nn as nn


class HMABackbone(nn.Module):
    """
    A simple placeholder backbone for skeleton-based ISLR.

    This is NOT the original HMA implementation. It is a minimal MLP-based
    encoder + classifier used to demonstrate how LBOR can be integrated.

    Expected input shape:
        (B, T, J, C)  or already flattened as (B, input_dim).

    The config is expected to provide:
        - input_dim:  T * J * C, if you feed flattened inputs
                      (otherwise we will flatten inside forward).
        - feat_dim:   embedding dimension used by LBOR.
        - hidden_dim: hidden dimension inside MLP (optional).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        feat_dim: int = 512,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.feat_dim = int(feat_dim)
        hidden_dim = int(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape (B, T, J, C) or (B, input_dim).

        Returns
        -------
        logits : Tensor, shape (B, num_classes)
        features : Tensor, shape (B, feat_dim)
        """
        if x.dim() > 2:
            # Flatten temporal and joint dimensions.
            x = x.view(x.size(0), -1)

        assert (
            x.size(1) == self.input_dim
        ), f"Expected input_dim={self.input_dim}, got {x.size(1)}"

        feat = self.mlp(x)
        logits = self.classifier(feat)
        return logits, feat

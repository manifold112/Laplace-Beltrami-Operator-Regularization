import torch
import torch.nn as nn


class SignBERTBackbone(nn.Module):
    """
    Placeholder backbone for a SignBERT-style model.

    This is a simple MLP-based encoder used as a template.
    Replace it with your actual SignBERT implementation when ready.
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
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        assert (
            x.size(1) == self.input_dim
        ), f"Expected input_dim={self.input_dim}, got {x.size(1)}"
        feat = self.mlp(x)
        logits = self.classifier(feat)
        return logits, feat

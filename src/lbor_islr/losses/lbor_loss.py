import torch
import torch.nn as nn
import torch.nn.functional as F


class LBORLoss(nn.Module):
    """
    LBOR: Laplace–Beltrami Operator Regularization for skeleton-based ISLR.

    This loss is designed to be used together with a standard classification loss
    (e.g., cross-entropy). It adds:

    1) A within-class Laplacian regularization term (Dirichlet energy) that
       encourages smooth and connected manifolds for each class in the feature space.

    2) A center-level margin term that enforces a minimum distance between
       class centers in the feature space.

    Usage:
        criterion = LBORLoss(num_classes=C, feat_dim=D, lambda_lap=1.0, mu_margin=0.1)
        total_loss, loss_dict = criterion(logits, features, labels)

    Arguments
    ---------
    num_classes : int
        Total number of classes in the dataset.
    feat_dim : int
        Dimensionality of the feature embedding z.
    lambda_lap : float
        Weight of the within-class Laplacian regularization term.
    mu_margin : float
        Weight of the center-level margin term.
    margin_M : float
        Target squared distance between class centers (margin).
    tau : float
        Temperature / bandwidth parameter for the Gaussian kernel on edges.
    use_knn : bool
        If True, build a within-class kNN graph; otherwise use fully connected
        within-class graphs.
    knn_k : int
        Number of neighbours when using kNN graphs.
    center_momentum : float
        Momentum for updating running class centers (EMA).
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        lambda_lap: float = 1.0,
        mu_margin: float = 0.1,
        margin_M: float = 1.0,
        tau: float = 1.0,
        use_knn: bool = False,
        knn_k: int = 5,
        center_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)

        self.lambda_lap = float(lambda_lap)
        self.mu_margin = float(mu_margin)
        self.margin_M = float(margin_M)
        self.tau = float(tau)
        self.use_knn = bool(use_knn)
        self.knn_k = int(knn_k)
        self.center_momentum = float(center_momentum)

        # Cross-entropy for the base classification loss.
        self.ce = nn.CrossEntropyLoss()

        # Running (EMA) class centers; used mainly for stability / visualization.
        self.register_buffer(
            "running_centers",
            torch.zeros(self.num_classes, self.feat_dim),
        )
        self.register_buffer(
            "centers_initialized",
            torch.zeros(self.num_classes, dtype=torch.bool),
        )

    # --------------------------------------------------------------------- #
    # Public forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Parameters
        ----------
        logits : Tensor, shape (B, C)
            Output of the classification head.
        features : Tensor, shape (B, D)
            Embedding vectors of the samples.
        labels : Tensor, shape (B,)
            Ground-truth class indices.

        Returns
        -------
        total_loss : Tensor (scalar)
            Overall loss = CE + lambda_lap * L_lap + mu_margin * L_margin.
        loss_dict : dict
            Dictionary with detached scalars for logging.
        """
        if features.dim() != 2:
            raise ValueError(f"features must be 2D (B, D), got shape {features.shape}")
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D (B, C), got shape {logits.shape}")
        if labels.dim() != 1:
            raise ValueError(f"labels must be 1D (B,), got shape {labels.shape}")

        # Standard classification loss.
        base_loss = self.ce(logits, labels)

        # Laplacian regularization within each class.
        lap_loss = self.within_class_laplacian_loss(features, labels)

        # Center-level margin between class centers.
        margin_loss = self.center_margin_loss(features, labels)

        total_loss = base_loss
        total_loss = total_loss + self.lambda_lap * lap_loss
        total_loss = total_loss + self.mu_margin * margin_loss

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_ce": base_loss.detach(),
            "loss_lap": lap_loss.detach(),
            "loss_margin": margin_loss.detach(),
        }
        return total_loss, loss_dict

    # --------------------------------------------------------------------- #
    # (1) Within-class Laplacian regularization
    # --------------------------------------------------------------------- #
    def within_class_laplacian_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the within-class Laplacian (Dirichlet) energy term:

            L_lap = 1/2 * sum_{i,j} A_ij * ||z_i - z_j||^2

        where A_ij > 0 only if samples i and j belong to the same class
        (and optionally are kNN neighbours).

        To keep the scale stable, the energy is normalized by the batch size.
        """
        device = features.device
        B, D = features.shape

        if B <= 1:
            return torch.zeros((), device=device)

        # Pairwise squared Euclidean distances: shape (B, B).
        # torch.cdist is simple and reasonably efficient for moderate B.
        dist2 = torch.cdist(features, features, p=2.0) ** 2

        # Same-class mask; remove self-connections on the diagonal.
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=device)
        same_class_mask = label_eq & (~eye)

        if self.use_knn:
            # kNN inside each class (per-row top-k).
            large_val = 1e9
            masked_dist = dist2.clone()
            masked_dist[~same_class_mask] = large_val

            # k cannot exceed B-1 (exclude self).
            k = max(1, min(self.knn_k, B - 1))

            knn_dist, knn_idx = torch.topk(
                masked_dist, k=k, dim=1, largest=False
            )  # (B, k)

            # Build adjacency matrix A with Gaussian weights.
            A = torch.zeros_like(dist2)
            row_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(knn_idx)

            knn_weight = torch.exp(-knn_dist / self.tau)
            A[row_idx, knn_idx] = knn_weight

            # Symmetrize, since kNN is directional.
            A = 0.5 * (A + A.t())
        else:
            # Fully-connected within-class graph with Gaussian weights.
            A = torch.exp(-dist2 / self.tau) * same_class_mask.float()

        # Dirichlet energy: 1/2 * sum_{i,j} A_ij * ||z_i - z_j||^2
        # Normalize by B to avoid scaling with batch size.
        energy = 0.5 * (A * dist2).sum() / float(B)

        return energy

    # --------------------------------------------------------------------- #
    # (2) Center-level margin
    # --------------------------------------------------------------------- #
    def center_margin_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Center-level margin term:

            L_margin = mean_{(c, c'), c != c'} max(0, M - ||mu_c - mu_c'||^2),

        where mu_c is the (batch) center of class c.

        This version:
        - Uses batch centers for backpropagation (so the gradient flows to
          features directly).
        - Updates running EMA centers for stability/visualization.
        """
        device = features.device
        batch_classes = labels.unique()
        num_batch_classes = int(batch_classes.numel())

        if num_batch_classes <= 1:
            # No class pairs in this batch.
            return torch.zeros((), device=device)

        # Compute batch centers for classes present in the batch.
        centers_batch = []
        valid_classes = []
        for c in batch_classes:
            mask = labels == c
            count = int(mask.sum().item())
            if count == 0:
                continue
            mu_c = features[mask].mean(dim=0)  # (D,)
            centers_batch.append(mu_c)
            valid_classes.append(int(c.item()))

        if not centers_batch:
            return torch.zeros((), device=device)

        centers_batch = torch.stack(centers_batch, dim=0)  # (Cb, D)
        num_centers = int(centers_batch.size(0))

        if num_centers <= 1:
            return torch.zeros((), device=device)

        # Update EMA running centers (no gradient).
        with torch.no_grad():
            for idx_in_batch, c_int in enumerate(valid_classes):
                mu_c_batch = centers_batch[idx_in_batch].detach()
                if not self.centers_initialized[c_int]:
                    self.running_centers[c_int] = mu_c_batch
                    self.centers_initialized[c_int] = True
                else:
                    m = self.center_momentum
                    self.running_centers[c_int] = (
                        (1.0 - m) * self.running_centers[c_int] + m * mu_c_batch
                    )

        # Compute pairwise squared distances between batch centers.
        center_dist2 = torch.cdist(centers_batch, centers_batch, p=2.0) ** 2

        # Use upper-triangular entries (c < c') to avoid duplicates and self-pairs.
        idx_i, idx_j = torch.triu_indices(num_centers, num_centers, offset=1)
        pair_dist2 = center_dist2[idx_i, idx_j]

        if pair_dist2.numel() == 0:
            return torch.zeros((), device=device)

        # Hinge-like margin: max(0, M - ||mu_c - mu_c'||^2)
        margin_term = F.relu(self.margin_M - pair_dist2)
        loss_margin = margin_term.mean()

        return loss_margin

from __future__ import annotations

import torch
from torch import nn


def _orthogonal_unit(vector: torch.Tensor, against: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    value = vector.clone()
    for basis in against:
        value = value - torch.dot(value, basis) * basis
    norm = value.norm()
    if norm <= eps:
        raise ValueError("fiducials do not span the requested canonical chart")
    return value / norm


def build_fiducials(class_count: int, ambient_dim: int, chart_dim: int, seed: int = 1729) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed semantic anchors and deterministic class tangent bases.

    The buffers are a public gauge. They are neither optimized nor federated.
    Calibration directions complete the semantic simplex when chart_dim exceeds
    its local rank.
    """
    generator = torch.Generator().manual_seed(seed)
    if ambient_dim < class_count - 1:
        raise ValueError("ambient_dim must contain the semantic simplex")
    centered = torch.eye(class_count, dtype=torch.float64) - torch.ones(
        class_count, class_count, dtype=torch.float64,
    ) / class_count
    values, vectors = torch.linalg.eigh(centered)
    simplex = centered @ vectors[:, values > 0.5]
    simplex = simplex / simplex.norm(dim=1, keepdim=True)
    random_subspace = torch.randn(ambient_dim, class_count - 1, generator=generator, dtype=torch.float64)
    embedding, _ = torch.linalg.qr(random_subspace, mode="reduced")
    c = (simplex @ embedding.T).float()
    calibration = torch.randn(max(chart_dim + 1, class_count), ambient_dim, generator=generator)
    bases = []
    for anchor in c:
        candidates = [v - torch.dot(v, anchor) * anchor for v in torch.cat([c, calibration], dim=0)]
        tangent: list[torch.Tensor] = []
        for candidate in candidates:
            try:
                tangent.append(_orthogonal_unit(candidate, tangent))
            except ValueError:
                continue
            if len(tangent) == chart_dim:
                break
        if len(tangent) != chart_dim:
            raise ValueError("ambient dimension is too small for requested chart")
        bases.append(torch.stack(tangent, dim=1))
    return c, torch.stack(bases)


class HypersphericalReferenceFrame(nn.Module):
    """Fixed HSRF from the paper's semantic reference-frame construction."""

    def __init__(self, class_count: int, ambient_dim: int, chart_dim: int, temperature: float = 0.12):
        super().__init__()
        anchors, bases = build_fiducials(class_count, ambient_dim, chart_dim)
        self.register_buffer("anchors", anchors, persistent=True)
        self.register_buffer("bases", bases, persistent=True)
        self.temperature = temperature

    def class_logits(self, unit: torch.Tensor) -> torch.Tensor:
        return unit @ self.anchors.T / self.temperature

    def log_map(self, unit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        anchor = self.anchors[labels]
        cosine = (unit * anchor).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cosine)
        tangent = unit - cosine.unsqueeze(-1) * anchor
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        return torch.bmm(tangent.unsqueeze(1), self.bases[labels]).squeeze(1) * theta.unsqueeze(-1)

    def log_map_all(self, unit: torch.Tensor) -> torch.Tensor:
        """Return class-conditional chart coordinates with shape [N, C, d_v]."""
        count = unit.shape[0]
        classes = self.anchors.shape[0]
        repeated = unit[:, None, :].expand(count, classes, -1).reshape(-1, unit.shape[-1])
        labels = torch.arange(classes, device=unit.device).repeat(count)
        return self.log_map(repeated, labels).reshape(count, classes, -1)

    def projection_residual(self, unit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        anchor = self.anchors[labels]
        tangent = unit - (unit * anchor).sum(-1, keepdim=True) * anchor
        basis = self.bases[labels]
        projected = torch.bmm(basis, torch.bmm(basis.transpose(1, 2), tangent.unsqueeze(-1))).squeeze(-1)
        return (tangent - projected).norm(dim=-1)

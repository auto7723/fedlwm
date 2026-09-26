from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .geometry import HypersphericalReferenceFrame


@dataclass
class ModelOutput:
    logits: torch.Tensor
    expert_logits: torch.Tensor
    reliability: torch.Tensor
    units: torch.Tensor


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, ambient_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, ambient_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(value), dim=-1)


class FedLWMModel(nn.Module):
    """Multimodal task model whose observer outputs live in the fixed HSRF."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.modality_names = tuple(config.modality_dims)
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim, config.hidden_dim, config.ambient_dim)
            for name, dim in config.modality_dims.items()
        })
        self.frame = HypersphericalReferenceFrame(
            config.class_count, config.ambient_dim, config.chart_dim, config.frame_temperature,
        )
        self.reliability = nn.ModuleDict({name: nn.Linear(config.ambient_dim, 1) for name in self.modality_names})
        self.task_head = nn.Linear(config.ambient_dim, config.class_count)

    def forward(self, features: dict[str, torch.Tensor], mask: torch.Tensor) -> ModelOutput:
        units = torch.stack([self.encoders[name](features[name]) for name in self.modality_names], dim=1)
        expert_logits = self.frame.class_logits(units)
        scores = torch.cat([
            self.reliability[name](units[:, index]) for index, name in enumerate(self.modality_names)
        ], dim=1)
        scores = scores.masked_fill(mask <= 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        fused = F.normalize((weights.unsqueeze(-1) * units).sum(dim=1), dim=-1)
        semantic_logits = self.frame.class_logits(fused)
        logits = semantic_logits + self.config.residual_scale * self.task_head(fused)
        return ModelOutput(logits, expert_logits, weights, units)

    def chart_coordinates(self, output: ModelOutput, labels: torch.Tensor) -> torch.Tensor:
        batch, modalities, _ = output.units.shape
        expanded = labels[:, None].expand(batch, modalities).reshape(-1)
        chart = self.frame.log_map(output.units.reshape(-1, output.units.shape[-1]), expanded)
        return chart.reshape(batch, modalities, -1)

    def all_class_coordinates(self, output: ModelOutput) -> torch.Tensor:
        charts = [self.frame.log_map_all(output.units[:, index]) for index in range(output.units.shape[1])]
        return torch.stack(charts, dim=2)

    def frame_loss(self, output: ModelOutput, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = labels[:, None].expand(-1, output.expert_logits.shape[1]).reshape(-1)
        losses = F.cross_entropy(output.expert_logits.reshape(-1, self.config.class_count), target, reduction="none")
        classification = (losses.reshape_as(mask) * mask).sum() / mask.sum().clamp_min(1.0)
        expanded = target
        residual = self.frame.projection_residual(output.units.reshape(-1, output.units.shape[-1]), expanded)
        projection = (residual.reshape_as(mask) * mask).sum() / mask.sum().clamp_min(1.0)
        return classification + 0.05 * projection

    def shared_state(self, scope: str) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        if scope == "full":
            return {key: value.detach().cpu().clone() for key, value in state.items()}
        return {
            key: value.detach().cpu().clone() for key, value in state.items()
            if key.startswith("encoders.") or key.startswith("reliability.") or key.startswith("frame.")
        }

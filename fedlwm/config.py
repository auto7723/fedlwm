from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    class_count: int
    modality_dims: dict[str, int]
    hidden_dim: int = 128
    ambient_dim: int = 32
    chart_dim: int = 4
    primitive_count: int = 2
    frame_temperature: float = 0.12
    residual_scale: float = 0.20


@dataclasses.dataclass(frozen=True)
class WorldConfig:
    lag: int = 4
    covariance_floor: float = 1e-3
    ridge: float = 1e-2
    out_of_window_decay: float = 0.5
    transition_shrinkage: float = 0.90
    min_primitive_count: float = 2.0
    ephemeris_weight: float = 0.05
    anchor_period: int = 10


@dataclasses.dataclass(frozen=True)
class FederatedConfig:
    rounds: int = 100
    clients_per_round: int = 4
    local_epochs: int = 2
    frame_fit_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    parameter_channel: str = "fedavg"
    shared_scope: str = "full"
    server_learning_rate: float = 1.0
    staleness_power: float = 0.5
    buffer_size: int = 4
    server_beta1: float = 0.9
    server_beta2: float = 0.99
    server_epsilon: float = 1e-8
    seed: int = 8


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    missing_rate: float = 0.5
    world_alpha_global: float = 0.50
    world_alpha_missing: float = 0.50
    world_alpha_personal: float = 0.0
    world_horizon_global: int = 1
    world_horizon_missing: int = 1
    unseen_support_size: int = 8
    unseen_steps: int = 20
    unseen_learning_rate: float = 0.03
    unseen_hidden_dim: int = 112
    unseen_expert_weight: float = 0.35
    unseen_consistency_weight: float = 0.05
    unseen_frame_weight: float = 0.50
    unseen_prediction_temperature: float = 3.0
    world_on_unseen: bool = False
    headline_tail_windows: int = 5


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    world: WorldConfig
    federated: FederatedConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        allowed = {"model", "world", "federated", "evaluation"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        cfg = cls(
            model=ModelConfig(**raw["model"]),
            world=WorldConfig(**raw.get("world", {})),
            federated=FederatedConfig(**raw.get("federated", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        m = self.model
        if m.class_count < 2 or not m.modality_dims:
            raise ValueError("at least two classes and one modality are required")
        if not 1 <= m.chart_dim < m.ambient_dim:
            raise ValueError("chart_dim must be positive and below ambient_dim")
        if m.ambient_dim < m.class_count - 1:
            raise ValueError("ambient_dim must contain the class simplex")
        if m.primitive_count < 1:
            raise ValueError("primitive_count must be positive")
        if self.world.lag < 1 or self.world.anchor_period < 1:
            raise ValueError("lag and anchor_period must be positive")
        if self.federated.parameter_channel not in {"none", "fedavg", "fedasync", "fedbuff_fedadam"}:
            raise ValueError("unsupported parameter_channel")
        if self.federated.shared_scope not in {"full", "backbone"}:
            raise ValueError("shared_scope must be full or backbone")

    def canonical_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

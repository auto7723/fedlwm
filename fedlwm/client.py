from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import FeatureBundle, Record, RecordDataset, collate_records
from .model import FedLWMModel
from .state import Ephemeris, WorldState
from .statistics import SufficientStatistics, collect_statistics


def state_digest(state: WorldState) -> str:
    digest = hashlib.sha256()
    for value in (
        state.mass, state.mean, state.second_moment, state.covariance,
        state.offsets, state.emission_covariance,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass
class ClientUpdate:
    client_id: str
    base_version: int
    sample_count: int
    arrival_time: int
    parameter_delta: dict[str, torch.Tensor]
    statistics: SufficientStatistics


class FederatedClient:
    def __init__(self, client_id: str, model: FedLWMModel, config: ExperimentConfig, device: torch.device):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.config = config
        self.device = device

    def synchronize(self, shared: dict[str, torch.Tensor]) -> None:
        current = self.model.state_dict()
        for key, value in shared.items():
            if key in current:
                current[key] = value.to(current[key].device)
        self.model.load_state_dict(current)

    def _ephemeris_loss(
        self, chart: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, ephemeris: Ephemeris,
    ) -> torch.Tensor:
        if ephemeris.effective_weight <= 0:
            return chart.sum() * 0.0
        mixture = torch.as_tensor(ephemeris.state.mixture, device=chart.device, dtype=chart.dtype)
        means = torch.as_tensor(ephemeris.state.mean, device=chart.device, dtype=chart.dtype)
        target = (mixture[labels].unsqueeze(-1) * means[labels]).sum(dim=1)
        observed = (chart * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return 0.5 * ephemeris.effective_weight * F.mse_loss(observed, target)

    def train_event(
        self, bundle: FeatureBundle, records: list[Record], ephemeris: Ephemeris,
        shared: dict[str, torch.Tensor], base_version: int,
    ) -> ClientUpdate:
        self.synchronize(shared)
        before = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        generator = torch.Generator().manual_seed(self.config.federated.seed + ephemeris.use_time)
        loader = DataLoader(
            RecordDataset(bundle, records), batch_size=self.config.federated.batch_size,
            shuffle=True, generator=generator, collate_fn=collate_records,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.federated.learning_rate)
        self.model.train()
        for _ in range(self.config.federated.local_epochs):
            for batch in loader:
                features = {name: value.to(self.device) for name, value in batch["features"].items()}
                labels = batch["labels"].to(self.device)
                mask = batch["mask"].to(self.device)
                output = self.model(features, mask)
                chart = self.model.chart_coordinates(output, labels)
                loss = (
                    F.cross_entropy(output.logits, labels)
                    + 0.10 * self.model.frame_loss(output, labels, mask)
                    + self._ephemeris_loss(chart, labels, mask, ephemeris)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

        self.model.eval()
        all_chart, all_labels, all_masks = [], [], []
        with torch.no_grad():
            stat_loader = DataLoader(
                RecordDataset(bundle, records), batch_size=self.config.federated.batch_size,
                shuffle=False, collate_fn=collate_records,
            )
            for batch in stat_loader:
                features = {name: value.to(self.device) for name, value in batch["features"].items()}
                labels = batch["labels"].to(self.device)
                mask = batch["mask"].to(self.device)
                output = self.model(features, mask)
                all_chart.append(self.model.chart_coordinates(output, labels).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
        chart = np.concatenate(all_chart)
        labels = np.concatenate(all_labels)
        masks = np.concatenate(all_masks)
        statistics = collect_statistics(
            chart, labels, masks, ephemeris.state,
            semantic_time=ephemeris.use_time,
            arrival_time=max(record.arrival_time for record in records),
            prior_digest=state_digest(ephemeris.state),
            include_emission_statistics=ephemeris.anchor,
        )
        if self.config.federated.parameter_channel == "none":
            # A disabled parameter channel carries no neural payload. The
            # sufficient statistics below remain the sole client upload.
            delta = {}
        else:
            after = self.model.shared_state(self.config.federated.shared_scope)
            delta = {
                key: value - before[key]
                for key, value in after.items()
                if torch.is_floating_point(value) and key in before
            }
        return ClientUpdate(
            self.client_id, base_version, len(records), statistics.arrival_time, delta, statistics,
        )

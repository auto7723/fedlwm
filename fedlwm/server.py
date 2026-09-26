from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import torch

from .client import ClientUpdate, state_digest
from .config import ExperimentConfig
from .dynamics import StructuredDynamics
from .fixed_lag import FixedLagInformationSmoother
from .model import FedLWMModel
from .state import Ephemeris, WorldState
from .statistics import assemble_information


@dataclass
class ServerDiagnostics:
    processed_updates: int = 0
    rejected_prior_mismatches: int = 0
    delayed_updates: int = 0
    parameter_aggregations: int = 0


class FedLWMServer:
    """Server with independent world and optional parameter channels."""

    def __init__(self, model: FedLWMModel, config: ExperimentConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        m = config.model
        initial = WorldState.initialize(
            m.class_count, m.primitive_count, len(m.modality_dims), m.chart_dim,
            config.world.covariance_floor,
        )
        self.dynamics = StructuredDynamics.identity(
            initial, config.world.transition_shrinkage, config.world.covariance_floor,
        )
        self.smoother = FixedLagInformationSmoother(
            initial, self.dynamics, config.world.lag, config.world.covariance_floor,
            config.world.out_of_window_decay,
        )
        self.version = 0
        self.queue: list[tuple[int, int, ClientUpdate]] = []
        self.sequence = 0
        self.recorded_priors: dict[str, WorldState] = {}
        self.anchor_history: list[WorldState] = []
        self.diagnostics = ServerDiagnostics()
        self.parameter_buffer: list[ClientUpdate] = []
        self.server_m: dict[str, torch.Tensor] = {}
        self.server_v: dict[str, torch.Tensor] = {}
        self.server_step = 0
        self.curvature_calibration: dict[tuple[int, ...], np.ndarray] = {}

    def shared_parameters(self) -> dict[str, torch.Tensor]:
        if self.config.federated.parameter_channel == "none":
            return {}
        return self.model.shared_state(self.config.federated.shared_scope)

    def broadcast(self, use_time: int) -> Ephemeris:
        anchor = self.smoother.current_time % self.config.world.anchor_period == 0
        horizon = 0 if anchor else max(use_time - self.smoother.current_time, 0)
        forecast = self.dynamics.forecast(self.smoother.current, horizon)
        uncertainty = float(forecast.uncertainty.mean())
        effective = 0.0 if anchor else self.config.world.ephemeris_weight / (1.0 + uncertainty)
        digest = state_digest(forecast)
        self.recorded_priors[digest] = forecast.copy()
        return Ephemeris(
            forecast, self.smoother.current_time, use_time,
            effective, uncertainty, anchor,
        )

    def submit(self, update: ClientUpdate) -> None:
        heapq.heappush(self.queue, (update.arrival_time, self.sequence, update))
        self.sequence += 1

    def process_until(self, time: int) -> None:
        self.smoother.advance_to(time)
        due: list[ClientUpdate] = []
        while self.queue and self.queue[0][0] <= time:
            _, _, update = heapq.heappop(self.queue)
            due.append(update)
        if not due:
            return
        for update in due:
            self._assimilate_world(update)
        if self.config.federated.parameter_channel == "fedavg":
            self._aggregate_fedavg(due)
        elif self.config.federated.parameter_channel == "fedasync":
            for update in due:
                self._aggregate_fedasync(update)
        elif self.config.federated.parameter_channel == "fedbuff_fedadam":
            self.parameter_buffer.extend(due)
            while len(self.parameter_buffer) >= self.config.federated.buffer_size:
                batch = self.parameter_buffer[:self.config.federated.buffer_size]
                del self.parameter_buffer[:self.config.federated.buffer_size]
                self._aggregate_fedbuff_fedadam(batch)

    def close_round(self, time: int) -> None:
        """Apply anchor-only estimators after every upload for the round is processed."""
        if time % self.config.world.anchor_period == 0:
            self.anchor_history.append(self.smoother.current.copy())
            self.dynamics.fit(self.anchor_history[-8:], self.config.world.ridge)

    def drain(self) -> None:
        while self.queue:
            time = self.queue[0][0]
            self.process_until(time)
            self.close_round(time)
        if self.parameter_buffer:
            self._aggregate_fedbuff_fedadam(self.parameter_buffer)
            self.parameter_buffer = []

    def _assimilate_world(self, update: ClientUpdate) -> None:
        prior = self.recorded_priors.get(update.statistics.recorded_prior_digest)
        if prior is None:
            self.diagnostics.rejected_prior_mismatches += 1
            return
        increments = assemble_information(
            update.statistics, prior, self.config.world.ridge,
            self.config.world.min_primitive_count, self.config.world.covariance_floor,
            self.curvature_calibration.get(update.statistics.modality_mask),
        )
        if update.statistics.semantic_time < self.smoother.current_time:
            self.diagnostics.delayed_updates += 1
        self.smoother.assimilate(increments)
        if update.statistics.has_emission_statistics:
            self._refit_curvature_calibration(update)
        self.diagnostics.processed_updates += 1

    def _refit_curvature_calibration(self, update: ClientUpdate) -> None:
        stats = update.statistics
        previous = self.curvature_calibration.get(
            stats.modality_mask, np.ones(self.config.model.class_count),
        )
        estimate = previous.copy()
        observed = stats.calibration_count > 0
        residual = np.zeros_like(estimate)
        residual[observed] = (
            stats.calibration_residual_sum[observed] / stats.calibration_count[observed]
        )
        estimate[observed] = np.clip(1.0 / (1.0 + residual[observed]), 0.1, 1.0)
        self.curvature_calibration[stats.modality_mask] = 0.8 * previous + 0.2 * estimate

    def _apply_delta(self, delta: dict[str, torch.Tensor], scale: float) -> None:
        state = self.model.state_dict()
        for key, value in delta.items():
            if key in state and torch.is_floating_point(state[key]):
                state[key] = state[key] + scale * value.to(state[key].device)
        self.model.load_state_dict(state)

    def _aggregate_fedavg(self, updates: list[ClientUpdate]) -> None:
        total = sum(update.sample_count for update in updates)
        if total <= 0:
            return
        combined: dict[str, torch.Tensor] = {}
        for update in updates:
            weight = update.sample_count / total
            for key, value in update.parameter_delta.items():
                combined[key] = combined.get(key, torch.zeros_like(value)) + weight * value
        self._apply_delta(combined, self.config.federated.server_learning_rate)
        self.version += 1
        self.diagnostics.parameter_aggregations += 1

    def _aggregate_fedasync(self, update: ClientUpdate) -> None:
        staleness = max(self.version - update.base_version, 0)
        scale = self.config.federated.server_learning_rate / ((1.0 + staleness) ** self.config.federated.staleness_power)
        self._apply_delta(update.parameter_delta, scale)
        self.version += 1
        self.diagnostics.parameter_aggregations += 1

    def _aggregate_fedbuff_fedadam(self, updates: list[ClientUpdate]) -> None:
        total_weight = 0.0
        combined: dict[str, torch.Tensor] = {}
        for update in updates:
            staleness = max(self.version - update.base_version, 0)
            weight = update.sample_count / ((1.0 + staleness) ** self.config.federated.staleness_power)
            total_weight += weight
            for key, value in update.parameter_delta.items():
                combined[key] = combined.get(key, torch.zeros_like(value)) + weight * value
        if total_weight <= 0:
            return
        self.server_step += 1
        beta1 = self.config.federated.server_beta1
        beta2 = self.config.federated.server_beta2
        adaptive: dict[str, torch.Tensor] = {}
        for key, value in combined.items():
            gradient = value / total_weight
            self.server_m[key] = beta1 * self.server_m.get(key, torch.zeros_like(gradient)) + (1 - beta1) * gradient
            self.server_v[key] = beta2 * self.server_v.get(key, torch.zeros_like(gradient)) + (1 - beta2) * gradient.square()
            m_hat = self.server_m[key] / (1 - beta1 ** self.server_step)
            v_hat = self.server_v[key] / (1 - beta2 ** self.server_step)
            adaptive[key] = m_hat / (v_hat.sqrt() + self.config.federated.server_epsilon)
        self._apply_delta(adaptive, self.config.federated.server_learning_rate)
        self.version += 1
        self.diagnostics.parameter_aggregations += 1

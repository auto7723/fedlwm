from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import WorldState, nearest_psd


def _ridge_map(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        return np.eye(x.shape[-1]), np.zeros(x.shape[-1])
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    regularizer = ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return weights[:-1].T, weights[-1]


@dataclass
class StructuredDynamics:
    """Constraint-preserving semantic dynamics C from the method section."""

    mean_map: np.ndarray
    mean_bias: np.ndarray
    offset_map: np.ndarray
    offset_bias: np.ndarray
    covariance_map: np.ndarray
    covariance_q: np.ndarray
    transition: np.ndarray
    shrinkage: float
    floor: float

    @classmethod
    def identity(cls, state: WorldState, shrinkage: float, floor: float) -> "StructuredDynamics":
        c, k, d = state.mean.shape
        m = state.modality_count
        eye = np.broadcast_to(np.eye(d), (c, d, d)).copy()
        transition = np.broadcast_to(np.eye(k), (c, k, k)).copy()
        return cls(
            mean_map=eye.copy(), mean_bias=np.zeros((c, k, d)),
            offset_map=eye.copy(), offset_bias=np.zeros((c, m, d)),
            covariance_map=eye.copy(), covariance_q=np.zeros((c, k, d, d)),
            transition=transition, shrinkage=shrinkage, floor=floor,
        )

    def fit(self, history: list[WorldState], ridge: float) -> None:
        if len(history) < 3:
            return
        for cls in range(history[0].class_count):
            x_mean = np.concatenate([state.mean[cls] for state in history[:-1]], axis=0)
            y_mean = np.concatenate([state.mean[cls] for state in history[1:]], axis=0)
            learned, bias = _ridge_map(x_mean, y_mean, ridge)
            identity = np.eye(history[0].chart_dim)
            self.mean_map[cls] = self.shrinkage * identity + (1.0 - self.shrinkage) * learned
            self.mean_bias[cls] = np.broadcast_to((1.0 - self.shrinkage) * bias, self.mean_bias[cls].shape)

            x_offset = np.concatenate([state.offsets[cls].reshape(-1, state.chart_dim) for state in history[:-1]])
            y_offset = np.concatenate([state.offsets[cls].reshape(-1, state.chart_dim) for state in history[1:]])
            learned_offset, offset_bias = _ridge_map(x_offset, y_offset, ridge)
            self.offset_map[cls] = self.shrinkage * identity + (1.0 - self.shrinkage) * learned_offset
            self.offset_bias[cls] = np.broadcast_to(
                (1.0 - self.shrinkage) * offset_bias, self.offset_bias[cls].shape,
            )

            floor_eye = self.floor * identity
            x_covariance = np.stack([
                state.covariance[cls, primitive] - floor_eye
                for state in history[:-1] for primitive in range(state.primitive_count)
            ])
            y_covariance = np.stack([
                state.covariance[cls, primitive] - floor_eye
                for state in history[1:] for primitive in range(state.primitive_count)
            ])
            numerator = float(np.sum(x_covariance * y_covariance))
            denominator = float(np.sum(x_covariance * x_covariance) + ridge)
            contraction = float(np.clip(numerator / denominator, 0.0, 1.0))
            self.covariance_map[cls] = np.sqrt(contraction) * identity
            residual = y_covariance - contraction * x_covariance
            q = nearest_psd(residual.mean(axis=0), 0.0)
            self.covariance_q[cls] = np.broadcast_to(q, self.covariance_q[cls].shape)

            transitions = np.full_like(self.transition[cls], ridge)
            for previous, current in zip(history[:-1], history[1:]):
                transitions += np.outer(current.mixture[cls], previous.mixture[cls])
            self.transition[cls] = transitions / transitions.sum(axis=0, keepdims=True)

    def step(self, state: WorldState) -> WorldState:
        result = state.copy()
        eye = np.eye(state.chart_dim)
        for cls in range(state.class_count):
            for primitive in range(state.primitive_count):
                result.mean[cls, primitive] = (
                    self.mean_map[cls] @ state.mean[cls, primitive] + self.mean_bias[cls, primitive]
                )
                centered = state.covariance[cls, primitive] - self.floor * eye
                result.covariance[cls, primitive] = nearest_psd(
                    self.floor * eye + self.covariance_map[cls] @ centered @ self.covariance_map[cls].T
                    + self.covariance_q[cls, primitive], self.floor,
                )
                for modality in range(state.modality_count):
                    result.offsets[cls, primitive, modality] = (
                        self.offset_map[cls] @ state.offsets[cls, primitive, modality]
                        + self.offset_bias[cls, modality]
                    )
            result.mass[cls] = self.transition[cls] @ state.mixture[cls]
        for cls in range(state.class_count):
            gain = max(
                np.linalg.norm(self.mean_map[cls], ord=2) ** 2,
                np.linalg.norm(self.covariance_map[cls], ord=2) ** 2,
            )
            injection = np.asarray([
                np.trace(self.covariance_q[cls, primitive]) / state.chart_dim
                for primitive in range(state.primitive_count)
            ])
            result.uncertainty[cls] = gain * state.uncertainty[cls] + injection + self.floor
        return result.project(self.floor)

    def forecast(self, state: WorldState, horizon: int) -> WorldState:
        result = state.copy()
        for _ in range(max(int(horizon), 0)):
            result = self.step(result)
        return result

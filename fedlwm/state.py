from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def nearest_psd(matrix: np.ndarray, floor: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


@dataclass
class WorldState:
    """Structured latent state B in fixed canonical chart coordinates."""

    mass: np.ndarray
    mean: np.ndarray
    second_moment: np.ndarray
    covariance: np.ndarray
    offsets: np.ndarray
    emission_covariance: np.ndarray
    uncertainty: np.ndarray

    @classmethod
    def initialize(
        cls, class_count: int, primitive_count: int, modality_count: int,
        chart_dim: int, covariance_floor: float,
    ) -> "WorldState":
        shape = (class_count, primitive_count)
        covariance = np.broadcast_to(
            np.eye(chart_dim) * max(covariance_floor, 1.0), shape + (chart_dim, chart_dim),
        ).copy()
        emission_dim = modality_count * chart_dim
        emission = np.broadcast_to(
            np.eye(emission_dim) * max(covariance_floor, 1.0), shape + (emission_dim, emission_dim),
        ).copy()
        return cls(
            mass=np.full(shape, 1.0 / primitive_count),
            mean=np.zeros(shape + (chart_dim,)),
            second_moment=covariance.copy(),
            covariance=covariance,
            offsets=np.zeros(shape + (modality_count, chart_dim)),
            emission_covariance=emission,
            uncertainty=np.ones(shape),
        )

    def copy(self) -> "WorldState":
        return WorldState(*(np.array(value, copy=True) for value in (
            self.mass, self.mean, self.second_moment, self.covariance, self.offsets,
            self.emission_covariance, self.uncertainty,
        )))

    @property
    def class_count(self) -> int:
        return self.mass.shape[0]

    @property
    def primitive_count(self) -> int:
        return self.mass.shape[1]

    @property
    def modality_count(self) -> int:
        return self.offsets.shape[2]

    @property
    def chart_dim(self) -> int:
        return self.mean.shape[-1]

    @property
    def mixture(self) -> np.ndarray:
        denominator = self.mass.sum(axis=1, keepdims=True)
        return self.mass / np.maximum(denominator, 1e-12)

    def project(self, floor: float) -> "WorldState":
        self.mass = np.maximum(self.mass, 1e-8)
        self.mass /= self.mass.sum(axis=1, keepdims=True)
        for cls in range(self.class_count):
            for primitive in range(self.primitive_count):
                self.covariance[cls, primitive] = nearest_psd(self.covariance[cls, primitive], floor)
                mean = self.mean[cls, primitive]
                self.second_moment[cls, primitive] = self.covariance[cls, primitive] + np.outer(mean, mean)
                self.emission_covariance[cls, primitive] = nearest_psd(
                    self.emission_covariance[cls, primitive], floor,
                )
                weighted = self.offsets[cls, primitive].mean(axis=0, keepdims=True)
                self.offsets[cls, primitive] -= weighted
        self.uncertainty = np.maximum(self.uncertainty, floor)
        return self

    def block_vector(self, cls: int, primitive: int) -> np.ndarray:
        d = self.chart_dim
        m = self.modality_count
        upper = np.triu_indices(d)
        # The last modality offset is implied by the centering constraint.
        return np.concatenate([
            self.mean[cls, primitive],
            self.covariance[cls, primitive][upper],
            self.offsets[cls, primitive, :m - 1].reshape(-1),
            np.log(np.maximum(self.mass[cls, primitive], 1e-12))[None],
        ])

    def set_block_vector(self, cls: int, primitive: int, vector: np.ndarray, floor: float) -> None:
        d = self.chart_dim
        m = self.modality_count
        upper = np.triu_indices(d)
        covariance_terms = len(upper[0])
        cursor = 0
        self.mean[cls, primitive] = vector[cursor:cursor + d]
        cursor += d
        covariance = np.zeros((d, d), dtype=np.float64)
        covariance[upper] = vector[cursor:cursor + covariance_terms]
        covariance[(upper[1], upper[0])] = covariance[upper]
        self.covariance[cls, primitive] = nearest_psd(covariance, floor)
        self.second_moment[cls, primitive] = self.covariance[cls, primitive] + np.outer(
            self.mean[cls, primitive], self.mean[cls, primitive],
        )
        cursor += covariance_terms
        if m > 1:
            offsets = vector[cursor:cursor + (m - 1) * d].reshape(m - 1, d)
            self.offsets[cls, primitive, :m - 1] = offsets
            self.offsets[cls, primitive, m - 1] = -offsets.sum(axis=0)
            cursor += (m - 1) * d
        self.mass[cls, primitive] = float(np.exp(np.clip(vector[cursor], -20.0, 20.0)))

    def class_prior(self) -> np.ndarray:
        masses = self.mass.sum(axis=1)
        return masses / np.maximum(masses.sum(), 1e-12)


@dataclass(frozen=True)
class Ephemeris:
    state: WorldState
    semantic_time: int
    use_time: int
    effective_weight: float
    predictive_uncertainty: float
    anchor: bool

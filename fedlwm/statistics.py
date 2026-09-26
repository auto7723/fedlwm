from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import WorldState, nearest_psd


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


@dataclass
class SufficientStatistics:
    """Compact state stream; no client primitive position is transmitted."""

    count: np.ndarray
    coordinate_sum: np.ndarray
    second_sum: np.ndarray
    modality_count: np.ndarray
    modality_residual_sum: np.ndarray
    joint_residual_second_sum: np.ndarray
    semantic_time: int
    arrival_time: int
    modality_mask: tuple[int, ...]
    recorded_prior_digest: str
    has_emission_statistics: bool
    calibration_count: np.ndarray
    calibration_residual_sum: np.ndarray


@dataclass(frozen=True)
class InformationIncrement:
    cls: int
    primitive: int
    semantic_time: int
    arrival_time: int
    precision: np.ndarray
    score: np.ndarray
    recorded_prior: np.ndarray
    emission_covariance: np.ndarray | None
    evidence_count: float


def responsibilities(
    chart: np.ndarray, labels: np.ndarray, masks: np.ndarray, state: WorldState,
) -> np.ndarray:
    """Class-conditional E-step using the observed joint-emission marginal."""
    n = len(labels)
    k = state.primitive_count
    d = state.chart_dim
    result = np.zeros((n, k), dtype=np.float64)
    for row, cls in enumerate(labels):
        visible_modalities = np.flatnonzero(masks[row] > 0)
        observed = np.concatenate([
            np.arange(modality * d, (modality + 1) * d) for modality in visible_modalities
        ])
        for primitive in range(k):
            expected = np.stack([
                state.mean[cls, primitive] + state.offsets[cls, primitive, modality]
                for modality in visible_modalities
            ]).reshape(-1)
            value = chart[row, visible_modalities].reshape(-1)
            diff = value - expected
            covariance = state.emission_covariance[cls, primitive][np.ix_(observed, observed)]
            covariance = covariance + np.eye(len(observed)) * 1e-5
            precision = np.linalg.pinv(covariance)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                logdet = 0.0
            result[row, primitive] = (
                np.log(max(state.mixture[cls, primitive], 1e-12))
                - 0.5 * (diff @ precision @ diff + logdet)
            )
    return _softmax(result)


def collect_statistics(
    chart: np.ndarray, labels: np.ndarray, masks: np.ndarray, state: WorldState,
    semantic_time: int, arrival_time: int, prior_digest: str,
    include_emission_statistics: bool,
) -> SufficientStatistics:
    """Compute the paper's n, s, Q, modal residual and cross-modal blocks."""
    n, modalities, d = chart.shape
    fused = (chart * masks[..., None]).sum(axis=1) / np.maximum(masks.sum(axis=1, keepdims=True), 1.0)
    rho = responsibilities(chart, labels, masks, state)
    shape = (state.class_count, state.primitive_count)
    count = np.zeros(shape)
    coordinate_sum = np.zeros(shape + (d,))
    second_sum = np.zeros(shape + (d, d))
    modality_count = np.zeros(shape + (modalities,))
    modality_residual_sum = np.zeros(shape + (modalities, d))
    joint_dim = modalities * d
    joint_second = np.zeros(shape + (joint_dim, joint_dim))
    for row in range(n):
        cls = int(labels[row])
        for primitive in range(state.primitive_count):
            weight = rho[row, primitive]
            value = fused[row]
            count[cls, primitive] += weight
            coordinate_sum[cls, primitive] += weight * value
            second_sum[cls, primitive] += weight * np.outer(value, value)
            residuals = np.zeros((modalities, d))
            for modality in range(modalities):
                if masks[row, modality] > 0:
                    residuals[modality] = chart[row, modality] - value
                    modality_count[cls, primitive, modality] += weight
                    modality_residual_sum[cls, primitive, modality] += weight * residuals[modality]
            visible = np.repeat(masks[row], d)
            residual = residuals.reshape(-1) * visible
            if include_emission_statistics:
                joint_second[cls, primitive] += weight * np.outer(residual, residual)
    active_mask = tuple(int(x) for x in (masks.sum(axis=0) > 0))
    calibration_count = np.zeros(state.class_count)
    calibration_residual_sum = np.zeros(state.class_count)
    if include_emission_statistics:
        for row, cls in enumerate(labels):
            visible = masks[row] > 0
            calibration_count[int(cls)] += 1.0
            calibration_residual_sum[int(cls)] += float(
                ((chart[row, visible] - fused[row]) ** 2).sum(axis=1).mean()
            )
    return SufficientStatistics(
        count, coordinate_sum, second_sum, modality_count,
        modality_residual_sum, joint_second, semantic_time, arrival_time,
        active_mask, prior_digest, include_emission_statistics,
        calibration_count, calibration_residual_sum,
    )


def assemble_information(
    stats: SufficientStatistics, recorded_prior: WorldState, ridge: float,
    min_count: float, covariance_floor: float,
    curvature_calibration: np.ndarray | None = None,
) -> list[InformationIncrement]:
    """Assemble score/curvature at the recorded prior, not a client position average."""
    increments: list[InformationIncrement] = []
    d = recorded_prior.chart_dim
    m = recorded_prior.modality_count
    upper = np.triu_indices(d)
    for cls in range(recorded_prior.class_count):
        for primitive in range(recorded_prior.primitive_count):
            count = float(stats.count[cls, primitive])
            if count < min_count:
                continue
            mean = stats.coordinate_sum[cls, primitive] / count
            raw_second = stats.second_sum[cls, primitive] / count
            covariance = nearest_psd(raw_second - np.outer(mean, mean), covariance_floor)
            offsets = np.zeros((m, d))
            for modality in range(m):
                modal_count = stats.modality_count[cls, primitive, modality]
                if modal_count > 0:
                    offsets[modality] = stats.modality_residual_sum[cls, primitive, modality] / modal_count
            offsets -= offsets.mean(axis=0, keepdims=True)
            target = np.concatenate([
                mean, covariance[upper], offsets[:m - 1].reshape(-1), np.log(count + 1e-8)[None],
            ])
            prior = recorded_prior.block_vector(cls, primitive)
            size = len(target)
            precision = np.eye(size) * (ridge + count)
            # Preserve observed modality directions in the mean-coordinate curvature.
            precision[:d, :d] = count * np.linalg.pinv(covariance) + ridge * np.eye(d)
            if curvature_calibration is not None:
                precision *= float(curvature_calibration[cls])
            score = precision @ (target - prior)
            joint_count = max(count, 1.0)
            emission_covariance = None
            if stats.has_emission_statistics:
                emission_covariance = nearest_psd(
                    stats.joint_residual_second_sum[cls, primitive] / joint_count,
                    covariance_floor,
                )
            increments.append(InformationIncrement(
                cls, primitive, stats.semantic_time, stats.arrival_time,
                precision, score, prior.copy(), emission_covariance, count,
            ))
    return increments

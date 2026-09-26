from __future__ import annotations

import numpy as np

from .state import WorldState


def _logsumexp(value: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(value, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.exp(value - maximum).sum(axis=axis) + 1e-12)


def world_log_likelihood(charts: np.ndarray, masks: np.ndarray, state: WorldState) -> np.ndarray:
    """Class energy with exact observed-block marginalization of joint emissions.

    charts has shape [N, C, M, d_v]. Missing modality coordinates are ignored by
    selecting the corresponding covariance principal submatrix.
    """
    n, classes, modalities, d = charts.shape
    result = np.zeros((n, classes), dtype=np.float64)
    class_prior = np.maximum(state.class_prior(), 1e-12)
    mixture = np.maximum(state.mixture, 1e-12)
    for row in range(n):
        visible_modalities = np.flatnonzero(masks[row] > 0)
        observed = np.concatenate([np.arange(modality * d, (modality + 1) * d) for modality in visible_modalities])
        for cls in range(classes):
            component_scores = []
            for primitive in range(state.primitive_count):
                expected = np.stack([
                    state.mean[cls, primitive] + state.offsets[cls, primitive, modality]
                    for modality in visible_modalities
                ]).reshape(-1)
                value = charts[row, cls, visible_modalities].reshape(-1)
                covariance = state.emission_covariance[cls, primitive][np.ix_(observed, observed)]
                covariance = covariance + np.eye(len(observed)) * 1e-5
                precision = np.linalg.pinv(covariance)
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0:
                    logdet = 0.0
                difference = value - expected
                component_scores.append(
                    np.log(mixture[cls, primitive])
                    - 0.5 * (difference @ precision @ difference + logdet)
                )
            result[row, cls] = np.log(class_prior[cls]) + _logsumexp(np.asarray(component_scores), axis=0)
    # Only relative class evidence is consumed by the task posterior.
    return result - result.mean(axis=1, keepdims=True)


def fuse_log_probabilities(task_probabilities: np.ndarray, world_log_score: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0.0:
        return np.asarray(task_probabilities)
    logits = np.log(np.maximum(task_probabilities, 1e-12)) + alpha * world_log_score
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)

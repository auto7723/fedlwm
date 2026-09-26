from __future__ import annotations

import math

import numpy as np


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray, class_count: int) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != (len(labels), class_count):
        raise ValueError("probability shape does not match labels and class_count")
    predictions = probabilities.argmax(axis=1)
    supports = np.bincount(labels, minlength=class_count).astype(np.float64)
    per_class_f1 = np.zeros(class_count, dtype=np.float64)
    recalls = np.zeros(class_count, dtype=np.float64)
    for cls in range(class_count):
        tp = np.sum((predictions == cls) & (labels == cls))
        fp = np.sum((predictions == cls) & (labels != cls))
        fn = np.sum((predictions != cls) & (labels == cls))
        per_class_f1[cls] = 2 * tp / max(2 * tp + fp + fn, 1)
        recalls[cls] = tp / max(tp + fn, 1)
    weighted = float(np.dot(per_class_f1, supports) / max(supports.sum(), 1))
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {
        "weighted_f1": 100.0 * weighted,
        "macro_f1": 100.0 * float(per_class_f1.mean()),
        "uar": 100.0 * float(recalls.mean()),
        "accuracy": 100.0 * float((predictions == labels).mean()),
        "nll": float(-np.log(clipped).mean()),
        "per_class_f1": (100.0 * per_class_f1).tolist(),
        "class_coverage": int(np.sum(per_class_f1 > 0)),
        "samples": int(len(labels)),
    }

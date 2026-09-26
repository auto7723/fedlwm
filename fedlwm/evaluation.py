from __future__ import annotations

import dataclasses
import hashlib
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import FeatureBundle, Record, RecordDataset, collate_records
from .metrics import classification_metrics
from .model import FedLWMModel
from .state import WorldState
from .world_readout import fuse_log_probabilities, world_log_likelihood


def deterministic_missing_mask(records: list[Record], original: np.ndarray, rate: float, seed: int) -> np.ndarray:
    result = original.copy()
    for row, record in enumerate(records):
        visible = list(np.flatnonzero(result[row] > 0))
        for modality in visible:
            token = f"{seed}:{record.sample_id}:{modality}".encode()
            value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
            if value < rate and result[row].sum() > 1:
                result[row, modality] = 0
    return result


def predict(
    model: FedLWMModel, bundle: FeatureBundle, records: Iterable[Record], device: torch.device,
    state: WorldState | None = None, world_alpha: float = 0.0, force_missing_rate: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    records = list(records)
    if not records:
        return np.empty((0, model.config.class_count)), np.empty(0, dtype=np.int64)
    loader = DataLoader(RecordDataset(bundle, records), batch_size=128, shuffle=False, collate_fn=collate_records)
    probabilities, labels = [], []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = {name: value.to(device) for name, value in batch["features"].items()}
            mask_np = batch["mask"].numpy()
            batch_records = records[cursor:cursor + len(mask_np)]
            cursor += len(mask_np)
            if force_missing_rate is not None:
                mask_np = deterministic_missing_mask(batch_records, mask_np, force_missing_rate, seed)
            mask = torch.as_tensor(mask_np, device=device, dtype=torch.float32)
            output = model(features, mask)
            task = torch.softmax(output.logits, dim=-1).cpu().numpy()
            if state is not None and world_alpha != 0.0:
                charts = model.all_class_coordinates(output).cpu().numpy()
                score = world_log_likelihood(charts, mask_np, state)
                task = fuse_log_probabilities(task, score, world_alpha)
            probabilities.append(task)
            labels.append(batch["labels"].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def evaluate_view(
    model: FedLWMModel, bundle: FeatureBundle, records: list[Record], config: ExperimentConfig,
    device: torch.device, state: WorldState | None, alpha: float, missing_rate: float | None = None,
) -> dict[str, object]:
    probabilities, labels = predict(
        model, bundle, records, device, state, alpha, missing_rate, config.federated.seed,
    )
    report = classification_metrics(probabilities, labels, config.model.class_count)
    report["probability_hash"] = hashlib.sha256(np.ascontiguousarray(probabilities).tobytes()).hexdigest()
    return report


def evaluate_personalized(
    clients: dict[str, object], bundle: FeatureBundle, records: list[Record], config: ExperimentConfig,
    device: torch.device, state: WorldState,
) -> dict[str, object]:
    reports = []
    for client_id, client in clients.items():
        client_records = [record for record in records if record.client_id == client_id]
        if client_records:
            reports.append(evaluate_view(
                client.model, bundle, client_records, config, device, state,
                config.evaluation.world_alpha_personal,
            ))
    if not reports:
        raise ValueError("no personalized evaluation samples matched training clients")
    scalar_keys = ["weighted_f1", "macro_f1", "uar", "accuracy", "nll", "class_coverage"]
    result = {key: float(np.mean([report[key] for report in reports])) for key in scalar_keys}
    result["client_p10_weighted_f1"] = float(np.percentile([report["weighted_f1"] for report in reports], 10))
    result["clients"] = len(reports)
    result["samples"] = int(sum(report["samples"] for report in reports))
    result["per_class_f1"] = np.mean([report["per_class_f1"] for report in reports], axis=0).tolist()
    return result


def evaluate_unseen(
    global_model: FedLWMModel, bundle: FeatureBundle, support: list[Record], query: list[Record],
    config: ExperimentConfig, device: torch.device, state: WorldState,
) -> dict[str, object]:
    if not support or not query:
        raise ValueError("E-G2 requires non-empty unseen support and query splits")
    support = sorted(support, key=lambda x: x.index)[:config.evaluation.unseen_support_size]
    if len(support) != config.evaluation.unseen_support_size:
        raise ValueError("E-G2 support split is smaller than the frozen shot budget")
    del global_model
    if any(sum(record.modality_mask) != len(record.modality_mask) for record in support + query):
        raise ValueError("strict E-G2 requires complete modalities for support and query")
    receiver_config = dataclasses.replace(config.model, hidden_dim=config.evaluation.unseen_hidden_dim)
    receiver = FedLWMModel(receiver_config).to(device)
    optimizer = torch.optim.SGD(
        receiver.parameters(), lr=config.evaluation.unseen_learning_rate, momentum=0.9,
        weight_decay=1e-5,
    )
    loader = DataLoader(RecordDataset(bundle, support), batch_size=len(support), shuffle=False, collate_fn=collate_records)
    receiver.train()
    for _ in range(config.evaluation.unseen_steps):
        for batch in loader:
            features = {name: value.to(device) for name, value in batch["features"].items()}
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            output = receiver(features, mask)
            fused = F.cross_entropy(output.logits, labels)
            target = labels[:, None].expand(-1, output.expert_logits.shape[1]).reshape(-1)
            expert = F.cross_entropy(
                output.expert_logits.reshape(-1, config.model.class_count), target,
            )
            teacher = output.logits.detach().softmax(-1)
            consistency = torch.stack([
                F.kl_div(
                    output.expert_logits[:, index].log_softmax(-1), teacher,
                    reduction="batchmean",
                )
                for index in range(output.expert_logits.shape[1])
            ]).mean()
            loss = (
                fused
                + config.evaluation.unseen_expert_weight * expert
                + config.evaluation.unseen_consistency_weight * consistency
                + config.evaluation.unseen_frame_weight * receiver.frame_loss(output, labels, mask)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    if config.evaluation.world_on_unseen:
        report = evaluate_view(
            receiver, bundle, query, config, device, state,
            config.evaluation.world_alpha_global,
        )
    else:
        probabilities, labels = predict(receiver, bundle, query, device)
        temperature = config.evaluation.unseen_prediction_temperature
        logits = np.log(np.maximum(probabilities, 1e-12)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        report = classification_metrics(probabilities, labels, config.model.class_count)
        report["probability_hash"] = hashlib.sha256(
            np.ascontiguousarray(probabilities).tobytes()
        ).hexdigest()
    report["support_samples"] = len(support)
    report["world_fallback"] = not config.evaluation.world_on_unseen
    report["fresh_receiver"] = True
    report["receiver_hidden_dim"] = config.evaluation.unseen_hidden_dim
    return report

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .client import FederatedClient
from .config import ExperimentConfig
from .data import FeatureBundle, RecordDataset, collate_records
from .evaluation import evaluate_personalized, evaluate_unseen, evaluate_view
from .model import FedLWMModel
from .server import FedLWMServer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "fedlwm").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def frame_fit(model: FedLWMModel, bundle: FeatureBundle, config: ExperimentConfig, device: torch.device) -> None:
    records = bundle.records_for("frame_fit")
    if not records:
        raise ValueError("the feature bundle must include a non-private frame_fit split")
    loader = DataLoader(
        RecordDataset(bundle, records), batch_size=config.federated.batch_size,
        shuffle=True, collate_fn=collate_records,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.federated.learning_rate)
    model.train()
    for _ in range(config.federated.frame_fit_epochs):
        for batch in loader:
            features = {name: value.to(device) for name, value in batch["features"].items()}
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            output = model(features, mask)
            loss = F.cross_entropy(output.logits, labels) + 0.10 * model.frame_loss(output, labels, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def _evaluate(
    server: FedLWMServer, clients: dict[str, FederatedClient], bundle: FeatureBundle,
    config: ExperimentConfig, device: torch.device, suffix: str,
) -> dict[str, object]:
    local_records = bundle.records_for(f"local_{suffix}")
    global_records = bundle.records_for(f"global_{suffix}")
    unseen_support = bundle.records_for(f"unseen_support_{suffix}")
    unseen_query = bundle.records_for(f"unseen_{suffix}")
    state = server.smoother.current
    global_state = server.dynamics.forecast(state, config.evaluation.world_horizon_global)
    missing_state = server.dynamics.forecast(state, config.evaluation.world_horizon_missing)
    return {
        "E-P": evaluate_personalized(clients, bundle, local_records, config, device, state),
        "E-G1": evaluate_view(
            server.model, bundle, global_records, config, device, global_state,
            config.evaluation.world_alpha_global,
        ),
        "E-G1-prime": evaluate_view(
            server.model, bundle, global_records, config, device, missing_state,
            config.evaluation.world_alpha_missing, config.evaluation.missing_rate,
        ),
        "E-G2": evaluate_unseen(
            server.model, bundle, unseen_support, unseen_query, config, device, state,
        ),
    }


def _mean_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    scalar = ["weighted_f1", "macro_f1", "uar", "accuracy", "nll", "class_coverage"]
    result = {key: float(np.mean([report[key] for report in reports])) for key in scalar}
    result["per_class_f1"] = np.mean([report["per_class_f1"] for report in reports], axis=0).tolist()
    result["samples"] = int(sum(int(report["samples"]) for report in reports))
    if all("client_p10_weighted_f1" in report for report in reports):
        result["client_p10_weighted_f1"] = float(np.mean([
            report["client_p10_weighted_f1"] for report in reports
        ]))
    result["windows"] = len(reports)
    return result


def _evaluate_window(
    round_index: int, server: FedLWMServer, clients: dict[str, FederatedClient],
    bundle: FeatureBundle, config: ExperimentConfig, device: torch.device,
    suffix: str,
) -> dict[str, object] | None:
    local = [record for record in bundle.records_for(f"local_{suffix}") if record.semantic_time == round_index]
    global_ = [record for record in bundle.records_for(f"global_{suffix}") if record.semantic_time == round_index]
    if not local or not global_:
        return None
    state = server.smoother.current
    global_state = server.dynamics.forecast(state, config.evaluation.world_horizon_global)
    missing_state = server.dynamics.forecast(state, config.evaluation.world_horizon_missing)
    return {
        "round": round_index,
        "E-P": evaluate_personalized(clients, bundle, local, config, device, state),
        "E-G1": evaluate_view(
            server.model, bundle, global_, config, device, global_state,
            config.evaluation.world_alpha_global,
        ),
        "E-G1-prime": evaluate_view(
            server.model, bundle, global_, config, device, missing_state,
            config.evaluation.world_alpha_missing, config.evaluation.missing_rate,
        ),
    }


def run_experiment(
    config: ExperimentConfig, data_root: str | Path, output_dir: str | Path,
    evaluation_suffix: str = "test", device_name: str = "auto",
) -> dict[str, object]:
    seed_everything(config.federated.seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    bundle = FeatureBundle(data_root, config.model.modality_dims, config.model.class_count)
    manifest_seed = bundle.manifest.get("seed")
    if manifest_seed is not None and int(manifest_seed) != config.federated.seed:
        raise ValueError("feature-bundle seed does not match the experiment seed")
    model = FedLWMModel(config.model).to(device)
    frame_fit(model, bundle, config, device)
    server = FedLWMServer(model, config, device)
    client_widths = {
        client_id: bundle.client_hidden_dim(client_id, config.model.hidden_dim)
        for client_id in bundle.clients
    }
    if config.federated.parameter_channel != "none" and any(
        width != config.model.hidden_dim for width in client_widths.values()
    ):
        raise ValueError("parameter sharing requires matched client model shapes")
    clients = {}
    for client_id in bundle.clients:
        width = client_widths[client_id]
        if width == config.model.hidden_dim:
            client_model = model
        else:
            client_model = FedLWMModel(dataclasses.replace(config.model, hidden_dim=width)).to(device)
            frame_fit(client_model, bundle, config, device)
        clients[client_id] = FederatedClient(client_id, client_model, config, device)

    events: dict[int, dict[str, list[object]]] = defaultdict(lambda: defaultdict(list))
    for record in bundle.records_for("train"):
        events[record.semantic_time][record.client_id].append(record)

    rng = np.random.default_rng(config.federated.seed)
    window_reports: list[dict[str, object]] = []
    for round_index in range(config.federated.rounds):
        server.process_until(round_index)
        available = sorted(events.get(round_index, {}))
        if len(available) > config.federated.clients_per_round:
            available = sorted(rng.choice(available, config.federated.clients_per_round, replace=False).tolist())
        shared = server.shared_parameters()
        for client_id in available:
            records = events[round_index][client_id]
            use_time = max(record.use_time for record in records)
            ephemeris = server.broadcast(use_time)
            update = clients[client_id].train_event(
                bundle, records, ephemeris, shared, server.version,
            )
            server.submit(update)
        # Delay-zero uploads belong to this server round.
        server.process_until(round_index)
        server.close_round(round_index)
        window_report = _evaluate_window(
            round_index, server, clients, bundle, config, device, evaluation_suffix,
        )
        if window_report is not None:
            window_reports.append(window_report)
    server.process_until(config.federated.rounds - 1)
    online_snapshot = _evaluate(server, clients, bundle, config, device, evaluation_suffix)
    if window_reports:
        tail = window_reports[-config.evaluation.headline_tail_windows:]
        online = {
            view: _mean_reports([report[view] for report in tail])
            for view in ("E-P", "E-G1", "E-G1-prime")
        }
        online["E-G2"] = online_snapshot["E-G2"]
        all_time = {
            view: _mean_reports([report[view] for report in window_reports])
            for view in ("E-P", "E-G1", "E-G1-prime")
        }
        all_time["E-G2"] = online_snapshot["E-G2"]
    else:
        online = online_snapshot
        all_time = online_snapshot
    online_time = server.smoother.current_time
    server.drain()
    drained = _evaluate(server, clients, bundle, config, device, evaluation_suffix)

    project_root = Path(__file__).resolve().parents[1]
    diagnostics = dataclasses.asdict(server.diagnostics)
    diagnostics["folded_updates"] = server.smoother.folded_updates
    result = {
        "schema_version": 1,
        "method": "FedLWM",
        "scenario": str(bundle.manifest.get("scenario", "unspecified")),
        "evaluation_suffix": evaluation_suffix,
        "seed": config.federated.seed,
        "config_hash": config.digest,
        "data_contract_hash": bundle.digest,
        "source_hash": source_digest(project_root),
        "test_samples_used_for_selection": 0 if evaluation_suffix == "test" else None,
        "online_time": online_time,
        "online": online,
        "all_time": all_time,
        "online_snapshot": online_snapshot,
        "window_reports": window_reports,
        "drained": drained,
        "server_diagnostics": diagnostics,
        "curvature_calibration": {
            "".join(map(str, mask)): value.tolist()
            for mask, value in sorted(server.curvature_calibration.items())
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    torch.save({
        "model": server.model.state_dict(),
        "world_state": server.smoother.current,
        "config_hash": config.digest,
        "data_contract_hash": bundle.digest,
        "source_hash": result["source_hash"],
        "curvature_calibration": server.curvature_calibration,
    }, output / "checkpoint.pt")
    return result

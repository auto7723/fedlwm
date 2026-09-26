import json

import numpy as np

from fedlwm.config import ExperimentConfig
from fedlwm.runner import run_experiment
from fedlwm.server import FedLWMServer


def create_synthetic_bundle(root):
    splits = []
    for index in range(6):
        splits.append(("frame_fit", "public", index % 3, 0))
    for time in range(2):
        for client_index in range(2):
            for offset in range(3):
                splits.append(("train", f"c{client_index}", (time + client_index + offset) % 3, time))
    for client_index in range(2):
        for offset in range(3):
            splits.append(("local_test", f"c{client_index}", (client_index + offset) % 3, 2))
    for offset in range(6):
        splits.append(("global_test", "global", offset % 3, 2))
    for offset in range(2):
        splits.append(("unseen_support_test", "new", offset % 3, 2))
    for offset in range(6):
        splits.append(("unseen_test", "new", offset % 3, 2))

    rng = np.random.default_rng(4)
    labels = np.asarray([item[2] for item in splits], dtype=np.int64)
    audio = rng.normal(size=(len(splits), 4)).astype(np.float32) + labels[:, None] * 0.3
    text = rng.normal(size=(len(splits), 5)).astype(np.float32) + labels[:, None] * 0.3
    np.save(root / "audio.npy", audio)
    np.save(root / "text.npy", text)
    np.save(root / "labels.npy", labels)
    manifest = {
        "schema_version": 1, "class_count": 3,
        "modalities": {
            "audio": {"path": "audio.npy", "dim": 4},
            "text": {"path": "text.npy", "dim": 5},
        },
        "labels": {"path": "labels.npy"}, "records": "records.jsonl",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = []
    for index, (split, client, _, time) in enumerate(splits):
        records.append(json.dumps({
            "index": index, "sample_id": f"s{index}", "client_id": client,
            "split": split, "semantic_time": time, "arrival_time": time + 1,
            "use_time": time + 1, "modality_mask": [1, 1], "sequence_id": "q0",
        }))
    (root / "records.jsonl").write_text("\n".join(records), encoding="utf-8")


def test_complete_four_view_path(tmp_path):
    data = tmp_path / "bundle"
    output = tmp_path / "run"
    data.mkdir()
    create_synthetic_bundle(data)
    config = ExperimentConfig.from_dict({
        "model": {
            "class_count": 3, "modality_dims": {"audio": 4, "text": 5},
            "hidden_dim": 8, "ambient_dim": 8, "chart_dim": 2,
            "primitive_count": 1,
        },
        "world": {"min_primitive_count": 0.1, "anchor_period": 2},
        "federated": {
            "rounds": 2, "clients_per_round": 2, "local_epochs": 1,
            "frame_fit_epochs": 1, "batch_size": 3, "parameter_channel": "fedavg",
        },
        "evaluation": {"unseen_support_size": 2, "unseen_steps": 1},
    })
    result = run_experiment(config, data, output, "test", "cpu")
    assert set(result["online"]) == {"E-P", "E-G1", "E-G1-prime", "E-G2"}
    assert result["online"]["E-G2"]["world_fallback"] is True
    assert (output / "checkpoint.pt").is_file()


def test_world_only_channel_has_no_neural_payload_and_accepts_heterogeneous_clients(tmp_path, monkeypatch):
    data = tmp_path / "bundle"
    output = tmp_path / "run"
    data.mkdir()
    create_synthetic_bundle(data)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["client_profiles"] = {
        "c0": {"hidden_dim": 6},
        "c1": {"hidden_dim": 10},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    submitted_payloads = []
    original_submit = FedLWMServer.submit

    def record_submit(self, update):
        submitted_payloads.append(update.parameter_delta)
        original_submit(self, update)

    monkeypatch.setattr(FedLWMServer, "submit", record_submit)
    config = ExperimentConfig.from_dict({
        "model": {
            "class_count": 3, "modality_dims": {"audio": 4, "text": 5},
            "hidden_dim": 8, "ambient_dim": 8, "chart_dim": 2,
            "primitive_count": 1,
        },
        "world": {"min_primitive_count": 0.1, "anchor_period": 2},
        "federated": {
            "rounds": 2, "clients_per_round": 2, "local_epochs": 1,
            "frame_fit_epochs": 1, "batch_size": 3, "parameter_channel": "none",
        },
        "evaluation": {"unseen_support_size": 2, "unseen_steps": 1},
    })
    result = run_experiment(config, data, output, "test", "cpu")

    assert submitted_payloads
    assert all(payload == {} for payload in submitted_payloads)
    assert result["server_diagnostics"]["processed_updates"] > 0
    assert result["server_diagnostics"]["parameter_aggregations"] == 0

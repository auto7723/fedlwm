import json

import numpy as np
import pytest

from fedlwm.data import FeatureBundle


def make_bundle(root, forbidden=False):
    np.save(root / "audio.npy", np.zeros((1, 2), dtype=np.float32))
    np.save(root / "text.npy", np.zeros((1, 3), dtype=np.float32))
    np.save(root / "labels.npy", np.zeros(1, dtype=np.int64))
    manifest = {
        "schema_version": 1,
        "class_count": 2,
        "modalities": {
            "audio": {"path": "audio.npy", "dim": 2},
            "text": {"path": "text.npy", "dim": 3},
        },
        "labels": {"path": "labels.npy"},
        "records": "records.jsonl",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = {
        "index": 0, "sample_id": "s0", "client_id": "c0", "split": "train",
        "semantic_time": 0, "arrival_time": 0, "use_time": 0, "modality_mask": [1, 1],
    }
    if forbidden:
        record["speaker_name"] = "private"
    (root / "records.jsonl").write_text(json.dumps(record), encoding="utf-8")


def test_bundle_loads_only_precomputed_features(tmp_path):
    make_bundle(tmp_path)
    bundle = FeatureBundle(tmp_path, {"audio": 2, "text": 3}, 2)
    assert len(bundle.records) == 1


def test_bundle_rejects_identity_fields(tmp_path):
    make_bundle(tmp_path, forbidden=True)
    with pytest.raises(ValueError, match="privacy-sensitive"):
        FeatureBundle(tmp_path, {"audio": 2, "text": 3}, 2)

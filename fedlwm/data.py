from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


FORBIDDEN_RECORD_FIELDS = {
    "name", "speaker_name", "email", "raw_path", "media_path", "image_path",
    "audio_path", "video_path", "text", "transcript",
}
REQUIRED_RECORD_FIELDS = {
    "index", "sample_id", "client_id", "split", "semantic_time", "arrival_time",
    "use_time", "modality_mask",
}
ALLOWED_SPLITS = {
    "frame_fit", "train", "local_dev", "global_dev", "unseen_support_dev", "unseen_dev",
    "local_test", "global_test", "unseen_support_test", "unseen_test",
}


@dataclass(frozen=True)
class Record:
    index: int
    sample_id: str
    client_id: str
    split: str
    semantic_time: int
    arrival_time: int
    use_time: int
    modality_mask: tuple[int, ...]
    sequence_id: str = ""


class FeatureBundle:
    """Reader for a privacy-preserving, pre-extracted feature artifact.

    V10 intentionally contains no raw-media decoder or feature extractor. All
    paths in the manifest must be relative to the bundle root.
    """

    def __init__(self, root: str | Path, expected_dims: dict[str, int], class_count: int):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing feature-bundle manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(expected_dims, class_count)
        self.modalities = {
            name: np.load(self._safe_path(spec["path"]), mmap_mode="r")
            for name, spec in self.manifest["modalities"].items()
        }
        self.labels = np.load(self._safe_path(self.manifest["labels"]["path"]), mmap_mode="r")
        records_path = self._safe_path(self.manifest["records"])
        self.records = self._load_records(records_path, tuple(expected_dims))
        self._validate_arrays(expected_dims, class_count)

    def _safe_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            raise ValueError("feature manifests may not contain absolute paths")
        resolved = (self.root / path).resolve()
        if self.root not in resolved.parents and resolved != self.root:
            raise ValueError("feature path escapes the bundle root")
        return resolved

    def _validate_manifest(self, expected_dims: dict[str, int], class_count: int) -> None:
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported feature-bundle schema")
        if self.manifest.get("class_count") != class_count:
            raise ValueError("class_count differs between bundle and experiment")
        specs = self.manifest.get("modalities", {})
        if set(specs) != set(expected_dims):
            raise ValueError("modality names differ between bundle and experiment")
        for name, dim in expected_dims.items():
            if specs[name].get("dim") != dim:
                raise ValueError(f"dimension mismatch for modality {name}")
            self._safe_path(specs[name]["path"])
        self._safe_path(self.manifest["labels"]["path"])
        self._safe_path(self.manifest["records"])

    @staticmethod
    def _load_records(path: Path, modalities: tuple[str, ...]) -> list[Record]:
        records: list[Record] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = json.loads(line)
            leaked = FORBIDDEN_RECORD_FIELDS.intersection(raw)
            if leaked:
                raise ValueError(f"privacy-sensitive fields at records line {line_number}: {sorted(leaked)}")
            missing = REQUIRED_RECORD_FIELDS - set(raw)
            if missing:
                raise ValueError(f"missing fields at records line {line_number}: {sorted(missing)}")
            mask = tuple(int(x) for x in raw["modality_mask"])
            if len(mask) != len(modalities) or any(x not in (0, 1) for x in mask):
                raise ValueError(f"invalid modality mask at records line {line_number}")
            if not any(mask):
                raise ValueError(f"empty modality mask at records line {line_number}")
            if raw["split"] not in ALLOWED_SPLITS:
                raise ValueError(f"unknown split at records line {line_number}: {raw['split']}")
            if int(raw["semantic_time"]) > int(raw["use_time"]):
                raise ValueError(f"use_time precedes semantic observation at records line {line_number}")
            if int(raw["use_time"]) > int(raw["arrival_time"]):
                raise ValueError(f"arrival_time precedes use_time at records line {line_number}")
            records.append(Record(
                index=int(raw["index"]), sample_id=str(raw["sample_id"]),
                client_id=str(raw["client_id"]), split=str(raw["split"]),
                semantic_time=int(raw["semantic_time"]), arrival_time=int(raw["arrival_time"]),
                use_time=int(raw["use_time"]), modality_mask=mask,
                sequence_id=str(raw.get("sequence_id", "")),
            ))
        return records

    def _validate_arrays(self, expected_dims: dict[str, int], class_count: int) -> None:
        n = len(self.records)
        if self.labels.shape != (n,):
            raise ValueError("labels must have shape [N]")
        if sorted(record.index for record in self.records) != list(range(n)):
            raise ValueError("record indices must be a permutation of [0, N)")
        if np.any(self.labels < 0) or np.any(self.labels >= class_count):
            raise ValueError("label outside configured class range")
        for name, array in self.modalities.items():
            if array.shape != (n, expected_dims[name]):
                raise ValueError(f"feature array shape mismatch for {name}")
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite feature values in {name}")

    def records_for(self, split: str, client_id: str | None = None) -> list[Record]:
        return [r for r in self.records if r.split == split and (client_id is None or r.client_id == client_id)]

    @property
    def clients(self) -> list[str]:
        return sorted({r.client_id for r in self.records if r.split.startswith("train")})

    def client_hidden_dim(self, client_id: str, default: int) -> int:
        profiles = self.manifest.get("client_profiles", {})
        profile = profiles.get(client_id, {})
        hidden_dim = int(profile.get("hidden_dim", default))
        if hidden_dim <= 0:
            raise ValueError(f"invalid hidden_dim for client {client_id}")
        return hidden_dim

    @property
    def digest(self) -> str:
        payload = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class RecordDataset(Dataset):
    def __init__(self, bundle: FeatureBundle, records: Iterable[Record]):
        self.bundle = bundle
        self.records = list(records)
        self.modality_names = tuple(bundle.modalities)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, object]:
        record = self.records[item]
        return {
            "features": {
                name: torch.as_tensor(np.array(self.bundle.modalities[name][record.index], copy=True), dtype=torch.float32)
                for name in self.modality_names
            },
            "label": torch.tensor(int(self.bundle.labels[record.index]), dtype=torch.long),
            "mask": torch.tensor(record.modality_mask, dtype=torch.float32),
            "record": record,
        }


def collate_records(items: list[dict[str, object]]) -> dict[str, object]:
    names = tuple(items[0]["features"])
    return {
        "features": {name: torch.stack([x["features"][name] for x in items]) for name in names},
        "labels": torch.stack([x["label"] for x in items]),
        "mask": torch.stack([x["mask"] for x in items]),
        "records": [x["record"] for x in items],
    }


def chronological_batches(records: Iterable[Record], batch_size: int) -> Iterator[list[Record]]:
    ordered = sorted(records, key=lambda x: (x.semantic_time, x.arrival_time, x.index))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start:start + batch_size]

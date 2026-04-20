"""Data preparation utilities for the bearing fault diagnosis project."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATASET_SOURCES = {
    0: {
        "name": "normal",
        "path": Path("data/Normal Baseline Data"),
        "ids": [97, 98, 99, 100],
    },
    1: {
        "name": "inner_race_fault",
        "path": Path("data/12k Drive End Bearing Fault Data/内圈故障"),
        "ids": [105, 106, 107, 108, 169, 170, 171, 172, 209, 210, 211, 212],
    },
    2: {
        "name": "ball_fault",
        "path": Path("data/12k Drive End Bearing Fault Data/滚动体故障"),
        "ids": [118, 119, 120, 121, 185, 186, 187, 188, 222, 223, 224, 225],
    },
    3: {
        "name": "outer_race_fault",
        "path": Path("data/12k Drive End Bearing Fault Data/外圈故障"),
        "ids": [130, 131, 132, 133, 197, 198, 199, 200, 234, 235, 236, 237],
    },
}


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32).reshape(-1)
    min_value = float(signal.min())
    max_value = float(signal.max())
    if max_value == min_value:
        return np.zeros_like(signal, dtype=np.float32)
    return (signal - min_value) / (max_value - min_value)


def _mat_channel_key(data_id: int, suffix: str) -> str:
    return f"X{data_id:03d}_{suffix}"


def load_segmented_sample(mat_path: Path, data_id: int, sample_length: int) -> np.ndarray:
    """Load one .mat sample and segment it into non-overlapping windows."""

    mat = loadmat(mat_path)
    de_signal = _normalize_signal(mat[_mat_channel_key(data_id, "DE_time")])
    fe_signal = _normalize_signal(mat[_mat_channel_key(data_id, "FE_time")])

    usable = (len(de_signal) // sample_length) * sample_length
    stacked = np.column_stack((de_signal[:usable], fe_signal[:usable]))
    return stacked.reshape(-1, sample_length, 2)


def build_processed_dataset(
    project_root: Path,
    sample_length: int = 1000,
    per_class_limit: int = 1400,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the processed 4-class dataset from the raw CWRU files."""

    data_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []

    for label, spec in DATASET_SOURCES.items():
        class_samples = []
        for data_id in spec["ids"]:
            mat_path = project_root / spec["path"] / f"{data_id}.mat"
            class_samples.append(load_segmented_sample(mat_path, data_id, sample_length))

        merged = np.concatenate(class_samples, axis=0)[:per_class_limit].astype(np.float32)
        labels = np.full((merged.shape[0],), label, dtype=np.int64)
        data_parts.append(merged)
        label_parts.append(labels)

    data = np.concatenate(data_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    return data, labels


def save_processed_dataset(output_dir: Path, data: np.ndarray, labels: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_data.npy", data)
    np.save(output_dir / "label.npy", labels)


def load_processed_dataset(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(output_dir / "train_data.npy").astype(np.float32)
    labels = np.load(output_dir / "label.npy").astype(np.int64)
    return data, labels


def summarize_labels(labels: np.ndarray) -> Counter:
    return Counter(labels.tolist())

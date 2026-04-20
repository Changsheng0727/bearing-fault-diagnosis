"""Training utilities for bearing fault diagnosis experiments."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .models import CNNResNetModel, SimpleCNNModel


@dataclass
class ExperimentResult:
    name: str
    accuracy: float
    runtime_seconds: float
    parameter_count: int | None = None
    epochs_ran: int | None = None
    best_val_acc: float | None = None
    classification_report: str | None = None


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def stratified_split(
    data: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.3,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        data,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )


def run_random_forest_baseline(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    model_path: Path | None = None,
    seed: int = 99,
) -> ExperimentResult:
    start = time.time()
    classifier = RandomForestClassifier(
        n_estimators=50,
        min_samples_split=5,
        min_samples_leaf=4,
        max_depth=5,
        random_state=seed,
        n_jobs=-1,
    )
    classifier.fit(train_data.reshape(train_data.shape[0], -1), train_labels)
    predictions = classifier.predict(val_data.reshape(val_data.shape[0], -1))
    runtime = time.time() - start

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(classifier, model_path)

    return ExperimentResult(
        name="random_forest",
        accuracy=float(accuracy_score(val_labels, predictions)),
        runtime_seconds=runtime,
        classification_report=classification_report(val_labels, predictions, digits=4),
    )


def train_keras_model(
    name: str,
    model: tf.keras.Model,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    epochs: int,
    batch_size: int,
    model_path: Path | None = None,
) -> ExperimentResult:
    start = time.time()
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    predictions = np.argmax(model.predict(val_data, verbose=0), axis=1)
    runtime = time.time() - start

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

    return ExperimentResult(
        name=name,
        accuracy=float(accuracy_score(val_labels, predictions)),
        runtime_seconds=runtime,
        parameter_count=int(model.count_params()),
        epochs_ran=len(history.history["loss"]),
        best_val_acc=float(max(history.history["val_acc"])),
        classification_report=classification_report(val_labels, predictions, digits=4),
    )


def run_simple_cnn_baseline(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    model_path: Path | None = None,
    learning_rate: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 16,
) -> ExperimentResult:
    model = SimpleCNNModel(label_count=len(np.unique(train_labels))).build(learning_rate=learning_rate)
    return train_keras_model(
        name="simple_cnn",
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        model_path=model_path,
    )


def run_cnn_resnet_experiment(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    model_path: Path | None = None,
    num_blocks: int = 5,
    learning_rate: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 32,
) -> ExperimentResult:
    model = CNNResNetModel(
        label_count=len(np.unique(train_labels)),
        num_blocks=num_blocks,
        data_shape=train_data.shape[1:],
    ).build(learning_rate=learning_rate)
    return train_keras_model(
        name=f"cnn_resnet_b{num_blocks}",
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        model_path=model_path,
    )


def save_results(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

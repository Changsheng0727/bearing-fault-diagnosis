"""Train baseline and optimized bearing-fault models on the processed dataset."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bearing_fault_diagnosis.data import load_processed_dataset
from bearing_fault_diagnosis.training import (
    run_cnn_resnet_experiment,
    run_random_forest_baseline,
    run_simple_cnn_baseline,
    save_results,
    set_global_seed,
    stratified_split,
)


def main() -> None:
    project_root = PROJECT_ROOT
    artifacts_dir = project_root / "artifacts"
    data_dir = artifacts_dir / "datasets"
    models_dir = artifacts_dir / "models"
    reports_dir = artifacts_dir / "reports"

    set_global_seed(99)
    data, labels = load_processed_dataset(data_dir)
    train_data, val_data, train_labels, val_labels = stratified_split(data, labels, test_size=0.3, seed=99)

    results = [
        run_random_forest_baseline(
            train_data,
            train_labels,
            val_data,
            val_labels,
            model_path=models_dir / "random_forest.joblib",
            seed=99,
        ),
        run_simple_cnn_baseline(
            train_data,
            train_labels,
            val_data,
            val_labels,
            model_path=models_dir / "simple_cnn.keras",
            learning_rate=1e-3,
            epochs=20,
            batch_size=16,
        ),
        run_cnn_resnet_experiment(
            train_data,
            train_labels,
            val_data,
            val_labels,
            model_path=models_dir / "cnn_resnet_original.keras",
            num_blocks=5,
            learning_rate=1e-4,
            epochs=20,
            batch_size=32,
        ),
        run_cnn_resnet_experiment(
            train_data,
            train_labels,
            val_data,
            val_labels,
            model_path=models_dir / "cnn_resnet_optimized.keras",
            num_blocks=1,
            learning_rate=3e-4,
            epochs=12,
            batch_size=32,
        ),
    ]

    save_results(results, reports_dir / "benchmark_results.json")

    print("\nBenchmark summary")
    print("-" * 60)
    for result in results:
        extras = []
        if result.parameter_count is not None:
            extras.append(f"params={result.parameter_count}")
        if result.epochs_ran is not None:
            extras.append(f"epochs={result.epochs_ran}")
        extras.append(f"time={result.runtime_seconds:.2f}s")
        print(f"{result.name:22s} acc={result.accuracy:.4f}  " + "  ".join(extras))


if __name__ == "__main__":
    main()

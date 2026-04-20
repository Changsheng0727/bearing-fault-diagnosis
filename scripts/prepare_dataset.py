"""Build the processed bearing-fault dataset from the raw CWRU files."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bearing_fault_diagnosis.data import (
    build_processed_dataset,
    save_processed_dataset,
    summarize_labels,
)


def main() -> None:
    project_root = PROJECT_ROOT
    output_dir = project_root / "artifacts" / "datasets"

    data, labels = build_processed_dataset(project_root=project_root)
    save_processed_dataset(output_dir=output_dir, data=data, labels=labels)

    print("Processed dataset saved to:", output_dir)
    print("Data shape:", data.shape)
    print("Label shape:", labels.shape)
    print("Class distribution:", summarize_labels(labels))


if __name__ == "__main__":
    main()

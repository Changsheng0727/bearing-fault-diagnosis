# Bearing Fault Diagnosis

An open, cleaned-up research project for rolling bearing fault diagnosis using both classical machine learning and deep learning.

This repository reproduces a compact end-to-end workflow based on the Case Western Reserve University bearing dataset:

- load raw vibration `.mat` files
- build a four-class diagnostic dataset
- train baseline and deep models
- compare model performance with both notebooks and reproducible scripts

## Why This Repo

- Reorganized for public GitHub sharing instead of a local coursework-style layout
- Covers the full path from raw signal data to model evaluation
- Includes both a traditional baseline and neural-network models
- Keeps large datasets, logs, checkpoints, and private files out of version control
- Adds a lighter optimized residual model after rerunning and benchmarking the full project

## Task Definition

The project builds a four-class bearing fault diagnosis task:

- Normal
- Inner-race fault
- Ball fault
- Outer-race fault

## Models

- Random Forest
- 1D CNN
- CNN + ResNet, original deeper setting
- CNN + ResNet, optimized lighter setting

## Latest Benchmark

The table below reflects the latest local benchmark produced by `scripts/train_models.py` on the processed dataset:

| Model | Validation Accuracy |
| --- | ---: |
| Random Forest | 0.7167 |
| 1D CNN | 0.8423 |
| CNN + ResNet, 5 residual blocks | 0.9970 |
| Optimized CNN + ResNet, 1 residual block | 1.0000 |

## Optimization Summary

After rerunning the project end-to-end and benchmarking multiple residual-depth settings, the best practical tradeoff was not the deepest network.

- Original residual setting: `5` residual blocks, `162,532` parameters, about `55.31s`
- Optimized setting: `1` residual block, `61,668` parameters, about `16.08s`
- Result: same or better validation accuracy on the current split, with much lower training cost

This means the optimized model cuts parameter count by about `62%` and reduces training time by about `71%`, while still reaching `1.0000` validation accuracy on the current benchmark.

## Visual Preview

| Sample Distribution | CNN + ResNet Training Curve |
| --- | --- |
| ![Sample distribution](assets/figures/sample-distribution.png) | ![CNN ResNet training curve](assets/figures/cnn-resnet-training.png) |

| Random Forest Score | CNN + ResNet Score |
| --- | --- |
| ![Random Forest score](assets/figures/random-forest-score.png) | ![CNN ResNet score](assets/figures/cnn-resnet-score.png) |

## Repository Layout

```text
.
|-- assets/
|   `-- figures/                         # Lightweight result figures kept for the repo page
|-- artifacts/                           # Local outputs, ignored by Git
|   `-- README.md
|-- data/                                # Local raw datasets and documents, ignored by Git
|   `-- README.md
|-- notebooks/
|   |-- 01_data_preparation.ipynb        # Build processed samples from raw .mat files
|   `-- 02_model_training.ipynb          # Explore training from notebooks
|-- scripts/
|   |-- prepare_dataset.py               # Reproducible dataset-generation entry point
|   `-- train_models.py                  # Reproducible benchmarking entry point
|-- src/
|   `-- bearing_fault_diagnosis/
|       |-- __init__.py
|       |-- data.py
|       |-- models.py
|       `-- training.py
|-- n_model.py                           # Compatibility shim for older notebook code
|-- requirements.txt
`-- README.md
```

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the raw data

Place the CWRU bearing dataset files under the local `data/` directory.

This repository does not commit the raw dataset because it is large and should stay separate from the public source tree.

### 3. Build the processed dataset

```bash
python scripts/prepare_dataset.py
```

This regenerates:

- `artifacts/datasets/train_data.npy`
- `artifacts/datasets/label.npy`

### 4. Train and benchmark the models

```bash
python scripts/train_models.py
```

This script trains and compares:

- Random Forest
- the original simple CNN baseline
- the original deeper CNN + ResNet setting
- the optimized lighter CNN + ResNet setting

It also writes a benchmark summary to:

- `artifacts/reports/benchmark_results.json`

### 5. Explore the notebooks

The original notebook workflow is still available:

1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_model_training.ipynb`

The notebooks were updated to follow the cleaned project structure and now read or write generated files under `artifacts/`.

## Implementation Notes

- Preferred source code lives in `src/bearing_fault_diagnosis/`.
- `scripts/prepare_dataset.py` is the clean entry point for dataset generation.
- `scripts/train_models.py` is the clean entry point for reproducible model comparison.
- `n_model.py` is kept only for backward compatibility with the original notebook import pattern.
- Generated arrays, trained weights, TensorBoard logs, and benchmark reports are written to `artifacts/`.
- Large local data and generated artifacts are excluded through `.gitignore`.

## Dataset Note

This project is organized around the Case Western Reserve University bearing fault dataset. If you publish or redistribute derived work, please make sure your usage complies with the dataset's original terms and citation expectations.

## License

This repository is released under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

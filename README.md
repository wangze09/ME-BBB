# ME-BBB: Mechanism-Enhanced Blood–Brain Barrier Permeability Prediction

[中文说明](README_zh-CN.md) · [Workflow](docs/workflow.md) · [Environment](docs/environment.md) · [Input data guide](data/README.md) · [Model acquisition](models/README.md) · [Third-party licenses](THIRD_PARTY_NOTICES.md)

ME-BBB integrates an ECFP4/Morgan radius-2 fingerprint (2,048 bits) with five transport/permeability expert probability scores for **BCRP, MDR1, MRP, PAMPA, and Caco-2**. Each expert uses TabPFN 3.5 with RDKit2DNormalized descriptors; an ExtraTrees classifier combines the five predictions with the molecular fingerprint for BBB permeability classification. The research workflow uses predefined Bemis–Murcko scaffold partitions.

## Contents

This is a **source-code release**. It contains Python scripts for mechanistic-expert training, baseline comparison, BBB model fitting, feature ablation, and manuscript Figures 2–6, plus configuration and basic smoke tests. Molecular datasets, per-split files, generated expert scores, model checkpoints, fitted models, SHAP caches, and manuscript figure outputs are **not included**. The scripts are therefore **not runnable end-to-end out of the box**; users must supply appropriately licensed input data, models and dependencies. No claim of independent reproduction of the manuscript's numerical results is made by this code-only release.

## Repository map

| Task | Script |
| --- | --- |
| TabPFN expert model selection | `scripts/train_experts/tune_tabpfn_experts.py` |
| Train full-data mechanistic experts | `scripts/train_experts/train_full_tabpfn_experts.py` |
| Classical-model expert comparison | `scripts/train_experts/tune_classical_experts.py` |
| Expert model evaluation | `scripts/train_experts/evaluate_selected_experts.py` |
| BBB model training and tuning | `scripts/train_bbb/tune_and_train_bbb.py` |
| Feature ablation | `scripts/analyses/ablation/run_ablation.py` |
| Manuscript Figures 2–6 | `scripts/figures/figure2/` through `scripts/figures/figure6/` |

The current figure mapping follows the revised manuscript numbering: feature attribution is Figure 5 and case studies are Figure 6.

## Environment

The supplied GPU-machine metadata (November 3, 2026) is:

| Component | Value |
| --- | --- |
| Python | 3.11.17 |
| PyTorch | 2.7.1+cu118 |
| PyTorch CUDA build/runtime | 11.8 |
| NVIDIA driver | 580.95.05 |
| CUDA version reported by `nvidia-smi` | 13.0 (driver capability, **not** the PyTorch CUDA build) |
| GPUs | 2 × NVIDIA Tesla T4, 15,360 MiB each |

The precise TabPFN, RDKit, scikit-learn, and other training-library versions are **not yet verified**. See [environment details](docs/environment.md). The scripts have not been re-trained on this exact GPU configuration as part of this public release.

## Getting started

```bash
conda create -n me-bbb python=3.11 -y
conda activate me-bbb
python -m pip install -r requirements/dev.txt
python -m pytest tests -q
```

The default development installation is CPU-friendly and excludes TabPFN. To train the expert models, install `requirements/tabpfn.txt` in a compatible GPU environment and access the official TabPFN 3.5 checkpoint under its own license. `torch==2.7.1+cu118` describes the supplied machine; do not assume a generic `pip install torch` reproduces that build. See [the upstream PyTorch install guide](https://pytorch.org/get-started/previous-versions/) for platform-specific wheels and [Prior Labs](https://huggingface.co/Prior-Labs/tabpfn_3_5) for official checkpoint access.

To inspect script interfaces after installing dependencies:

```bash
python scripts/train_bbb/tune_and_train_bbb.py --help
python scripts/train_experts/tune_tabpfn_experts.py --help
python scripts/figures/figure4/plot_figure4.py --help
```

The scripts expect data files laid out as described in [data/README.md](data/README.md). Outputs default to `outputs/` (excluded from version control).

## License and citation

The repository's original code is distributed under the project [MIT license](LICENSE). This **does not relicense** TabPFN 3.5 weights or derivatives, third-party packages, molecular data, or separately restricted model outputs. TabPFN 3.5 is obtained from its official provider under a separate non-commercial license; see [third-party notices](THIRD_PARTY_NOTICES.md). Formal paper citation and DOI will be added when available.

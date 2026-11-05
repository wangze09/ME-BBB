# ME-BBB computational environments and provenance

ME-BBB used **two separate Conda environments**, as reported by the researcher:

- **`base`** — conventional machine learning (including final BBB ExtraTrees training), feature ablation, SHAP attribution, and figure generation. These workflows were run mainly on CPU, although CUDA-capable PyTorch is installed in this environment.
- **`tabpfn35`** — TabPFN 3.5 mechanistic expert fitting and expert-score inference, with GPU support.

The following versions were read from the two user-supplied `me_bbb_environment_report*.json` audit reports in November 2026. These are **observed installed environments**, not immutable lockfiles or proof that every original historical experiment used precisely these package builds. Both audit reports have an empty `historical_metadata` array.

## Recorded software versions

| Component | Conventional ML / analysis (`base`) | TabPFN experts (`tabpfn35`) |
| --- | --- | --- |
| Conda environment | `base` | `tabpfn35` |
| Python | 3.13.9 | 3.11.17 |
| TabPFN Python package | Not installed | 9.2.0 |
| TabPFN model selected in code | Not applicable | `ModelVersion.V3_5` |
| scikit-learn | 1.6.1 | 1.9.1 |
| XGBoost | 3.1.2 | Not installed |
| RDKit | 2025.09.3 | 2026.09.1 |
| descriptastorus | 2.8.0 | 2.8.0 |
| SHAP | 0.51.0 | Not installed |
| NumPy | 2.1.3 | 2.4.6 |
| pandas | 2.2.3 | 3.1.0 |
| PyTorch (imported version) | 2.9.1+cu128 | 2.7.1+cu118 |
| SciPy | 1.15.3 | 1.17.1 |
| Matplotlib | 3.10.0 | 3.11.2 |
| joblib | 1.4.2 | 1.6.0 |
| CUDA build used by PyTorch | 12.8 | 11.8 |
| NVIDIA driver | 580.95.05 | 580.95.05 |
| GPU hardware reported by `nvidia-smi` | 2 × Tesla T4, 15,360 MiB each | 2 × Tesla T4, 15,360 MiB each |

The hardware is shared by the two environments; this table **does not mean** that conventional ML or SHAP was GPU-accelerated. In both audited environments, `torch.cuda.is_available()` returned `True`; that only establishes CUDA availability. The `CUDA Version` value shown by `nvidia-smi` (previously reported as 13.0) represents a driver capability, **not** either environment's PyTorch CUDA build.

The model configuration identifies the upstream checkpoint as `tabpfn-v3.5-20260909.safetensors`; its checksum and exact historical file identity have not been independently checked. The user-provided version strings and source labels are also recorded in [`configs/audited_environments.json`](../configs/audited_environments.json).

## Which environment belongs to each workflow?

| Workflow | Environment | Important notes |
| --- | --- | --- |
| TabPFN scaffold benchmarks and five expert models | `tabpfn35` | TabPFN 3.5, RDKit2DNormalized / descriptastorus |
| TabPFN expert-score generation | `tabpfn35` | Export probabilities; downstream scripts read the resulting score CSV |
| Conventional expert model comparisons | `base` | scikit-learn / XGBoost |
| BBB fingerprint + mechanism fusion, ExtraTrees and ablations | `base` | ECFP4 (Morgan radius 2; 2,048 bits) + five expert scores |
| SHAP and permutation attribution | `base` | SHAP 0.51.0 present in the audit |
| Manuscript figure generation | `base` | Matplotlib and other plotting dependencies |

The expert models and final BBB estimator should normally be **loaded with environments compatible with the ones used for serialization**. In particular, different RDKit and scikit-learn versions can affect feature generation or deserialization. Do not load untrusted pickle/joblib files.

## Installation versus historical provenance

The root [`environment.yml`](../environment.yml) is a **convenience development / smoke-test environment**, not the author's original `base` or `tabpfn35` environment. The `requirements/*.txt` files are **unpinned dependency groups**, not reproductions of the audited versions. Do not state that they can exactly recreate the reported experiments.

To provision separate environments for subsequent work (approximate installation guides only):

```bash
# Conventional ML / SHAP / plotting
conda create -n me-bbb-analysis python=3.13 -y
conda activate me-bbb-analysis
python -m pip install -r requirements/figures.txt

# In a separate shell or after deactivating, TabPFN experts
conda create -n me-bbb-tabpfn python=3.11 -y
conda activate me-bbb-tabpfn
# Install a compatible PyTorch/CUDA build using the official PyTorch instructions first.
python -m pip install -r requirements/tabpfn.txt
```

These commands intentionally do **not** claim to lock exact package builds. Check CUDA wheels, TabPFN 3.5 access/license, and binary compatibility before fitting or unpickling. The pretrained checkpoint is obtained separately from [Prior Labs](https://huggingface.co/Prior-Labs/tabpfn_3_5).

For code-only syntax/data-layout checks, a Python 3.11 environment is sufficient:

```bash
conda env create -f environment.yml
conda activate me-bbb-dev
python -m unittest discover -s tests -v
```

## Audit historical runs before claiming exact reproduction

The original scripts can save provenance files that are more specific than a present-day package audit:

- `rdkit2d_<endpoint>_full.json`: expert `tabpfn_package_version`, model version, seed, input hash, and training device.
- BBB `run_config.json`: RDKit and scikit-learn versions, split/input hashes, and modeling configuration.
- SHAP `run_config.json`: RDKit, scikit-learn, and SHAP versions; settings and input hashes.

For each relevant original environment, export:

```bash
python -m pip freeze > local_pip_freeze.txt
conda list --export > local_conda_list.txt
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)"
nvidia-smi
```

Review exports for local absolute paths, private URLs, credentials and other confidential data before publishing. Model results and historical package versions have **not** been revalidated simply by editing these documents.

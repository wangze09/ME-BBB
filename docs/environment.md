# Environment provenance

## GPU machine reported by the author (2026-11-03)

| Component | Value | Meaning |
| --- | --- | --- |
| Python | 3.11.17 | Python interpreter |
| PyTorch | 2.7.1+cu118 | Installed PyTorch binary |
| `torch.version.cuda` | 11.8 | CUDA runtime with which PyTorch was built |
| NVIDIA driver | 580.95.05 | Host driver |
| `nvidia-smi` CUDA | 13.0 | Maximum CUDA driver API capability, not the active PyTorch build |
| GPU 0, GPU 1 | Tesla T4, 15,360 MiB each | Compute devices reported by `nvidia-smi` |

These are machine-level observations; they do not prove that every archived experiment was run with exactly this environment. In particular, running processes shown in `nvidia-smi` can belong to unrelated Conda environments.

## Versions to verify before claiming exact experiment reproduction

`tabpfn`, selected official checkpoint build, `rdkit`, `descriptastorus`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `xgboost`, `shap`, `matplotlib`, `joblib`, model settings and seeds.

Export from the **actual training environment**:

```bash
python -m pip freeze > local_pip_freeze.txt
conda list --export > local_conda_list.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
```

Review exported files for sensitive local paths or credentials before adding anything to Git. The provided `requirements/*.txt` are unpinned installation groups, **not** a frozen original training environment.

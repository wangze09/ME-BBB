# Model files are not included

No TabPFN 3.5 upstream checkpoint, fitted mechanistic-expert model, or final BBB ExtraTrees estimator is included in this repository.

- **TabPFN 3.5** must be obtained from [Prior Labs' official model page](https://huggingface.co/Prior-Labs/tabpfn_3_5) under its separate license; do not copy model weights from this repository.
- The scripts support a project-local path or official library-managed access; see `--model_path` and `TABPFN_MODEL_PATH` where implemented.
- `manifest.json` records the currently unpublished artifacts. `python models/download_models.py` only downloads entries with a verified published URL, size and SHA-256. Current entries are intentionally not published.

The repository MIT license covers original source code only and does not confer rights in TabPFN checkpoint files, model derivatives or outputs. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

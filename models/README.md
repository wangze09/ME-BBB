# Model acquisition: Zenodo archive

**Zenodo DOI:** [10.5281/zenodo.23908803](https://doi.org/10.5281/zenodo.23908803)

Five fitted TabPFN 3.5 mechanistic experts (BCRP, Caco-2, MDR1, MRP and PAMPA) and the final ExtraTrees BBB model are associated with this archive. These files are hosted separately from the GitHub repository. The record DOI was provided by the depositor; the public file inventory, exact filenames, sizes and SHA-256 checksums have not yet been independently verified.

No TabPFN 3.5 upstream checkpoint, fitted mechanistic-expert model, or final BBB ExtraTrees estimator is included in this repository.

- **TabPFN 3.5** must be obtained from [Prior Labs' official model page](https://huggingface.co/Prior-Labs/tabpfn_3_5) under its separate license; do not copy model weights from this repository.
- The scripts support a project-local path or official library-managed access; see `--model_path` and `TABPFN_MODEL_PATH` where implemented.
- `manifest.json` records the Zenodo DOI and the planned artifacts. `python models/download_models.py` requires a verified direct HTTPS file URL, byte size and SHA-256 for each artifact and a `published` manifest status. Those direct file details are not yet verified, so automatic downloading remains disabled; consult the DOI record for available files.

The repository MIT license covers original source code only and does not confer rights in TabPFN checkpoint files, model derivatives or outputs. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

# Third-party dependencies and licensing

ME-BBB's repository MIT license applies to its original project code only. It does not override separate licenses applicable to dependencies, checkpoints, fitted models, third-party datasets or restricted model outputs.

## TabPFN 3.5

- Official model page: https://huggingface.co/Prior-Labs/tabpfn_3_5
- Official model license: https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE
- Official inference-code repository: https://github.com/PriorLabs/TabPFN

The TabPFN 3.5 model is governed by a **separate non-commercial license**, with distinct provisions for models, derivatives and outputs. **This GitHub repository does not bundle fitted TabPFN models**; the trained model archive is linked at [Zenodo DOI 10.5281/zenodo.23908803](https://doi.org/10.5281/zenodo.23908803). It does publish TabPFN-derived expert-score tables and selected attribution caches under `data/derived/`. The complete official license and applicable notices must accompany authorized distribution of covered model artifacts in the separate archive. A project-local notice template is present in `models/licenses/NOTICE.txt`, but is not a substitute for the full upstream model license.

Other dependencies include RDKit (BSD-3-Clause), scikit-learn (BSD-3-Clause), XGBoost (Apache-2.0), SHAP (MIT), NumPy, pandas, SciPy, Matplotlib, Pillow, PyTorch, joblib and descriptastorus, each under its own terms. Verify current dependency licensing at the respective upstream repositories before distributing bundled code or binaries.

## Data

The public `data/` tree includes reference molecular data, predefined scaffold splits, derived expert scores, and attribution caches. The MIT source-code license does not automatically apply to those research datasets or third-party model outputs. Check their provenance and all applicable licenses before reuse.

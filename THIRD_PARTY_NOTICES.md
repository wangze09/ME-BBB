# Third-party dependencies and licensing

ME-BBB's repository MIT license applies to its original project code only. It does not override separate licenses applicable to dependencies, checkpoints, fitted models, third-party datasets or restricted model outputs.

## TabPFN 3.5

- Official model page: https://huggingface.co/Prior-Labs/tabpfn_3_5
- Official model license: https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE
- Official inference-code repository: https://github.com/PriorLabs/TabPFN

The TabPFN 3.5 model is governed by a **separate non-commercial license**, with distinct provisions for models, derivatives and outputs. **This repository does not redistribute its weights, fitted derivatives, or expert-score tables.** The official license and any required notices must accompany future authorized distribution of covered model artifacts. A project-local notice template is present in `models/licenses/NOTICE.txt`, but is not a substitute for the full upstream model license.

Other dependencies include RDKit (BSD-3-Clause), scikit-learn (BSD-3-Clause), XGBoost (Apache-2.0), SHAP (MIT), NumPy, pandas, SciPy, Matplotlib, Pillow, PyTorch, joblib and descriptastorus, each under its own terms. Verify current dependency licensing at the respective upstream repositories before distributing bundled code or binaries.

## Data

No proprietary or third-party labeled molecular datasets, derived prediction scores, or cached attribution results are distributed in this source-code release. Users must obtain and use their own appropriately licensed data.

# Input data layout (user supplied)

This directory intentionally contains no molecular data. Data redistribution permissions have not been established for the archived datasets.

Expected paths include:

```text
data/reference/bbb_dataset.csv
data/splits/experts/{bcrp,caco2,mdr1,mrp,pampa}/endpoint_subset.csv
data/splits/experts/{bcrp,caco2,mdr1,mrp,pampa}/scaffold{1..5}.csv
data/splits/bbb/scaffold_00/{bbb_train,bbb_val,bbb_test}.csv
... through scaffold_09
data/derived/expert_scores/bbb_tabpfn35_scores_full_data.csv
data/derived/ablation/ablation_by_split.csv
data/derived/figure5/feature_attribution/per_split/scaffold0_shap.npz
...
data/derived/figure6/case_data_v35.csv
```

**Column expectations (examples):** expert `endpoint_subset.csv` has `STSMILES`, `label`; expert split files have `STSMILES`, `GROUP` (`train`/`val`/`test`). BBB split files have `STSMILES`, `BBB`. The BBB expert score table contains molecule identifiers/SMILES and the five prediction columns expected by the BBB scripts. For precise headers, consult the corresponding loader in the relevant Python script.

Files and directories containing raw or derived data are excluded from Git by `.gitignore`. Add only data whose source license and redistribution rights have been confirmed.

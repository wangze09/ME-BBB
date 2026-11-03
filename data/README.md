# ME-BBB data directory

This directory contains the research data and derived results currently committed to this repository. The pretrained TabPFN checkpoint and fitted model weights are distributed separately (model-weight release on Zenodo is pending); see [models/README.md](../models/README.md).

## Directory layout

```text
data/
├── README.md
├── reference/
│   └── bbb_dataset.csv
├── splits/
│   ├── bbb/
│   │   └── scaffold_00/ ... scaffold_09/
│   │       ├── bbb_train.csv
│   │       ├── bbb_val.csv
│   │       ├── bbb_test.csv
│   │       ├── split_meta.json
│   │       └── split_report.txt
│   └── experts/
│       ├── BCRP/
│       ├── Caco-2/
│       ├── MDR1/
│       ├── MRP/
│       └── PAMPA/
│           (each endpoint: endpoint_subset.csv,
│            scaffold1.csv ... scaffold5.csv, split_meta.json)
├── derived/
│   ├── expert_scores/bbb_tabpfn35_scores_full_data.csv
│   ├── ablation/ablation_by_split.csv
│   ├── ablation/ablation_design.csv
│   ├── figure5/feature_attribution/
│   │   ├── individual_permutation_all_repeats.csv
│   │   └── per_split/scaffold0_shap.npz ... scaffold9_shap.npz
│   └── figure6/case_data_v35.csv
└── provenance/
    └── source_hashes.json
```

**Current completeness (checked November 3, 2026, `main`):** all five expert endpoint datasets and their five scaffold assignment files are present; all ten BBB scaffold directories include train, validation, test, and metadata files. The BBB root-level source/audit files (including `BBB_labels_corrected.csv` and `split_protocol.json`) are not published; their absence does not affect loading the released precomputed partitions, but workflows explicitly requiring those original files must use separately supplied inputs.

## File schemas and interpretation

| Dataset | Required / principal fields | Purpose |
| --- | --- | --- |
| `reference/bbb_dataset.csv` | Molecular SMILES and endpoint columns; inspect the CSV header | Reference table for dataset and mechanism figures |
| `splits/experts/<endpoint>/endpoint_subset.csv` | `STSMILES`, `label` | Binary labeled molecules for one mechanistic endpoint |
| `splits/experts/<endpoint>/scaffold1.csv` ... `scaffold5.csv` | `STSMILES`, `GROUP` | Five preassigned endpoint scaffold partitions; `GROUP` is `train`, `val`, or `test` |
| `splits/bbb/scaffold_XX/bbb_{train,val,test}.csv` | `STSMILES`, `BBB` and, when stored, additional endpoint/score columns | Predefined BBB partitions, using the directory number 00–09 |
| `splits/bbb/scaffold_XX/split_meta.json` | Split method and partition metadata | Documents how that particular BBB split was constructed |
| `derived/expert_scores/bbb_tabpfn35_scores_full_data.csv` | Molecular identifiers/SMILES plus five TabPFN 3.5 expert score columns | Mechanistic features consumed by the BBB modeling scripts |
| `derived/ablation/*.csv` | Experiment-specific metrics / design fields | Figure 4 and ablation summaries |
| `derived/figure5/feature_attribution/` | Permutation summary and ten SHAP `.npz` caches | Figure 5 |
| `derived/figure6/case_data_v35.csv` | Case, SMILES, transporter scores and BBB predictions | Figure 6 |

The primary BBB classifier uses Morgan/ECFP4 radius 2 (2,048 bits) together with five mechanistic expert scores, for a total of 2,053 features. Figure cache names and case-table names may retain historical `V35` or `V39` identifiers; do not treat these filenames as interchangeable training checkpoints.

The `scaffold_XX` BBB directories and `scaffold1`–`scaffold5` endpoint files describe **precomputed, distinct partition collections**. Do not regenerate or randomly resplit these files when reproducing reported analyses. Some original endpoint split metadata also mentions earlier random partitions; these are not distributed in the current GitHub tree and are not required by the scaffold-only main workflow.

## Important path compatibility note

The expert directories currently published on GitHub use their original case-sensitive names: **`BCRP`, `Caco-2`, `MDR1`, `MRP`, `PAMPA`**. Earlier versions of several training scripts looked up lowercase endpoint names; this has been corrected. The four expert training/evaluation scripts now use the published endpoint directory names verbatim, so the default paths resolve on case-sensitive Linux. Do not rename input directories or modify scientific CSV contents.

For BBB training, the script translates a logical split label such as `scaffold7` into `splits/bbb/scaffold_07`. All ten partition directories now contain the three required BBB CSV files. Model-weight availability, dependencies, input schemas and expert-score provenance still determine whether end-to-end training succeeds.

## Provenance and usage

`provenance/source_hashes.json` records checksums of the original release inputs. Published copies can be compared with that inventory after checking its path mappings. Data and cached model outputs are **not covered by the project's MIT source-code license**; check the original dataset permissions and [third-party notices](../THIRD_PARTY_NOTICES.md), especially for TabPFN-derived outputs. The trained model files and upstream TabPFN weights are not included here; see [models/README.md](../models/README.md).

This README describes the files **currently visible on GitHub**, not a claim that every experiment can already be reproduced without further input or dependency setup.

# ME-BBB pipeline

1. **Endpoint curation and scaffold assignments:** prepare labeled endpoint datasets for BCRP, MDR1, MRP, PAMPA and Caco-2 with a `STSMILES` column, binary endpoint labels and predefined train/val/test groups.
2. **Expert model selection:** compare TabPFN 3.5 / RDKit2DNormalized descriptors against classical baselines, selecting by predefined scaffold-validation criteria.
3. **Expert fitting:** train the selected five experts and generate their prediction probabilities for the BBB dataset.
4. **BBB model:** concatenate the ECFP4 (radius 2, 2,048 bits, `includeChirality=False`) fingerprint with five expert probabilities (2,053 features in total), then tune/train an ExtraTrees classifier across ten predefined scaffold partitions.
5. **Ablation and manuscript figures:** run the fixed ablation and plotting scripts using the expected CSV and cache schemas.

Scripts are kept close to the archived research implementation. Scaffold split generation and raw data curation are not included in this source-only release. The individual script `--help` options are authoritative for their supported inputs and outputs.

## Script entrances

```text
scripts/train_experts/tune_tabpfn_experts.py
scripts/train_experts/train_full_tabpfn_experts.py
scripts/train_experts/tune_classical_experts.py
scripts/train_experts/evaluate_selected_experts.py
scripts/train_bbb/tune_and_train_bbb.py
scripts/analyses/ablation/run_ablation.py
scripts/figures/figure2/plot_figure2.py
scripts/figures/figure3/plot_figure3.py
scripts/figures/figure4/plot_figure4.py
scripts/figures/figure5/plot_figure5.py
scripts/figures/figure6/plot_case_studies.py
```

No raw labels, prediction cache files, trained weights or scientific result tables are distributed in this code-only repository. Output directories are automatically created under `outputs/` and are gitignored.

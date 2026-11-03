#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ME-BBB mechanistic-expert selection: B-only / five-scaffold validation AUROC.

Purpose
-------
For each endpoint, identify ONE fixed combination of representation, algorithm,
and hyperparameters using the HIGHEST MEAN VALIDATION AUROC over scaffold1-5.
Then fit one final model on ALL labeled molecules of that endpoint.

IMPORTANT:
- The original scaffold1..scaffold5.csv files are READ, never modified.
- Each configuration is fitted on the current split's TRAIN and scored on VAL.
- Test labels/metrics are NEVER used for parameter selection or reporting.
- Full-data final fit happens ONLY AFTER all five validation results are ready.
- Mean validation AUROC is a selection score, NOT an unbiased test AUROC.
- This program only trains the conventional ML candidates, not TabPFN.

Required layout (for each endpoint):
  split_root/mrp/endpoint_subset.csv: STSMILES,label
  split_root/mrp/scaffold1.csv:       STSMILES,GROUP (train/val/test)
  ... scaffold2.csv to scaffold5.csv ...

Run (defaults already point to user's Linux directories):
  python train_mechanistic_experts_scaffold_AUROC_B_only.py --n_jobs 8

Smoke test (one endpoint, all five scaffolds, small grid subset):
  python train_mechanistic_experts_scaffold_AUROC_B_only.py \\
    --endpoints MRP --reps maccs --models extratrees \\
    --save_dir ./MRP_B_only_smoke --n_jobs 4

Search only / final fit from saved search files:
  --phase search
  --phase final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

LOG = logging.getLogger("ME-BBB-B-only")
SEED = 42
ENDPOINTS = ("BCRP", "Caco-2", "MDR1", "MRP", "PAMPA")
SPLITS = tuple(f"scaffold{i}" for i in range(1, 6))
REPS = ("ecfp", "maccs", "rdk")
MODELS = ("rf", "extratrees", "xgb", "svm_rbf")
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# RF/ExtraTrees: the original 1000-tree balanced ensemble is INCLUDED in grid.
# Four families * three representations = 90 configurations per split.
PARAM_GRIDS = {
    "rf": {
        "n_estimators": [500, 1000],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", 0.2],
    },
    "extratrees": {
        "n_estimators": [500, 1000],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", 0.2],
    },
    "xgb": {
        "n_estimators": [200, 500],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.1],
    },
    "svm_rbf": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto"],
    },
}


def params_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_csv_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.csv")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def write_json_atomic(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def read_labeled(endpoint_dir: Path) -> pd.DataFrame:
    path = endpoint_dir / "endpoint_subset.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"STSMILES": "string"})
    df.columns = df.columns.str.strip()
    if not {"STSMILES", "label"}.issubset(df.columns):
        raise ValueError(f"{path}: expected STSMILES,label columns")
    df = df[["STSMILES", "label"]].copy()
    if df.isna().any().any():
        raise ValueError(f"{path}: missing SMILES or labels")
    df["STSMILES"] = df["STSMILES"].astype(str).str.strip()
    if (df.STSMILES == "").any() or df.STSMILES.duplicated().any():
        raise ValueError(f"{path}: blank/duplicate STSMILES")
    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if not df.label.isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be binary 0/1")
    df["label"] = df.label.astype(int)
    if df.label.nunique() != 2:
        raise ValueError(f"{path}: training requires both classes")
    return df


def read_groups(endpoint_dir: Path, split: str, all_data: pd.DataFrame) -> pd.DataFrame:
    path = endpoint_dir / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    grp = pd.read_csv(path, dtype={"STSMILES": "string"})
    grp.columns = grp.columns.str.strip()
    if not {"STSMILES", "GROUP"}.issubset(grp.columns):
        raise ValueError(f"{path}: expected STSMILES,GROUP columns")
    grp = grp[["STSMILES", "GROUP"]].copy()
    if grp.isna().any().any():
        raise ValueError(f"{path}: blank group/SMILES")
    grp["STSMILES"] = grp.STSMILES.astype(str).str.strip()
    grp["GROUP"] = grp.GROUP.astype(str).str.strip().str.lower()
    if grp.STSMILES.duplicated().any():
        raise ValueError(f"{path}: duplicated STSMILES")
    if not grp.GROUP.isin(["train", "val", "test"]).all():
        raise ValueError(f"{path}: unexpected group: {sorted(grp.GROUP.unique())}")
    merged = all_data.merge(grp, on="STSMILES", how="inner", validate="one_to_one")
    if len(merged) != len(all_data) or len(merged) != len(grp):
        raise ValueError(f"{path}: mismatched SMILES; full={len(all_data)}, group={len(grp)}, merged={len(merged)}")
    if not {"train", "val"}.issubset(set(merged.GROUP.unique())):
        raise ValueError(f"{path}: missing train or val group")
    return merged


@lru_cache(maxsize=100000)
def scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi!r}")
    result = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    # Match the original expert scaffold splitter: all acyclic SMILES share a scaffold.
    return result if result else "NO_SCAFFOLD"


def scaffold_audit(df: pd.DataFrame, endpoint: str, split: str) -> dict:
    groups = {
        k: {scaffold(s) for s in df.loc[df.GROUP == k, "STSMILES"]}
        for k in ("train", "val", "test")
    }
    result = {
        "endpoint": endpoint, "split": split,
        "train_n": int((df.GROUP == "train").sum()),
        "val_n": int((df.GROUP == "val").sum()),
        "test_n_unused": int((df.GROUP == "test").sum()),
        "train_val_scaffold_overlap": len(groups["train"] & groups["val"]),
        "train_test_scaffold_overlap_info_only": len(groups["train"] & groups["test"]),
        "val_test_scaffold_overlap_info_only": len(groups["val"] & groups["test"]),
    }
    # Because this is B-only selection, only Train vs Val disjointness is
    # necessary for uncontaminated validation scoring. Test is never scored.
    if result["train_val_scaffold_overlap"]:
        raise ValueError(
            f"{endpoint}/{split}: Train and Val share "
            f"{result['train_val_scaffold_overlap']} Murcko scaffolds. "
            "Existing split left unchanged; cannot use it for strict scaffold validation."
        )
    if result["val_test_scaffold_overlap_info_only"] or result["train_test_scaffold_overlap_info_only"]:
        LOG.warning("%s/%s: test shares scaffolds with other groups, but test is "
                    "NOT accessed for model selection or evaluation.", endpoint, split)
    return result


@lru_cache(maxsize=90000)
def fp_one(smi: str, representation: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi!r}")
    if representation == "ecfp":
        bits = MORGAN.GetFingerprint(mol)
        n = 2048
    elif representation == "maccs":
        bits = MACCSkeys.GenMACCSKeys(mol)
        n = 167
    elif representation == "rdk":
        bits = Chem.RDKFingerprint(mol, fpSize=2048)
        n = 2048
    else:
        raise ValueError(f"Unknown fingerprint {representation}")
    arr = np.zeros(n, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bits, arr)
    if arr.shape != (n,):
        raise RuntimeError(f"Unexpected {representation} size: {arr.shape}, expected {n}")
    return arr


def featurize(series, representation: str) -> np.ndarray:
    return np.stack([fp_one(str(s), representation) for s in series]).astype(np.float32)


def estimator(name: str, n_jobs: int, final_fit: bool = False):
    if name == "rf":
        return RandomForestClassifier(
            class_weight="balanced", random_state=SEED, n_jobs=n_jobs
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            class_weight="balanced", random_state=SEED, n_jobs=n_jobs
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is needed when using --models xgb") from exc
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=n_jobs, verbosity=0,
        )
    if name == "svm_rbf":
        # Fast decision_function scores suffice for validation AUROC.
        # In the final fitted model, enable predict_proba for ME-BBB fusion.
        return SVC(kernel="rbf", class_weight="balanced",
                   probability=bool(final_fit), random_state=SEED)
    raise ValueError(name)


def positive_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return model.predict_proba(X)[:, classes.index(1)]
    return model.decision_function(X)


def grid_keys(reps, models):
    return [(rep, model, params_json(params))
            for rep in reps for model in models for params in ParameterGrid(PARAM_GRIDS[model])]


def run_search(args, endpoint: str, split: str, all_data: pd.DataFrame) -> None:
    dst = args.save_dir / "grid_by_split" / f"{endpoint}__{split}.csv"
    merged = read_groups(args.split_root / endpoint.lower(), split, all_data)
    audit = scaffold_audit(merged, endpoint, split)
    tr = merged.loc[merged.GROUP == "train"]
    va = merged.loc[merged.GROUP == "val"]
    ytr, yva = tr.label.to_numpy(dtype=int), va.label.to_numpy(dtype=int)
    if len(tr) == 0 or len(va) == 0 or len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
        raise ValueError(f"{endpoint}/{split}: missing class/empty Train or Val; AUROC undefined")

    audit_path = args.save_dir / "scaffold_audit" / f"{endpoint}__{split}.json"
    write_json_atomic(audit, audit_path)
    expected = grid_keys(args.reps, args.models)
    cached = {}
    if args.force_search:
        dst.unlink(missing_ok=True)
    elif dst.exists():
        old = pd.read_csv(dst, dtype={"params_json": str})
        for rec in old.to_dict("records"):
            key = (rec["representation"], rec["model"], rec["params_json"])
            if key in cached:
                raise RuntimeError(f"Duplicated grid row in {dst}: {key}")
            if not np.isfinite(float(rec["validation_auc"])):
                raise RuntimeError(f"Invalid cached validation AUROC in {dst}")
            cached[key] = rec
        LOG.info("RESUME %s/%s: %d/%d configurations", endpoint, split, len(cached), len(expected))
    extra = set(cached) - set(expected)
    if extra:
        raise RuntimeError(f"Unexpected grid configs in {dst}; use another --save_dir")
    if len(cached) == len(expected):
        LOG.info("SKIP %s/%s: all %d validation configs already scored", endpoint, split, len(expected))
        return

    LOG.info("SEARCH %s/%s: train=%d, val=%d, configs=%d", endpoint, split, len(tr), len(va), len(expected))
    for rep in args.reps:
        Xtr = featurize(tr.STSMILES, rep)
        Xva = featurize(va.STSMILES, rep)
        for model_name in args.models:
            for parameters in ParameterGrid(PARAM_GRIDS[model_name]):
                pjs = params_json(parameters)
                key = (rep, model_name, pjs)
                if key in cached:
                    continue
                tic = time.monotonic()
                model = estimator(model_name, args.n_jobs)
                model.set_params(**parameters)
                model.fit(Xtr, ytr)
                auc = float(roc_auc_score(yva, positive_score(model, Xva)))
                if not np.isfinite(auc):
                    raise ValueError(f"Non-finite validation AUROC {endpoint}/{split}/{key}")
                cached[key] = {
                    "endpoint": endpoint, "split": split,
                    "representation": rep, "model": model_name,
                    "params_json": pjs, "validation_auc": auc,
                    "fit_seconds": round(time.monotonic() - tic, 3),
                    "seed": SEED,
                }
                # Save after EVERY combination; a terminated process can resume.
                write_csv_atomic(pd.DataFrame([cached[k] for k in expected if k in cached]), dst)
        LOG.info("%s/%s completed fingerprint %s", endpoint, split, rep)


def consolidate(args) -> None:
    paths = sorted((args.save_dir / "grid_by_split").glob("*.csv"))
    if paths:
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        write_csv_atomic(df, args.save_dir / "all_grid_validation_results.csv")
    audits = sorted((args.save_dir / "scaffold_audit").glob("*.json"))
    if audits:
        df = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in audits])
        write_csv_atomic(df, args.save_dir / "scaffold_audit.csv")


def select_and_train_final(args, endpoint: str, all_data: pd.DataFrame):
    keys = grid_keys(args.reps, args.models)
    expected_keys = set(keys)
    frames = []
    for split in SPLITS:
        src = args.save_dir / "grid_by_split" / f"{endpoint}__{split}.csv"
        if not src.exists():
            LOG.warning("FINAL DEFERRED %s: missing validation results for %s", endpoint, split)
            return False
        df = pd.read_csv(src, dtype={"params_json": str})
        got = list(zip(df.representation, df.model, df.params_json))
        if len(got) != len(expected_keys) or len(set(got)) != len(got) or set(got) != expected_keys:
            LOG.warning("FINAL DEFERRED %s: partial validation grid in %s (%d/%d)",
                        endpoint, src.name, len(got), len(expected_keys))
            return False
        if (df["split"] != split).any() or (df["endpoint"] != endpoint).any():
            raise ValueError(f"Wrong split or endpoint contents in {src}")
        if not np.isfinite(df.validation_auc.to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid validation AUROC in {src}")
        frames.append(df)

    all_val = pd.concat(frames, ignore_index=True)
    ranking = (all_val.groupby(["representation", "model", "params_json"], as_index=False,
                                sort=False)
               .agg(mean_validation_auc=("validation_auc", "mean"),
                    sd_validation_auc=("validation_auc", "std"),
                    n_scaffold_splits=("split", "nunique")))
    if len(ranking) != len(expected_keys) or not (ranking.n_scaffold_splits == 5).all():
        raise ValueError(f"{endpoint}: incomplete final five-split validation ranking")
    ranking = ranking.sort_values(
        ["mean_validation_auc", "sd_validation_auc", "representation", "model", "params_json"],
        ascending=[False, True, True, True, True], kind="mergesort"
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.insert(0, "endpoint", endpoint)

    outdir = args.save_dir / "final_models"
    outdir.mkdir(parents=True, exist_ok=True)
    rank_out = outdir / f"{endpoint}_all_config_mean_validation.csv"
    write_csv_atomic(ranking, rank_out)

    best = ranking.iloc[0]
    family, rep = str(best["model"]), str(best["representation"])
    params = json.loads(best["params_json"])
    final_model_file = outdir / f"{endpoint}_full_data_expert.joblib"
    final_config_file = outdir / f"{endpoint}_selected_config.json"
    if final_model_file.exists() and final_config_file.exists() and not args.force_final:
        old = json.loads(final_config_file.read_text(encoding="utf-8"))
        if (old.get("representation") == rep and old.get("model") == family
                and old.get("hyperparameters") == params
                and old.get("full_data_sha256") == sha256_file(args.split_root / endpoint.lower() / "endpoint_subset.csv")):
            LOG.info("FINAL SKIP %s: matching final model already exists", endpoint)
            return True
        raise RuntimeError(f"{endpoint}: saved model configuration/data changed; use --force_final")

    LOG.info("FINAL FIT %s: %s+%s, mean Val AUROC=%.4f (SD=%.4f), N=%d",
             endpoint, rep, family, best["mean_validation_auc"], best["sd_validation_auc"], len(all_data))
    X = featurize(all_data.STSMILES, rep)
    model = estimator(family, args.n_jobs, final_fit=True)
    model.set_params(**params)
    model.fit(X, all_data.label.to_numpy(dtype=int))
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(f"Final model {family} has no predict_proba")

    package = {
        "model": model, "endpoint": endpoint, "representation": rep,
        "selected_params": params, "selection_metric": "mean validation AUROC over 5 scaffold splits",
        "mean_validation_auc": float(best["mean_validation_auc"]),
        "sd_validation_auc": float(best["sd_validation_auc"]),
        "train_scope": "all labeled samples of this endpoint",
        "n_labeled": int(len(all_data)), "n_features": int(X.shape[1]),
        "seed": SEED, "fingerprint_length": int(X.shape[1]),
        "svm_probability": bool(family == "svm_rbf"),
    }
    temp_model = final_model_file.with_suffix(".tmp.joblib")
    joblib.dump(package, temp_model, compress=3)
    os.replace(temp_model, final_model_file)
    write_json_atomic({
        "endpoint": endpoint, "model": family, "representation": rep,
        "hyperparameters": params,
        "selection_rule": "highest mean validation AUROC of identical configs across 5 scaffold splits",
        "mean_validation_auc": float(best["mean_validation_auc"]),
        "sd_validation_auc": float(best["sd_validation_auc"]),
        "seed": SEED, "n_full_training_samples": int(len(all_data)),
        "model_file": str(final_model_file),
        "full_data_sha256": sha256_file(args.split_root / endpoint.lower() / "endpoint_subset.csv"),
        "heldout_test_metrics": None,
        "note": "Test sets NOT evaluated; final model fitted on all endpoint labels after validation selection.",
    }, final_config_file)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split_root", type=Path,
                   default=REPO_ROOT / "data" / "splits" / "experts")
    p.add_argument("--save_dir", type=Path,
                   default=REPO_ROOT / "outputs" / "expert_classical")
    p.add_argument("--endpoints", nargs="+", choices=ENDPOINTS, default=list(ENDPOINTS))
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument("--reps", nargs="+", choices=REPS, default=list(REPS))
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--n_jobs", type=int, default=8, help="Threads per fitted estimator; grid loop is serial")
    p.add_argument("--phase", choices=("all", "search", "final"), default="all",
                   help="all=search then full fit; search=validation search only; final=from completed saved grids")
    p.add_argument("--force_search", action="store_true", help="Recompute each searched grid from scratch")
    p.add_argument("--force_final", action="store_true", help="Overwrite saved final fitted model")
    args = p.parse_args()
    if args.n_jobs < 1:
        p.error("--n_jobs must be >=1")
    if len(set(args.splits)) != len(args.splits) or len(set(args.endpoints)) != len(args.endpoints):
        p.error("Duplicate split or endpoint names are not allowed")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    # Strict run manifest prevents resuming from an incompatible parameter/representation search.
    manifest = {
        "script_protocol": "ME-BBB-B-only-mean-validation-AUROC-v1",
        "seed": SEED,
        "endpoints": list(args.endpoints), "splits": list(args.splits),
        "reps": list(args.reps), "models": list(args.models),
        "grids": {m: PARAM_GRIDS[m] for m in args.models},
        "split_root": str(args.split_root.resolve()),
        "selection": "same fixed configuration averaged over five scaffold validation splits",
        "no_test_evaluation": True,
        "dimensions": {"maccs": 167, "ecfp": 2048, "rdk": 2048},
        "rdkit_version": rdBase.rdkitVersion,
        "sklearn_version": sklearn.__version__,
    }
    conf = args.save_dir / "run_config.json"
    if conf.exists():
        prior = json.loads(conf.read_text(encoding="utf-8"))
        if prior != manifest:
            p.error("Output directory belongs to a different run configuration. Use a new --save_dir.")
    else:
        write_json_atomic(manifest, conf)

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        level=logging.INFO, datefmt="%H:%M:%S")
    LOG.info("B-only | root=%s | output=%s | fixed seed=%d", args.split_root, args.save_dir, SEED)
    failed = []
    fitted = []
    tic = time.monotonic()
    for ep in args.endpoints:
        try:
            data = read_labeled(args.split_root / ep.lower())
            if args.phase in ("search", "all"):
                for split in args.splits:
                    run_search(args, ep, split, data)
                consolidate(args)
            if args.phase in ("final", "all"):
                if set(args.splits) != set(SPLITS):
                    LOG.warning("FINAL DEFERRED %s: requires all scaffold1..scaffold5; subset requested", ep)
                elif select_and_train_final(args, ep, data):
                    fitted.append(ep)
        except Exception as exc:
            LOG.exception("FAILED %s: %s", ep, exc)
            failed.append({"endpoint": ep, "error": str(exc)})
        finally:
            consolidate(args)

    if fitted:
        csvs = sorted((args.save_dir / "final_models").glob("*_selected_config.json"))
        items = []
        for path in csvs:
            conf_data = json.loads(path.read_text(encoding="utf-8"))
            items.append({k: conf_data.get(k) for k in (
                "endpoint", "representation", "model", "mean_validation_auc",
                "sd_validation_auc", "n_full_training_samples", "model_file")})
        write_csv_atomic(pd.DataFrame(items), args.save_dir / "final_model_summary.csv")

    if failed:
        write_csv_atomic(pd.DataFrame(failed), args.save_dir / "failures.csv")
        raise SystemExit(1)
    LOG.info("DONE: %.2f min | final models generated/confirmed for %d endpoints",
             (time.monotonic() - tic) / 60, len(fitted))
    LOG.info("Outputs: %s", args.save_dir)


if __name__ == "__main__":
    main()

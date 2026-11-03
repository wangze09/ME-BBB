#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ME-BBB | Full-data TabPFN v3.5 mechanistic experts + optional BBB prediction.

Based on the user's TabPFN v3.5 RDKit2DNormalized benchmark script.

Differences from the scaffold benchmark:
  - No random/scaffold splits, no val/test model selection.
  - Fit ONE TabPFN v3.5 classifier per endpoint on ALL labeled samples in
    <split_root>/<endpoint>/endpoint_subset.csv (STSMILES, label).
  - Persist as rdkit2d_<endpoint>_full.pkl, NEVER overwrite *_best.pkl.
  - Optionally score a BBB SMILES CSV with all 5 experts.

All input features use descriptastorus RDKit2DNormalized, exactly as in the
benchmark script: process(smiles)[1:] as float32, invalid SMILES -> zero vector.

IMPORTANT: Scores on compounds included in the endpoint training data are
in-sample expert predictions, NOT leakage-free out-of-fold scores. Audit
cross-task overlap before using the scores in BBB model evaluation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("TabPFN-v3.5-Full")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "experts"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs" / "tabpfn_full_experts"
DEFAULT_CHECKPOINT = "auto"
ALL_ENDPOINTS = ("BCRP", "Caco-2", "MDR1", "MRP", "PAMPA")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_endpoint_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing endpoint dataset: {path}")
    df = pd.read_csv(path, dtype={"STSMILES": "string"})
    df.columns = df.columns.str.strip()
    if not {"STSMILES", "label"}.issubset(df.columns):
        raise ValueError(f"{path}: expected STSMILES and label columns")
    df = df[["STSMILES", "label"]].copy()
    if df.isna().any().any():
        raise ValueError(f"{path}: missing STSMILES or label")
    df["STSMILES"] = df.STSMILES.astype(str).str.strip()
    if (df.STSMILES == "").any():
        raise ValueError(f"{path}: blank SMILES")
    df["label"] = pd.to_numeric(df.label, errors="raise")
    if not df.label.isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be 0 or 1")
    df["label"] = df.label.astype(int)

    # Match the intention of endpoint_subset: one molecule, one label.
    if df.STSMILES.duplicated().any():
        conflicts = df.groupby("STSMILES")["label"].nunique()
        conflicts = conflicts[conflicts > 1]
        if not conflicts.empty:
            raise ValueError(f"{path}: duplicate SMILES with conflicting labels (n={len(conflicts)})")
        n_before = len(df)
        df = df.drop_duplicates(subset=["STSMILES"]).reset_index(drop=True)
        LOG.warning("%s: removed %d exact duplicate SMILES rows", path, n_before - len(df))
    if df.label.nunique() < 2:
        raise ValueError(f"{path}: cannot fit binary classifier with only one class")
    return df


def read_prediction_csv(path: Path, smiles_col: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")
    df = pd.read_csv(path, dtype={smiles_col: "string"})
    if smiles_col not in df.columns:
        raise ValueError(f"{path}: missing SMILES column {smiles_col!r}")
    if df[smiles_col].isna().any() or (df[smiles_col].astype(str).str.strip() == "").any():
        raise ValueError(f"{path}: missing/empty SMILES in {smiles_col}")
    return df


class RDKit2DFeaturizer:
    """Same feature definition/invalid-SMILES behavior as the uploaded script."""
    def __init__(self):
        from descriptastorus.descriptors import rdNormalizedDescriptors
        self.generator = rdNormalizedDescriptors.RDKit2DNormalized()
        dummy = self.generator.process("C")
        if dummy is None or not dummy[0]:
            raise RuntimeError("RDKit2DNormalized initialization failed")
        self.n_features = len(dummy[1:])

    def transform(self, smiles, strict_smiles: bool = False) -> np.ndarray:
        rows = []
        invalid = []
        for i, s in enumerate(smiles):
            result = self.generator.process(str(s))
            if result is None or result[0] == 0:
                invalid.append(i)
                rows.append(np.zeros(self.n_features, dtype=np.float32))
            else:
                arr = np.asarray(result[1:], dtype=np.float32)
                if arr.shape != (self.n_features,):
                    raise RuntimeError(f"Row {i}: expected {self.n_features} features; got {arr.shape}")
                rows.append(arr)
        if invalid:
            if strict_smiles:
                raise ValueError(f"Invalid SMILES at row positions {invalid[:12]}; n={len(invalid)}")
            LOG.warning("%d invalid SMILES converted to zero RDKit2D vectors (same as benchmark)", len(invalid))
        if not rows:
            raise ValueError("Empty SMILES list")
        return np.stack(rows, axis=0)


def positive_probability(classifier, X: np.ndarray) -> np.ndarray:
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(f"Positive label 1 missing from fitted classes: {classes}")
    result = classifier.predict_proba(X)
    result = np.asarray(result)
    if result.ndim != 2 or result.shape[0] != len(X):
        raise ValueError(f"Unexpected predict_proba shape: {result.shape}")
    prob = result[:, classes.index(1)].astype(np.float64)
    if not np.isfinite(prob).all() or (prob < -1e-5).any() or (prob > 1 + 1e-5).any():
        raise ValueError("Non-finite/out-of-range predicted probabilities")
    return np.clip(prob, 0, 1)


def batched_predict(clf, X: np.ndarray, batch_size: int) -> np.ndarray:
    parts = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        LOG.info("  inference batch %d:%d / %d", start, end, len(X))
        parts.append(positive_probability(clf, X[start:end]))
    return np.concatenate(parts)


def make_tabpfn(device: str, seed: int, ckpt: Path | None):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    if not hasattr(ModelVersion, "V3_5"):
        raise RuntimeError(
            "This TabPFN installation lacks ModelVersion.V3_5. "
            "Run in the SAME working v3.5 environment as your benchmark."
        )
    kwargs = {"random_state": seed, "device": device}
    if ckpt is not None:
        kwargs["model_path"] = str(ckpt)
    return TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, **kwargs)


def resolve_ckpt(raw: str) -> Path | None:
    if raw.lower() in ("auto", "none"):
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Provide --model_path /actual/path/to/tabpfn-v3.5-20260909.safetensors "
            "or use --model_path auto to use TabPFN's model cache/download."
        )
    return path.resolve()


def verify_env(device: str) -> str:
    import torch
    try:
        ver = package_version("tabpfn")
    except Exception:
        ver = "unknown"
    LOG.info("tabpfn version=%s | torch=%s | cuda_available=%s | device=%s",
             ver, torch.__version__, torch.cuda.is_available(), device)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu or correct PyTorch/CUDA installation")
        if ":" in device:
            index = int(device.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise RuntimeError(f"Device {device} unavailable: GPU count={torch.cuda.device_count()}")
    return ver


def maybe_empty_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def fit_one_endpoint(args, endpoint: str, featurizer: RDKit2DFeaturizer,
                     ckpt: Path | None, tabpfn_version: str,
                     predict_features: np.ndarray | None) -> tuple[dict, np.ndarray | None]:
    data_path = args.split_root / endpoint.lower() / "endpoint_subset.csv"
    df = load_endpoint_csv(data_path)
    sha = sha256(data_path)
    out_models = args.save_dir / "full_models"
    out_models.mkdir(parents=True, exist_ok=True)
    model_file = out_models / f"rdkit2d_{endpoint}_full.pkl"
    meta_file = out_models / f"rdkit2d_{endpoint}_full.json"

    should_fit = True
    if model_file.exists() or meta_file.exists():
        if args.force:
            LOG.warning("%s: overwriting existing full model (--force)", endpoint)
        elif model_file.is_file() and meta_file.is_file():
            old = json.loads(meta_file.read_text(encoding="utf-8"))
            expected = (
                old.get("data_sha256") == sha
                and old.get("seed") == args.seed
                and old.get("model_version") == "V3_5"
                and old.get("representation") == "RDKit2DNormalized"
                and old.get("model_path") == (str(ckpt) if ckpt else "auto")
            )
            if not expected:
                raise RuntimeError(f"{endpoint}: previous model metadata does not match settings; pass --force or change --save_dir")
            should_fit = False
            LOG.info("%s: skipping existing full fitted model: %s", endpoint, model_file)
        else:
            raise RuntimeError(f"{endpoint}: incomplete saved model or metadata; pass --force")

    clf = None
    if should_fit:
        tic = time.perf_counter()
        LOG.info("%s: full fit N=%d (positive=%d / negative=%d) | no splits",
                 endpoint, len(df), int(df.label.sum()), int((df.label == 0).sum()))
        X = featurizer.transform(df.STSMILES, strict_smiles=args.strict_smiles)
        clf = make_tabpfn(args.device, args.seed, ckpt)
        clf.fit(X, df.label.to_numpy(dtype=int))
        if not hasattr(clf, "classes_") or 1 not in list(clf.classes_):
            raise RuntimeError(f"{endpoint}: fitted model has invalid classes_")
        # Saving with pickle matches the previous benchmark's persisted .pkl model.
        tmp = model_file.with_name(model_file.name + ".tmp")
        try:
            with tmp.open("wb") as f:
                pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, model_file)
        finally:
            tmp.unlink(missing_ok=True)
        metadata = {
            "endpoint": endpoint,
            "representation": "RDKit2DNormalized",
            "n_features": int(X.shape[1]),
            "n_full_training_samples": int(len(df)),
            "n_pos": int(df.label.sum()),
            "n_neg": int((df.label == 0).sum()),
            "model_version": "V3_5",
            "tabpfn_package_version": tabpfn_version,
            "model_path": str(ckpt) if ckpt else "auto",
            "seed": int(args.seed),
            "device_at_training": str(args.device),
            "train_scope": "all labeled samples in endpoint_subset.csv, no split filtering",
            "data_path": str(data_path.resolve()),
            "data_sha256": sha,
            "model_file": str(model_file.resolve()),
            "save_format": "pickle",
            "full_fit_seconds": round(time.perf_counter() - tic, 2),
            "test_metrics": None,
            "note": "Not a held-out test fit; use for downstream prediction, not independent evaluation.",
        }
        write_json_atomic(meta_file, metadata)
        LOG.info("%s: saved full fitted model %s | %.1fs", endpoint, model_file, metadata["full_fit_seconds"])
        if args.save_official:
            from tabpfn.model_loading import save_fitted_tabpfn_model
            official_path = out_models / f"rdkit2d_{endpoint}_full.tabpfn_fit"
            if official_path.exists():
                if official_path.is_dir():
                    import shutil
                    shutil.rmtree(official_path)
                else:
                    official_path.unlink()
            save_fitted_tabpfn_model(clf, official_path)
            LOG.info("%s: also saved official fitted format at %s", endpoint, official_path)
    else:
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))

    if args.verify_reload or predict_features is not None:
        if clf is not None:
            del clf
            clf = None
            maybe_empty_cuda()
        LOG.info("%s: loading persisted model for %s", endpoint,
                 "full BBB inference" if predict_features is not None else "reload sanity check")
        with model_file.open("rb") as f:
            clf = pickle.load(f)
        if args.verify_reload:
            # A small prediction sanity check, NOT training/validation accuracy.
            probe = featurizer.transform(df.STSMILES.iloc[:min(3, len(df))], strict_smiles=args.strict_smiles)
            p = positive_probability(clf, probe)
            LOG.info("%s: reload + predict_proba OK; first %d probabilities: %s", endpoint, len(p), np.round(p, 5))
        predictions = batched_predict(clf, predict_features, args.batch_size) if predict_features is not None else None
        del clf
        maybe_empty_cuda()
        return metadata, predictions

    if clf is not None:
        del clf
    maybe_empty_cuda()
    return metadata, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_root", type=Path, default=DEFAULT_SPLIT_ROOT,
                    help="Directory containing each endpoint/endpoint_subset.csv")
    ap.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    ap.add_argument("--endpoints", nargs="+", choices=ALL_ENDPOINTS, default=list(ALL_ENDPOINTS))
    ap.add_argument("--device", default="cuda:0", help="cuda:0, cuda:1, or cpu; default cuda:0")
    ap.add_argument("--model_path", default=os.environ.get("TABPFN_MODEL_PATH", DEFAULT_CHECKPOINT),
                    help="Local .safetensors file; set 'auto' for cache/download")
    ap.add_argument("--seed", type=int, default=0, help="Same seed as original v3.5 benchmark")
    ap.add_argument("--batch_size", type=int, default=256, help="Number of BBB molecules per prediction batch")
    ap.add_argument("--strict_smiles", action="store_true", help="Fail on invalid SMILES instead of original zero-vector fallback")
    ap.add_argument("--force", action="store_true", help="Re-fit and overwrite full model if it already exists")
    ap.add_argument("--verify_reload", action="store_true", help="Reload saved pickle and predict a few train molecules as smoke test")
    ap.add_argument("--save_official", action="store_true", help="Additionally use PriorLabs save_fitted_tabpfn_model to create .tabpfn_fit")
    ap.add_argument("--predict_csv", type=Path, default=None,
                    help="Optional BBB SMILES CSV to score with every fitted expert")
    ap.add_argument("--smiles_col", default="STSMILES", help="SMILES column in --predict_csv")
    ap.add_argument("--predict_out", type=Path, default=None, help="Output CSV for BBB mechanism predictions")
    args = ap.parse_args(argv)

    if len(args.endpoints) != len(set(args.endpoints)):
        ap.error("Duplicate endpoints are not allowed")
    if args.batch_size < 1:
        ap.error("--batch_size must be positive")
    if args.predict_out is not None and args.predict_csv is None:
        ap.error("--predict_out requires --predict_csv")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    # Must be set before importing tabpfn.
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(PROJECT_ROOT / "models" / "cache"))
    ckpt = resolve_ckpt(str(args.model_path))
    ver = verify_env(args.device)
    featurizer = RDKit2DFeaturizer()
    LOG.info("RDKit2DNormalized dimensions=%d | endpoint root=%s", featurizer.n_features, args.split_root)
    LOG.info("Checkpoint=%s", ckpt if ckpt is not None else "TabPFN official auto cache")

    predictions_df = None
    X_predict = None
    if args.predict_csv is not None:
        predictions_df = read_prediction_csv(args.predict_csv, args.smiles_col)
        X_predict = featurizer.transform(predictions_df[args.smiles_col].astype(str), strict_smiles=args.strict_smiles)
        LOG.info("BBB prediction input: %d molecules, %d features", len(predictions_df), X_predict.shape[1])

    records = []
    for endpoint in args.endpoints:
        record, prob = fit_one_endpoint(args, endpoint, featurizer, ckpt, ver, X_predict)
        records.append(record)
        if predictions_df is not None:
            predictions_df[f"{endpoint}_score_TabPFN_V3_5"] = prob
            # Incremental checkpoint; unfinished endpoints are not fabricated.
            pred_out = args.predict_out or (args.save_dir / "BBB_five_mechanistic_scores_TabPFN_V3_5.csv")
            pred_out.parent.mkdir(parents=True, exist_ok=True)
            tmp = pred_out.with_name(pred_out.stem + ".tmp.csv")
            predictions_df.to_csv(tmp, index=False, encoding="utf-8-sig")
            os.replace(tmp, pred_out)
            LOG.info("Predicted score column saved: %s", endpoint)

    summaries = [{k: r[k] for k in ("endpoint", "n_full_training_samples", "n_pos", "n_neg", "n_features", "model_file", "model_version", "seed")}
                 for r in records]
    pd.DataFrame(summaries).to_csv(args.save_dir / "full_model_summary.csv", index=False, encoding="utf-8-sig")
    LOG.info("DONE: %d full fitted expert models saved under %s", len(records), args.save_dir / "full_models")
    if predictions_df is not None:
        LOG.info("BBB prediction CSV: %s", pred_out)
        LOG.warning("Check exact/canonical molecular overlap and OOF needs before using the predictions for BBB validation/test analysis")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Full expert fit failed")
        sys.exit(1)

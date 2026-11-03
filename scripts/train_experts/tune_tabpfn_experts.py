# -*- coding: utf-8 -*-
"""
ME-BBB mechanistic endpoint benchmark | TabPFN-3.5 | RDKit2DNormalized

Based on: 训练专家TabPFN_GPU_new_rdkit2D(1).py

Default experiment (unchanged from original):
  endpoint = MDR1; rep = rdkit2d; splits = scaffold1..5; seed = 0;
  validation AUC selects best model; every run is also evaluated on test.

Changes for v3.5:
  1. Use official ModelVersion.V3_5 (do not load v2.6 .ckpt).
  2. Separate model-weight cache and result directory from v2.6.
  3. Accept optional local .safetensors checkpoint (--model_path).
  4. Save best fitted models as .pkl, preserving downstream filenames.
  5. Resume-safe best-model tracking and scaffold-overlap warning.

Usage:
  python 训练专家TabPFN_V3_5_GPU_RDKit2D.py
  python 训练专家TabPFN_V3_5_GPU_RDKit2D.py --device cuda:0
  python 训练专家TabPFN_V3_5_GPU_RDKit2D.py --endpoints BCRP MDR1 PAMPA Caco-2 --seeds 0 1 2
  python 训练专家TabPFN_V3_5_GPU_RDKit2D.py --model_path /path/to/tabpfn-v3.5-20260909.safetensors

Dependencies:
  python -m pip install 'tabpfn==9.2.0' descriptastorus rdkit pandas numpy scikit-learn

Note: For headless environments, use TABPFN_TOKEN as an external environment variable.
Do not hard-code your token into a script.
"""

import os

# ---------------- Paths (edit here if moving the experiment) ----------------
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = str(PROJECT_ROOT / "data" / "splits" / "experts")
SAVE_DIR = str(PROJECT_ROOT / "outputs" / "tabpfn_scaffold_experts")
MODEL_CACHE_DIR = str(PROJECT_ROOT / "models" / "cache")

# Must be set before importing tabpfn. A user-supplied environment variable wins.
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", MODEL_CACHE_DIR)

import argparse
from importlib.metadata import version as package_version
import pickle
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

# ---------------- Original experiment defaults ----------------
# ENDPOINTS = ["MDR1"]
# ENDPOINTS = ["BCRP", "Caco-2", "MDR1", "MRP", "PAMPA"]
ENDPOINTS = ["BCRP", "Caco-2", "MDR1", "MRP", "PAMPA"]


SPLITS = [f"scaffold{i}" for i in range(1, 6)]
SEEDS = [0]
REPRESENTATIONS = ["rdkit2d"]
DEVICE = "cuda:0"  # First GPU by default; --device can select another GPU or CPU.

RESULT_COLUMNS = [
    "endpoint", "representation", "split", "split_type", "seed",
    "val_auc", "val_ap", "val_mcc", "val_acc", "val_f1",
    "val_tp", "val_tn", "val_fp", "val_fn",
    "auc", "ap", "mcc", "acc", "f1",
    "tp", "tn", "fp", "fn", "support_pos", "support_neg",
]


def log(msg):
    print(msg, flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_type(name):
    return "random" if name.startswith("random") else "scaffold"


# The *same* molecular representation as in the v2.6 script.
_rdkit2d_gen = rdNormalizedDescriptors.RDKit2DNormalized()
_DUMMY_FEATURE_COUNT = len(_rdkit2d_gen.process("C")[1:])


def featurize_rdkit2d(smiles_list):
    feats = []
    invalid_count = 0
    for smiles in smiles_list:
        result = _rdkit2d_gen.process(smiles)
        if result is None or result[0] == 0:
            # Keep the previous script's invalid-SMILES treatment unchanged.
            feats.append(np.zeros(_DUMMY_FEATURE_COUNT, dtype=np.float32))
            invalid_count += 1
        else:
            feats.append(np.asarray(result[1:], dtype=np.float32))
    if invalid_count:
        log(f"[WARN] {invalid_count} invalid SMILES were replaced with zero vectors (as before).")
    return np.stack(feats, axis=0)


def featurize(smiles_list, rep):
    if rep == "rdkit2d":
        return featurize_rdkit2d(smiles_list)
    raise ValueError(f"Unsupported representation: {rep}")


def compute_metrics(y_true, y_score, score_is_prob=True):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    pred = (y_score >= (0.5 if score_is_prob else 0.0)).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    # Keep the original single-class convention for backwards comparability.
    auc, ap = 0.5, 0.5
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, pred) if len(np.unique(y_true)) > 1 else 0.0

    return dict(
        auc=float(auc), ap=float(ap), mcc=float(mcc),
        acc=float(accuracy_score(y_true, pred)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        tp=tp, tn=tn, fp=fp, fn=fn,
        support_pos=int((y_true == 1).sum()),
        support_neg=int((y_true == 0).sum()),
    )


def load_split(ep_dir, split_name):
    base = pd.read_csv(os.path.join(ep_dir, "endpoint_subset.csv"))[["STSMILES", "label"]]
    grp = pd.read_csv(os.path.join(ep_dir, f"{split_name}.csv"))[["STSMILES", "GROUP"]]
    m = base.merge(grp, on="STSMILES", how="inner")
    tr = m[m.GROUP == "train"][["STSMILES", "label"]].reset_index(drop=True)
    va = m[m.GROUP == "val"][["STSMILES", "label"]].reset_index(drop=True)
    te = m[m.GROUP == "test"][["STSMILES", "label"]].reset_index(drop=True)
    return tr, va, te


def warn_on_split_overlap(tr, va, te, split_name, check_scaffold=True):
    """Only report overlap; do not change the input split or silently 'fix' it."""
    sets = {name: set(df.STSMILES) for name, df in (("train", tr), ("val", va), ("test", te))}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(sets[a] & sets[b])
        if n:
            log(f"[WARN] {split_name}: {a}/{b} share {n} identical SMILES. Input split unchanged.")

    if not check_scaffold or not split_name.startswith("scaffold"):
        return

    def scaffolds(smiles_values):
        result = set()
        for smiles in smiles_values:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                result.add(scaf)
        return result

    groups = {name: scaffolds(ss) for name, ss in sets.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(groups[a] & groups[b])
        if n:
            log(f"[WARN] {split_name}: {a}/{b} share {n} Murcko scaffolds. "
                "This split is NOT strictly scaffold-disjoint; no automatic repair was made.")


def make_classifier(seed, device, model_path=None):

    model_path = model_path or os.environ.get("TABPFN_MODEL_PATH")
    if model_path and model_path.lower() in ("auto", "none"):
        model_path = None
    if model_path and not Path(model_path).expanduser().is_file():
        raise FileNotFoundError(f"TabPFN-3.5 checkpoint not found: {model_path}")

    kwargs = dict(random_state=seed, device=device)
    if model_path:
        kwargs["model_path"] = str(Path(model_path).expanduser().resolve())
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, **kwargs)

    return clf


def train_and_eval_tabpfn(Xtr, ytr, Xva, yva, Xte, yte, seed, device, model_path=None):
    np.random.seed(seed)
    if len(np.unique(ytr)) < 2:
        log("[WARN] Training data contains only one class; no fitted classifier for this run.")
        va_zero = np.zeros(len(yva), dtype=np.float32)
        te_zero = np.zeros(len(yte), dtype=np.float32)
        return (
            compute_metrics(yva, va_zero, score_is_prob=False),
            compute_metrics(yte, te_zero, score_is_prob=False),
            None,
        )

    clf = make_classifier(seed=seed, device=device, model_path=model_path)
    clf.fit(Xtr, ytr)
    val_score = clf.predict_proba(Xva)[:, 1]
    test_score = clf.predict_proba(Xte)[:, 1]
    return compute_metrics(yva, val_score), compute_metrics(yte, test_score), clf


def load_done_keys(out_csv):
    if not os.path.exists(out_csv):
        return set()
    df = pd.read_csv(out_csv, usecols=["endpoint", "representation", "split", "seed"])
    return set(zip(df.endpoint, df.representation, df.split, df.seed.astype(int)))


def append_row(out_csv, row_dict):
    pd.DataFrame([row_dict], columns=RESULT_COLUMNS).to_csv(
        out_csv, mode="a", header=not os.path.exists(out_csv), index=False
    )


class BestModelTracker:
    """Keep best-by-val-AUC models, including when a run is resumed."""

    def __init__(self, model_dir, summary_path):
        self.model_dir = model_dir
        self.summary_path = summary_path
        ensure_dir(model_dir)
        self._store = {}
        if os.path.isfile(summary_path):
            df = pd.read_csv(summary_path)
            for _, row in df.iterrows():
                fname = str(row["model_file"])
                path = os.path.join(model_dir, fname)
                if not os.path.isfile(path):
                    continue
                self._store[(str(row["representation"]), str(row["endpoint"]))] = {
                    "val_auc": float(row["best_val_auc"]),
                    "split": str(row["best_split"]),
                    "seed": int(row["best_seed"]),
                    "model_file": fname,
                }

    def update(self, rep, endpoint, split, seed, val_auc, clf):
        key = (rep, endpoint)
        if clf is None:
            return
        if key in self._store and val_auc <= self._store[key]["val_auc"]:
            return

        fname = f"{rep}_{endpoint}_best.pkl"  # Preserve existing downstream filename.
        final_path = os.path.join(self.model_dir, fname)
        tmp_path = final_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, final_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        self._store[key] = dict(
            val_auc=float(val_auc), split=split, seed=int(seed), model_file=fname
        )
        self.save_summary()  # Persist immediately, not only at normal program exit.
        log(f"[SAVE] New best: {rep}/{endpoint}, val AUC={val_auc:.4f} -> {final_path}")

    def save_summary(self):
        rows = []
        for (rep, endpoint), info in sorted(self._store.items()):
            rows.append(dict(
                representation=rep, endpoint=endpoint,
                best_split=info["split"], best_seed=info["seed"],
                best_val_auc=info["val_auc"], model_file=info["model_file"],
            ))
        if rows:
            pd.DataFrame(rows).to_csv(self.summary_path, index=False)


def check_environment(device):
    log(f"[INFO] tabpfn package version : {package_version('tabpfn')}")
    log("[INFO] requested model        : ModelVersion.V3_5 (standard model)")
    log(f"[INFO] checkpoint cache       : {os.environ['TABPFN_MODEL_CACHE_DIR']}")
    log(f"[INFO] target device          : {device}")
    try:
        import torch
        log(f"[INFO] PyTorch/CUDA           : {torch.__version__} / {torch.cuda.is_available()}")
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; pass --device cpu or check CUDA/PyTorch installation.")
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"{device} does not exist (available CUDA GPU count: {torch.cuda.device_count()}). "
                    "Use --device cuda:0 if there is only one GPU."
                )
    except ImportError:
        raise RuntimeError("PyTorch is not installed in this Python environment.")


def main():
    parser = argparse.ArgumentParser(description="ME-BBB / TabPFN-3.5 RDKit2D endpoint training")
    parser.add_argument("--split_root", default=SPLIT_ROOT)
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--out_name", default="summary_tabpfn_by_endpoint_splitType_V3_5.csv")
    parser.add_argument("--endpoints", nargs="*", default=ENDPOINTS)
    parser.add_argument("--splits", nargs="*", default=SPLITS)
    parser.add_argument("--reps", nargs="*", default=REPRESENTATIONS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--model_path", default=None,
                        help="Optional local TabPFN-3.5 .safetensors path; by default use official cache/download.")
    parser.add_argument("--skip_scaffold_check", action="store_true",
                        help="Do not print Murcko scaffold overlap warnings (splits are never modified).")
    args = parser.parse_args()

    check_environment(args.device)
    ensure_dir(args.save_dir)
    ensure_dir(os.environ["TABPFN_MODEL_CACHE_DIR"])
    model_dir = os.path.join(args.save_dir, "best_models")
    out_csv = os.path.join(args.save_dir, args.out_name)
    best_summary = os.path.join(args.save_dir, "best_model_summary.csv")
    done = load_done_keys(out_csv)
    tracker = BestModelTracker(model_dir, best_summary)

    log(f"[INFO] split root           : {args.split_root}")
    log(f"[INFO] output CSV           : {out_csv}")
    log(f"[INFO] best models          : {model_dir}")
    log(f"[INFO] already completed    : {len(done)} runs")
    if args.model_path:
        log(f"[INFO] local model override : {args.model_path}")
    start_time = time.time()

    for rep in args.reps:
        log(f"\n===== Representation: {rep.upper()} =====")
        for endpoint in args.endpoints:
            ep_dir = os.path.join(args.split_root, endpoint)
            subset_path = os.path.join(ep_dir, "endpoint_subset.csv")
            if not os.path.isfile(subset_path):
                log(f"[WARN] skip missing endpoint: {subset_path}")
                continue

            for split in args.splits:
                split_path = os.path.join(ep_dir, f"{split}.csv")
                if not os.path.isfile(split_path):
                    log(f"[WARN] skip missing split: {split_path}")
                    continue
                remaining = [seed for seed in args.seeds if (endpoint, rep, split, int(seed)) not in done]
                if not remaining:
                    log(f"[SKIP] {endpoint}/{rep}/{split}: all seeds already in CSV")
                    continue

                try:
                    tr, va, te = load_split(ep_dir, split)
                    log(f"\n[{endpoint}/{rep}/{split}] train={len(tr)} val={len(va)} test={len(te)}")
                    if min(len(tr), len(va), len(te)) == 0:
                        raise ValueError("An empty train/validation/test partition was found.")
                    warn_on_split_overlap(tr, va, te, split, not args.skip_scaffold_check)
                    Xtr = featurize(tr.STSMILES, rep)
                    Xva = featurize(va.STSMILES, rep)
                    Xte = featurize(te.STSMILES, rep)
                    ytr, yva, yte = tr.label.values, va.label.values, te.label.values
                except Exception as exc:
                    log(f"[ERROR] Failed to load/featurize {endpoint}/{split}: {exc}")
                    log(traceback.format_exc())
                    continue

                for seed in remaining:
                    try:
                        val_m, test_m, clf = train_and_eval_tabpfn(
                            Xtr, ytr, Xva, yva, Xte, yte, int(seed),
                            device=args.device, model_path=args.model_path,
                        )
                        row = dict(
                            endpoint=endpoint, representation=rep, split=split,
                            split_type=split_type(split), seed=int(seed),
                            val_auc=val_m["auc"], val_ap=val_m["ap"],
                            val_mcc=val_m["mcc"], val_acc=val_m["acc"],
                            val_f1=val_m["f1"], val_tp=val_m["tp"],
                            val_tn=val_m["tn"], val_fp=val_m["fp"],
                            val_fn=val_m["fn"],
                            **{key: test_m[key] for key in (
                                "auc", "ap", "mcc", "acc", "f1", "tp", "tn",
                                "fp", "fn", "support_pos", "support_neg",
                            )},
                        )
                        # Save model first: a resumed run never marks an unsaved best model as done.
                        tracker.update(rep, endpoint, split, seed, val_m["auc"], clf)
                        append_row(out_csv, row)
                        done.add((endpoint, rep, split, int(seed)))
                        log(f"  seed={seed} | val AUC={val_m['auc']:.4f} "
                            f"| test AUC={test_m['auc']:.4f} "
                            f"| test MCC={test_m['mcc']:.4f} "
                            f"| elapsed={(time.time()-start_time)/60:.1f} min")
                    except Exception as exc:
                        log(f"[ERROR] {endpoint}/{rep}/{split}/seed={seed}: {repr(exc)}")
                        log(traceback.format_exc())

    tracker.save_summary()
    log("\n===== ALL DONE =====")
    if os.path.isfile(out_csv):
        df = pd.read_csv(out_csv)
        log("\n--- Test AUC mean by (representation, split_type) ---")
        log(str(df.groupby(["representation", "split_type"])["auc"].mean().round(4)))
        log("\n--- Test AUC mean by (representation, endpoint) ---")
        log(str(df.groupby(["representation", "endpoint"])["auc"].mean().round(4)))
    else:
        log("[WARN] No results produced. See warnings/errors above.")


if __name__ == "__main__":
    main()

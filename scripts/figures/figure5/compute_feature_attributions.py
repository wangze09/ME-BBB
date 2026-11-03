#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ME-BBB Figure 5 V39.3 — audited single-tree numerical SHAP repair option.

One command: fit 10 selected ExtraTrees models, compute held-out interventional
probability SHAP with TRAIN-only background, rank 15 ECFP4 bits by macro-mean
|SHAP|, separately permute those bits plus all five expert scores, and plot
matched 20-feature teal/coral Figure 5a-b; joint permutations are supplementary.

No previous script, SHAP .npz cache or pre-trained model is required.
Models, SHAP, and permutation calculations are cached independently by split.
This is descriptive repeated scaffold holdout interpretation, not an independent
or causal mechanistic confirmation. Test SHAP is used to select illustrated bits.

V39 defaults: ALL held-out TEST molecules (shap_samples=0); 10 permutation
repeats per feature; no explanatory footnote below the manuscript figure.
Interventional SHAP is cached by batch for safe resumption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

LOG = logging.getLogger('ME-BBB-V39')
REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / 'data' / 'splits' / 'bbb'
OUT = REPO_ROOT / 'outputs' / 'feature_attribution_computation'
DEFAULT_SCORES = REPO_ROOT / 'data' / 'derived' / 'expert_scores' / 'bbb_tabpfn35_scores_full_data.csv'

def split_directory(label: str) -> str:
    return f"scaffold_{int(label.removeprefix('scaffold')):02d}"
EXPERTS = ('BCRP', 'Caco-2', 'MDR1', 'MRP', 'PAMPA')
COLS = {e: f'{e}_score_TabPFN_V3_5' for e in EXPERTS}
NFP = 2048
FEATURES = [f'ECFP4_{i:04d}' for i in range(NFP)] + list(EXPERTS)
INDEX = {name: i for i, name in enumerate(FEATURES)}
PARAMS = dict(n_estimators=500, min_samples_leaf=3,
              max_features='sqrt', class_weight='balanced', random_state=42)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NFP,
                                                  includeChirality=False)
BLOCKS = {'ECFP4 (all bits)': list(range(NFP)),
          'All five experts': list(range(NFP, NFP+5)),
          'Efflux-associated': [2048, 2050, 2051],
          'Permeability-associated': [2049, 2052]}
TEAL = '#238493'; CORAL = '#DC7668'; DARK = '#253B4A'
BLUE = '#347DAC'; LIGHT = '#D7E4E9'; GREY = '#6B7E88'


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, path)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def save_npz_atomic(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as out:
        np.savez_compressed(out, **kwargs)
    os.replace(tmp, path)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for b in iter(lambda: src.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, low_memory=False, dtype={'STSMILES': 'string'})
    d.columns = [str(x).replace('\ufeff', '').strip() for x in d.columns]
    if not d.columns.is_unique:
        raise ValueError(f'Duplicate columns: {path}')
    return d


@lru_cache(maxsize=120000)
def canon(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid molecule: {smiles!r}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=120000)
def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or 'NO_SCAFFOLD'


@lru_cache(maxsize=120000)
def fingerprint(smiles: str) -> np.ndarray:
    ar = np.zeros(NFP, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(MORGAN.GetFingerprint(Chem.MolFromSmiles(smiles)), ar)
    return ar.astype(np.float32)


def load_score_lookup(path: Path) -> pd.DataFrame:
    d = read_csv(path)
    expected = {'STSMILES', *COLS.values()}
    if not expected.issubset(d.columns):
        raise ValueError(f'{path}: missing {sorted(expected-set(d.columns))}')
    d['_canon'] = [canon(str(z).strip()) for z in d.STSMILES]
    for col in COLS.values():
        d[col] = pd.to_numeric(d[col], errors='coerce')
        if not np.isfinite(d[col].to_numpy(float)).all() or not d[col].between(0, 1).all():
            raise ValueError(f'{path}: invalid values in {col}')
    if d._canon.duplicated().any():
        for name, grp in d.groupby('_canon'):
            a = grp[list(COLS.values())].to_numpy(float)
            if not np.allclose(a, a[0], atol=1e-7, rtol=1e-7):
                raise ValueError(f'Conflicting prediction scores for canonical molecule {name}')
        d = d.drop_duplicates('_canon')
    return d.set_index('_canon')[list(COLS.values())]


def load_part(path: Path, lookup: pd.DataFrame | None) -> pd.DataFrame:
    d = read_csv(path)
    label = 'label' if 'label' in d.columns else 'BBB'
    if 'STSMILES' not in d.columns or label not in d.columns:
        raise ValueError(f'{path}: requires STSMILES and BBB/label')
    if 'BBB' in d and 'label' in d:
        raise ValueError(f'{path}: ambiguous BBB and label columns')
    d['__label'] = pd.to_numeric(d[label], errors='coerce')
    if not d.__label.isin([0, 1]).all():
        raise ValueError(f'{path}: invalid BBB labels')
    d['_canon'] = [canon(str(z).strip()) for z in d.STSMILES]
    if d._canon.duplicated().any():
        raise ValueError(f'{path}: repeated canonical SMILES')
    if lookup is None:
        if not set(COLS.values()).issubset(d.columns):
            raise ValueError(f'{path}: no complete V3.5 mechanism predictions; use --scores_csv')
        raw = d[list(COLS.values())].copy()
    else:
        raw = lookup.reindex(d._canon.tolist()).reset_index(drop=True)
    raw = raw.apply(pd.to_numeric, errors='coerce')
    scores = raw.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError(f'{path}: missing/out-of-range predictions; check score coverage')
    for e in EXPERTS:
        d[COLS[e]] = raw[COLS[e]].to_numpy(dtype=np.float32)
    return d


def matrices(root: Path, split: str, lookup: pd.DataFrame | None):
    dfs = {p: load_part(root/split_directory(split)/f'bbb_{p}.csv', lookup) for p in ('train', 'val', 'test')}
    for a,b in [('train','val'), ('train','test'), ('val','test')]:
        aa, bb = set(dfs[a]._canon), set(dfs[b]._canon)
        if aa.intersection(bb):
            raise ValueError(f'{split}: exact molecule overlap {a} / {b}')
        shared = {scaffold(z) for z in aa}.intersection({scaffold(z) for z in bb})
        if shared:
            raise ValueError(f'{split}: achiral scaffold overlap {a} / {b}: {len(shared)}')
    for p in ('val', 'test'):
        d = dfs[p]
        if set(EXPERTS).issubset(d.columns) and d[list(EXPERTS)].notna().any().any():
            raise ValueError(f'{split}/{p}: actual endpoint labels present (not cold-start)')
    X, y = {}, {}
    for p, d in dfs.items():
        X[p] = np.ascontiguousarray(np.concatenate([
            np.stack([fingerprint(s) for s in d._canon]),
            d[list(COLS.values())].to_numpy(np.float32)], axis=1), dtype=np.float32)
        y[p] = d.__label.to_numpy(np.int64)
        if np.unique(y[p]).size != 2:
            raise ValueError(f'{split}/{p}: BBB labels must contain both classes')
    return X, y, dfs


def _positive_tree_shap(explainer, row):
    """Return positive-class probability attributions for exactly one row."""
    raw = explainer.shap_values(row, check_additivity=False)
    if isinstance(raw, list):
        vals = np.asarray(raw[1], dtype=np.float64)
    else:
        vals = np.asarray(raw, dtype=np.float64)
        if vals.ndim == 3 and vals.shape[-1] == 2:
            vals = vals[:, :, 1]
    baselines = np.asarray(explainer.expected_value, dtype=np.float64).ravel()
    baseline = float(baselines[1] if len(baselines) == 2 else baselines[0])
    if vals.shape != row.shape or not np.isfinite(vals).all():
        raise ValueError(f'Invalid single-tree SHAP: {vals.shape} != {row.shape}')
    return vals[0], baseline


def _repair_one_anomalous_tree(model, row, background, forest_phi, forest_base, prob,
                               tree_index=298, permutations=8, seed=20261030):
    """Replace ONE faulty tree's SHAP, not the model output or forest baseline.

    Uses model-agnostic permutation Shapley approximations over the *split-used*
    features of one tree, with exactly the original interventional background.
    The same missing-feature intervention game is retained. A strict check
    confirms this tree accounts for the ensemble's reconstruction error.

    Do NOT use for every tree or change SHAP reference distribution.
    """
    import shap
    row = np.asarray(row, dtype=np.float32).reshape(1, -1)
    bg = np.asarray(background, dtype=np.float32)
    if not (0 <= tree_index < len(model.estimators_)):
        raise IndexError(f'Invalid tree index {tree_index}')
    tree = model.estimators_[tree_index]
    ntrees = len(model.estimators_)
    tree_exp = shap.TreeExplainer(tree, data=bg,
                                  feature_perturbation='interventional',
                                  model_output='probability')
    bad_phi, tree_base = _positive_tree_shap(tree_exp, row)
    tree_prob = float(tree.predict_proba(row)[0, list(tree.classes_).index(1)])
    resid_tree = float(tree_base + bad_phi.sum(dtype=np.float64) - tree_prob)
    resid_ensemble = float(forest_base + forest_phi.sum(dtype=np.float64) - prob)
    if abs(resid_ensemble - resid_tree / ntrees) > 1e-5:
        raise RuntimeError('Tree-specific SHAP error does not explain ensemble residual: '
                           f'forest={resid_ensemble:+.9g}, '
                           f'tree/n={resid_tree/ntrees:+.9g}. Refusing repair.')
    features = np.unique(tree.tree_.feature[tree.tree_.feature >= 0]).astype(int)
    if features.size < 1:
        raise RuntimeError('Anomalous tree has no split-used features')

    # This projection is exact for this tree: no other feature is read by it.
    bg_small = np.ascontiguousarray(bg[:, features], dtype=np.float32)
    x_small = np.ascontiguousarray(row[:, features], dtype=np.float32)
    template = np.asarray(bg[0], dtype=np.float32)

    def tree_positive_on_used_features(z):
        z = np.asarray(z, dtype=np.float32)
        full = np.broadcast_to(template, (len(z), row.shape[1])).copy()
        full[:, features] = z
        return tree.predict_proba(full)[:, list(tree.classes_).index(1)]

    # Direct SHAP-permutation attribution is a *different numerical estimator*
    # of the same interventional Shapley game. Approximate but locally additive.
    perm_exp = shap.PermutationExplainer(tree_positive_on_used_features,
                                         bg_small, seed=seed)
    budget = int(permutations) * (2 * len(features) + 1)
    explained = perm_exp(x_small, max_evals=budget, silent=True)
    perm_vals = np.asarray(explained.values, dtype=np.float64)
    if perm_vals.shape != (1, len(features)):
        raise RuntimeError(f'Unexpected permutation SHAP output {perm_vals.shape}')
    perm_base = float(np.asarray(explained.base_values, dtype=np.float64).ravel()[0])
    true_tree_background = float(tree.predict_proba(bg)[:, list(tree.classes_).index(1)].mean())
    if abs(perm_base - true_tree_background) > 1e-7 or abs(tree_base - true_tree_background) > 1e-7:
        raise RuntimeError('Permutation/tree SHAP baselines differ from training background mean')
    reconstruction = float(perm_base + perm_vals.sum(dtype=np.float64))
    if abs(reconstruction - tree_prob) > 1e-7:
        raise RuntimeError(f'Permutation SHAP failed additivity: {reconstruction-tree_prob:+.9g}')

    phi_new_tree = np.zeros(row.shape[1], dtype=np.float64)
    phi_new_tree[features] = perm_vals[0]
    repaired = np.asarray(forest_phi, dtype=np.float64) + (phi_new_tree - bad_phi) / ntrees
    resid_after = float(forest_base + repaired.sum(dtype=np.float64) - prob)
    if abs(resid_after) > 1e-5:
        raise RuntimeError('Updated forest explanation still fails additivity: '
                           f'{resid_after:+.9g} (before={resid_ensemble:+.9g})')
    audit = {
        'tree_index': tree_index, 'tree_max_depth': int(tree.tree_.max_depth),
        'n_trees': ntrees, 'n_used_features': int(len(features)),
        'permutations': int(permutations), 'background_rows': int(len(bg)),
        'forest_residual_before': resid_ensemble,
        'single_tree_residual_before': resid_tree,
        'tree_baseline': tree_base, 'tree_probability': tree_prob,
        'permutation_baseline': perm_base,
        'single_tree_residual_after': float(reconstruction-tree_prob),
        'forest_residual_after': resid_after,
        'correction_l1': float(np.abs(repaired - forest_phi).sum()),
        'correction_linf': float(np.abs(repaired - forest_phi).max()),
        'method': 'single-tree permutation SHAP, independent TRAIN background',
    }
    return repaired, audit


def positive_shap(model, X: np.ndarray, background: np.ndarray, batch_size: int,
                  batch_cache: Path | None = None, *,
                  repair_tree_index: int | None = None,
                  repair_permutations: int = 8):
    """Probability SHAP on held-out TEST, with restartable batch caches.

    Every batch is checked against model.predict_proba[:, 1]. Existing chunks
    are re-validated before use. Source/config hashes are managed by prepare_run.
    """
    import shap
    if repair_permutations < 1:
        raise ValueError('repair_permutations must be at least 1')
    if repair_tree_index is not None and not (0 <= repair_tree_index < len(model.estimators_)):
        raise ValueError(f'Invalid repair_tree_index={repair_tree_index}')
    if batch_cache is not None:
        batch_cache.mkdir(parents=True, exist_ok=True)
    explainer = None
    base = None
    values = []
    max_error = 0.0
    n_batches = (len(X) + batch_size - 1) // batch_size
    for batch_no, start in enumerate(range(0, len(X), batch_size)):
        end = min(start + batch_size, len(X))
        chunk = X[start:end]
        predicted = model.predict_proba(chunk)[:, list(model.classes_).index(1)]
        cache = (batch_cache / f'batch_{batch_no:05d}.npz') if batch_cache is not None else None
        if cache is not None and cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                sv = np.asarray(z['shap'], dtype=np.float64)
                cached_prob = np.asarray(z['prob'], dtype=np.float64)
                batch_base = float(np.asarray(z['base']).reshape(-1)[0])
                if sv.shape != chunk.shape or cached_prob.shape != predicted.shape:
                    raise RuntimeError(f'Invalid cached SHAP batch shape: {cache}')
                if not np.allclose(cached_prob, predicted, atol=1e-8, rtol=0):
                    raise RuntimeError(f'Cached SHAP batch predictions mismatch: {cache}')
            LOG.info('      Reusing SHAP batch %d/%d', batch_no+1, n_batches)
        else:
            if explainer is None:
                explainer = shap.TreeExplainer(model, data=background,
                                  feature_perturbation='interventional',
                                  model_output='probability')
            expected = np.asarray(explainer.expected_value, dtype=np.float64).ravel()
            if expected.size == 2:
                batch_base = float(expected[1])
            elif expected.size == 1:
                batch_base = float(expected[0])
            else:
                raise RuntimeError('Unexpected SHAP expected-value shape')
            sv = explainer.shap_values(chunk, check_additivity=True)
            if isinstance(sv, list):
                if len(sv) != 2:
                    raise RuntimeError('Unexpected SHAP class list')
                sv = np.asarray(sv[1], dtype=np.float64)
            else:
                sv = np.asarray(sv, dtype=np.float64)
                if sv.ndim == 3 and sv.shape[2] == 2:
                    sv = sv[:, :, 1]
        if sv.shape != chunk.shape or not np.isfinite(sv).all() or not np.isfinite(batch_base):
            raise RuntimeError(f'Invalid SHAP batch shape/nonfinite values at rows {start}:{end}')
        if base is None:
            base = batch_base
        elif not np.isclose(base, batch_base, atol=1e-6, rtol=0):
            raise RuntimeError('SHAP baseline differs across chunks; caches may be incompatible')
        residuals = sv.sum(axis=1, dtype=np.float64)+batch_base-predicted
        repairs = []
        if np.any(np.abs(residuals) > 1e-4):
            bad = np.flatnonzero(np.abs(residuals) > 1e-4)
            if repair_tree_index is None:
                raise RuntimeError(f'SHAP additivity failed in batch {batch_no}; '
                                   f'bad rows within batch: {bad.tolist()}, max={max(abs(residuals)):.6g}. '
                                   'No repair requested.')
            for local_i in bad:
                repaired_phi, audit = _repair_one_anomalous_tree(
                    model, chunk[local_i], background, sv[local_i], batch_base,
                    float(predicted[local_i]), tree_index=repair_tree_index,
                    permutations=repair_permutations)
                sv[local_i] = repaired_phi
                audit.update({'batch': batch_no, 'local_row': int(local_i),
                              'test_index_0based': int(start+local_i)})
                repairs.append(audit)
                LOG.warning('Audited single-tree repair for test index %s: '
                            'residual %.6g -> %.6g, tree %s',
                            start+local_i, audit['forest_residual_before'],
                            audit['forest_residual_after'], audit['tree_index'])
            if repairs and batch_cache is not None:
                save_csv(pd.DataFrame(repairs),
                         batch_cache/f'batch_{batch_no:05d}_single_tree_repair_audit.csv')
        error = float(np.max(np.abs(sv.sum(axis=1, dtype=np.float64)+batch_base-predicted)))
        max_error = max(max_error, error)
        if error > 1e-4:
            raise RuntimeError(f'SHAP probability additivity failed in batch {batch_no}: {error:.4g} > 1e-4')
        if cache is not None and not cache.exists():
            save_npz_atomic(cache, shap=sv.astype(np.float32),
                            prob=np.asarray(predicted, dtype=np.float64),
                            base=np.array([batch_base], dtype=np.float64))
        values.append(sv.astype(np.float32))
        LOG.info('      SHAP batch %d/%d (%d/%d samples), max err %.2g',
                 batch_no+1, n_batches, end, len(X), error)
    return np.concatenate(values, axis=0), float(base), max_error


def seed_for(split: str, salt: int = 0) -> int:
    return 42 + int(split.replace('scaffold',''))*1009 + salt


def permutations(model, X, y, split, blocks, repeats, kind, baseline_auc=None):
    # The same shuffles for each feature within a split, for fairer comparison.
    rng = np.random.default_rng(seed_for(split, 103 if kind=='single' else 401))
    shuffled = [rng.permutation(len(X)) for _ in range(repeats)]
    base = (float(baseline_auc) if baseline_auc is not None else
            float(roc_auc_score(y, model.predict_proba(X)[:,list(model.classes_).index(1)])))
    rows = []
    for block, cols in blocks.items():
        for rep, perm in enumerate(shuffled):
            xx = X.copy()
            xx[:, cols] = X[perm][:,cols]
            p = model.predict_proba(xx)[:,list(model.classes_).index(1)]
            perm_auc=float(roc_auc_score(y,p))
            rows.append({'split':split, 'feature':block, 'repeat':rep,
                         'baseline_auc':base, 'permuted_auc':perm_auc,
                         'delta_auc':base-perm_auc,
                         'n_perturbed_columns':len(cols), 'type':kind})
    return pd.DataFrame(rows)


def summarize_permutation(rows: pd.DataFrame) -> pd.DataFrame:
    per_split = rows.groupby(['feature','split'],as_index=False)['delta_auc'].mean()
    summ = per_split.groupby('feature')['delta_auc'].agg(['mean','std','count']).reset_index()
    summ.columns = ['feature','mean_delta_auc','sd_across_splits','n_splits']
    return summ.sort_values('mean_delta_auc',ascending=False,kind='stable'), per_split


def read_shap(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = {'x', 'shap', 'prob', 'base'}
        if not required.issubset(z.files):
            raise ValueError(f'{path}: incomplete cached SHAP keys')
        X = np.asarray(z['x'],dtype=np.float32)
        sv = np.asarray(z['shap'],dtype=np.float32)
        prob = np.asarray(z['prob'],dtype=np.float64)
        base = float(np.asarray(z['base']).ravel()[0])
        if X.shape!=sv.shape or X.ndim!=2 or X.shape[1]!=2053:
            raise ValueError(f'{path}: invalid cached SHAP shape')
        if not (np.isfinite(X).all() and np.isfinite(sv).all() and np.isfinite(prob).all()):
            raise ValueError(f'{path}: nonfinite cached SHAP')
        err = np.max(np.abs(np.sum(sv,axis=1,dtype=np.float64)+base-prob))
        if err>1e-4:
            raise ValueError(f'{path}: cached SHAP probability reconstruction error {err}')
        return X, sv


def fit_or_load_model(split: str, Xtrain, ytrain, model_file: Path, n_jobs: int, save: bool):
    from joblib import dump, load
    if save and model_file.exists():
        model=load(model_file)
        if (model.n_features_in_ != 2053 or model.n_estimators != PARAMS['n_estimators']
                or model.min_samples_leaf != PARAMS['min_samples_leaf']
                or model.max_features != PARAMS['max_features']):
            raise RuntimeError(f'{split}: cached model incompatible with chosen parameters')
        LOG.info('   Loaded cached model %s', model_file.name)
        return model
    model = ExtraTreesClassifier(**PARAMS, n_jobs=n_jobs)
    model.fit(Xtrain, ytrain)
    if save:
        tmp = model_file.with_name(model_file.name+'.tmp')
        dump(model, tmp, compress=0)
        os.replace(tmp,model_file)
    return model


def prepare_run(args):
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'per_split').mkdir(exist_ok=True)
    source_files=[args.data_root/split_directory(s)/f'bbb_{p}.csv' for s in args.splits for p in ('train','val','test')]
    if args.scores_csv: source_files.append(args.scores_csv)
    for src in source_files:
        if not src.is_file(): raise FileNotFoundError(src)
    import shap
    manifest={'protocol':'V39_standalone_fulltest_interventional_2026-10-28',
              'root':str(args.data_root.resolve()),
              'scores_csv':str(args.scores_csv.resolve()) if args.scores_csv else None,
              'splits':args.splits, 'params':PARAMS,
              'fingerprint':{'radius':2,'bits':2048,'chirality':False},
              'shap_samples':args.shap_samples,
              'background_samples':args.shap_background_samples,
              'shap_batch_size':args.shap_batch_size,
              'top_ecfp':args.top_ecfp,
              'perm_repeats':args.perm_repeats,
              'version':{'rdkit':rdBase.rdkitVersion,'sklearn':sklearn_version,'shap':shap.__version__},
              'source_sha256':{str(z.resolve()):hash_file(z) for z in source_files}}
    cfg=args.out_dir/'run_config.json'
    if cfg.exists() and json.loads(cfg.read_text(encoding='utf-8'))!=manifest:
        raise RuntimeError(f'Output directory contains a different model/data/configuration: {cfg}. '
                           'Use a fresh --out_dir to prevent mixing caches.')
    save_json(manifest,cfg)
    return manifest


def stage_shap(args, lookup):
    for split in args.splits:
        dest=args.out_dir/'per_split'
        s_npz=dest/f'{split}_shap.npz'
        s_joint=dest/f'{split}_group_permutation.csv'
        s_metrics=dest/f'{split}_metrics.json'
        model_file=dest/f'{split}_model.joblib'
        if s_npz.exists() and s_joint.exists() and s_metrics.exists():
            read_shap(s_npz)
            LOG.info('%s: SHAP and group permutations already cached. Skip.',split)
            continue
        LOG.info('%s: load features and train/load ExtraTrees',split)
        tic=time.monotonic()
        X,y,_=matrices(args.data_root,split,lookup)
        model=fit_or_load_model(split,X['train'],y['train'],model_file,args.n_jobs,args.save_models)
        auc=float(roc_auc_score(y['test'],model.predict_proba(X['test'])[:,1]))
        ap=float(average_precision_score(y['test'],model.predict_proba(X['test'])[:,1]))
        if not s_metrics.exists():
            save_json({'split':split,'train_n':len(y['train']), 'test_n':len(y['test']),
                       'test_auc':auc,'test_ap':ap,'test_positive_rate':float(y['test'].mean())},s_metrics)
        else:
            ref=json.loads(s_metrics.read_text(encoding='utf-8'))
            if abs(ref['test_auc']-auc)>1e-8: raise RuntimeError(f'{split}: model changed')
        LOG.info('%s: AUROC=%.5f AP=%.5f',split,auc,ap)
        if not s_npz.exists():
            rg=np.random.default_rng(seed_for(split))
            idx=(np.arange(len(y['test']), dtype=int) if args.shap_samples == 0 else
                 np.sort(rg.choice(len(y['test']),size=min(args.shap_samples,len(y['test'])),replace=False)))
            bgidx=np.sort(np.random.default_rng(seed_for(split,1)).choice(
                len(y['train']),size=min(args.shap_background_samples,len(y['train'])),replace=False))
            rescue_index = (298 if args.repair_scaffold6_tree298 and split == 'scaffold6' else None)
            vals, base, err=positive_shap(model,X['test'][idx],X['train'][bgidx],args.shap_batch_size,
                                         batch_cache=dest/f'{split}_shap_batches',
                                         repair_tree_index=rescue_index,
                                         repair_permutations=args.repair_permutations)
            save_npz_atomic(s_npz,x=X['test'][idx],shap=vals,prob=model.predict_proba(X['test'][idx])[:,1],
                            base=np.array([base]),label=y['test'][idx],test_indices=idx,
                            background_indices=bgidx,additivity_error=np.array([err]))
            LOG.info('   %s: SHAP cached, max additivity error %.2g',split,err)
        if not s_joint.exists():
            group=permutations(model,X['test'],y['test'],split,BLOCKS,args.perm_repeats,'group',auc)
            save_csv(group,s_joint)
        LOG.info('%s: stage1 finished in %.1f sec',split,time.monotonic()-tic)


def select_features(args):
    foldrank=[]; all_data=[]
    for split in args.splits:
        xx,sv=read_shap(args.out_dir/'per_split'/f'{split}_shap.npz')
        foldrank.append(np.abs(sv).mean(axis=0))
        all_data.append((split,xx,sv))
    macro=np.mean(np.stack(foldrank,axis=0),axis=0)
    indices=np.argsort(-macro[:2048],kind='stable')[:args.top_ecfp].tolist()+list(range(2048,2053))
    ranks=pd.DataFrame({'feature':FEATURES,'feature_index':np.arange(2053),
                        'mean_abs_shap_macro':macro}).sort_values('mean_abs_shap_macro',ascending=False)
    save_csv(ranks,args.out_dir/'all_SHAP_features_rank.csv')
    selected=ranks[ranks.feature.isin([FEATURES[j] for j in indices])]
    save_csv(selected,args.out_dir/'selected_features.csv')
    LOG.info('Selected %d ECFP bits by across-split mean |SHAP|: %s',args.top_ecfp,
             ', '.join(FEATURES[j] for j in indices[:args.top_ecfp]))
    return indices, all_data


def stage_individual_permutation(args, lookup, indices):
    feature_blocks = {FEATURES[i]:[i] for i in indices}
    ref_names=set(feature_blocks)
    frames=[]
    for split in args.splits:
        dest=args.out_dir/'per_split'
        target=dest/f'{split}_individual_permutation.csv'
        model_file=dest/f'{split}_model.joblib'
        existing=read_csv(target) if target.exists() else pd.DataFrame()
        if not existing.empty:
            if not {'feature','repeat','split','delta_auc'}.issubset(existing.columns):
                raise ValueError(f'{target}: malformed individual permutation cache')
            existing=existing[existing.feature.isin(ref_names)].copy()
        rows=[]; done=set()
        for f in ref_names:
            sub=existing[existing.feature==f] if not existing.empty else pd.DataFrame()
            if len(sub)==args.perm_repeats and sorted(sub['repeat'].tolist())==list(range(args.perm_repeats)):
                done.add(f)
                rows.extend(sub.to_dict('records'))
        missing={k:v for k,v in feature_blocks.items() if k not in done}
        if not missing:
            LOG.info('%s: 20 individual features already cached',split)
            frames.append(pd.DataFrame(rows));continue
        LOG.info('%s: calculate %d missing individual-feature permutations',split,len(missing))
        X,y,_=matrices(args.data_root,split,lookup)
        model=fit_or_load_model(split,X['train'],y['train'],model_file,args.n_jobs,args.save_models)
        auc=float(roc_auc_score(y['test'],model.predict_proba(X['test'])[:,1]))
        ref=json.loads((dest/f'{split}_metrics.json').read_text(encoding='utf-8'))
        if abs(ref['test_auc']-auc)>1e-8:
            raise RuntimeError(f'{split}: trained/reloaded model does not reproduce stage1 Test AUC')
        for feature,columns in missing.items():
            df=permutations(model,X['test'],y['test'],split,{feature:columns},
                            args.perm_repeats,'single',auc)
            # reuse IDENTICAL random shuffle sequence for all individual features
            rows.extend(df.to_dict('records'))
            save_csv(pd.DataFrame(rows),target)
            LOG.info('     done %s',feature)
        frames.append(pd.DataFrame(rows))
    merged=pd.concat(frames,ignore_index=True)
    save_csv(merged,args.out_dir/'individual_permutation_all_repeats.csv')
    return merged


def make_plot(args, single, group, selected_indices, all_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.labelcolor':DARK,'xtick.color':DARK,'ytick.color':DARK})
    summ,by_split=summarize_permutation(single)
    group_summ,group_split=summarize_permutation(group)
    save_csv(summ,args.out_dir/'individual_permutation_summary.csv')
    save_csv(by_split,args.out_dir/'individual_permutation_by_split.csv')
    save_csv(group_summ,args.out_dir/'group_permutation_summary.csv')
    save_csv(group_split,args.out_dir/'group_permutation_by_split.csv')

    names=summ.feature.tolist()
    order=[INDEX[name] for name in names]
    xs=np.concatenate([d[1] for d in all_data],axis=0)[:,order]
    svs=np.concatenate([d[2] for d in all_data],axis=0)[:,order]
    h=max(8.4, .35*len(names)+1.65)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17.4,h),
                gridspec_kw={'width_ratios':[1.05,1.28],'wspace':.40},facecolor='white')
    yy=np.arange(len(names))
    mean=summ.mean_delta_auc.to_numpy(float)
    sd=summ.sd_across_splits.fillna(0).to_numpy(float)
    colors=[CORAL if x in EXPERTS else TEAL for x in names]
    ax1.barh(yy,mean,color=colors,height=.67,zorder=3)
    ax1.errorbar(mean,yy,xerr=sd,fmt='none',elinewidth=1.1,capsize=2.5,ecolor=DARK,zorder=4)
    ax1.set_yticks(yy, names,fontsize=10)
    ax1.set_ylim(len(names)-.5,-.5)
    ax1.axvline(0,color='#607480',lw=.9)
    ax1.grid(axis='x',color='#E5EDF0',lw=.8)
    ax1.set_axisbelow(True)
    ax1.set_xlabel('Single-feature permutation importance (ΔAUROC)',fontsize=11,weight='bold')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax1.set_title('a   Permutation importance',loc='left',fontsize=15,weight='bold',color=DARK,pad=18)
    ax1.legend(handles=[Patch(facecolor=TEAL,label='ECFP4 bit'),
                        Patch(facecolor=CORAL,label='Mechanistic expert')],
               loc='lower right',frameon=False,fontsize=9)

    cm=LinearSegmentedColormap.from_list('ME_BBB_low_high',[BLUE,'#D9E0E4','#DA5F6C'])
    rng=np.random.default_rng(2053)
    # A compact distribution plot; dots are actual held-out SHAP attributions.
    for j, name in enumerate(names):
        v=svs[:,j]
        f=xs[:,j]
        if name.startswith('ECFP4_'):
            # Exact zero/one colors for rare binary fingerprint bits.
            norm=np.clip(f,0.0,1.0)
        else:
            lo,hi=np.quantile(f,[.01,.99])
            norm=np.full(len(f),.5) if hi-lo<=1e-10 else np.clip((f-lo)/(hi-lo),0,1)
        # jitter represents density only, not SHAP value
        bins=np.digitize(v,np.linspace(float(svs.min()),float(svs.max()),70))
        jit=np.zeros(len(v))
        for b in np.unique(bins):
            ids=np.flatnonzero(bins==b)
            rng.shuffle(ids)
            offsets=np.arange(len(ids))-(len(ids)-1)/2
            jit[ids]=np.clip(offsets*.018 + rng.normal(0,.016,size=len(ids)),-.29,.29)
        ax2.scatter(v,j+jit,c=norm,cmap=cm,vmin=0,vmax=1,s=10,alpha=.65,
                    linewidths=0,rasterized=True)
    ax2.set_yticks(yy,names,fontsize=10)
    ax2.set_ylim(len(names)-.5,-.5)
    ax2.axvline(0,color='#607480',lw=.9)
    ax2.grid(axis='y',color='#E5EDF0',ls=':',lw=.75)
    ax2.set_axisbelow(True)
    ax2.set_xlabel('SHAP contribution to BBB-positive probability',fontsize=11,weight='bold')
    ax2.set_title('b   SHAP value distribution',loc='left',fontsize=15,weight='bold',color=DARK,pad=18)
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cm),ax=ax2,shrink=.74,pad=.02)
    cb.set_ticks([0,1]);cb.set_ticklabels(['Low','High'])
    cb.set_label('Feature value',fontsize=10,weight='bold')
    cb.outline.set_visible(False)
    # Publication-ready main figure: no in-figure explanatory footnote.
    fig.subplots_adjust(left=.14,right=.94,bottom=.075,top=.91)
    for ext in args.formats:
        fig.savefig(args.out_dir/f'Figure5_V39_Top{args.top_ecfp}ECFP_5Experts.{ext}',
                    dpi=args.dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)

    # Figure S: joint perturbations remain independent from single-feature ranking.
    group_order=['ECFP4 (all bits)','All five experts','Efflux-associated','Permeability-associated']
    group_summ=group_summ.set_index('feature').loc[group_order]
    fig2,ax=plt.subplots(figsize=(8.5,4.3),facecolor='white')
    gy=np.arange(len(group_order)); gm=group_summ.mean_delta_auc.to_numpy(float)
    err=group_summ.sd_across_splits.fillna(0).to_numpy(float)
    ax.barh(gy,gm,color=[TEAL,CORAL,'#B39DAF','#B39DAF'],height=.65)
    ax.errorbar(gm,gy,xerr=err,fmt='none',ecolor=DARK,capsize=2.5)
    ax.set_yticks(gy,['ECFP4 (all 2,048 bits)','All five expert scores',
                      'Efflux-associated experts','Permeability-associated experts'])
    ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.9)
    ax.grid(axis='x',color='#E5EDF0');ax.set_axisbelow(True)
    ax.set_xlabel('Joint permutation importance (ΔAUROC)',weight='bold')
    ax.set_title('Grouped permutation importance',loc='left',weight='bold',pad=14,color=DARK)
    fig2.tight_layout()
    for ext in args.formats:
        fig2.savefig(args.out_dir/f'FigureS_V39_GroupPermutation.{ext}',
                     dpi=args.dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig2)
    LOG.info('Done plotting Figure 5 and Figure S in %s',args.out_dir)


def cli():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root',type=Path,default=ROOT)
    ap.add_argument('--scores_csv',type=Path,default=None,
                    help='CSV holding five TabPFN v3.5 scores; omit if already in each split CSV')
    ap.add_argument('--out_dir',type=Path,default=OUT)
    ap.add_argument('--splits',nargs='+',default=[f'scaffold{i}' for i in range(10)])
    ap.add_argument('--shap_samples',type=int,default=0,
                    help='0 (default) explains the ENTIRE TEST set for each scaffold; positive N subsamples')
    ap.add_argument('--shap_background_samples',type=int,default=40)
    ap.add_argument('--shap_batch_size',type=int,default=10)
    ap.add_argument('--top_ecfp',type=int,default=15)
    ap.add_argument('--perm_repeats',type=int,default=10)
    ap.add_argument('--n_jobs',type=int,default=16)
    ap.add_argument('--reuse_cached_config', action='store_true',
                    help='Load original V39 SHAP/background/batch/split settings from existing run_config.json')
    ap.add_argument('--repair_scaffold6_tree298', action='store_true',
                    help='Only when audited: repair failing scaffold6 rows by replacing tree 298 SHAP '
                         'with model-agnostic permutation attributions for the same TRAIN background')
    ap.add_argument('--repair_permutations', type=int, default=8,
                    help='Number of antithetic permutation cycles for affected tree only (default 8)')
    ap.add_argument('--save_models',action=argparse.BooleanOptionalAction,default=True,
                    help='Cache each fitted model once (may require hundreds of MB across ten splits)')
    ap.add_argument('--formats',nargs='+',choices=['png','svg','pdf'],default=['png'])
    ap.add_argument('--dpi',type=int,default=320)
    ap.add_argument('--plot_only',action='store_true',
                    help='Re-export figures using existing complete SHAP and permutation caches')
    args=ap.parse_args()
    if args.repair_permutations < 1: ap.error('--repair_permutations must be positive')
    if not 1<=args.top_ecfp<=200 or args.shap_samples<0 or min(args.shap_background_samples,
                                       args.shap_batch_size,args.perm_repeats,args.n_jobs,args.dpi)<1:
        ap.error('Invalid sample sizes, top_ecfp, repeats, threads, or dpi')
    if len(args.splits)!=len(set(args.splits)) or any(not re.fullmatch(r'scaffold\d+',s) for s in args.splits):
        ap.error('Splits must be unique scaffold0, scaffold1, ...')
    return args


def main():
    args=cli()
    if args.reuse_cached_config:
        cached_path = args.out_dir / 'run_config.json'
        if not cached_path.is_file():
            raise FileNotFoundError(f'--reuse_cached_config needs {cached_path}')
        old = json.loads(cached_path.read_text(encoding='utf-8'))
        mapped = [('shap_samples','shap_samples'),
                  ('shap_background_samples','background_samples'),
                  ('shap_batch_size','shap_batch_size'),
                  ('top_ecfp','top_ecfp'),
                  ('perm_repeats','perm_repeats'),
                  ('splits','splits')]
        for dest, src in mapped:
            if src in old: setattr(args, dest, old[src])
        if old.get('root'): args.data_root = Path(old['root'])
        if old.get('scores_csv'): args.scores_csv = Path(old['scores_csv'])
        LOG.warning('Reusing V39 SHAP/background/split settings from %s', cached_path)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%H:%M:%S')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'per_split').mkdir(exist_ok=True)
    if not args.plot_only:
        prepare_run(args)
        lookup=load_score_lookup(args.scores_csv) if args.scores_csv else None
        stage_shap(args,lookup)
    else:
        lookup=None
    indices,data=select_features(args)
    if not args.plot_only:
        singles=stage_individual_permutation(args,lookup,indices)
    else:
        singles=read_csv(args.out_dir/'individual_permutation_all_repeats.csv')
    grouped=[]
    for split in args.splits:
        grouped.append(read_csv(args.out_dir/'per_split'/f'{split}_group_permutation.csv'))
    grouped=pd.concat(grouped,ignore_index=True)
    save_csv(grouped,args.out_dir/'group_permutation_all_repeats.csv')
    make_plot(args,singles,grouped,indices,data)
    LOG.info('SUCCESS | %s | %d splits | %d+5 features',args.out_dir,len(args.splits),args.top_ecfp)

if __name__=='__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ME-BBB fixed-hyperparameter ablations on ten ACHIRAL Murcko scaffold splits.

Based on ME_BBB_RF_四组特征消融_BBB原始定义_双标准审计版(2).py,
but uses the previously selected BBB best classifier:
    ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                         max_features='sqrt', class_weight='balanced',
                         random_state=42)

All features: ECFP4 (Morgan radius=2, 2048 bits, includeChirality=False)
and/or five explicitly named TabPFN v3.5 mechanistic expert probabilities.
No hyperparameter tuning or model switching is performed using Val or Test.

Default ten ablations:
 full                     ECFP4 + all five experts (the selected BBB configuration)
 ecfp_only                ECFP4 only
 experts_only             all five experts only
 ecfp_efflux              ECFP4 + BCRP, MDR1, MRP
 ecfp_permeability        ECFP4 + Caco-2, PAMPA
 drop_BCRP / drop_Caco-2 / drop_MDR1 / drop_MRP / drop_PAMPA
                         ECFP4 + remaining four expert scores

Note: 'permeability' is an operational grouping. PAMPA primarily reflects
passive diffusion; Caco-2 apparent permeability can include transporter/efflux
contributions, so do NOT call Caco-2 a strictly passive diffusion assay.

Usage:
 python ME_BBB_ExtraTrees_ECFP4_V35_10组消融.py \\
   --data_root <local-data-path> \\
   --scores_csv /path/to/BBB_TabPFN35_scores_full_data.csv \\
   --n_jobs 16

If each split CSV already contains the five v3.5 probability columns,
--scores_csv can be omitted. Only the split's TRAIN trains the BBB estimator,
VAL and TEST are scored with the same trained estimator. Scores for all
ablations share EXACTLY the same split and random_state=42.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, matthews_corrcoef, roc_auc_score,
)

LOG = logging.getLogger('ME-BBB-ExtraTrees-V35-ablation')
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / 'data' / 'splits' / 'bbb'
DEFAULT_OUT = REPO_ROOT / 'outputs' / 'ablation'
DEFAULT_SCORES = REPO_ROOT / 'data' / 'derived' / 'expert_scores' / 'bbb_tabpfn35_scores_full_data.csv'
SPLITS = tuple(f'scaffold{i}' for i in range(10))

def split_directory(label: str) -> str:
    return f"scaffold_{int(label.removeprefix('scaffold')):02d}"
ENDPOINTS = ('BCRP', 'Caco-2', 'MDR1', 'MRP', 'PAMPA')
EFFLUX = ('BCRP', 'MDR1', 'MRP')
PERMEABILITY = ('Caco-2', 'PAMPA')
SCORE_COLS = {ep: f'{ep}_score_TabPFN_V3_5' for ep in ENDPOINTS}
INTERNAL = {ep: f'__mech_{ep}' for ep in ENDPOINTS}
SEED = 42
PARAMS = dict(n_estimators=500, min_samples_leaf=3, max_features='sqrt',
              class_weight='balanced', random_state=SEED)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=False,
)

# Keep expert order identical to the previous BBB parameter search.
# (Name, include fingerprint, expert-name tuple, scientific interpretation.)
ABLATIONS = OrderedDict([
    ('full', (True, ENDPOINTS, 'ECFP4 plus all five expert scores')),
    ('ecfp_only', (True, (), 'Structural fingerprint only')),
    ('experts_only', (False, ENDPOINTS, 'Five expert scores only')),
    ('ecfp_efflux', (True, EFFLUX, 'ECFP4 plus efflux-associated BCRP/MDR1/MRP')),
    ('ecfp_permeability', (True, PERMEABILITY, 'ECFP4 plus permeability-related Caco-2/PAMPA')),
    ('drop_BCRP', (True, tuple(ep for ep in ENDPOINTS if ep != 'BCRP'), 'Leave BCRP out')),
    ('drop_Caco-2', (True, tuple(ep for ep in ENDPOINTS if ep != 'Caco-2'), 'Leave Caco-2 out')),
    ('drop_MDR1', (True, tuple(ep for ep in ENDPOINTS if ep != 'MDR1'), 'Leave MDR1 out')),
    ('drop_MRP', (True, tuple(ep for ep in ENDPOINTS if ep != 'MRP'), 'Leave MRP out')),
    ('drop_PAMPA', (True, tuple(ep for ep in ENDPOINTS if ep != 'PAMPA'), 'Leave PAMPA out')),
])
METRICS = ('auc', 'ap', 'mcc', 'acc', 'f1', 'brier')
PARTS = ('train', 'val', 'test')


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    df.to_csv(temp, index=False, encoding='utf-8-sig')
    os.replace(temp, path)


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(2 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False, dtype={'STSMILES': 'string'})
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    if not df.columns.is_unique:
        raise ValueError(f'{path}: duplicated CSV column names')
    return df


@lru_cache(maxsize=120000)
def canonical(smi: str) -> str:
    if not isinstance(smi, str) or not smi.strip():
        raise ValueError(f'Empty or invalid SMILES {smi!r}')
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        raise ValueError(f'Invalid SMILES {smi!r}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=120000)
def achiral_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f'Invalid canonical SMILES: {smi}')
    key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return key if key else 'NO_SCAFFOLD'


@lru_cache(maxsize=120000)
def ecfp4(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f'Invalid canonical SMILES: {smi}')
    bits = MORGAN.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bits, arr)
    return arr.astype(np.float32)


def check_scores(df: pd.DataFrame, where: str) -> pd.DataFrame:
    for ep in ENDPOINTS:
        col = INTERNAL[ep]
        x = pd.to_numeric(df[col], errors='coerce')
        if not x.notna().all() or not np.isfinite(x.to_numpy(dtype=float)).all() or not x.between(0, 1).all():
            raise ValueError(f'{where}: {ep} v3.5 score missing, nonnumeric or outside [0,1]')
        df[col] = x.to_numpy(dtype=np.float32)
    return df


def scores_lookup(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    required = {'STSMILES', *SCORE_COLS.values()}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: expected STSMILES and NEW v3.5 columns: {sorted(required)}')
    df['_canon'] = [canonical(str(s)) for s in df['STSMILES']]
    for ep in ENDPOINTS:
        df[INTERNAL[ep]] = df[SCORE_COLS[ep]]
    check_scores(df, str(path))
    df = df[['_canon', *(INTERNAL.values())]]
    # Identical structures with identical scores may appear multiple times in an
    # external predictions file. Contradicting scores are never silently accepted.
    if df['_canon'].duplicated().any():
        for _, group in df.groupby('_canon', sort=False):
            arr = group[list(INTERNAL.values())].to_numpy(dtype=float)
            if not np.allclose(arr, arr[0], atol=1e-7, rtol=1e-7):
                raise ValueError(f'{path}: conflicting v3.5 expert scores for duplicate canonical SMILES')
        LOG.warning('Deduplicating %d identical canonical SMILES in scores CSV',
                    len(df) - df['_canon'].nunique())
        df = df.drop_duplicates('_canon')
    return df.set_index('_canon')


def read_partition(path: Path, lookup: pd.DataFrame | None) -> pd.DataFrame:
    df = load_csv(path)
    if 'STSMILES' not in df:
        raise ValueError(f'{path}: missing STSMILES')
    if 'label' in df and 'BBB' in df:
        raise ValueError(f'{path}: both label and BBB columns exist; ambiguous')
    if 'BBB' in df:
        df = df.rename(columns={'BBB': 'label'})
    if 'label' not in df:
        raise ValueError(f'{path}: missing BBB/label')
    df['label'] = pd.to_numeric(df['label'], errors='raise')
    if df['label'].isna().any() or not df['label'].isin((0, 1)).all():
        raise ValueError(f'{path}: BBB must be nonmissing binary 0/1')
    df['label'] = df['label'].astype(int)
    df['_canon'] = [canonical(str(s)) for s in df['STSMILES']]
    if df['_canon'].duplicated().any():
        raise ValueError(f'{path}: duplicate canonical SMILES within a partition')
    if lookup is None:
        missing = [c for c in SCORE_COLS.values() if c not in df]
        if missing:
            raise ValueError(f'{path}: missing v3.5 expert columns {missing}. Pass --scores_csv; OLD *_prediction_score is not accepted.')
        for ep in ENDPOINTS:
            df[INTERNAL[ep]] = df[SCORE_COLS[ep]]
    else:
        matched = lookup.reindex(df['_canon'].tolist())
        for ep in ENDPOINTS:
            df[INTERNAL[ep]] = matched[INTERNAL[ep]].to_numpy()
    check_scores(df, str(path))
    return df


def read_split(root: Path, split: str, lookup: pd.DataFrame | None):
    frames = {p: read_partition(root / split_directory(split) / f'bbb_{p}.csv', lookup) for p in PARTS}
    audit = {'split': split}
    keys = {p: set(frames[p]['_canon']) for p in PARTS}
    scaffolds = {p: {achiral_scaffold(k) for k in keys[p]} for p in PARTS}
    for p in PARTS:
        df = frames[p]
        audit[f'{p}_n'] = int(len(df))
        audit[f'{p}_positive_rate'] = float(df['label'].mean())
        audit[f'{p}_n_scaffolds'] = len(scaffolds[p])
        if df.empty or df.label.nunique() != 2:
            raise ValueError(f'{split}/{p}: empty or single-class partition (AUROC undefined)')
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        n_keys = len(keys[a] & keys[b])
        n_scaf = len(scaffolds[a] & scaffolds[b])
        audit[f'{a}_{b}_canonical_overlap'] = n_keys
        audit[f'{a}_{b}_achiral_scaffold_overlap'] = n_scaf
        if n_keys or n_scaf:
            raise ValueError(f'{split}: {a}/{b} overlap: canonical={n_keys}, achiral Murcko={n_scaf}. '
                             'Use corrected achiral BBB splits, not the original stereochemical split.')
    # Audit truly labeled mechanism endpoints (NOT the model probability columns).
    for part in ('val', 'test'):
        df = frames[part]
        missing = set(ENDPOINTS) - set(df.columns)
        if missing:
            LOG.warning('%s/%s: cold-start audit incomplete: real endpoint-label columns missing: %s',
                        split, part, sorted(missing))
            audit[f'{part}_coldstart_check'] = 'unverifiable_missing_real_endpoint_columns'
        elif df[list(ENDPOINTS)].notna().any().any():
            raise ValueError(f'{split}/{part}: contains one or more REAL mechanism endpoint labels; cold-start violated')
        else:
            audit[f'{part}_coldstart_check'] = 'pass'
    return frames, audit


def feature_matrix(frame: pd.DataFrame, ablation: str, fp_cache: np.ndarray | None = None) -> np.ndarray:
    include_fp, included_experts, _ = ABLATIONS[ablation]
    blocks = []
    if include_fp:
        blocks.append(fp_cache if fp_cache is not None else np.stack([ecfp4(s) for s in frame['_canon']]))
    if included_experts:
        blocks.append(frame[[INTERNAL[ep] for ep in included_experts]].to_numpy(dtype=np.float32))
    if not blocks:
        raise ValueError(f'Empty features for ablation: {ablation}')
    X = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)
    expected = (2048 if include_fp else 0) + len(included_experts)
    if X.shape[1] != expected or not np.isfinite(X).all():
        raise ValueError(f'{ablation}: bad features shape={X.shape} expected={expected}')
    return X


def evaluate(y: np.ndarray, proba: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= 0.5).astype(int)
    return dict(
        auc=float(roc_auc_score(y, proba)),
        ap=float(average_precision_score(y, proba)),
        mcc=float(matthews_corrcoef(y, pred)),
        acc=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)),
        brier=float(brier_score_loss(y, proba)),
    )


def definitions_frame() -> pd.DataFrame:
    rows = []
    for name, (is_fp, eps, desc) in ABLATIONS.items():
        rows.append(dict(
            ablation=name,
            ecfp4=int(is_fp),
            included_experts=','.join(eps) if eps else 'NONE',
            excluded_experts=','.join(ep for ep in ENDPOINTS if ep not in eps) or 'NONE',
            n_experts=len(eps),
            n_features=(2048 if is_fp else 0) + len(eps),
            description=desc,
        ))
    return pd.DataFrame(rows)


def summary_outputs(rows: pd.DataFrame, dest: Path):
    if rows.empty:
        return
    keys = set(zip(rows['split'], rows['ablation']))
    # A CSV can exist after a kill; only compute paired analysis for groups
    # that are present at the same split.
    data = rows.groupby('ablation', sort=False).agg(
        n_splits=('split', 'nunique'),
        **{f'{part}_{metric}_{stat}': (f'{part}_{metric}', stat)
           for part in ('val', 'test') for metric in METRICS for stat in ('mean', 'std')},
    ).reset_index()
    # Preserve ablation presentation order.
    data['_order'] = data.ablation.map({a:i for i,a in enumerate(ABLATIONS)})
    data = data.sort_values('_order').drop(columns='_order')
    save_csv(data, dest/'ablation_summary_mean_sd.csv')
    comparisons = []
    for split in rows['split'].unique():
        for baseline in ABLATIONS:
            if baseline == 'full' or (split, baseline) not in keys or (split, 'full') not in keys:
                continue
            full = rows.loc[(rows['split'] == split) & (rows['ablation'] == 'full')].iloc[0]
            base = rows.loc[(rows['split'] == split) & (rows['ablation'] == baseline)].iloc[0]
            row = dict(split=split, comparison=f'full_minus_{baseline}', reference='full', baseline=baseline)
            for part in ('val', 'test'):
                for metric in METRICS:
                    row[f'delta_{part}_{metric}'] = float(full[f'{part}_{metric}'] - base[f'{part}_{metric}'])
            comparisons.append(row)
        for add in ('ecfp_efflux', 'ecfp_permeability'):
            if (split, add) not in keys or (split, 'ecfp_only') not in keys:
                continue
            plus = rows.loc[(rows.split == split) & (rows.ablation == add)].iloc[0]
            fp = rows.loc[(rows.split == split) & (rows.ablation == 'ecfp_only')].iloc[0]
            row = dict(split=split, comparison=f'{add}_minus_ecfp_only', reference=add, baseline='ecfp_only')
            for part in ('val', 'test'):
                for metric in METRICS:
                    row[f'delta_{part}_{metric}'] = float(plus[f'{part}_{metric}']-fp[f'{part}_{metric}'])
            comparisons.append(row)
    if comparisons:
        df = pd.DataFrame(comparisons)
        save_csv(df, dest/'ablation_paired_differences_by_split.csv')
        cols = [c for c in df if c.startswith('delta_')]
        summary = df.groupby(['comparison','reference','baseline'], sort=False).agg(
            n_splits=('split', 'nunique'),
            **{f'{c}_{stat}': (c, stat) for c in cols for stat in ('mean','std')},
        ).reset_index()
        save_csv(summary, dest/'ablation_paired_differences_summary.csv')
    LOG.info('Current ablation summary (ten splits expected):\n%s',
             data[['ablation', 'n_splits', 'test_auc_mean','test_mcc_mean','test_ap_mean']]
             .round(5).to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--save_dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--scores_csv', type=Path, default=None,
                   help='Optional external ALL-BBB TabPFN v3.5 probability CSV; supersedes any scores in split files')
    p.add_argument('--splits', nargs='+', default=list(SPLITS),
                   help='Default: scaffold0 ... scaffold9')
    p.add_argument('--ablations', nargs='+', choices=list(ABLATIONS), default=list(ABLATIONS),
                   help='Subset for quick testing; default all 10')
    p.add_argument('--n_jobs', type=int, default=16)
    p.add_argument('--save_predictions', action='store_true',
                   help='Save row-level predicted probabilities for each Val/Test partition')
    args = p.parse_args()
    if args.n_jobs < 1:
        p.error('--n_jobs must be >= 1')
    if len(set(args.splits)) != len(args.splits) or len(set(args.ablations)) != len(args.ablations):
        p.error('duplicated split/ablation argument')
    if args.scores_csv and not args.scores_csv.is_file():
        p.error(f'--scores_csv file missing: {args.scores_csv}')
    for split in args.splits:
        for part in PARTS:
            path = args.data_root / split_directory(split) / f'bbb_{part}.csv'
            if not path.is_file():
                p.error(f'Missing {path}')
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')
    output = args.save_dir
    output.mkdir(parents=True, exist_ok=True)
    source_files = [args.data_root/split_directory(s)/f'bbb_{part}.csv'
                    for s in args.splits for part in PARTS]
    if args.scores_csv:
        source_files.append(args.scores_csv)
    manifest = dict(
        protocol='ME-BBB-ECFP4-ExtraTrees-V35-10-ablation-v1',
        source_script='ME_BBB_RF_四组特征消融_BBB原始定义_双标准审计版(2).py',
        data_root=str(args.data_root.resolve()),
        scores_csv=str(args.scores_csv.resolve()) if args.scores_csv else None,
        splits=args.splits,
        ablations=args.ablations,
        random_seed=SEED,
        save_predictions=args.save_predictions,
        model='ExtraTreesClassifier',
        hyperparameters=PARAMS,
        fingerprint='Morgan radius 2 fpSize 2048 includeChirality False',
        scaffold='MurckoScaffoldSmiles(includeChirality=False)',
        features=definitions_frame().to_dict('records'),
        rdkit_version=rdBase.rdkitVersion,
        sklearn_version=sklearn.__version__,
        source_file_sha256={str(path.resolve()):file_hash(path) for path in sorted(source_files)},
    )
    cfg = output/'run_config.json'
    if cfg.is_file():
        previous = json.loads(cfg.read_text(encoding='utf-8'))
        if previous != manifest:
            raise RuntimeError(f'{output} contains an incompatible past run; use another --save_dir')
    else:
        save_json(manifest, cfg)
    save_csv(definitions_frame(), output/'ablation_definitions.csv')
    lookup = scores_lookup(args.scores_csv) if args.scores_csv else None
    results_path = output/'ablation_by_split.csv'
    if results_path.is_file():
        old = load_csv(results_path)
        if not old.empty:
            if old.duplicated(['split','ablation']).any():
                raise RuntimeError(f'{results_path}: duplicated split/ablation')
            if set(old['split']) - set(args.splits) or set(old['ablation']) - set(args.ablations):
                raise RuntimeError(f'{results_path}: unexpected split or ablation')
            rows = old.to_dict('records')
        else:
            rows = []
    else:
        rows = []
    done = {(str(r['split']), str(r['ablation'])) for r in rows}
    audits_path = output/'scaffold_audit.csv'
    audits = {}
    if audits_path.is_file():
        for r in load_csv(audits_path).to_dict('records'):
            audits[r['split']] = r
    t0 = time.monotonic()
    LOG.info('Protocol: %d splits, %d feature groups, %d planned ExtraTrees fits, seed=%d',
             len(args.splits), len(args.ablations), len(args.splits)*len(args.ablations), SEED)
    LOG.info('Fixed parameters: %s', PARAMS)
    for split in args.splits:
        frames, audit = read_split(args.data_root, split, lookup)
        audits[split] = audit
        save_csv(pd.DataFrame([audits[s] for s in args.splits if s in audits]), audits_path)
        y = {part: frames[part]['label'].to_numpy(dtype=int) for part in PARTS}
        fp = {part: np.stack([ecfp4(k) for k in frames[part]['_canon']]) for part in PARTS}
        LOG.info('%s train=%d val=%d test=%d | Test positive rate=%.3f | remaining=%d',
                 split, *(len(frames[p]) for p in PARTS), y['test'].mean(),
                 sum((split, a) not in done for a in args.ablations))
        for ablation in args.ablations:
            if (split, ablation) in done:
                continue
            X = {part: feature_matrix(frames[part], ablation, fp_cache=fp[part]) for part in PARTS}
            clf = ExtraTreesClassifier(**PARAMS, n_jobs=args.n_jobs)
            tic = time.monotonic()
            clf.fit(X['train'], y['train'])
            record = dict(
                split=split, ablation=ablation, model='extratrees', seed=SEED,
                n_features=int(X['train'].shape[1]), n_train=len(y['train']),
                n_val=len(y['val']), n_test=len(y['test']),
                fit_seconds=round(time.monotonic()-tic, 3),
            )
            for part in ('val','test'):
                proba = clf.predict_proba(X[part])[:, list(clf.classes_).index(1)]
                record.update({f'{part}_{name}':value for name,value in evaluate(y[part], proba).items()})
                if args.save_predictions:
                    pred_path = output/'predictions'/f'{split}__{ablation}__{part}.csv'
                    save_csv(pd.DataFrame({
                        'split':split, 'ablation':ablation,
                        'STSMILES':frames[part]['STSMILES'].astype(str).to_numpy(),
                        'canonical_smiles':frames[part]['_canon'].to_numpy(),
                        'label':y[part], 'probability_BBB_1':proba,
                    }), pred_path)
            rows.append(record)
            done.add((split, ablation))
            save_csv(pd.DataFrame(rows), results_path)
            LOG.info('%s / %-20s | Val AUROC=%.4f | Test AUROC=%.4f MCC=%.4f AP=%.4f (%.1fs)',
                     split, ablation, record['val_auc'],record['test_auc'],
                     record['test_mcc'],record['test_ap'],time.monotonic()-tic)
        summary_outputs(pd.DataFrame(rows), output)
    LOG.info('COMPLETE: %.2f minutes; results=%s', (time.monotonic()-t0)/60, output)
    LOG.warning('This is fixed-config repeated-scaffold Test ablation, not independent external validation. '
                'All five expert scores must come from the same TabPFN v3.5 models as the BBB grid search.')


if __name__ == '__main__':
    main()

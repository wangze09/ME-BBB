#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ME-BBB final classifier: B-only grid search on 10 scaffold splits.

Inputs: (MACCS 167-bit OR ECFP4 2048-bit) + 5 TabPFN v3.5 expert probabilities.
Models and hyperparameter grids EXACTLY match the uploaded expert B-only code:
    RF / ExtraTrees / XGBoost / SVM-RBF (8 / 8 / 8 / 6 = 30 configs/model-family sum).

Model selection: choose ONE FIXED (representation, learner, hyperparameters) by the
highest MEAN VALIDATION AUROC over scaffold0..9. Test is NEVER consulted during
search or selection. After selection, refit on ALL BBB-labeled compounds.
Optionally, --eval_test re-trains that ONE selected fixed config on each scaffold
Train set and reports exploratory held-out Test AUROC/AP/MCC/ACC/F1.

Example:
  python train_BBB_ML_scaffold_AUROC_B_only_V35.py \\
    --data_root <local-data-path> \\
    --scores_csv /path/to/BBB_TabPFN35_scores_full_data.csv \\
    --n_jobs 8 --eval_test

If the newly split BBB CSVs already contain all 5 v3.5 score columns, you can
omit --scores_csv. The default full-data CSV location is
    DATA_ROOT/BBB_labels_corrected.csv
(or provide --full_data_csv explicitly).

Note: full-data experts may produce in-sample scores on some BBB Train molecules.
These results require upstream expert/BBB overlap auditing and are not a strict
nested, independent external evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, matthews_corrcoef, f1_score)
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

LOG = logging.getLogger('ME-BBB-B-only-V3.5')
SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / 'data' / 'splits' / 'bbb'
DEFAULT_SAVE = REPO_ROOT / 'outputs' / 'bbb_model_selection'
DEFAULT_SCORES = REPO_ROOT / 'data' / 'derived' / 'expert_scores' / 'bbb_tabpfn35_scores_full_data.csv'
DEFAULT_SPLITS = tuple(f'scaffold{i}' for i in range(10))

def split_directory(label: str) -> str:
    return f"scaffold_{int(label.removeprefix('scaffold')):02d}"
REPS = ('maccs', 'ecfp')  # RDK removed intentionally; exactly one fingerprint + experts
MODELS = ('rf', 'extratrees', 'xgb', 'svm_rbf')
ENDPOINTS = ('BCRP', 'Caco-2', 'MDR1', 'MRP', 'PAMPA')
SCORE_COLS = tuple(f'{ep}_score_TabPFN_V3_5' for ep in ENDPOINTS)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)

PARAM_GRIDS = {
    'rf': {
        'n_estimators': [500, 1000],
        'min_samples_leaf': [1, 3],
        'max_features': ['sqrt', 0.2],
    },
    'extratrees': {
        'n_estimators': [500, 1000],
        'min_samples_leaf': [1, 3],
        'max_features': ['sqrt', 0.2],
    },
    'xgb': {
        'n_estimators': [200, 500],
        'max_depth': [3, 5],
        'learning_rate': [0.03, 0.1],
    },
    'svm_rbf': {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto'],
    },
}


def write_csv(df: pd.DataFrame, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, target)


def write_json(obj, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, target)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.replace('\ufeff', '', regex=False).str.strip()
    if not df.columns.is_unique:
        raise ValueError(f'{path}: duplicated column names')
    return df


@lru_cache(maxsize=120000)
def canonical(s: str) -> str:
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise ValueError(f'Invalid SMILES: {s!r}')
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=120000)
def scaffold(s: str) -> str:
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise ValueError(f'Invalid SMILES: {s!r}')
    # Same achiral grouping rule as the current BBB and expert scaffold splits.
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    return scaf if scaf else 'NO_SCAFFOLD'


@lru_cache(maxsize=120000)
def fingerprint(s: str, rep: str) -> np.ndarray:
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise ValueError(f'Invalid SMILES: {s!r}')
    if rep == 'maccs':
        bits, length = MACCSkeys.GenMACCSKeys(m), 167
    elif rep == 'ecfp':
        bits, length = MORGAN.GetFingerprint(m), 2048
    else:
        raise ValueError(f'Unknown representation: {rep}')
    arr = np.zeros(length, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bits, arr)
    return arr.astype(np.float32)


def load_scores(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    missing = set(('STSMILES', *SCORE_COLS)) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing TabPFN v3.5 score columns: {sorted(missing)}')
    df = df[['STSMILES', *SCORE_COLS]].copy()
    df['_key'] = [canonical(str(s).strip()) for s in df.STSMILES]
    if df._key.duplicated().any():
        raise ValueError('scores_csv: duplicate canonical SMILES (please de-duplicate)')
    check_scores(df, f'scores_csv {path}')
    return df[['_key', *SCORE_COLS]]


def check_scores(df: pd.DataFrame, desc: str):
    for c in SCORE_COLS:
        if c not in df:
            raise ValueError(f'{desc}: missing NEW v3.5 score {c}; legacy predictions are not accepted')
        x = pd.to_numeric(df[c], errors='coerce')
        if x.isna().any() or not np.isfinite(x.to_numpy(dtype=float)).all() or not x.between(0, 1).all():
            raise ValueError(f'{desc}: {c} contains missing/invalid/out-of-range probabilities')
        df[c] = x.astype(np.float32)


def prep_frame(path: Path, score_lookup: pd.DataFrame | None) -> pd.DataFrame:
    df = load_csv(path)
    if 'STSMILES' not in df or not ({'BBB','label'} & set(df.columns)):
        raise ValueError(f'{path}: expected STSMILES + (BBB or label)')
    if 'BBB' in df and 'label' in df:
        raise ValueError(f'{path}: both BBB and label present (ambiguous)')
    if 'BBB' in df:
        df = df.rename(columns={'BBB': 'label'})
    df['label'] = pd.to_numeric(df.label, errors='raise')
    if df.label.isna().any() or not df.label.isin((0, 1)).all():
        raise ValueError(f'{path}: BBB labels must be 0/1 with no missing')
    df['label'] = df.label.astype(int)
    df['_key'] = [canonical(str(s).strip()) for s in df.STSMILES]
    if df._key.duplicated().any():
        raise ValueError(f'{path}: duplicate canonical molecular SMILES')
    if score_lookup is not None:
        df = df.drop(columns=list(SCORE_COLS), errors='ignore').merge(
            score_lookup, on='_key', how='left', validate='one_to_one', sort=False
        )
    check_scores(df, path.name)
    return df


def split_data(root: Path, split: str, lookup: pd.DataFrame | None):
    parts = {part: prep_frame(root / split_directory(split) / f'bbb_{part}.csv', lookup)
             for part in ('train', 'val', 'test')}
    for part, df in parts.items():
        if not len(df):
            raise ValueError(f'{split}/{part}: empty split')
        if df.label.nunique() < 2:
            raise ValueError(f'{split}/{part}: single-class BBB labels; AUROC undefined')
    # BBB cold-start is critical; only enforce if actual mechanism label cols exist.
    for part in ('val', 'test'):
        df = parts[part]
        missing_labels = set(ENDPOINTS) - set(df.columns)
        if missing_labels:
            LOG.warning('%s/%s: cannot audit cold-start: missing real label columns %s',
                        split, part, sorted(missing_labels))
        elif df[list(ENDPOINTS)].notna().any().any():
            raise ValueError(f'{split}/{part}: has REAL mechanistic endpoint labels; cold-start failed')
    keys = {p: set(d['_key']) for p, d in parts.items()}
    scaffs = {p: {scaffold(s) for s in d['_key']} for p,d in parts.items()}
    audit = {'split': split, **{f'{p}_n': len(parts[p]) for p in parts}}
    for a, b in (('train','val'),('train','test'),('val','test')):
        nk = len(keys[a] & keys[b]); ns = len(scaffs[a] & scaffs[b])
        audit[f'{a}_{b}_canonical_overlap'] = nk
        audit[f'{a}_{b}_achiral_scaffold_overlap'] = ns
        if nk or ns:
            raise ValueError(f'{split} {a}/{b}: {nk} shared canonical SMILES, {ns} achiral Murcko scaffolds')
    return parts, audit


def feature_matrix(df: pd.DataFrame, rep: str) -> np.ndarray:
    xfp = np.stack([fingerprint(s, rep) for s in df['_key']])
    xmech = df[list(SCORE_COLS)].to_numpy(dtype=np.float32)
    return np.ascontiguousarray(np.concatenate((xfp, xmech), axis=1), dtype=np.float32)


def make_estimator(name: str, n_jobs: int, final: bool = False):
    # Exactly the same defaults as the supplied expert B-only code.
    if name == 'rf':
        return RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=n_jobs)
    if name == 'extratrees':
        return ExtraTreesClassifier(class_weight='balanced', random_state=SEED, n_jobs=n_jobs)
    if name == 'xgb':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError('xgboost must be installed for --models xgb') from exc
        return XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=n_jobs, verbosity=0)
    if name == 'svm_rbf':
        return SVC(kernel='rbf', class_weight='balanced', probability=final, random_state=SEED)
    raise ValueError(name)


def score_vector(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(x)[:, list(model.classes_).index(1)]
    return model.decision_function(x)


def eval_metrics(y, pred_scores, score_is_probability: bool) -> dict:
    yy = np.asarray(y, dtype=int)
    z = np.asarray(pred_scores, dtype=float)
    m = {'auc': float(roc_auc_score(yy, z)), 'ap': float(average_precision_score(yy, z))}
    predicted = (z >= (0.5 if score_is_probability else 0.0)).astype(int)
    m.update(mcc=float(matthews_corrcoef(yy, predicted)),
             acc=float(accuracy_score(yy, predicted)),
             f1=float(f1_score(yy, predicted, zero_division=0)))
    return m


def key_params(p):
    return json.dumps(p, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def config_keys(reps, models):
    return [(rep, model, key_params(p)) for rep in reps
            for model in models for p in ParameterGrid(PARAM_GRIDS[model])]


def search_split(args, split: str, lookup: pd.DataFrame | None):
    parts, audit = split_data(args.data_root, split, lookup)
    write_json(audit, args.save_dir / 'scaffold_audit' / f'{split}.json')
    out = args.save_dir / 'grid_by_split' / f'{split}.csv'
    expected = config_keys(args.reps, args.models)
    expected_set = set(expected)
    existing = {}
    if out.exists():
        old = load_csv(out)
        for r in old.to_dict('records'):
            k = (r['representation'], r['model'], r['params_json'])
            if k in existing:
                raise RuntimeError(f'Duplicate configuration in cached file {out}: {k}')
            existing[k] = r
        if set(existing) - expected_set:
            raise RuntimeError(f'Incompatible saved grid in {out}; use another --save_dir')
    if len(existing) == len(expected):
        LOG.info('SKIP %s all %d configurations already done', split, len(expected))
        return
    tr, va = parts['train'], parts['val']
    LOG.info('%s train=%d val=%d test(unused)=%d remaining=%d/%d',
             split, len(tr), len(va), len(parts['test']), len(expected)-len(existing), len(expected))
    for rep in args.reps:
        xtr, xva = feature_matrix(tr, rep), feature_matrix(va, rep)
        LOG.info('%s fingerprint=%s + 5 experts | n_features=%d', split, rep, xtr.shape[1])
        for model_name in args.models:
            for params in ParameterGrid(PARAM_GRIDS[model_name]):
                ps = key_params(params)
                key = (rep, model_name, ps)
                if key in existing:
                    continue
                tic = time.monotonic()
                clf = make_estimator(model_name, args.n_jobs)
                clf.set_params(**params)
                clf.fit(xtr, tr.label.to_numpy())
                z = score_vector(clf, xva)
                vm = eval_metrics(va.label.to_numpy(), z, score_is_probability=hasattr(clf,'predict_proba'))
                existing[key] = dict(split=split, representation=rep, model=model_name,
                                     params_json=ps, validation_auc=vm['auc'],
                                     validation_ap=vm['ap'], validation_mcc=vm['mcc'],
                                     validation_acc=vm['acc'], validation_f1=vm['f1'],
                                     fit_seconds=round(time.monotonic()-tic, 3), seed=SEED)
                write_csv(pd.DataFrame([existing[k] for k in expected if k in existing]), out)
    LOG.info('FINISHED %s, %d validation configurations', split, len(existing))


def select_config(args):
    expected_set = set(config_keys(args.reps, args.models))
    pieces = []
    for split in args.splits:
        path = args.save_dir / 'grid_by_split' / f'{split}.csv'
        if not path.is_file():
            raise ValueError(f'{split}: no validation grid {path}')
        df = load_csv(path)
        got = list(zip(df.representation, df.model, df.params_json))
        if len(got) != len(expected_set) or set(got) != expected_set or len(set(got)) != len(got):
            raise ValueError(f'{path}: incomplete or inconsistent grid; expected {len(expected_set)}')
        if set(df.split) != {split}:
            raise ValueError(f'{path}: contains unexpected split names')
        pieces.append(df)
    all_val = pd.concat(pieces, ignore_index=True)
    write_csv(all_val, args.save_dir / 'all_grid_validation_results.csv')
    ranked = (all_val.groupby(['representation','model','params_json'], sort=False, as_index=False)
              .agg(mean_val_auc=('validation_auc','mean'), sd_val_auc=('validation_auc','std'),
                   mean_val_ap=('validation_ap','mean'), mean_val_mcc=('validation_mcc','mean'),
                   n_splits=('split','nunique')))
    ranked = ranked.sort_values(['mean_val_auc','sd_val_auc','representation','model','params_json'],
                                ascending=[False,True,True,True,True], kind='mergesort').reset_index(drop=True)
    ranked.insert(0, 'rank', np.arange(1,len(ranked)+1))
    write_csv(ranked, args.save_dir/'all_config_mean_validation_ranking.csv')
    best = ranked.iloc[0]
    selection = dict(representation=str(best.representation), model=str(best.model),
                     hyperparameters=json.loads(best.params_json),
                     mean_validation_auc=float(best.mean_val_auc),
                     sd_validation_auc=float(best.sd_val_auc),
                     selection_rule=f'highest mean validation AUROC across {len(args.splits)} scaffold splits',
                     splits=list(args.splits), seed=SEED,
                     feature_order='fingerprint_bits_then_'+','.join(SCORE_COLS),
                     test_not_used_for_selection=True)
    write_json(selection, args.save_dir/'selected_config.json')
    LOG.info('BEST: %s + %s + five experts; mean Val AUC %.6f +/- %.6f; %s',
             selection['model'], selection['representation'],
             selection['mean_validation_auc'], selection['sd_validation_auc'],
             selection['hyperparameters'])
    return selection


def load_selection(args):
    path = args.save_dir/'selected_config.json'
    if not path.is_file():
        raise FileNotFoundError(f'{path} missing: finish complete search and selection first')
    selection = json.loads(path.read_text(encoding='utf-8'))
    if selection['splits'] != list(args.splits):
        raise ValueError('Stored selection uses different splits')
    if (selection['representation'] not in args.reps or selection['model'] not in args.models):
        raise ValueError('Stored selection not in current model/representation set')
    return selection


def final_fit(args, lookup):
    sel = load_selection(args)
    data_file = args.full_data_csv or (args.data_root / 'BBB_labels_corrected.csv')
    if not data_file.is_file():
        raise FileNotFoundError(f'For final full-data fit, supply --full_data_csv (not found: {data_file})')
    dest_dir = args.save_dir/'final_model'
    dest_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = dest_dir/'full_BBB_selected_config.json'
    model_path = dest_dir/'BBB_selected_full_data.joblib'
    full_sha = file_hash(data_file)
    if model_path.is_file() and cfg_path.is_file() and not args.force_final:
        old = json.loads(cfg_path.read_text(encoding='utf-8'))
        if old.get('full_data_sha256') == full_sha and old.get('selected_config') == sel:
            LOG.info('FINAL: full-data model exists and matches config/data; skip')
            return
        raise RuntimeError('Existing final model differs from current run. Use --force_final if intentional.')
    df = prep_frame(data_file, lookup)
    if df.label.nunique() < 2:
        raise ValueError('Full-data BBB labels contain only one class')
    X = feature_matrix(df, sel['representation'])
    model = make_estimator(sel['model'], args.n_jobs, final=True)
    model.set_params(**sel['hyperparameters'])
    LOG.info('FINAL FIT: %d BBB rows, %d features, %s/%s',
             len(df), X.shape[1], sel['model'], sel['representation'])
    model.fit(X, df.label.to_numpy())
    if not hasattr(model, 'predict_proba'):
        raise RuntimeError('Final model does not support predict_proba')
    bundle = dict(model=model, representation=sel['representation'], selected=sel,
                  score_columns=list(SCORE_COLS), train_n=len(df), n_features=X.shape[1],
                  train_scope='all BBB-labeled molecules', seed=SEED)
    tmp = model_path.with_suffix('.tmp.joblib')
    joblib.dump(bundle, tmp, compress=3)
    os.replace(tmp, model_path)
    write_json(dict(selected_config=sel, full_data_sha256=full_sha,
                    scores_sha256=file_hash(args.scores_csv) if args.scores_csv else None,
                    full_data_n=int(len(df)), n_features=int(X.shape[1]),
                    model_file=str(model_path), no_full_data_test_metrics=True), cfg_path)
    LOG.info('FINAL SAVED: %s', model_path)


def exploratory_test(args, lookup):
    sel = load_selection(args)
    out = args.save_dir/'selected_config_exploratory_test_by_split.csv'
    existing = load_csv(out) if out.is_file() else pd.DataFrame()
    rows = existing.to_dict('records')
    done = set(existing.split) if not existing.empty else set()
    for split in args.splits:
        if split in done:
            continue
        parts, _ = split_data(args.data_root, split, lookup)
        Xtr = feature_matrix(parts['train'], sel['representation'])
        Xte = feature_matrix(parts['test'], sel['representation'])
        clf = make_estimator(sel['model'], args.n_jobs, final=False)
        clf.set_params(**sel['hyperparameters'])
        clf.fit(Xtr, parts['train'].label.to_numpy())
        z = score_vector(clf, Xte)
        m = eval_metrics(parts['test'].label.to_numpy(), z, hasattr(clf,'predict_proba'))
        rows.append(dict(split=split, representation=sel['representation'],
                         model=sel['model'], params_json=key_params(sel['hyperparameters']),
                         n_train=len(parts['train']), n_test=len(parts['test']),
                         **{f'test_{k}': v for k,v in m.items()}))
        write_csv(pd.DataFrame(rows), out)
        LOG.info('EXPLORATORY TEST %s: AUROC %.4f MCC %.4f AP %.4f', split,
                 m['auc'],m['mcc'],m['ap'])
    df = load_csv(out)
    metrics_cols = ('test_auc','test_ap','test_mcc','test_acc','test_f1')
    summary = {'n_splits':len(df), 'representation':sel['representation'],'model':sel['model']}
    for c in metrics_cols:
        summary[f'{c}_mean'] = float(df[c].mean())
        summary[f'{c}_sd'] = float(df[c].std())
    write_csv(pd.DataFrame([summary]), args.save_dir/'selected_config_exploratory_test_summary.csv')
    LOG.warning('Test performance is exploratory: selection used all validation splits, '
                'splits may share molecules across repeats, and expert training-inclusion must be audited.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--save_dir', type=Path, default=DEFAULT_SAVE)
    ap.add_argument('--scores_csv', type=Path, default=DEFAULT_SCORES,
                    help='Optional all-BBB v3.5 scores file; overrides scores embedded in the split/full CSVs')
    ap.add_argument('--full_data_csv', type=Path, default=None,
                    help='Corrected full BBB labels. Default DATA_ROOT/BBB_labels_corrected.csv')
    ap.add_argument('--splits', nargs='+', default=list(DEFAULT_SPLITS))
    ap.add_argument('--reps', nargs='+', choices=REPS, default=list(REPS))
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--n_jobs', type=int, default=8)
    ap.add_argument('--phase', choices=('all','search','select','final','test'), default='all')
    ap.add_argument('--eval_test', action='store_true',
                    help='After selecting config, evaluate selected config on each Test split (exploratory only)')
    ap.add_argument('--force_final', action='store_true')
    args = ap.parse_args()
    if args.n_jobs < 1:
        ap.error('n_jobs must be >=1')
    if len(set(args.splits)) != len(args.splits):
        ap.error('duplicate split names')
    if len(set(args.reps)) != len(args.reps) or len(set(args.models)) != len(args.models):
        ap.error('duplicate reps/models')
    for split in args.splits:
        for part in ('train','val','test'):
            path = args.data_root/split_directory(split)/f'bbb_{part}.csv'
            if not path.is_file():
                ap.error(f'Missing split input file: {path}')
    full_path = args.full_data_csv or args.data_root/'BBB_labels_corrected.csv'
    if args.phase in ('all','final') and not full_path.is_file():
        ap.error(f'Full-data fit requires {full_path}. Pass --full_data_csv or use --phase search first.')
    source_files = [args.data_root/split_directory(s)/f'bbb_{p}.csv'
                    for s in args.splits for p in ('train','val','test')]
    if args.scores_csv:
        source_files.append(args.scores_csv)
    if full_path.is_file():
        source_files.append(full_path)
    fingerprint_manifest = {str(p.resolve()): file_hash(p) for p in sorted(source_files)}
    manifest = dict(protocol='ME-BBB-B-only-V35-achiral-v1',
                    data_root=str(args.data_root.resolve()),
                    full_data_csv=str(full_path.resolve()) if full_path.is_file() else None,
                    splits=list(args.splits), reps=list(args.reps), models=list(args.models),
                    grids={m: PARAM_GRIDS[m] for m in args.models},
                    random_seed=SEED, fingerprint_types=['MACCS167','MorganECFP4_2048'],
                    features='one fingerprint + five explicit TabPFN v3.5 probabilities',
                    expert_score_columns=list(SCORE_COLS),
                    scaffold_definition='MurckoScaffoldSmiles(includeChirality=False)',
                    rdkit_version=rdBase.rdkitVersion, sklearn_version=sklearn.__version__,
                    input_sha256=fingerprint_manifest)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')
    cfg = args.save_dir/'run_config.json'
    if cfg.exists():
        previous = json.loads(cfg.read_text(encoding='utf-8'))
        if previous != manifest:
            ap.error(f'Output {args.save_dir} belongs to a different input/config; choose a NEW --save_dir')
    else:
        write_json(manifest, cfg)
    lookup = load_scores(args.scores_csv) if args.scores_csv else None
    LOG.info('BBB B-only | splits=%d | configs/split=%d | planned fits=%d | seed=%d',
             len(args.splits), len(config_keys(args.reps, args.models)),
             len(args.splits)*len(config_keys(args.reps, args.models)), SEED)
    LOG.info('Features: fingerprint + %s', ', '.join(SCORE_COLS))
    start = time.monotonic()
    if args.phase in ('all','search'):
        for split in args.splits:
            search_split(args, split, lookup)
    if args.phase in ('all','select'):
        select_config(args)
    if args.phase in ('all','final'):
        final_fit(args, lookup)
    if args.phase == 'test' or (args.phase == 'all' and args.eval_test):
        exploratory_test(args, lookup)
    LOG.info('FINISHED in %.1f min; output=%s', (time.monotonic()-start)/60, args.save_dir)


if __name__ == '__main__':
    main()

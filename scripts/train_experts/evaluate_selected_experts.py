#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory Test diagnostics for ME-BBB B-only selected experts.

USE AFTER train_mechanistic_experts_scaffold_AUROC_B_only.py has selected a
single fixed configuration for every endpoint from five Validation AUROC scores.

For every endpoint and scaffold split:
    1. load the already chosen algorithm, fingerprint, and params;
    2. fit on the split's TRAIN only (same fit evaluates Val and Test);
    3. report Val/Test AUROC, AP, MCC, Accuracy, and F1;
    4. summarize average +/- SD over the five splits.

These five split tests are *exploratory*, not independent/untouched test results:
all five validation sets were used for the global configuration selection, and
molecules can move among groups across different scaffold splits.

Never use a full-data fitted expert for evaluation on these test partitions.
Never use exploratory Test scores to reselect the configuration.

Keep this file beside train_mechanistic_experts_scaffold_AUROC_B_only.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from tune_classical_experts import (
    ENDPOINTS,
    SPLITS,
    SEED,
    estimator,
    featurize,
    params_json,
    positive_score,
    read_groups,
    read_labeled,
    scaffold_audit,
    sha256_file,
    write_csv_atomic,
)

LOG = logging.getLogger('ME-BBB-exploratory-test')


def metrics(model, x, y):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(model.predict(x), dtype=int)
    scores = positive_score(model, x)
    if len(y) == 0:
        raise ValueError('Empty evaluation partition')
    if np.unique(y).size < 2:
        auc = float('nan')
        ap = float('nan')
        LOG.warning('Single-class evaluation partition: AUROC/AP set to NaN')
    else:
        auc = float(roc_auc_score(y, scores))
        ap = float(average_precision_score(y, scores))
    return {
        'auc': auc,
        'ap': ap,
        'mcc': float(matthews_corrcoef(y, pred)),
        'accuracy': float(accuracy_score(y, pred)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'n': len(y),
        'positive_n': int(y.sum()),
        'negative_n': int(len(y) - y.sum()),
    }


def run_one_endpoint(args, endpoint: str):
    cfg_path = args.save_dir / 'final_models' / f'{endpoint}_selected_config.json'
    if not cfg_path.is_file():
        raise FileNotFoundError(f'{cfg_path} not found; finish B-only final selection first')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    rep = cfg['representation']
    family = cfg['model']
    params = cfg['hyperparameters']
    all_data = read_labeled(args.split_root / endpoint)
    data_hash = sha256_file(args.split_root / endpoint / 'endpoint_subset.csv')
    if cfg.get('full_data_sha256') and cfg['full_data_sha256'] != data_hash:
        raise ValueError(f'{endpoint}: endpoint_subset.csv differs from data used during selection')
    detail = []
    for split in SPLITS:
        merged = read_groups(args.split_root / endpoint, split, all_data)
        # Audit uses existing file unchanged; does not block on Test overlap,
        # but blocks on Train/Validation overlap per the B-only protocol.
        audit = scaffold_audit(merged, endpoint, split)
        if audit['train_test_scaffold_overlap_info_only'] or audit['val_test_scaffold_overlap_info_only']:
            LOG.warning('%s/%s: Test has scaffold overlap with another group; NOT strict scaffold holdout',
                        endpoint, split)
        train = merged[merged.GROUP == 'train']
        val = merged[merged.GROUP == 'val']
        test = merged[merged.GROUP == 'test']
        if len(test) == 0:
            raise ValueError(f'{endpoint}/{split}: no test samples')
        model = estimator(family, args.n_jobs, final_fit=False)
        model.set_params(**params)
        model.fit(featurize(train.STSMILES, rep), train.label.to_numpy(dtype=int))
        val_res = metrics(model, featurize(val.STSMILES, rep), val.label.to_numpy(dtype=int))
        test_res = metrics(model, featurize(test.STSMILES, rep), test.label.to_numpy(dtype=int))

        # Diagnostic: same fit and Val AUROC should match the earlier grid result.
        old_path = args.save_dir / 'grid_by_split' / f'{endpoint}__{split}.csv'
        cached_auc = float('nan')
        if old_path.exists():
            old = pd.read_csv(old_path, dtype={'params_json': str})
            sel = old.loc[
                (old['representation'] == rep) &
                (old['model'] == family) &
                (old['params_json'] == params_json(params))
            ]
            if len(sel) == 1:
                cached_auc = float(sel.iloc[0]['validation_auc'])
                if not np.isclose(cached_auc, val_res['auc'], atol=1e-8, rtol=0):
                    raise RuntimeError(
                        f'{endpoint}/{split}: selected Val AUROC does not reproduce: '
                        f'grid={cached_auc:.10f} now={val_res["auc"]:.10f}'
                    )
        row = {
            'endpoint': endpoint,
            'split': split,
            'representation': rep,
            'model': family,
            'params_json': params_json(params),
            'seed': SEED,
            'cached_val_auc': cached_auc,
            **{f'val_{k}': v for k, v in val_res.items()},
            **{f'test_{k}': v for k, v in test_res.items()},
            'auc_gap_val_minus_test': val_res['auc'] - test_res['auc'],
            'train_n': len(train),
            'train_test_scaffold_overlap': audit['train_test_scaffold_overlap_info_only'],
            'val_test_scaffold_overlap': audit['val_test_scaffold_overlap_info_only'],
            'evaluation_type': 'exploratory_nonindependent_scaffold_test',
        }
        detail.append(row)
        LOG.info('%s/%s | selected %s+%s | Val AUROC %.4f | Test AUROC %.4f | Test MCC %.4f',
                 endpoint, split, rep, family, val_res['auc'], test_res['auc'], test_res['mcc'])
    return detail


def summarise(details: pd.DataFrame) -> pd.DataFrame:
    keys = ['endpoint', 'representation', 'model', 'params_json']
    cols = [
        'val_auc', 'test_auc', 'val_ap', 'test_ap', 'val_mcc', 'test_mcc',
        'val_accuracy', 'test_accuracy', 'val_f1', 'test_f1',
        'auc_gap_val_minus_test',
    ]
    summary = details.groupby(keys, dropna=False, as_index=False).agg(
        n_scaffold_splits=('split', 'nunique'),
        n_valid_test_auc=('test_auc', 'count'),
        n_test_scaffold_overlap_splits=('train_test_scaffold_overlap', lambda x: int((x > 0).sum())),
        n_val_test_scaffold_overlap_splits=('val_test_scaffold_overlap', lambda x: int((x > 0).sum())),
    )
    for col in cols:
        agg = details.groupby(keys, dropna=False)[col].agg(['mean', 'std']).reset_index()
        summary = summary.merge(agg.rename(columns={
            'mean': f'{col}_mean', 'std': f'{col}_sd'
        }), on=keys, validate='one_to_one')
    summary['evaluation_type'] = 'exploratory_nonindependent_scaffold_test'
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split_root', type=Path, default=REPO_ROOT / "data" / "splits" / "experts")
    p.add_argument('--save_dir', type=Path, default=REPO_ROOT / "outputs" / "expert_classical")
    p.add_argument('--endpoints', nargs='+', choices=ENDPOINTS, default=list(ENDPOINTS))
    p.add_argument('--n_jobs', type=int, default=8)
    args = p.parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
    if args.n_jobs < 1:
        p.error('--n_jobs must be at least 1')
    all_rows, fail = [], []
    for endpoint in args.endpoints:
        try:
            all_rows.extend(run_one_endpoint(args, endpoint))
        except Exception as exc:
            LOG.exception('FAILED %s: %s', endpoint, exc)
            fail.append((endpoint, str(exc)))
    if all_rows:
        out = args.save_dir / 'exploratory_test_diagnostics'
        out.mkdir(parents=True, exist_ok=True)
        detail_df = pd.DataFrame(all_rows)
        write_csv_atomic(detail_df, out / 'selected_config_scaffold_val_test_details.csv')
        summary_df = summarise(detail_df)
        write_csv_atomic(summary_df, out / 'selected_config_scaffold_val_test_summary.csv')
        LOG.info('Wrote exploratory Val/Test diagnostics to %s', out)
        print('\n' + summary_df[[
            'endpoint', 'model', 'representation', 'val_auc_mean',
            'test_auc_mean', 'test_auc_sd', 'test_mcc_mean', 'test_mcc_sd',
            'test_ap_mean', 'test_ap_sd'
        ]].to_string(index=False))
    if fail:
        p.exit(1, 'Failed endpoints: ' + repr(fail) + '\n')


if __name__ == '__main__':
    main()

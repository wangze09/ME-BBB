#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3 a–f: experimental BBB/endpoint labels vs TabPFN v3.5 endpoint scores.

Primary inputs (latest V3.5 setup):
  --data   data.csv                               Observed BBB + 5 endpoint labels
  --scores BBB_TabPFN35_scores_full_data.csv       Five *_score_TabPFN_V3_5 predictions
  --original score_new_TabPFN.csv                 OPTIONAL legacy QA only

Panels (a,b,c): real endpoint labels from --data, restricted to BBB-annotated rows.
Panels (d,e,f): expert predictions (default cutoff 0.5) from --scores.
SMILES matching is by stripped exact STSMILES, not row position.
The 289 extra legacy MDR1 annotations are never backfilled into data.csv.

Example:
  python figure3_TabPFN_V35_teal_coral_compatible.py \
    --data 'data(1).csv' \
    --scores 'BBB_TabPFN35_scores_full_data.csv' \
    --original 'score_new_TabPFN(2).csv' \
    --outdir Figure3_V35 --dpi 300

Output: 6 English-only panels, composite PNG/PDF, source/label audits, and stats.
Dependencies: pandas numpy matplotlib scipy rdkit
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

EPS = ['BCRP', 'MDR1', 'MRP', 'PAMPA', 'Caco-2']
PROPS = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotB', 'Rings', 'FracCSP3', 'HeavyAtoms']
EFFLUX = {'BCRP', 'MDR1', 'MRP'}
QUAD_COLS = ['EP1_BBB1', 'EP1_BBB0', 'EP0_BBB1', 'EP0_BBB0']
# Figure 3 Teal Coral palette; matches Figure 2 teal_coral scheme.
# To match journal appearance: text ink (#293A46); teal (#2B798A), coral (#CF7667).
PALETTE = {
    'ink': '#293A46', 'muted': '#5E7079', 'grid': '#E6EBED', 'light': '#F3F7F7',
    'teal': '#2B798A', 'teal_light': '#A5CDD3',
    'coral': '#CF7667', 'coral_light': '#F0BDAA',
    'connector': '#A9B6BD', 'bar_edge': '#FFFFFF',
}
QUAD_COLORS = [PALETTE['teal'], PALETTE['coral'],
               PALETTE['teal_light'], PALETTE['coral_light']]
# Negative Cliff's delta = higher descriptors for BBB=1; positive = BBB=0.
CMAP = LinearSegmentedColormap.from_list(
    'teal_white_coral', ['#277487', '#A6D0D6', '#FFFFFF', '#F1C0AD', '#CF7667'], 256
)
plt.rcParams.update({
    'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 11, 'axes.titlesize': 11.7, 'axes.labelsize': 11.5,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'axes.labelcolor': PALETTE['ink'], 'text.color': PALETTE['ink'],
    'axes.edgecolor': '#AFC1C8', 'axes.linewidth': .8,
    'xtick.color': PALETTE['ink'], 'ytick.color': PALETTE['ink'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})



def cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--data', type=Path, default=REPO_ROOT / 'data' / 'reference' / 'bbb_dataset.csv',
                   help='Authoritative experimental labels: STSMILES, BBB and five endpoint columns')
    p.add_argument('--scores', '--new', dest='scores', type=Path,
                   default=REPO_ROOT / 'data' / 'derived' / 'expert_scores' / 'bbb_tabpfn35_scores_full_data.csv',
                   help='Latest five TabPFN V3.5 expert scores; --new retained as alias')
    p.add_argument('--original', type=Path, default=None,
                   help='Optional earlier score_new_TabPFN.csv for label/score audit ONLY; never plotted')
    p.add_argument('--outdir', type=Path, default=REPO_ROOT / 'outputs' / 'figures' / 'figure3')
    p.add_argument('--thresholds', default='',
                   help='Comma-separated per-endpoint thresholds: MDR1=0.4,PAMPA=0.6 (default 0.5)')
    p.add_argument('--dpi', type=int, default=450)
    p.add_argument('--font-scale', type=float, default=1.15,
                   help='Global font scaling factor; default 1.15, try 1.25 for manuscript use')
    return p.parse_args()


def parse_thresholds(raw):
    thresholds = {ep: 0.5 for ep in EPS}
    for s in raw.split(','):
        if not s.strip():
            continue
        if '=' not in s:
            raise ValueError(f'Invalid threshold {s!r}: expected EP=value')
        key, v = map(str.strip, s.split('=', 1))
        if key not in EPS:
            raise ValueError(f'Unknown endpoint {key!r}; choices: {EPS}')
        val = float(v)
        if not np.isfinite(val) or not 0 <= val <= 1:
            raise ValueError(f'Invalid threshold for {key}: {val}')
        thresholds[key] = val
    return thresholds


def load_inputs(args, thresholds):
    # Source of truth: data.csv; NEVER merge true labels from score files into it.
    label_cols = ['BBB'] + EPS
    data = pd.read_csv(args.data, low_memory=False)
    scores = pd.read_csv(args.scores, low_memory=False)
    for df in (data, scores):
        df.columns = [str(x).replace('\ufeff', '').strip() for x in df.columns]
        if not df.columns.is_unique:
            raise ValueError('Duplicate columns after trimming/BOM cleanup')
    for name, df in [('data', data), ('primary_score', scores)]:
        if 'STSMILES' not in df:
            raise ValueError(f'{name} must contain STSMILES')
        df['STSMILES'] = df['STSMILES'].astype('string').str.strip()
        df.dropna(subset=['STSMILES'], inplace=True)
        df.drop(df.index[df['STSMILES'].eq('')], inplace=True)
        if df['STSMILES'].duplicated().any():
            examples = df.loc[df.STSMILES.duplicated(), 'STSMILES'].head(3).tolist()
            raise ValueError(f'{name} has duplicate STSMILES; examples: {examples}')
        print(f'[INPUT] {name}: {len(df)} compounds')
    missing_labels = [c for c in label_cols if c not in data]
    if missing_labels:
        raise ValueError(f'data.csv lacks required experimental labels: {missing_labels}')

    score_cols = {}
    for ep in EPS:
        candidate = [f'{ep}_score_TabPFN_V3_5', f'{ep}_prediction_score',
                     f'{ep}_score_TabPFN', f'{ep}_prediction']
        hits = [c for c in candidate if c in scores]
        if not hits:
            raise ValueError(f'Score file lacks {ep}. Expected one of: {candidate}; available: {list(scores.columns)}')
        score_cols[ep] = hits[0]

    model_version = ('TabPFN V3.5' if all(score_cols[e].endswith('_score_TabPFN_V3_5') for e in EPS)
                     else 'legacy/mixed predictor columns')
    print(f'[SOURCE] Prediction columns: {score_cols} | detected version: {model_version}')
    if model_version != 'TabPFN V3.5':
        warnings.warn('Non-V3.5 score columns selected; do NOT call this figure the V3.5 result', RuntimeWarning)

    # Only bring predicted scores and optional QA annotations from score file;
    # BBB and experimental EP labels remain data.csv columns throughout.
    qa_label_cols = [c for c in label_cols if c in scores]
    score_frame = scores[['STSMILES'] + list(dict.fromkeys(score_cols.values())) + qa_label_cols].copy()
    # Avoid potential collisions with experimental columns from data.csv.
    score_frame.rename(columns={c: f'__score_label_{c}' for c in qa_label_cols}, inplace=True)
    matched = data[['STSMILES'] + label_cols].merge(
        score_frame, on='STSMILES', how='inner', validate='one_to_one')
    n_data, n_score = len(data), len(scores)
    print(f'[MATCH] STSMILES matched={len(matched)} / data={n_data} / scores={n_score}')
    if matched.empty:
        raise ValueError('No exact STSMILES matches between data.csv and score file')
    if len(matched) != n_score:
        print(f'[WARN] {n_score-len(matched)} molecules in score file have no data.csv match')

    matched['BBB'] = pd.to_numeric(matched['BBB'], errors='coerce')
    matched = matched[matched.BBB.isin([0, 1])].copy()
    matched['BBB'] = matched.BBB.astype(int)
    if matched.empty:
        raise ValueError('No BBB experimental labels found in data.csv for matched molecules')

    label_audit = []
    audit_counts = {}
    for ep in EPS:
        matched[ep] = pd.to_numeric(matched[ep], errors='coerce')
        invalid = matched[ep].notna() & ~matched[ep].isin([0, 1])
        if invalid.any():
            raise ValueError(f'data.csv contains {int(invalid.sum())} invalid experimental {ep} labels')
        audit_col = f'__score_label_{ep}'
        if audit_col in matched:
            score_lab = pd.to_numeric(matched[audit_col], errors='coerce')
            extra = score_lab.notna() & matched[ep].isna()
            conflict = score_lab.notna() & matched[ep].notna() & score_lab.ne(matched[ep])
            missing = score_lab.isna() & matched[ep].notna()
            if extra.any() or conflict.any() or missing.any():
                for i in matched.index[extra | conflict | missing]:
                    label_audit.append({
                        'STSMILES': matched.at[i, 'STSMILES'],
                        'Endpoint': ep, 'data_label': matched.at[i, ep],
                        'score_file_label': score_lab.at[i],
                        'Issue': ('extra_in_score' if extra.at[i] else
                                  'conflicting_labels' if conflict.at[i] else 'missing_in_score')
                    })
            audit_counts[ep] = {
                'data_overlap_with_BBB': int(matched[ep].notna().sum()),
                'score_file_observed_labels': int(score_lab.notna().sum()),
                'extra_in_score_ignored': int(extra.sum()),
                'conflicts_ignored': int(conflict.sum()),
                'missing_in_score': int(missing.sum()),
            }
        else:
            audit_counts[ep] = {'data_overlap_with_BBB': int(matched[ep].notna().sum()),
                                'score_file_observed_labels': None,
                                'extra_in_score_ignored': None,
                                'conflicts_ignored': None,
                                'missing_in_score': None}
        print(f'[REAL] {ep:7s}: observed from data.csv={int(matched[ep].notna().sum())}'
              f'; score-file extra labels ignored={audit_counts[ep]["extra_in_score_ignored"]}')

        score = pd.to_numeric(matched[score_cols[ep]], errors='coerce')
        bad_score = score.notna() & (~np.isfinite(score) | ~score.between(0, 1))
        if bad_score.any():
            raise ValueError(f'Invalid {ep} scores outside [0,1], n={int(bad_score.sum())}')
        matched[f'{ep}_new_score'] = score
        matched[f'{ep}_new_pred'] = (score >= thresholds[ep]).where(score.notna(), np.nan)
        print(f'[THRESHOLD] {ep:7s}: {thresholds[ep]:.3f}; predicted score n={int(score.notna().sum())}')

    if '__score_label_BBB' in matched:
        score_bbb = pd.to_numeric(matched['__score_label_BBB'], errors='coerce')
        mismatch = score_bbb.notna() & score_bbb.ne(matched['BBB'])
        if mismatch.any():
            raise ValueError(f'BBB labels differ across data.csv and score file: {int(mismatch.sum())} rows')
    matched.drop(columns=[c for c in matched if c.startswith('__score_label_')], inplace=True)

    # Optional legacy audit. This cannot overwrite real labels or primary V3.5 scores.
    legacy_audit_rows = []
    legacy_score_comparison = []
    if args.original is not None:
        legacy = pd.read_csv(args.original, low_memory=False)
        legacy.columns = [str(c).replace('\ufeff', '').strip() for c in legacy.columns]
        if 'STSMILES' not in legacy:
            raise ValueError('--original input lacks STSMILES')
        legacy['STSMILES'] = legacy['STSMILES'].astype('string').str.strip()
        if legacy.STSMILES.duplicated().any():
            raise ValueError('--original has duplicate STSMILES')
        paired = data[['STSMILES'] + label_cols].merge(legacy, on='STSMILES', how='inner',
                                                      suffixes=('_data','_old'), validate='one_to_one')
        for ep in EPS:
            real_col = ep + '_data' if ep + '_data' in paired else ep
            old_col = ep + '_old' if ep + '_old' in paired else None
            if old_col:
                real = pd.to_numeric(paired[real_col], errors='coerce')
                earlier = pd.to_numeric(paired[old_col], errors='coerce')
                for condition, issue in [(real.isna() & earlier.notna(), 'extra_in_legacy_ignored'),
                                         (real.notna() & earlier.notna() & real.ne(earlier), 'conflicting_legacy_ignored'),
                                         (real.notna() & earlier.isna(), 'missing_in_legacy')]:
                    for smi, v_real, v_old in zip(paired.loc[condition,'STSMILES'],
                                                   real.loc[condition], earlier.loc[condition]):
                        legacy_audit_rows.append({'STSMILES':smi, 'Endpoint':ep,
                            'data_label':v_real, 'legacy_label':v_old, 'Issue':issue})
                print(f'[LEGACY AUDIT] {ep:7s}: extra={int((real.isna() & earlier.notna()).sum())}, '
                      f'conflicts={int((real.notna() & earlier.notna() & real.ne(earlier)).sum())}')
            old_score = f'{ep}_prediction_score'
            if old_score in legacy.columns:
                new_col = score_cols[ep]
                paired_scores = scores[['STSMILES',new_col]].merge(
                    legacy[['STSMILES',old_score]],on='STSMILES',how='inner',validate='one_to_one')
                v35 = pd.to_numeric(paired_scores[new_col],errors='coerce')
                historic = pd.to_numeric(paired_scores[old_score],errors='coerce')
                valid = v35.notna() & historic.notna()
                legacy_score_comparison.append({'Endpoint':ep, 'n':int(valid.sum()),
                    'mean_absolute_difference':float((v35[valid]-historic[valid]).abs().mean()),
                    'spearman_rho':float(v35[valid].corr(historic[valid],method='spearman'))})
        print(f'[AUDIT] Optional legacy file has {len(legacy)} rows; NEVER used for Figure 3 panels')

    # Molecular descriptors for the BBB-labeled molecules.
    RDLogger.DisableLog('rdApp.error')
    descs, invalid_smiles = [], []
    for smi in matched.STSMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles.append(smi)
            continue
        descs.append({
            'STSMILES': smi,
            'MW': float(Descriptors.MolWt(mol)),
            'LogP': float(Crippen.MolLogP(mol)),
            'TPSA': float(rdMolDescriptors.CalcTPSA(mol)),
            'HBD': int(Lipinski.NumHDonors(mol)),
            'HBA': int(Lipinski.NumHAcceptors(mol)),
            'RotB': int(Lipinski.NumRotatableBonds(mol)),
            'Rings': int(rdMolDescriptors.CalcNumRings(mol)),
            'FracCSP3': float(rdMolDescriptors.CalcFractionCSP3(mol)),
            'HeavyAtoms': int(mol.GetNumHeavyAtoms()),
        })
    print(f'[RDKIT] valid molecules={len(descs)}; invalid={len(invalid_smiles)}')
    if not descs:
        raise ValueError('No valid SMILES for descriptor/Cliff analysis')
    matched = matched.merge(pd.DataFrame(descs), on='STSMILES', how='inner', validate='one_to_one')
    return matched, {
        'data_source': str(args.data),
        'prediction_source': str(args.scores),
        'prediction_version_detected': model_version,
        'real_labels_authoritative': 'data.csv (provided by --data)',
        'score_file_real_labels_used_in_analysis': False,
        'data_rows': n_data, 'score_rows': n_score,
        'matched_valid_smiles': len(matched), 'invalid_smiles': len(invalid_smiles),
        'current_BBB_positive': int((matched.BBB == 1).sum()),
        'current_BBB_negative': int((matched.BBB == 0).sum()),
        'thresholds': thresholds, 'prediction_source_columns': score_cols,
        'experimental_label_audit': audit_counts,
        '_audit_rows': label_audit,
        '_legacy_audit_rows': legacy_audit_rows,
        '_legacy_score_comparison': legacy_score_comparison,
    }


def bh_qvalues(pvalues):
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), p, 1)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    sorted_q = np.minimum.accumulate((sorted_p * n / np.arange(1, n+1))[::-1])[::-1]
    q = np.zeros(n)
    q[order] = np.minimum(sorted_q, 1)
    return q


def compute_stats(matched, label_set):
    """label_set=real (a/b/c) or pred (d/e/f). Original FDR: 9 tests per EP/EPvalue."""
    counts, details = [], []
    delta = {0: pd.DataFrame(index=PROPS, columns=EPS, dtype=float),
             1: pd.DataFrame(index=PROPS, columns=EPS, dtype=float)}
    qvals = {0: pd.DataFrame(index=PROPS, columns=EPS, dtype=float),
             1: pd.DataFrame(index=PROPS, columns=EPS, dtype=float)}
    for ep in EPS:
        col = ep if label_set == 'real' else f'{ep}_new_pred'
        df = matched.loc[matched[col].isin([0, 1])].copy()
        if len(df) == 0:
            raise ValueError(f'Endpoint {ep} has no valid {label_set} labels')
        epv = df[col].astype(int)
        bbb = df.BBB.astype(int)
        a = int(((epv == 1) & (bbb == 1)).sum())
        b = int(((epv == 1) & (bbb == 0)).sum())
        c = int(((epv == 0) & (bbb == 1)).sum())
        d = int(((epv == 0) & (bbb == 0)).sum())
        n = a+b+c+d
        n1, n0 = a+b, c+d
        cons = b+c if ep in EFFLUX else a+d
        counts.append({
            'Endpoint': ep, 'Overlap_total': n, 'EP1_BBB1': a, 'EP1_BBB0': b,
            'EP0_BBB1': c, 'EP0_BBB0': d,
            'n_EP0': n0, 'n_EP1': n1, 'P_BBB1_given_EP0': c/n0 if n0 else np.nan,
            'P_BBB1_given_EP1': a/n1 if n1 else np.nan,
            'Mechanism_consistent': cons, 'Mechanism_conflict': n-cons,
            'Mechanism_consistent_ratio': cons/n if n else np.nan,
        })
        for ep_val in (1, 0):
            sub = df.loc[epv == ep_val]
            group0 = sub[sub.BBB == 0]
            group1 = sub[sub.BBB == 1]
            pv, stats = [], []
            for prop in PROPS:
                x = group0[prop].dropna().to_numpy(dtype=float)
                y = group1[prop].dropna().to_numpy(dtype=float)
                if not len(x) or not len(y):
                    p, delt = np.nan, np.nan
                else:
                    res = mannwhitneyu(x, y, alternative='two-sided')
                    p = float(res.pvalue)
                    # BBB=0 versus BBB=1 (positive => higher in BBB=0)
                    delt = 2.0 * float(res.statistic) / (len(x)*len(y)) - 1.0
                pv.append(p)
                stats.append((prop, len(x), len(y), np.median(x) if len(x) else np.nan,
                              np.median(y) if len(y) else np.nan, delt, p))
            qs = bh_qvalues(pv)
            for (prop, n_bbb0, n_bbb1, med0, med1, delt, p), q in zip(stats, qs):
                delta[ep_val].loc[prop, ep] = delt
                qvals[ep_val].loc[prop, ep] = q if np.isfinite(delt) else np.nan
                details.append({
                    'Data_type': label_set, 'Endpoint': ep, 'EP_value': ep_val,
                    'Property': prop, 'n_BBB0': n_bbb0, 'n_BBB1': n_bbb1,
                    'median_BBB0': med0, 'median_BBB1': med1,
                    'Cliffs_delta_BBB0_vs_BBB1': delt, 'MWU_p_value': p,
                    'FDR_BH_q_value': q if np.isfinite(delt) else np.nan,
                })
    return pd.DataFrame(counts), delta, qvals, pd.DataFrame(details)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    phat = k/n
    den = 1+z*z/n
    center = (phat+z*z/(2*n))/den
    half = z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    return center-half, center+half


def style_axes(ax, grid_axis='y'):
    ax.spines[['top', 'right']].set_visible(False)
    for loc in ('left', 'bottom'):
        ax.spines[loc].set_color('#AFBFC8')
        ax.spines[loc].set_linewidth(.85)
    ax.tick_params(axis='both', length=3.2, width=1.0, colors=PALETTE['ink'])
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE['grid'], linewidth=.85, zorder=0)
    ax.set_axisbelow(True)


def bar_panel(ax, counts, predicted=False):
    arr = counts[QUAD_COLS].to_numpy(dtype=float)
    total = arr.sum(axis=1)
    pc = 100.0 * arr / total[:, None]
    x = np.arange(len(EPS))
    bottoms = np.zeros(len(EPS))
    prefix = 'EP_pred' if predicted else 'EP'
    for i, col in enumerate(QUAD_COLORS):
        pval = 1 if i <= 1 else 0
        bval = 1 if i % 2 == 0 else 0
        ax.bar(x, pc[:, i], bottom=bottoms, color=col, width=.64,
               edgecolor=PALETTE['bar_edge'], linewidth=1.2, zorder=3,
               label=f'{prefix}={pval}, BBB={bval}')
        bottoms += pc[:, i]
    for j, n in enumerate(counts.Overlap_total):
        ax.text(j, 103.7, f'n = {int(n):,}', ha='center', va='bottom',
                fontsize=10.8, fontweight='bold', color=PALETTE['ink'])
    ax.set_ylim(0, 113.5)
    ax.set_xlim(-.55, len(EPS)-.45)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Percentage (%)', fontsize=12.0, fontweight='bold')
    ax.set_xticks(x, EPS)
    ax.legend(loc='lower right', fontsize=9.1, framealpha=.99,
              facecolor='white', edgecolor=PALETTE['grid'], borderpad=.45,
              handlelength=1.5, handletextpad=.45, labelspacing=.25)
    style_axes(ax)


def direction_panel(ax, counts, predicted=False):
    # Shape encodes EP class, color encodes assay family:
    # open circles = EP=0; solid circles = EP=1.
    for ix, r in counts.iterrows():
        ep = r.Endpoint
        color = PALETTE['coral'] if ep in EFFLUX else PALETTE['teal']
        x0, x1 = ix-.17, ix+.17
        n0, n1 = int(r.n_EP0), int(r.n_EP1)
        p0, p1 = r.P_BBB1_given_EP0, r.P_BBB1_given_EP1
        if np.isfinite(p0) and np.isfinite(p1):
            ax.plot([x0, x1], [p0, p1], color=PALETTE['connector'], lw=1.6, alpha=.95, zorder=1)
        for xi, prob, k, n, is_positive in [
            (x0, p0, int(r.EP0_BBB1), n0, False),
            (x1, p1, int(r.EP1_BBB1), n1, True),
        ]:
            if n:
                lo, hi = wilson(k, n)
                ax.vlines(xi, lo, hi, lw=2.1, color=color, zorder=2)
                ax.plot(xi, prob, 'o', markersize=6.8,
                        markerfacecolor=color if is_positive else 'white',
                        markeredgecolor=color, markeredgewidth=1.55, zorder=4)
        # Matplotlib MathText renders subscripts consistently across systems.
        # Avoid literal Unicode subscripts (₀, ₁), which can display as tofu/misaligned.
        sample_note = rf'$n_{{0}}$ = {n0:,}' + '\n' + rf'$n_{{1}}$ = {n1:,}'
        ax.text(ix, -.095, sample_note, ha='center', va='top',
                fontsize=10.0, color=PALETTE['ink'], linespacing=1.22,
                fontweight='bold')
    ax.set_xlim(-.52, len(EPS)-.48)
    # Slightly extend lower data space to give the two sample-count rows room.
    ax.set_ylim(-.26, 1.02)
    ax.set_xticks(range(len(EPS)), EPS)
    ax.set_yticks(np.arange(0, 1.01, .2))
    suffix = 'EP_pred' if predicted else 'EP'
    ax.set_ylabel(f'P(BBB = 1 | {suffix})', fontsize=12.0, fontweight='bold')
    style_axes(ax)
    handles = [
        Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=PALETTE['coral'],
               markeredgewidth=1.5, markersize=6.6, label='Efflux EP=0'),
        Line2D([], [], marker='o', ls='', markerfacecolor=PALETTE['coral'], markeredgecolor=PALETTE['coral'],
               markersize=6.6, label='Efflux EP=1'),
        Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=PALETTE['teal'],
               markeredgewidth=1.5, markersize=6.6, label='Permeability EP=0'),
        Line2D([], [], marker='o', ls='', markerfacecolor=PALETTE['teal'], markeredgecolor=PALETTE['teal'],
               markersize=6.6, label='Permeability EP=1'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=9.2,
              borderpad=.15, labelspacing=.2, handletextpad=.32, ncol=2, columnspacing=.9)


def heat_pair(fig, gs, delta, predicted=False):
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cbax = fig.add_subplot(gs[0, 2])
    for ax, val in [(ax0, 1), (ax1, 0)]:
        mat = delta[val][EPS].to_numpy(dtype=float)
        im = ax.imshow(mat, cmap=CMAP, vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(EPS)), EPS)
        ax.set_yticks(range(len(PROPS)))
        ax.set_yticklabels(PROPS if val == 1 else [])
        suffix = 'EP_pred' if predicted else 'EP'
        ax.set_title(f"Cliff's δ (BBB=0 vs BBB=1) | {suffix}={val}",
                     fontsize=12.0, fontweight='bold', pad=11, color=PALETTE['ink'])
        ax.tick_params(axis='both', length=0, labelsize=10.2, colors=PALETTE['ink'])
        # light white gutters between cells, easy to scan
        ax.set_xticks(np.arange(-.5, len(EPS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(PROPS), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.35)
        ax.tick_params(which='minor', bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                number = mat[i, j]
                ink = 'white' if np.isfinite(number) and abs(number) >= .55 else PALETTE['ink']
                ax.text(j, i, f'{number:+.2f}' if np.isfinite(number) else 'NA',
                        ha='center', va='center', fontsize=9.8,
                        color=ink, fontweight='bold')
    cb = fig.colorbar(im, cax=cbax)
    cb.set_ticks([-1, -.5, 0, .5, 1])
    cb.set_label("Cliff's δ (BBB=0 vs BBB=1)", fontsize=10.8, color=PALETTE['ink'], labelpad=10, fontweight='bold')
    cb.ax.tick_params(labelsize=9.8, length=2.5, colors=PALETTE['ink'])
    cb.outline.set_visible(False)
    return ax0, ax1


def save(fig, dest, dpi, font_scale=1.0):
    if font_scale != 1.0:
        import matplotlib as mpl
        for artist in fig.findobj(match=mpl.text.Text):
            artist.set_fontsize(artist.get_fontsize() * font_scale)
            try:
                fw = artist.get_fontweight()
                if fw in [None, 'normal', 'regular']:
                    artist.set_fontweight('medium')
            except Exception:
                pass
    fig.savefig(str(dest) + '.png', dpi=dpi, bbox_inches='tight', pad_inches=.13, facecolor='white')
    fig.savefig(str(dest) + '.pdf', bbox_inches='tight', pad_inches=.13, facecolor='white')
    plt.close(fig)


def set_panel_tag(ax, letter):
    ax.text(-.11, 1.055, letter, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=18.5, fontweight='bold', color=PALETTE['ink'], clip_on=False)


def plot_six_individual(outdir, counts_real, counts_pred, delta_real, delta_pred, dpi, font_scale):
    for tag, counts, kind, predicted in [
        ('a', counts_real, 'bar', False), ('b', counts_real, 'direction', False),
        ('d', counts_pred, 'bar', True), ('e', counts_pred, 'direction', True),
    ]:
        fig, ax = plt.subplots(figsize=(7.6, 5.65))
        fig.subplots_adjust(top=.91, bottom=.15, left=.14, right=.975)
        (bar_panel if kind == 'bar' else direction_panel)(ax, counts, predicted)
        set_panel_tag(ax, tag)
        save(fig, outdir / f'Figure3{tag}', dpi, font_scale)
    for tag, delta, predicted in [('c', delta_real, False), ('f', delta_pred, True)]:
        fig = plt.figure(figsize=(15.4, 4.95))
        grid = fig.add_gridspec(1, 3, left=.085, right=.955, bottom=.14, top=.86,
                                width_ratios=[1,1,.040], wspace=.22)
        ax, _ = heat_pair(fig, grid, delta, predicted)
        set_panel_tag(ax, tag)
        save(fig, outdir / f'Figure3{tag}', dpi, font_scale)


def plot_composite(outdir, counts_real, counts_pred, delta_real, delta_pred, dpi, font_scale):
    """Panel layout: a,b / c / d,e / f. Color system matches Figure 2 teal_coral."""
    fig = plt.figure(figsize=(16.0, 20.0), facecolor='white')
    outer = fig.add_gridspec(4, 2, left=.088, right=.965, top=.975, bottom=.045,
                             height_ratios=[1.12, .93, 1.12, .93],
                             hspace=.39, wspace=.30)
    for pos, counts, func, predicted, tag in [
        ((0,0), counts_real, bar_panel, False, 'a'),
        ((0,1), counts_real, direction_panel, False, 'b'),
        ((2,0), counts_pred, bar_panel, True, 'd'),
        ((2,1), counts_pred, direction_panel, True, 'e'),
    ]:
        ax = fig.add_subplot(outer[pos[0], pos[1]])
        func(ax, counts, predicted)
        set_panel_tag(ax, tag)
    for row, delta, predicted, letter in [(1, delta_real, False, 'c'), (3, delta_pred, True, 'f')]:
        g = outer[row, :].subgridspec(1, 3, width_ratios=[1, 1, .036], wspace=.24)
        ax, _ = heat_pair(fig, g, delta, predicted)
        set_panel_tag(ax, letter)
    save(fig, outdir / 'Figure3_all_panels_teal_coral', dpi, font_scale)


def main():
    args = cli()
    if args.dpi <= 0:
        raise ValueError('--dpi must be positive')
    args.outdir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    print(f'[START] output to {args.outdir.resolve()}')
    matched, qa = load_inputs(args, thresholds)
    audit_rows = qa.pop('_audit_rows', [])
    legacy_audit_rows = qa.pop('_legacy_audit_rows', [])
    legacy_comparison = qa.pop('_legacy_score_comparison', [])
    pd.DataFrame(legacy_audit_rows, columns=['STSMILES','Endpoint','data_label','legacy_label','Issue']).to_csv(
        args.outdir / 'legacy_vs_data_endpoint_label_audit.csv', index=False)
    pd.DataFrame(legacy_comparison, columns=['Endpoint','n','mean_absolute_difference','spearman_rho']).to_csv(
        args.outdir / 'legacy_vs_v35_score_comparison.csv', index=False)
    pd.DataFrame(audit_rows, columns=['STSMILES','Endpoint','data_label','score_file_label','Issue']).to_csv(
        args.outdir / 'score_vs_data_observed_label_audit.csv', index=False)
    counts_real, delta_real, q_real, detail_real = compute_stats(matched, 'real')
    counts_pred, delta_pred, q_pred, detail_pred = compute_stats(matched, 'pred')
    for typ, counts, delta, qvals in [
        ('real', counts_real, delta_real, q_real),
        ('TabPFN', counts_pred, delta_pred, q_pred),
    ]:
        counts.to_csv(args.outdir / f'quadrant_counts_{typ}.csv', index=False)
        for val in (1, 0):
            delta[val].to_csv(args.outdir / f'cliffs_delta_{typ}_EP{val}.csv')
            qvals[val].to_csv(args.outdir / f'cliffs_fdr_q_{typ}_EP{val}.csv')
    pd.concat([detail_real, detail_pred], ignore_index=True).to_csv(
        args.outdir / 'cliffs_full_statistics.csv', index=False)
    matched.to_csv(args.outdir / 'matched_labels_scores_and_rdkit.csv', index=False)
    with open(args.outdir / 'run_metadata.json', 'w', encoding='utf8') as f:
        json.dump(qa, f, indent=2, ensure_ascii=False)

    print('[COUNTS] Real EP labels:')
    print(counts_real[['Endpoint','Overlap_total']].to_string(index=False))
    print('[COUNTS] New predictions:')
    print(counts_pred[['Endpoint','Overlap_total','n_EP0','n_EP1',
                       'Mechanism_consistent_ratio']].to_string(index=False))
    plot_six_individual(args.outdir, counts_real, counts_pred, delta_real, delta_pred, args.dpi, args.font_scale)
    plot_composite(args.outdir, counts_real, counts_pred, delta_real, delta_pred, args.dpi, args.font_scale)
    print(f'[DONE] 6 individual panels + Figure 3 composite (PNG/PDF) + CSVs: {args.outdir.resolve()}')


if __name__ == '__main__':
    main()

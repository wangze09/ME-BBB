#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw ME-BBB Figure 5 from existing V39/V39.3 result caches only.

NO fitting, NO SHAP computation, NO model loading, NO permutation calculation.

Panel a: single-feature permutation ΔAUROC > 0 on average (per-split mean,
         then mean±SD across scaffold splits; the error bar may cross zero).
Panel b: actual saved probability-scale SHAP values, drawn as light violins
         plus a subsampled, density-spread beeswarm; continuous expert values
         and 0/1 ECFP4 bits use the same blue--coral low/high color meaning.

Usage:
  python plot_Figure5_V39_replot_only.py \
    --result_dir '<local-data-path>' \
    --out_dir ./Figure5_replot_v2 \
    --n_ecfp_shap 10 --max_points_per_feature 650 --dpi 400
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.patches import Patch

# Colors / typography
DARK = '#263948'
GRID = '#E4EBEE'
TEAL = '#238694'
CORAL = '#D5766A'
BLUE = '#347CA5'
LIGHT_BG = '#E5EAED'
LOW_HIGH = mcolors.LinearSegmentedColormap.from_list(
    'teal_coral_gradient', [BLUE, '#CDD9DE', '#D45D6B'], N=256)
EXPERTS = ('BCRP', 'Caco-2', 'MDR1', 'MRP', 'PAMPA')
FEATURES = [f'ECFP4_{i:04d}' for i in range(2048)] + list(EXPERTS)
INDEX = {f: i for i, f in enumerate(FEATURES)}


def init_style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':12.5,
        'axes.titlesize':17, 'axes.titleweight':'bold',
        'axes.labelsize':13, 'axes.labelweight':'bold',
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.edgecolor':'#AABAC1', 'axes.linewidth':1.0,
        'xtick.color':DARK, 'ytick.color':DARK,
        'axes.labelcolor':DARK, 'text.color':DARK,
        'savefig.facecolor':'white', 'figure.facecolor':'white',
        'pdf.fonttype':42, 'ps.fonttype':42,
        'svg.fonttype':'none',
    })


def read_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = {'x','shap','prob','base'}
        if not required.issubset(z.files):
            raise ValueError(f'{path} missing keys: {sorted(required-set(z.files))}')
        x = np.asarray(z['x'], dtype=np.float32)
        sv = np.asarray(z['shap'], dtype=np.float32)
        prob = np.asarray(z['prob'], dtype=np.float64).reshape(-1)
        base = float(np.asarray(z['base']).reshape(-1)[0])
    if x.shape != sv.shape or x.ndim != 2 or x.shape[1] != 2053:
        raise ValueError(f'{path}: unexpected matrix dimensions x={x.shape}, SHAP={sv.shape}')
    if len(prob)!=len(x) or not np.isfinite(x).all() or not np.isfinite(sv).all() or not np.isfinite(prob).all():
        raise ValueError(f'{path}: missing/non-finite values')
    residual = np.sum(sv,axis=1,dtype=np.float64)+base-prob
    max_error = float(np.max(np.abs(residual)))
    if max_error > 1e-4:
        raise ValueError(f'{path}: cached SHAP fails additivity: max error={max_error:.8g}; '
                         'Finish the audited V39.3 repair before replotting.')
    return x, sv, max_error


def load_permutation(result_dir: Path):
    candidate = result_dir / 'individual_permutation_all_repeats.csv'
    if not candidate.exists():
        raise FileNotFoundError(f'No cached single-feature permutation file: {candidate}')
    rows = pd.read_csv(candidate, encoding='utf-8-sig')
    required = {'split','feature','delta_auc'}
    if not required.issubset(rows.columns):
        raise ValueError(f'{candidate} missing columns {sorted(required-set(rows.columns))}')
    rows['delta_auc'] = pd.to_numeric(rows['delta_auc'], errors='raise')
    if not np.isfinite(rows.delta_auc).all():
        raise ValueError('Permutation contains non-finite ΔAUROC')
    if rows.duplicated(['split','feature','repeat']).any() if 'repeat' in rows.columns else False:
        raise ValueError('Duplicate split/feature/repeat in permutation cache')
    per_split = rows.groupby(['split','feature'], as_index=False).delta_auc.mean()
    summary = (per_split.groupby('feature').delta_auc
               .agg(mean_delta_auc='mean', sd_across_splits='std', n_splits='count')
               .reset_index())
    return summary, per_split


def load_shap_cache(result_dir: Path, requested_splits=None):
    files = sorted((result_dir/'per_split').glob('scaffold*_shap.npz'),
                   key=lambda p: int(p.name.split('_')[0].replace('scaffold','')))
    if requested_splits:
        keys = set(requested_splits)
        files = [p for p in files if p.name.split('_')[0] in keys]
    if not files:
        raise FileNotFoundError(f'No scaffold*_shap.npz in {result_dir}/per_split/; '
                                'A PNG image alone cannot reproduce genuine SHAP distributions.')
    xs, svs, audit, splits = [], [], [], []
    for path in files:
        x, s, err = read_npz(path)
        split = path.name.split('_')[0]
        xs.append(x);svs.append(s);splits.append(split)
        audit.append({'split':split,'n_samples':len(x),'max_additivity_error':err})
    return xs, svs, splits, pd.DataFrame(audit)


def macro_abs_shap(svs):
    # Equal weight for each scaffold split, not for each pooled test molecule.
    return np.mean(np.stack([np.abs(v).mean(axis=0) for v in svs]),axis=0)


def format_ax(ax):
    ax.grid(axis='x', color=GRID, lw=.95)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0,pad=10)
    ax.tick_params(axis='x', length=4.2, pad=8)
    for lab in ax.get_xticklabels()+ax.get_yticklabels():
        lab.set_color(DARK)
        lab.set_fontweight('bold')


def select_uniform(sample_count, n, rng):
    if n<=sample_count:
        return np.arange(n)
    return np.sort(rng.choice(n, sample_count, replace=False))


def jitter_by_density(shap_values, sampled_values, max_offset, rng):
    # Density-based vertical jitter: wider where the full distribution is denser.
    lo, hi = float(np.min(shap_values)), float(np.max(shap_values))
    if hi-lo < 1e-11:
        return rng.uniform(-.06,.06,len(sampled_values))
    counts, edges = np.histogram(shap_values, bins=55, range=(lo,hi))
    # Avoid artificial single-bin discontinuities with a small moving average.
    counts=np.convolve(counts, [0.2,0.6,0.2],mode='same')
    where=np.clip(np.digitize(sampled_values,edges)-1,0,len(counts)-1)
    max_count=max(1., float(counts.max()))
    width=max_offset * np.sqrt(np.clip(counts[where]/max_count, 0.02, 1.))
    return rng.uniform(-1,1,len(sampled_values))*width


def make_figure(result_dir: Path, out_dir: Path, top_bits: int, points: int, dpi: int,
                min_positive: float, fmt_list, suffix=''):
    init_style()
    summary, per_split = load_permutation(result_dir)
    xs, svs, splits, audit = load_shap_cache(result_dir)
    permsplits=set(per_split['split'].unique())
    if permsplits != set(splits):
        raise ValueError('The permutation and SHAP caches use DIFFERENT scaffold sets: '
                         f'permutation={sorted(permsplits)}, SHAP={sorted(splits)}. '
                         'For strict comparisons, use the same completed splits.')
    if len(splits)<2:
        raise ValueError('At least two completed splits needed for across-split SD.')
    if not (summary.n_splits == len(splits)).all():
        raise ValueError('Permutation file is incomplete: not all features have all split entries.')
    macro=macro_abs_shap(svs)
    available=set(summary.feature)
    candidates=[f for f in available if f in INDEX]
    positive = (summary.loc[summary.feature.isin(candidates) & (summary.mean_delta_auc>min_positive)]
                .sort_values('mean_delta_auc',ascending=False).copy())
    if positive.empty:
        raise ValueError('No positively associated permutation features: adjust --min_positive')
    # Show the informative ECFP4 bits, keep *all 5* mechanistic experts.
    eligible_bits = [f for f in candidates if f.startswith('ECFP4_')]
    eligible_bits.sort(key=lambda f: -macro[INDEX[f]])
    selected_b = sorted([e for e in EXPERTS if e in available],
                        key=lambda x:-macro[INDEX[x]]) + eligible_bits[:top_bits]
    if len([e for e in EXPERTS if e in available]) != 5:
        raise ValueError('Expected five mechanistic experts in the original 20-feature permutation cache.')
    if not selected_b:
        raise ValueError('No features chosen for panel b.')
    # Preserve scientific outputs even when figure is rearranged.
    audit.to_csv(out_dir/'Figure5_SHAP_cache_additivity_audit.csv',index=False)
    positive.to_csv(out_dir/'Figure5_positive_permutation_features.csv',index=False)
    pd.DataFrame({'feature':selected_b,'mean_abs_shap_macro':
                  [macro[INDEX[f]] for f in selected_b]}).to_csv(
                      out_dir/'Figure5_displayed_SHAP_feature_rank.csv',index=False)

    fig = plt.figure(figsize=(18.6, 9.8), facecolor='white')
    gs=fig.add_gridspec(1,2,width_ratios=[0.92,1.40],
                       left=.15,right=.93,top=.90,bottom=.11,wspace=.46)
    axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1])
    # -------- panel a: only positive *mean* ΔAUROC --------
    ys=np.arange(len(positive))
    means=positive.mean_delta_auc.to_numpy(dtype=float)
    sds=positive.sd_across_splits.fillna(0).to_numpy(dtype=float)
    names=positive.feature.astype(str).tolist()
    colors=[CORAL if f in EXPERTS else TEAL for f in names]
    axa.barh(ys,means,height=.60,color=colors,alpha=.98,zorder=3)
    axa.errorbar(means,ys,xerr=sds,fmt='none',ecolor=DARK,elinewidth=1.2,
                 capsize=3.0,zorder=4)
    left=min(-.0004, float(np.min(means-sds))*1.06)
    right=max(.0008, float(np.max(means+sds))*1.08)
    axa.set_xlim(left,right)
    axa.set_yticks(ys,names)
    axa.set_ylim(-.65,len(names)-.35)
    axa.invert_yaxis()
    axa.axvline(0,color='#617886',lw=1.1,zorder=2)
    axa.set_xlabel('Permutation importance (ΔAUROC)',labelpad=10)
    axa.set_title('a   Positive permutation importance',loc='left',pad=18,
                  color=DARK,fontsize=16.0,fontweight='bold')
    axa.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axa.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axa.legend(handles=[Patch(facecolor=CORAL,label='Mechanistic expert'),
                        Patch(facecolor=TEAL,label='ECFP4 bit')],
               loc='lower right',frameon=False,fontsize=10.8)
    format_ax(axa)

    # -------- panel b: low-clutter density violin + genuine SHAP points --------
    nrows=len(selected_b)
    b_indices=[INDEX[f] for f in selected_b]
    x_values=np.concatenate([a[:,b_indices] for a in xs],axis=0)
    shap_values=np.concatenate([a[:,b_indices] for a in svs],axis=0)
    rng=np.random.default_rng(20261030)
    all_swarm=[]
    for j,feat in enumerate(selected_b):
        v=shap_values[:,j].astype(float)
        f=x_values[:,j].astype(float)
        # Violin density uses all available SHAP values, not just plotted subsample.
        if np.std(v)>1e-9:
            parts=axb.violinplot([v],positions=[j],vert=False,widths=.66,
                                 showmeans=False,showmedians=False,showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor('#E6EEF0')
                body.set_edgecolor('none')
                body.set_alpha(.92)
                body.set_zorder(1)
        idx=select_uniform(points,len(v),rng)
        z=v[idx]
        if feat.startswith('ECFP4_'):
            norm=np.clip(f[idx],0,1)
        else:
            low,high=np.quantile(f,[.02,.98])
            norm=np.full(len(idx),.5) if high-low<1e-9 else np.clip((f[idx]-low)/(high-low),0,1)
        jit=jitter_by_density(v,z,.31,rng)
        axb.scatter(z,j+jit,c=norm,cmap=LOW_HIGH,vmin=0,vmax=1,
                    s=11.5,alpha=.82,linewidths=0,rasterized=True,zorder=3)
        all_swarm.append(z)
    minx=float(np.min(shap_values));maxx=float(np.max(shap_values))
    pad=max((maxx-minx)*.05,.002)
    axb.set_xlim(minx-pad,maxx+pad)
    axb.set_yticks(np.arange(nrows),selected_b)
    axb.set_ylim(-.65,nrows-.35)
    axb.invert_yaxis()
    axb.axvline(0,lw=1.2,color='#627985',zorder=2)
    axb.grid(axis='y',ls=':',color='#E2E8EB',lw=.9)
    axb.set_xlabel('SHAP contribution to P(BBB+)',labelpad=10)
    axb.set_title('b   SHAP value distributions',loc='left',pad=18,
                  color=DARK,fontsize=16.0,fontweight='bold')
    axb.xaxis.set_major_locator(MaxNLocator(nbins=4))
    format_ax(axb)
    # Semantic colorbar: align exactly with the vertical extent of panel b.
    # Expert columns are quantile-normalized, whereas ECFP4 bits remain actual 0/1.
    pos = axb.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.016, pos.height])
    norm=mcolors.Normalize(vmin=0,vmax=1)
    cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=LOW_HIGH),cax=cax)
    cb.set_ticks([0,1]);cb.set_ticklabels(['Low','High'])
    cb.set_label('Feature value',weight='bold',labelpad=9,fontsize=12)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0,labelsize=11,labelcolor=DARK)
    for tick in cb.ax.get_yticklabels():tick.set_fontweight('bold')

    for ext in fmt_list:
        fig.savefig(out_dir/f'Figure5_V39_postprocessed{suffix}.{ext}',dpi=dpi,
                    bbox_inches='tight',pad_inches=.13,facecolor='white')
    plt.close(fig)
    print(f'[OK] saved Figure5_V39_postprocessed{suffix} in {out_dir}')
    print(f'[OK] {len(positive)} features have positive mean permutation ΔAUROC; '
          f'SHAP panel shows {len(selected_b)} features, {sum(len(x) for x in xs)} real sample rows.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result_dir',type=Path,
                    default=REPO_ROOT/'data'/'derived'/'figure5'/'feature_attribution')
    ap.add_argument('--out_dir',type=Path,default=None)
    ap.add_argument('--n_ecfp_shap',type=int,default=10,
                    help='Number of ECFP4 features in SHAP panel (out of cached ones); default 10')
    ap.add_argument('--max_points_per_feature',type=int,default=650,
                    help='Downsampling affects visualization ONLY; not SHAP values or ranking')
    ap.add_argument('--min_positive',type=float,default=0.0,
                    help='Panel a retains features whose mean ΔAUROC is > this value')
    ap.add_argument('--dpi',type=int,default=400)
    ap.add_argument('--formats',nargs='+',choices=['png','svg','pdf'],default=['png','svg','pdf'])
    args=ap.parse_args()
    if args.n_ecfp_shap<1 or args.max_points_per_feature<10 or args.dpi<72:
        ap.error('Invalid --n_ecfp_shap, --max_points_per_feature, or --dpi')
    output=args.out_dir or (REPO_ROOT/'outputs'/'figures'/'figure5')
    output.mkdir(parents=True,exist_ok=True)
    make_figure(args.result_dir,output,args.n_ecfp_shap,args.max_points_per_feature,
                args.dpi,args.min_positive,args.formats)

if __name__=='__main__':main()

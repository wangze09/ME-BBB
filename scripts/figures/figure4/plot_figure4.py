#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ME-BBB Figure 4: manuscript-ready, paired scaffold ablation figure.

Usage:
    python plot_Figure4_V35_publication.py \
      --input 'ablation_by_split.csv' --outdir 'Figure4_publication'

Figure 4 (main):
  a: Full ME-BBB minus ECFP4-only test metrics for ten scaffold splits.
  b: ECFP4 + efflux/permeability/all-five vs ECFP4-only, same splits.
Figure S (supplement): Full ME-BBB minus each single-expert-removal model.

Dots are individual paired split differences. Diamonds are arithmetic means;
whiskers are ±1 sample SD, not confidence intervals or significance tests.
"""
from __future__ import annotations
import argparse
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter

# Consistent teal–coral palette, high contrast type
INK = '#243544'
MUTED = '#5B6C78'
GRID = '#E2E9EC'
ZERO = '#667682'
TEAL = '#247B88'
TEAL_DOT = '#70AEB6'
SOFT_TEAL = '#6F9EA9'
CORAL = '#D36C59'
SOFT_CORAL = '#E2A08F'

METRICS = ['auc','mcc','ap']
COL = {'auc':'AUROC', 'mcc':'MCC', 'ap':'PRAUC'}
AXIS = {
  'auc': {'xlim':(-0.0054,0.0171),'ticks':[-0.005,0,0.005,0.010,0.015],'fmt':'%.3f'},
  'mcc': {'xlim':(-0.050,0.085),'ticks':[-0.04,0,0.04,0.08],'fmt':'%.2f'},
  'ap': {'xlim':(-0.0036,0.0111),'ticks':[-0.002,0,0.004,0.008],'fmt':'%.3f'},
}
SI_AXIS = {
  'auc': {'xlim':(-0.0054,0.0062),'ticks':[-0.004,0,0.004],'fmt':'%.3f'},
  'mcc': {'xlim':(-0.055,0.081),'ticks':[-0.05,0,0.05],'fmt':'%.2f'},
  'ap': {'xlim':(-0.0043,0.0042),'ticks':[-0.004,-0.002,0,0.002,0.004],'fmt':'%.3f'},
}
GROUPS = [
  ('ecfp_efflux', 'Efflux-associated', SOFT_TEAL),
  ('ecfp_permeability','Permeability-related',CORAL),
  ('full','All five experts',TEAL),
]
DROPS = ['BCRP','MDR1','MRP','Caco-2','PAMPA']


def theme():
    plt.rcParams.update({
        'font.family':'DejaVu Sans',
        'font.size':12,
        'font.weight':'medium',
        'axes.titlesize':13,
        'axes.titleweight':'bold',
        'axes.labelsize':12,
        'axes.labelweight':'bold',
        'xtick.labelsize':10.5,
        'ytick.labelsize':11.5,
        'axes.edgecolor':'#C4D0D5',
        'xtick.color':INK,
        'ytick.color':INK,
        'text.color':INK,
        'axes.labelcolor':INK,
        'figure.facecolor':'white',
        'savefig.facecolor':'white',
        'pdf.fonttype':42,
        'svg.fonttype':'none',
        'axes.linewidth':0.9,
    })


def validate_read(path):
    df=pd.read_csv(path,encoding='utf-8-sig')
    expected={'full','ecfp_only','experts_only','ecfp_efflux','ecfp_permeability',
              'drop_BCRP','drop_MDR1','drop_MRP','drop_Caco-2','drop_PAMPA'}
    needed={'split','ablation','seed','model','n_test',
            'test_auc','test_mcc','test_ap'}
    if needed-set(df.columns):
        raise ValueError(f'Missing columns: {sorted(needed-set(df.columns))}')
    if df.duplicated(['split','ablation']).any():
        raise ValueError('Repeated (split,ablation) pairs in input')
    splits=sorted(df.split.unique(),key=lambda s:int(s.replace('scaffold','')))
    if len(splits)!=10 or len(df)!=100:
        raise ValueError(f'Expected ten scaffold splits × 10 feature configurations, got {len(splits)} splits and {len(df)} rows')
    for split, block in df.groupby('split'):
        if set(block.ablation)!=expected or len(block)!=10:
            raise ValueError(f'{split}: ablations incomplete')
        for col in ('model','seed','n_test'):
            if block[col].nunique()!=1:
                raise ValueError(f'{split}: {col} must match across ablations')
    if not np.isfinite(df[[f'test_{x}' for x in METRICS]].to_numpy(float)).all():
        raise ValueError('Non-finite test scores')
    return df


def deltas(df):
    p={m:df.pivot(index='split',columns='ablation',values=f'test_{m}').sort_index() for m in METRICS}
    return p


def axes_format(ax, metric, limits=AXIS):
    c=limits[metric]
    ax.set_xlim(*c['xlim'])
    ax.xaxis.set_major_locator(FixedLocator(c['ticks']))
    ax.xaxis.set_major_formatter(FormatStrFormatter(c['fmt']))
    ax.set_axisbelow(True)
    ax.grid(axis='x',color=GRID,lw=0.9,zorder=0)
    ax.axvline(0,color=ZERO,lw=1.35,zorder=1)
    for side in ('left','top','right'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#A9BAC2')
    ax.tick_params(axis='y',left=False,length=0,pad=10)
    ax.tick_params(axis='x',length=4,width=0.8,color='#90A4AE',pad=7)
    for t in ax.get_xticklabels():t.set_fontweight('bold')
    for t in ax.get_yticklabels():t.set_fontweight('bold')


def dotwhisker(ax, values, y, color, seed):
    vals=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    ys=y+rng.uniform(-0.115,0.115,len(vals))
    dotcolor=TEAL_DOT if color in (TEAL,SOFT_TEAL) else SOFT_CORAL
    ax.scatter(vals,ys,s=33,c=dotcolor,alpha=.8,edgecolors='white',linewidth=.48,zorder=3)
    avg=vals.mean(); sd=vals.std(ddof=1)
    ax.errorbar([avg],[y],xerr=[sd],fmt='D',color=color,ecolor=color,
                capsize=5,markersize=7,markeredgecolor='white',markeredgewidth=1,
                elinewidth=2.35,zorder=7)
    return avg,sd,(vals>1e-12).sum(),len(vals)


def write_files(fig,path,dpi):
    for ext in ('png','pdf','svg'):
        fig.savefig(path.with_suffix('.'+ext),dpi=dpi,
                    bbox_inches='tight',pad_inches=0.13,facecolor='white')
    plt.close(fig)


def make_main(p,outdir,dpi):
    # A fixed-size 3-column grid with independent axes; deliberately generous
    # horizontal spacing prevents tick-label collisions in adjacent plots.
    fig=plt.figure(figsize=(18.4,9.4))
    lefts=(.155,.455,.755)
    width=.206
    y_a=.652; h_a=.188
    y_b=.160; h_b=.292

    fig.text(.044,.951,'a',fontsize=21,fontweight='bold',ha='left',va='top')
    fig.text(.076,.951,'Gain from adding all five mechanistic experts',
             fontsize=16.5,fontweight='bold',va='top')
    fig.text(.044,.540,'b',fontsize=21,fontweight='bold',ha='left',va='top')
    fig.text(.076,.540,'Contribution of functional expert groups',
             fontsize=16.5,fontweight='bold',va='top')

    for ci,m in enumerate(METRICS):
        ax1=fig.add_axes([lefts[ci],y_a,width,h_a])
        a=p[m]['full']-p[m]['ecfp_only']
        ax1.set_ylim(-.5,.5)
        dotwhisker(ax1,a,0,TEAL,101+ci)
        ax1.set_yticks([0],['All five experts' if ci==0 else ''])
        axes_format(ax1,m)
        ax1.set_title('Test '+COL[m],fontweight='bold',pad=17,color=INK)
        ax1.set_xlabel('Δ'+COL[m],labelpad=11)


        ax2=fig.add_axes([lefts[ci],y_b,width,h_b])
        for ri,(ab,label,c) in enumerate(GROUPS):
            d=p[m][ab]-p[m]['ecfp_only']
            dotwhisker(ax2,d,ri,c,201+ci*17+ri)
        ax2.set_ylim(-.43,2.43)
        ax2.invert_yaxis()
        ax2.set_yticks(range(3),[z[1] for z in GROUPS] if ci==0 else ['','',''])
        axes_format(ax2,m)
        ax2.set_title('Test '+COL[m],fontweight='bold',pad=17,color=INK)
        ax2.set_xlabel('Δ'+COL[m],labelpad=11)

    write_files(fig,outdir/'Figure4_V35_publication',dpi)


def make_si(p,outdir,dpi):
    fig=plt.figure(figsize=(18.2,6.2))
    fig.text(.048,.935,'Single-expert removal',fontweight='bold',fontsize=17,va='top')
    lefts=(.153,.450,.747); width=.218
    for ci,m in enumerate(METRICS):
        ax=fig.add_axes([lefts[ci],.19,width,.59])
        for ri,ep in enumerate(DROPS):
            d=p[m]['full']-p[m]['drop_'+ep]
            dotwhisker(ax,d,ri,CORAL,402+ri*17+ci)
        ax.set_ylim(-.47,4.47);ax.invert_yaxis()
        ax.set_yticks(range(5),DROPS if ci==0 else ['']*5)
        axes_format(ax,m,limits=SI_AXIS)
        ax.set_title('Test '+COL[m],pad=18)
        ax.set_xlabel('Δ'+COL[m]+' (Full − without expert)',labelpad=11)
    write_files(fig,outdir/'FigureS_V35_single_expert_removal',dpi)


def make_stats(p,outdir):
    rows=[]
    for m in METRICS:
        for ab in ['full','ecfp_efflux','ecfp_permeability']:
            v=p[m][ab]-p[m]['ecfp_only']
            rows.append(dict(panel='main',metric=COL[m],contrast=ab+'_minus_ecfp_only',
                             mean=v.mean(),sd=v.std(ddof=1),wins=(v>1e-12).sum(),n=len(v)))
        for ep in DROPS:
            v=p[m]['full']-p[m]['drop_'+ep]
            rows.append(dict(panel='supplement',metric=COL[m],contrast='full_minus_drop_'+ep,
                             mean=v.mean(),sd=v.std(ddof=1),wins=(v>1e-12).sum(),n=len(v)))
    pd.DataFrame(rows).to_csv(outdir/'Figure4_paired_statistics.csv',index=False)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--input',type=Path,default=REPO_ROOT/'data'/'derived'/'ablation'/'ablation_by_split.csv')
    a.add_argument('--outdir',type=Path,default=REPO_ROOT/'outputs'/'figures'/'figure4')
    a.add_argument('--dpi',type=int,default=400)
    args=a.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    theme()
    p=deltas(validate_read(args.input))
    make_main(p,args.outdir,args.dpi)
    make_si(p,args.outdir,args.dpi)
    make_stats(p,args.outdir)
    print('Created Figure 4 and SI Figure in',args.outdir)

if __name__=='__main__': main()

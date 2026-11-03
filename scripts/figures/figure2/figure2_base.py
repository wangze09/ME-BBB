#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ME-BBB Figure 2, completely regenerated from the original archive's data.csv.

Replaces the original scripts fig1a, fig1b, fig1c, fig1d and incorporates
'fig1补充图_模仿达哥绘图_V2_GPT_经纬修改版.py' as an UpSet-style panel.

Main outputs:
  Figure2_standard_4panels.png/pdf  (a sample labels, b annotation degree,
                                    c directed overlap, d chemical t-SNE)
  Figure2_extended_5panels.png/pdf  (above plus e redesigned UpSet)
  Figure2_upset_only.png/pdf         (detailed UpSet with top 20 combinations)
  endpoint_stats.csv, ...

The t-SNE follows the original's fingerprint -> PCA(50) -> t-SNE procedure,
using full-set Morgan radius=2 2048-bit fingerprints; no molecules are sampled.
Cache keyed by SMILES hash, fingerprint parameters, and t-SNE settings.

Run:
    python ME_BBB_Figure2_redraw.py --data data.csv --outdir output --dpi 300

Dependencies: pandas numpy matplotlib rdkit scikit-learn
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

EP = ['BBB', 'PAMPA', 'Caco-2', 'MDR1', 'BCRP', 'MRP']

# --- EDIT ALL COLORS HERE: inspired by the user's revised pastel UpSet code ---
COLORS = {
    'white': '#FFFFFF',
    'text': '#283540',
    'subtext': '#60707A',
    'grid': '#E5EAED',
    'spine': '#B4BEC4',
    'band': '#F6F8F8',
    'missing': '#EBEFF0',
    'positive': '#638EA5',
    'negative': '#E4AE8E',
    'singleton': '#ACBCC4',
    'multiple': '#D78376',
    'inactive': '#E5EAEC',
    'active': '#596A76',
    'upset_bar': '#B5C9D6',
}
EP_COLORS = {
    'BBB': '#A7C49C',       # sage green
    'PAMPA': '#A8C7E3',     # sky blue
    'Caco-2': '#C9B8DC',    # lavender
    'MDR1': '#9CCBD0',      # teal
    'BCRP': '#E5BAC8',      # pink
    'MRP': '#E9CFB5',       # apricot
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'axes.edgecolor': COLORS['spine'],
    'axes.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=REPO_ROOT / 'data' / 'reference' / 'bbb_dataset.csv')
    p.add_argument('--outdir', type=Path, default=REPO_ROOT / 'outputs' / 'figures' / 'figure2' / 'base')
    p.add_argument('--updated-labels', type=Path, default=None, help='Optional new REAL endpoint labels (e.g., score_new_TabPFN.csv); fills only previously missing labels by STSMILES, never uses predicted scores')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--top-combinations', type=int, default=16, help='UpSet columns in extended version')
    p.add_argument('--tsne-perplexity', type=float, default=30.)
    p.add_argument('--tsne-iterations', type=int, default=1000)
    p.add_argument('--fp-bits', type=int, default=2048)
    p.add_argument('--tsne-seed', type=int, default=42)
    p.add_argument('--force-tsne', action='store_true')
    return p.parse_args()


def load_data(path, updates_path=None):
    df = pd.read_csv(path, low_memory=False)
    req = ['STSMILES'] + EP
    miss = set(req).difference(df.columns)
    if miss:
        raise ValueError('Missing required input columns: ' + ', '.join(sorted(miss)))
    df['STSMILES'] = df['STSMILES'].astype('string').str.strip()
    df = df.dropna(subset=['STSMILES'])
    df = df.loc[df['STSMILES'].ne('')].copy()
    for ep in EP:
        df[ep] = pd.to_numeric(df[ep], errors='coerce')
        if (~df[ep].isin([0, 1]) & df[ep].notna()).any():
            raise ValueError(f'{ep} has nonbinary labels')
    if df['STSMILES'].duplicated().any():
        # Aggregate per unique structure; contradictory 0/1 annotations are rejected.
        for ep in EP:
            u = df.dropna(subset=[ep]).groupby('STSMILES', sort=False)[ep].nunique()
            if (u > 1).any():
                raise ValueError(f'Conflicting labels for {ep} in the same STSMILES')
        df = df.groupby('STSMILES', sort=False, as_index=False)[EP].first()
    if updates_path is not None:
        updated = pd.read_csv(updates_path, low_memory=False)
        if 'STSMILES' not in updated.columns:
            raise ValueError('Updated-labels CSV needs STSMILES')
        updated['STSMILES'] = updated['STSMILES'].astype('string').str.strip()
        if updated['STSMILES'].duplicated().any():
            raise ValueError('Updated-labels CSV has duplicate STSMILES')
        keep = ['STSMILES'] + [ep for ep in EP if ep in updated.columns]
        tmp = df.merge(updated[keep], on='STSMILES', how='left',
                       validate='one_to_one', suffixes=('', '_update'))
        for ep in EP:
            if ep not in updated.columns:
                continue
            alt = pd.to_numeric(tmp[ep+'_update'], errors='coerce')
            if (alt.notna() & ~alt.isin([0,1])).any():
                raise ValueError(f'Updated {ep} contains labels other than 0/1')
            both = tmp[ep].notna() & alt.notna()
            if (tmp.loc[both,ep]!=alt.loc[both]).any():
                raise ValueError(f'Conflicting observed {ep} labels between input files')
            n_added = int((tmp[ep].isna() & alt.notna()).sum())
            if n_added:
                print(f'[UPDATE] Added {n_added} nonconflicting REAL {ep} labels from {updates_path.name}',flush=True)
            tmp[ep] = tmp[ep].fillna(alt)
            tmp = tmp.drop(columns=[ep+'_update'])
        df = tmp
    return df.reset_index(drop=True)


def compute_all(df, out):
    pres = df[EP].notna()
    n_total = len(df)
    rows = []
    for ep in EP:
        n1 = int(df[ep].eq(1).sum())
        n0 = int(df[ep].eq(0).sum())
        n = n0+n1
        rows.append({'Endpoint':ep, 'n_label_1':n1, 'n_label_0':n0,
                     'n_labeled':n, 'n_missing':n_total-n,
                     'positive_rate':n1/n if n else np.nan,
                     'coverage_rate':n/n_total})
    stats = pd.DataFrame(rows)
    degree = pres.sum(axis=1).astype(int)
    degree_counts = degree.value_counts().reindex(range(1,len(EP)+1),fill_value=0).sort_index()
    coverage = pd.DataFrame({'n_endpoints':degree_counts.index,
                             'n_molecules':degree_counts.values,
                             'fraction':degree_counts.values/n_total})
    counts = pres.to_numpy(dtype=np.int32).T @ pres.to_numpy(dtype=np.int32)
    sizes = np.diag(counts).astype(float)
    ratio = np.divide(counts,sizes[:,None],out=np.zeros_like(counts,dtype=float),where=sizes[:,None]>0)
    matrix_counts = pd.DataFrame(counts,index=EP,columns=EP)
    matrix_ratio = pd.DataFrame(ratio,index=EP,columns=EP)
    sig = pres.astype(int).astype(str).agg(''.join,axis=1)
    combo = sig.value_counts().reset_index()
    combo.columns=['signature','n_molecules']
    combo['endpoints'] = combo.signature.apply(lambda s: '+'.join(e for bit,e in zip(s,EP) if bit=='1'))
    combo['n_endpoints'] = combo.signature.str.count('1')
    combo['fraction'] = combo.n_molecules/n_total
    stats.to_csv(out/'endpoint_stats.csv',index=False)
    coverage.to_csv(out/'endpoint_label_degree.csv',index=False)
    matrix_counts.to_csv(out/'endpoint_overlap_counts.csv')
    matrix_ratio.to_csv(out/'endpoint_overlap_row_normalized.csv')
    combo.to_csv(out/'endpoint_combination_counts.csv',index=False)
    print('[DATA]',n_total,'molecules')
    print(stats[['Endpoint','n_labeled','positive_rate']].to_string(index=False))
    print('[DEGREE]',degree_counts.to_dict())
    return {'df':df,'stats':stats,'coverage':coverage,'counts':matrix_counts,
            'ratio':matrix_ratio,'combos':combo,'degree':degree}


def get_tsne(df, args):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    keyraw = '\n'.join(df.STSMILES.astype(str).tolist()) + f'|{args.fp_bits}|{args.tsne_perplexity}|{args.tsne_iterations}|{args.tsne_seed}'
    key = hashlib.sha256(keyraw.encode('utf-8')).hexdigest()
    cache = args.outdir/'fingerprint_tsne_cache.npz'
    if cache.exists() and not args.force_tsne:
        with np.load(cache) as arr:
            if str(arr['key'].item()) == key and arr['xy'].shape == (len(df),2):
                print('[t-SNE] cache hit:',cache)
                return arr['xy']
    print('[t-SNE] Calculating Morgan fingerprints, PCA(50), then t-SNE on ALL molecules...',flush=True)
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=args.fp_bits)
    matrix = np.empty((len(df),args.fp_bits),dtype=np.uint8)
    valid = []
    invalid = []
    for i,smi in enumerate(df.STSMILES):
        mol=Chem.MolFromSmiles(str(smi))
        if mol is None:
            invalid.append((i,smi))
            continue
        fp=generator.GetFingerprint(mol)
        vec=np.zeros(args.fp_bits,dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp,vec)
        matrix[i]=vec
        valid.append(i)
    if invalid:
        # Reject rather than silently changing the chemistry space sample.
        raise ValueError(f'{len(invalid)} invalid SMILES, first: {invalid[:3]}')
    print('[t-SNE] fingerprints:',matrix.shape, 'PCA step',flush=True)
    pre = PCA(n_components=50,svd_solver='randomized',random_state=args.tsne_seed)
    low=pre.fit_transform(matrix.astype(np.float32))
    del matrix
    model=TSNE(n_components=2,perplexity=args.tsne_perplexity,
               learning_rate='auto',init='pca',random_state=args.tsne_seed,
               max_iter=args.tsne_iterations,method='barnes_hut',verbose=1,n_jobs=4)
    xy=model.fit_transform(low)
    np.savez_compressed(cache,key=key,xy=xy)
    print('[t-SNE] computed and cached:', cache,flush=True)
    return xy


def minimalist(ax, grid_axis=None):
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color(COLORS['spine'])
    ax.tick_params(colors=COLORS['subtext'],length=2,width=.6)
    if grid_axis: ax.grid(axis=grid_axis,color=COLORS['grid'],lw=.75,zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, tag, x=-.15, y=1.08):
    ax.text(x,y,tag,ha='left',va='bottom',transform=ax.transAxes,
            fontsize=17,fontweight='bold',color=COLORS['text'],clip_on=False)


def title(ax,t):
    ax.set_title(t,fontsize=11,fontweight='semibold',color=COLORS['text'],pad=12)


def panel_a(fig, spec, d):
    sub=spec.subgridspec(1,2,width_ratios=[2.2,1],wspace=.13)
    ax=fig.add_subplot(sub[0,0]); rateax=fig.add_subplot(sub[0,1],sharey=ax)
    st=d['stats']
    y=np.arange(len(EP))
    # Show observed 0+1, with light-gray total-data context behind each bar.
    ax.barh(y,np.full(6,len(d['df'])),height=.62,color=COLORS['missing'],zorder=1)
    ax.barh(y,st.n_label_1,height=.62,color=COLORS['positive'],zorder=3,label='Label = 1')
    ax.barh(y,st.n_label_0,left=st.n_label_1,height=.62,color=COLORS['negative'],zorder=3,label='Label = 0')
    for i,n in enumerate(st.n_labeled):
        ax.text(n+len(d['df'])*.015,i,f'{n:,}',fontsize=8.6,va='center',color=COLORS['text'])
    ax.set_yticks(y,EP,fontsize=9.5);ax.invert_yaxis()
    ax.set_xlim(0,len(d['df'])*1.21)
    ax.set_xlabel('Number of molecules',fontsize=9.5)
    ax.set_xticks([0,5000,10000,15000],['0','5k','10k','15k'])
    ax.tick_params(axis='y',length=0)
    minimalist(ax,grid_axis='x')
    ax.legend(handles=[Patch(facecolor=COLORS['positive'],label='Label = 1'),
                       Patch(facecolor=COLORS['negative'],label='Label = 0'),
                       Patch(facecolor=COLORS['missing'],label='Unlabeled')],
              frameon=False,ncol=3,fontsize=8.0,loc='upper left',bbox_to_anchor=(-.02,-.19),handlelength=1.3,columnspacing=.85)
    rateax.scatter(st.positive_rate*100,y,s=48,c=[EP_COLORS[e] for e in EP],
                   edgecolor='white',linewidth=.75,zorder=5)
    for i,v in enumerate(st.positive_rate):
        rateax.text(min(v*100+5,102),i,f'{v*100:.1f}%',fontsize=8.3,va='center',color=COLORS['text'])
    rateax.set_xlim(0,116);rateax.set_xticks([0,50,100],['0','50','100'])
    rateax.set_yticks(y);rateax.tick_params(axis='y',left=False,labelleft=False)
    rateax.set_xlabel('Positive rate (%)',fontsize=9.5)
    rateax.grid(axis='x',color=COLORS['grid'],lw=.75)
    minimalist(rateax)
    for side in ['left','top','right']:rateax.spines[side].set_visible(False)
    title(ax,'Endpoint availability and class balance')
    panel_label(ax,'a',x=-.16,y=1.07)
    return ax


def panel_b(fig,spec,d):
    ax=fig.add_subplot(spec)
    t=d['coverage']
    x=t.n_endpoints.to_numpy()
    n=t.n_molecules.to_numpy()
    barcolors=[COLORS['singleton']]+[COLORS['multiple']]*5
    ax.bar(x,n,width=.65,color=barcolors,edgecolor='white',lw=.8,zorder=4)
    ax.set_yscale('log');ax.set_ylim(.8,68000)
    ax.set_xticks(np.arange(1,7))
    ax.set_xlabel('Annotated endpoints per molecule')
    ax.set_ylabel('Number of molecules (log scale)')
    for i,nn in enumerate(n):
        pct=100*nn/len(d['df'])
        ptext=f'{pct:.1f}%' if pct>=.05 else '<0.1%'
        ax.text(x[i],nn*1.33,f'{nn:,}\n({ptext})',ha='center',va='bottom',
                fontsize=8.3,color=COLORS['text'],linespacing=1.0)
    minimalist(ax,grid_axis='y')
    ax.set_yticks([1,10,100,1000,10000],['1','10','100','1k','10k'])
    title(ax,'Sparsity of endpoint annotations')
    panel_label(ax,'b',x=-.17,y=1.07)
    return ax


def panel_c(fig,spec,d):
    g=spec.subgridspec(1,2,width_ratios=[1,.045],wspace=.12)
    ax=fig.add_subplot(g[0,0]);cb=fig.add_subplot(g[0,1])
    mat=d['ratio'].to_numpy()
    cm=LinearSegmentedColormap.from_list('pastel_overlap',
         ['#F6F8F9','#E1EDF0','#AACCD7','#6599A9','#386782'],256)
    im=ax.imshow(mat,vmin=0,vmax=1,cmap=cm,aspect='equal',interpolation='nearest')
    ax.set_xticks(np.arange(6),EP,rotation=0,fontsize=9)
    ax.set_yticks(np.arange(6),EP,fontsize=9)
    ax.tick_params(length=0,pad=7)
    for s in ax.spines.values():s.set_visible(False)
    for i in range(6):
        for j in range(6):
            v=mat[i,j]
            ax.text(j,i,f'{v*100:.1f}%' if v<.1 else f'{v*100:.0f}%',
                    fontsize=8,ha='center',va='center',color='white' if v>.57 else COLORS['text'],
                    fontweight='medium' if i==j else 'normal')
    cbar=fig.colorbar(im,cax=cb)
    cbar.set_label('Row-normalized overlap',fontsize=9,color=COLORS['text'])
    cbar.set_ticks([0,.25,.50,.75,1]); cbar.set_ticklabels(['0%','25%','50%','75%','100%'])
    cbar.ax.tick_params(labelsize=8,colors=COLORS['subtext'])
    cbar.outline.set_visible(False)
    title(ax,'Pairwise endpoint overlap')
    panel_label(ax,'c',x=-.18,y=1.075)
    return ax


def panel_d(fig,spec,d,xy):
    ax=fig.add_subplot(spec)
    n=d['degree'].to_numpy()
    single=n==1; multiple=n>=2
    ax.scatter(xy[single,0],xy[single,1],s=5.3,color=COLORS['singleton'],
               alpha=.37,lw=0,rasterized=True,zorder=1)
    ax.scatter(xy[multiple,0],xy[multiple,1],s=10.5,color=COLORS['multiple'],
               alpha=.80,lw=0,rasterized=True,zorder=2)
    legend=[Line2D([],[],marker='o',linestyle='',markersize=5.2,
                   markerfacecolor=COLORS['singleton'],markeredgecolor='none',
                   label=f'Single endpoint (n={single.sum():,})'),
            Line2D([],[],marker='o',linestyle='',markersize=5.2,
                   markerfacecolor=COLORS['multiple'],markeredgecolor='none',
                   label=f'≥2 endpoints (n={multiple.sum():,})')]
    ax.legend(handles=legend,loc='lower left',fontsize=8,frameon=True,
              edgecolor=COLORS['grid'],facecolor='white',framealpha=.92,
              handletextpad=.35,borderpad=.6)
    ax.set_xlabel('t-SNE 1');ax.set_ylabel('t-SNE 2')
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    title(ax,'Chemical space of labeled compounds')
    panel_label(ax,'d',x=-.12,y=1.075)
    return ax


def panel_e(fig,spec,d,top_n=16,letter=True):
    # Inspired by the user's UpSet redesign, but drawn into a configurable panel.
    top=d['combos'].head(top_n)
    end=EP;sig=top.signature.to_list();nums=top.n_molecules.to_numpy()
    arr=np.array([[int(x) for x in c] for c in sig]).T
    g=spec.subgridspec(2,2,width_ratios=[1.12,5.5],height_ratios=[2.7,1.6],
                      wspace=.11,hspace=.08)
    topax=fig.add_subplot(g[0,1]);matax=fig.add_subplot(g[1,1]);leftax=fig.add_subplot(g[1,0],sharey=matax)
    x=np.arange(len(sig));y=np.arange(6)
    max_y=int(max(nums))
    topax.bar(x,nums,color=COLORS['upset_bar'],width=.64,edgecolor='white',lw=.6,zorder=3)
    for xx,nn in zip(x,nums):
        topax.text(xx,nn+max_y*.027,f'{nn:,}',ha='center',va='bottom',fontsize=7.8,color=COLORS['text'])
    topax.set_xlim(-.65,len(sig)-.35);topax.set_ylim(0,max_y*1.22)
    topax.set_xticks([]);topax.set_ylabel('Intersection size',fontsize=9.5)
    minimalist(topax,grid_axis='y')
    topax.spines['bottom'].set_visible(False)
    for r in range(6):
        if r%2==0:
            matax.axhspan(r-.5,r+.5,color=COLORS['band'],zorder=0)
            leftax.axhspan(r-.5,r+.5,color=COLORS['band'],zorder=0)
    for r in range(6):
        matax.scatter(x,np.full(len(sig),r),s=47,c=COLORS['inactive'],zorder=1,linewidths=0)
    for j in range(len(sig)):
        active=np.flatnonzero(arr[:,j])
        if len(active)>=2:
            matax.plot([j,j],[active.min(),active.max()],lw=2.1,color=COLORS['active'],zorder=2)
        matax.scatter(np.repeat(j,len(active)),active,c=COLORS['active'],s=57,zorder=3,linewidths=0)
    matax.set_xlim(-.65,len(sig)-.35);matax.set_ylim(5.5,-.5)
    matax.set_xticks([]);matax.set_yticks(y,EP,fontsize=9)
    matax.tick_params(length=0,pad=8)
    for s in matax.spines.values():s.set_visible(False)
    sizes=d['stats'].n_labeled.to_numpy()
    leftax.barh(y,sizes,height=.64,color=[EP_COLORS[e] for e in EP],edgecolor='none',zorder=3)
    for yi,ss in zip(y,sizes):
        leftax.text(ss+max(sizes)*.02,yi,f'{ss:,}',va='center',ha='right',fontsize=8.6,color=COLORS['text'])
    leftax.set_xlim(max(sizes)*1.38,0)
    leftax.set_xticks([0,max(sizes)],['0',f'{max(sizes):,}'])
    leftax.set_yticks(y);leftax.tick_params(axis='y',left=False,labelleft=False)
    leftax.set_xlabel('Set size',fontsize=9.3)
    minimalist(leftax)
    leftax.spines['left'].set_visible(False)
    leftax.spines['top'].set_visible(False)
    leftax.spines['right'].set_visible(False)
    topax.text(-.205,1.19,'Endpoint intersections (UpSet)',transform=topax.transAxes,
               fontsize=11,fontweight='semibold',color=COLORS['text'],ha='left',va='bottom')
    if letter:
        # Place at the same left margin as other figure panel labels.
        topax.text(-.205,1.38,'e',transform=topax.transAxes,
                   fontsize=17,fontweight='bold',color=COLORS['text'],ha='left',va='bottom')
    return topax


def export(fig,path,dpi):
    fig.savefig(path.with_suffix('.png'),dpi=dpi,bbox_inches='tight',pad_inches=.17,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.17,facecolor='white')
    plt.close(fig)


def draw_four(d,xy,outdir,dpi):
    fig=plt.figure(figsize=(15.8,10.5),facecolor='white')
    grid=fig.add_gridspec(2,2,left=.072,right=.975,top=.94,bottom=.075,
                          hspace=.38,wspace=.22,height_ratios=[1,1.07])
    panel_a(fig,grid[0,0],d);panel_b(fig,grid[0,1],d)
    panel_c(fig,grid[1,0],d);panel_d(fig,grid[1,1],d,xy)
    export(fig,outdir/'Figure2_standard_4panels',dpi)


def draw_five(d,xy,outdir,dpi,top_n):
    fig=plt.figure(figsize=(16.0,17.9),facecolor='white')
    grid=fig.add_gridspec(3,2,left=.07,right=.975,top=.963,bottom=.053,
                          hspace=.39,wspace=.22,height_ratios=[.98,1.02,1.31])
    panel_a(fig,grid[0,0],d);panel_b(fig,grid[0,1],d)
    panel_c(fig,grid[1,0],d);panel_d(fig,grid[1,1],d,xy)
    panel_e(fig,grid[2,:],d,top_n,letter=True)
    export(fig,outdir/'Figure2_extended_5panels',dpi)


def draw_upset(d,outdir,dpi):
    fig=plt.figure(figsize=(15.3,7.35),facecolor='white')
    panel_e(fig,fig.add_gridspec(1,1,left=.085,right=.975,top=.86,bottom=.10)[0],d,20,letter=False)
    export(fig,outdir/'Figure2_upset_only',dpi)


def main():
    a=cli()
    a.outdir.mkdir(parents=True,exist_ok=True)
    d=compute_all(load_data(a.data,a.updated_labels),a.outdir)
    d['df'].to_csv(a.outdir/'data_used_for_figure.csv',index=False)
    xy=get_tsne(d['df'],a)
    draw_four(d,xy,a.outdir,a.dpi)
    draw_five(d,xy,a.outdir,a.dpi,a.top_combinations)
    draw_upset(d,a.outdir,a.dpi)
    meta={'n':len(d['df']),'endpoints':EP,'fp_bits':a.fp_bits,'fingerprint':'Morgan ECFP radius=2',
          'pca_dim':50,'tsne_perplexity':a.tsne_perplexity,'tsne_iterations':a.tsne_iterations,
          'tsne_seed':a.tsne_seed,'updated_labels_source':str(a.updated_labels) if a.updated_labels else None,'upset_top_n':a.top_combinations,'colors':COLORS,'endpoint_colors':EP_COLORS}
    (a.outdir/'method_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
    print('[DONE] Main 4-panel, extended 5-panel, UpSet-only, PNG & PDF exported.',flush=True)

if __name__=='__main__':
    main()

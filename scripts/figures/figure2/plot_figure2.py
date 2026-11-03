#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nature-inspired Figure 2 redraw for ME-BBB.

Six reproducible editorial palettes (four newer options including two high-saturation variants).
Molecule/endpoint counts and t-SNE are computed by the validated accompanying
ME_BBB_Figure2_redraw.py module, not copied from figure images.

Typical run:
    python ME_BBB_Figure2_nature_style.py --data data.csv \
       --updated-labels score_new_TabPFN.csv --outdir nature_fig2 --palette both

Use --palette pastel_plum, teal_coral, vivid_plum or cobalt_sunset for a single style.
Use --palette both for the two NEW styles; use --palette all for legacy too.
The latest observed EP labels are merged by STSMILES only; predicted scores are
never treated as experimental labels. Refer to README for plot definitions.
Dependencies: numpy pandas matplotlib scikit-learn rdkit.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

import figure2_base as base

EP=base.EP
# Feature-level colors in panel (a) and endpoint colors in the set-size bars
# carry different meanings:
#   primary/secondary: binary experimental endpoint label (EP=1 / EP=0)
#   endpoints: fixed palette for six named endpoints (mainly panel e)
#   single/multi: compounds annotated for exactly 1 vs >=2 endpoints (b, d)
#   upset_bar: ONE color for ALL exclusive-intersection bars in panel (e)
# The values below are editable HEX codes. All are colorblind-conscious in that
# key class/endpoint meanings are also labeled with text, not color alone.
PALETTES = {
    'pastel_plum': {
        'name': 'Pastel Plum | slate blue, sand and dusty plum',
        'ink': '#303C49', 'muted': '#64717F',
        'grid': '#E8EBEF', 'light': '#F4F5F7',
        'primary': '#5C7FA3', 'secondary': '#DCB38B',
        'single': '#B2C0CD', 'multi': '#B87990',
        'inactive': '#E2E7EB', 'active': '#465463',
        'upset_bar': '#ABBBCB',
        'hm': ['#F8F9FC', '#E8ECF4', '#C9D5E4', '#91AAC8', '#4B709B'],
        'endpoints': {
            'BBB': '#BFD1AD', 'PAMPA': '#AFCBE4',
            'Caco-2': '#D3C3DE', 'MDR1': '#B7D8DC',
            'BCRP': '#E5C0CD', 'MRP': '#E8D5C5',
        },
    },
    'teal_coral': {
        'name': 'Teal Coral | muted teal and warm terracotta',
        'ink': '#293A46', 'muted': '#5E7079',
        'grid': '#E6EBED', 'light': '#F3F7F7',
        'primary': '#2B798A', 'secondary': '#E4AA8E',
        'single': '#AFC4CC', 'multi': '#CF7667',
        'inactive': '#E1E8EB', 'active': '#364E5C',
        'upset_bar': '#A8BECA',
        'hm': ['#F6FAFB', '#DDECEF', '#BAD7DE', '#77AAB8', '#277487'],
        'endpoints': {
            'BBB': '#B8D1BA', 'PAMPA': '#A5CAE0',
            'Caco-2': '#CAC1DE', 'MDR1': '#AAD6D5',
            'BCRP': '#DCAEBB', 'MRP': '#DECDBB',
        },
    },

    'vivid_plum': {
        'name': 'Vivid Plum | saturated teal, apricot and plum',
        'ink': '#24313B', 'muted': '#566672',
        'grid': '#E7EBEF', 'light': '#F5F6F8',
        'primary': '#2F7F93', 'secondary': '#E3A07B',
        'single': '#AFBEC9', 'multi': '#B55E84',
        'inactive': '#E1E7EC', 'active': '#394A59',
        'upset_bar': '#A8BAC8',
        'hm': ['#F8FBFC', '#E0ECF0', '#A9CED7', '#5E9FB0', '#23697D'],
        'endpoints': {
            'BBB': '#456F9A', 'PAMPA': '#5B97C2',
            'Caco-2': '#8F79B6', 'MDR1': '#D28A4A',
            'BCRP': '#C45E83', 'MRP': '#8B79B0',
        },
    },
    'cobalt_sunset': {
        'name': 'Cobalt Sunset | cobalt blue, coral and violet',
        'ink': '#23303A', 'muted': '#5B6A74',
        'grid': '#E6EAEE', 'light': '#F6F7F9',
        'primary': '#2A6FBB', 'secondary': '#E49A62',
        'single': '#AEBCCA', 'multi': '#D06D52',
        'inactive': '#E0E6EB', 'active': '#364755',
        'upset_bar': '#A7B8C9',
        'hm': ['#F8FAFD', '#DFE9F7', '#AFC7EA', '#5F93D0', '#245D9C'],
        'endpoints': {
            'BBB': '#2C6DB2', 'PAMPA': '#56A0D3',
            'Caco-2': '#7F96C9', 'MDR1': '#E0914C',
            'BCRP': '#C96A8E', 'MRP': '#8D78BA',
        },
    },
    # The previous two palettes remain available for backwards comparison.
    'marine': {
        'name': 'Marine | teal and warm coral (legacy)',
        'ink': '#24333E', 'muted': '#586B75', 'grid': '#E6EBEE', 'light': '#F3F6F7',
        'primary': '#226E80', 'secondary': '#E5A080',
        'single': '#AABAC2', 'multi': '#D26F61',
        'inactive': '#DEE6E9', 'active': '#334C5B',
        'upset_bar': '#ABBEC5',
        'hm': ['#F7FAFA', '#D7E5E8', '#9CC7D1', '#4B8C9D', '#1D5163'],
        'endpoints': {'BBB': '#3A6879', 'PAMPA': '#4B8EA8', 'Caco-2': '#7F9EB8',
                      'MDR1': '#CE8C5A', 'BCRP': '#C67585', 'MRP': '#977AA4'},
    },
    'earth': {
        'name': 'Earth | ink blue and golden rose (legacy)',
        'ink': '#303643', 'muted': '#656B7A', 'grid': '#E8E9EC', 'light': '#F6F6F7',
        'primary': '#466D97', 'secondary': '#D9A36B',
        'single': '#AFB8C5', 'multi': '#BE7187',
        'inactive': '#E2E5EA', 'active': '#505D76',
        'upset_bar': '#AEBAC8',
        'hm': ['#F9FAFC', '#E2E8F2', '#BDCDE0', '#7998BC', '#355A87'],
        'endpoints': {'BBB': '#526A86', 'PAMPA': '#638AA7', 'Caco-2': '#8DA9B9',
                      'MDR1': '#C98D60', 'BCRP': '#B66D89', 'MRP': '#877DA5'},
    },
}

def cli():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=REPO_ROOT/'data'/'reference'/'bbb_dataset.csv')
    p.add_argument('--updated-labels',type=Path,default=None,
                   help='Latest REAL observed labels; set to none to use only data.csv')
    p.add_argument('--outdir',type=Path,default=REPO_ROOT/'outputs'/'figures'/'figure2')
    p.add_argument('--palette',choices=['pastel_plum','teal_coral','vivid_plum','cobalt_sunset','marine','earth','both','saturated','all'],default='both')
    p.add_argument('--dpi',type=int,default=450)
    p.add_argument('--top-combinations',type=int,default=16)
    p.add_argument('--tsne-cache',type=Path,default=REPO_ROOT/'outputs'/'figures'/'figure2'/'cache'/'fingerprint_tsne_cache.npz')
    p.add_argument('--tsne-perplexity',type=float,default=30.0)
    p.add_argument('--tsne-iterations',type=int,default=1000)
    p.add_argument('--tsne-seed',type=int,default=42)
    p.add_argument('--fp-bits',type=int,default=2048)
    p.add_argument('--font-scale',type=float,default=1.0,
                   help='Global multiplier for all text sizes (e.g. 1.2)')
    p.add_argument('--figure-scale',type=float,default=1.0,
                   help='Global multiplier for figure canvas size (e.g. 1.15)')
    p.add_argument('--line-scale',type=float,default=1.0,
                   help='Global multiplier for line widths, spines and tick strokes')
    p.add_argument('--marker-scale',type=float,default=1.0,
                   help='Global multiplier for scatter/dot marker sizes')
    p.add_argument('--text-tone',choices=['normal','dark'],default='dark',
                   help='Use darker near-black text for better readability in manuscript figures')
    return p.parse_args()


def configure(P):
    plt.rcParams.update({
      'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],
      'font.size':12.0, 'axes.titlesize':14.0, 'axes.labelsize':12.2,
      'xtick.labelsize':11.2, 'ytick.labelsize':11.2,
      'axes.labelcolor':P['ink'], 'text.color':P['ink'], 'axes.edgecolor':'#ABB4BD',
      'axes.linewidth':.9,'xtick.color':P['muted'],'ytick.color':P['muted'],
      'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white', 'figure.facecolor':'white',
      'axes.facecolor':'white',
    })


def style(ax,P,axis=None):
    for k in ['top','right']: ax.spines[k].set_visible(False)
    for k in ['left','bottom']: ax.spines[k].set_color('#BBC4CB');ax.spines[k].set_linewidth(.85)
    ax.tick_params(length=3.2,width=.85)
    if axis: ax.grid(axis=axis,color=P['grid'],lw=.65,zorder=0)
    ax.set_axisbelow(True)


def mark(ax,label,P,xy=(-.13,1.095)):
    ax.text(*xy,label,transform=ax.transAxes,fontsize=19,fontweight='bold',
            ha='left',va='bottom',color=P['ink'],clip_on=False)


def heading(ax,text,P):
    ax.set_title(text,pad=12,fontsize=13.8,fontweight='bold',color=P['ink'])


def fig_a(fig,sp,D,P):
    g=sp.subgridspec(1,2,width_ratios=[3.9,1.15],wspace=.22)
    ax=fig.add_subplot(g[0,0]);rt=fig.add_subplot(g[0,1],sharey=ax)
    st=D['stats'];y=np.arange(6)
    good=st.n_label_1.to_numpy();bad=st.n_label_0.to_numpy();n=good+bad
    ax.barh(y,good,height=.58,color=P['primary'],zorder=3)
    ax.barh(y,bad,left=good,height=.58,color=P['secondary'],zorder=3)
    ax.set_yticks(y,EP);ax.invert_yaxis();ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,9300)
    ax.set_xticks([0,2000,4000,6000,8000],['0','2k','4k','6k','8k'])
    for i,v in enumerate(n):
        ax.text(v+135,i,f'{v:,}',va='center',fontsize=11.2,color=P['ink'],fontweight='medium')
    ax.set_xlabel('Annotated compounds (n)')
    style(ax,P,'x')
    # Keep the legend inside a naturally empty part of the axis so it cannot
    # collide with the next panel label (c) in a multi-row figure.
    ax.legend(handles=[Patch(fc=P['primary'],label='EP = 1'),Patch(fc=P['secondary'],label='EP = 0')],
              loc='lower right',bbox_to_anchor=(.992,.035),ncol=2,frameon=False,fontsize=10.6,
              handlelength=1.5,columnspacing=1.0)
    rates=100*st.positive_rate.to_numpy()
    rt.hlines(y,0,rates,lw=1.6,color=P['grid'])
    rt.scatter(rates,y,s=48,c=[P['endpoints'][e] for e in EP],edgecolors='white',linewidths=.9,zorder=5)
    for i,r in enumerate(rates):
        rt.text(min(r+5,102),i,f'{r:.1f}',va='center',fontsize=10.9,color=P['ink'],fontweight='medium')
    rt.set_xlim(0,115);rt.set_xticks([0,50,100]);rt.set_xlabel('EP = 1 (%)')
    rt.tick_params(axis='y',left=False,labelleft=False)
    style(rt,P)
    rt.spines['left'].set_visible(False)
    heading(ax,'Endpoint coverage and class balance',P)
    mark(ax,'a',P,xy=(-.13,1.09))


def fig_b(fig,sp,D,P):
    ax=fig.add_subplot(sp)
    dat=D['coverage'];x=dat.n_endpoints.to_numpy();counts=dat.n_molecules.to_numpy()
    col=[P['single']]+[P['multi']]*5
    ax.bar(x,counts,width=.60,color=col,edgecolor='white',lw=1.0,zorder=3)
    ax.set_yscale('log'); ax.set_ylim(.9,50000); ax.set_xlim(.5,6.5)
    ax.set_xticks(x)
    ax.set_yticks([1,10,100,1000,10000],['1','10','100','1k','10k'])
    ax.set_xlabel('Number of annotated endpoints')
    ax.set_ylabel('Compounds (log scale)')
    for i,v in enumerate(counts):
        pct=v/len(D['df'])*100
        ax.text(x[i],v*1.45,f'{v:,}\n{pct:.1f}%' if pct>=0.1 else f'{v:,}\n<0.1%',
                ha='center',va='bottom',fontsize=10.8,color=P['ink'],linespacing=1.15,fontweight='medium')
    style(ax,P,'y');heading(ax,'Sparse multi-endpoint annotation',P)
    mark(ax,'b',P,xy=(-.13,1.09))


def fig_c(fig,sp,D,P):
    g=sp.subgridspec(1,2,width_ratios=[1,.038],wspace=.045)
    ax=fig.add_subplot(g[0,0]);cax=fig.add_subplot(g[0,1])
    mat=D['ratio'].to_numpy().copy()
    offdiag=mat.copy();np.fill_diagonal(offdiag,np.nan)
    cap=.30
    cmap=LinearSegmentedColormap.from_list('journal_overlap',P['hm'],256)
    cmap.set_bad(P['light'])
    im=ax.imshow(np.ma.masked_invalid(offdiag),vmin=0,vmax=cap,cmap=cmap,aspect='equal')
    ax.set_xticks(np.arange(6),EP);ax.set_yticks(np.arange(6),EP)
    ax.tick_params(length=0,pad=5)
    # Fine white gutters between heatmap cells.
    ax.set_xticks(np.arange(-.5,6,.5)[::2],minor=True)
    ax.set_yticks(np.arange(-.5,6,.5)[::2],minor=True)
    ax.grid(which='minor',color='white',lw=2.0)
    for spine in ax.spines.values():spine.set_visible(False)
    for i in range(6):
        for j in range(6):
            if i==j:
                ax.text(j,i,'—',ha='center',va='center',fontsize=12,color='#ABB5BD')
            else:
                v=mat[i,j]; ax.text(j,i,f'{v*100:.1f}%',ha='center',va='center',
                    fontsize=10.2,color='white' if v>.18 else P['ink'],
                    fontweight='bold' if v>.18 else 'semibold')
    bar=fig.colorbar(im,cax=cax)
    bar.set_ticks([0,.1,.2,.3],labels=['0%','10%','20%','30%'])
    bar.ax.tick_params(labelsize=10.8,length=0)
    bar.outline.set_visible(False)
    bar.set_label('Row-normalized overlap',fontsize=11.4,labelpad=10)
    heading(ax,'Pairwise endpoint overlap',P)
    mark(ax,'c',P,xy=(-.20,1.09))


def fig_d(fig,sp,D,xy,P):
    ax=fig.add_subplot(sp)
    degree=D['degree'].to_numpy(); s=degree==1;m=degree>=2
    ax.scatter(xy[s,0],xy[s,1],s=5.0,color=P['single'],alpha=.48,lw=0,zorder=1,rasterized=True)
    ax.scatter(xy[m,0],xy[m,1],s=13.0,color=P['multi'],alpha=.82,lw=0,zorder=2,rasterized=True)
    h=[Line2D([],[],color='none',marker='o',markersize=6.2,markerfacecolor=P['single'],
              markeredgecolor='none',label=f'1 endpoint (n={sum(s):,})'),
       Line2D([],[],color='none',marker='o',markersize=6.2,markerfacecolor=P['multi'],
              markeredgecolor='none',label=f'≥2 endpoints (n={sum(m):,})')]
    ax.legend(handles=h,loc='lower left',fontsize=10.6,frameon=False,
              borderpad=.2,handletextpad=.4,labelspacing=.3)
    ax.set_xticks([]);ax.set_yticks([])
    for k in ax.spines:ax.spines[k].set_visible(False)
    ax.set_xlabel('t-SNE 1');ax.set_ylabel('t-SNE 2')
    heading(ax,'Chemical-space distribution',P)
    mark(ax,'d',P,xy=(-.09,1.09))


def fig_e(fig,sp,D,P,top_n=16,letter=True):
    top=D['combos'].head(top_n)
    sig=top.signature.to_list(); val=top.n_molecules.to_numpy()
    active=np.array([[int(t) for t in ss] for ss in sig]).T
    g=sp.subgridspec(2,2,width_ratios=[1.1,5.5],height_ratios=[2.0,1.22],
                     hspace=.035,wspace=.10)
    topax=fig.add_subplot(g[0,1]);mt=fig.add_subplot(g[1,1],sharex=topax)
    left=fig.add_subplot(g[1,0],sharey=mt)
    x=np.arange(len(val));rows=np.arange(6)
    # Uniform bar color: height communicates exact intersection size.
    # Previously the first six bars (single-endpoint combinations) were gray,
    # while multi-endpoint combinations were plum/coral. This difference was
    # not explained by the legend and looked like arbitrary coloring.
    topax.bar(x,val,color=P['upset_bar'],width=.68,lw=0,zorder=3)
    ymax=max(val);topax.set_xlim(-.60,len(val)-.38);topax.set_ylim(0,ymax*1.22)
    for idx,(xx,v) in enumerate(zip(x,val)):
        topax.text(xx,v+ymax*.025,f'{v:,}',fontsize=10.4,ha='center',va='bottom',color=P['ink'],fontweight='medium')
    topax.set_xticks([]);topax.set_ylabel('Exclusive intersection size (n)',labelpad=8)
    topax.set_yticks([0,2000,4000,6000],[0,'2k','4k','6k'])
    style(topax,P,'y');topax.spines['bottom'].set_visible(False)
    for i in range(6):
        if i%2==0:
            mt.axhspan(i-.50,i+.50,color=P['light'],lw=0,zorder=0)
            left.axhspan(i-.50,i+.50,color=P['light'],lw=0,zorder=0)
        mt.scatter(x,np.repeat(i,len(x)),s=40,color=P['inactive'],zorder=1,lw=0)
    for j in range(len(x)):
        ids=np.flatnonzero(active[:,j])
        if len(ids)>=2:
            mt.plot([j,j],[ids.min(),ids.max()],color=P['active'],lw=2.3,zorder=2)
        mt.scatter(np.repeat(j,len(ids)),ids,s=58,color=P['active'],zorder=3,lw=0)
    mt.set_ylim(5.52,-.53);mt.set_xlim(-.60,len(val)-.38)
    mt.set_xticks([]);mt.set_yticks(rows,EP);mt.tick_params(axis='y',length=0,pad=9)
    for spine in mt.spines.values():spine.set_visible(False)
    sz=D['stats'].n_labeled.to_numpy()
    left.barh(rows,sz,height=.58,color=[P['endpoints'][e] for e in EP],lw=0,zorder=3)
    left.set_xlim(10500,0)
    for y,v in zip(rows,sz):
        left.text(v+160,y,f'{v:,}',va='center',ha='right',fontsize=11.0,color=P['ink'],fontweight='medium')
    left.set_xticks([0,5000,10000],['0','5k','10k']);left.set_xlabel('Set size (n)')
    left.tick_params(axis='y',length=0,labelleft=False)
    for k in left.spines: left.spines[k].set_visible(False)
    if letter: mark(topax,'e',P,xy=(-.245,1.105))
    heading(topax,'Exact endpoint combinations (UpSet)',P)
    return topax



def apply_user_scaling(fig,a):
    """Post-hoc scaling of figure size, fonts, lines and markers.
    This keeps the original layout logic but allows easy tuning for manuscript readability.
    """
    # 1) Canvas size
    if getattr(a, 'figure_scale', 1.0) != 1.0:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * a.figure_scale, h * a.figure_scale, forward=True)

    # 2) Text sizes (titles, labels, annotations, legend texts, colorbar labels, etc.)
    fs = getattr(a, 'font_scale', 1.0)
    if fs != 1.0:
        for artist in fig.findobj(match=mpl.text.Text):
            try:
                size = artist.get_fontsize()
                if size is not None:
                    artist.set_fontsize(size * fs)
            except Exception:
                pass

    # 3) Lines / spines / ticks
    ls = getattr(a, 'line_scale', 1.0)
    if ls != 1.0:
        for ax in fig.axes:
            # spines
            for sp in ax.spines.values():
                try:
                    sp.set_linewidth(sp.get_linewidth() * ls)
                except Exception:
                    pass
            # ticks
            try:
                ax.tick_params(length=3.2 * ls, width=0.85 * ls)
            except Exception:
                pass
            # lines
            for ln in ax.lines:
                try:
                    ln.set_linewidth(ln.get_linewidth() * ls)
                except Exception:
                    pass
            # patches (bar edges, etc.)
            for pa in ax.patches:
                try:
                    pa.set_linewidth(pa.get_linewidth() * ls)
                except Exception:
                    pass

    # 4) Scatter / markers
    ms = getattr(a, 'marker_scale', 1.0)
    if ms != 1.0:
        for ax in fig.axes:
            for coll in ax.collections:
                # PathCollection scatter sizes are areas in pt^2
                try:
                    sizes = coll.get_sizes()
                    if sizes is not None and len(sizes) > 0:
                        coll.set_sizes(sizes * (ms ** 2))
                except Exception:
                    pass
            for ln in ax.lines:
                try:
                    if ln.get_markersize() is not None:
                        ln.set_markersize(ln.get_markersize() * ms)
                except Exception:
                    pass



def apply_text_tone(fig,a):
    """Darken text/ticks/spines for better contrast in manuscripts."""
    if getattr(a, 'text_tone', 'normal') != 'dark':
        return
    dark = '#1F2933'      # near-black blue-gray
    dark2 = '#3A4651'     # darker muted tick color
    spine = '#AEB8C2'
    for ax in fig.axes:
        # titles / labels
        try: ax.title.set_color(dark)
        except Exception: pass
        try: ax.xaxis.label.set_color(dark)
        except Exception: pass
        try: ax.yaxis.label.set_color(dark)
        except Exception: pass
        # ticks
        try:
            ax.tick_params(colors=dark2)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_color(dark2)
                # small boost to legibility
                if (lab.get_fontweight() in [None, 'normal', 'regular']):
                    lab.set_fontweight('medium')
        except Exception:
            pass
        # spine colors
        for k, sp in ax.spines.items():
            try:
                if k in ('left','bottom'):
                    sp.set_color(spine)
                else:
                    sp.set_color('none')
            except Exception:
                pass
    # all free text annotations / legends / colorbar texts
    for artist in fig.findobj(match=mpl.text.Text):
        try:
            current = artist.get_color()
            if current is None or str(current).lower() in ('#abb5bd','#bbc4cb'):
                artist.set_color(dark2)
            else:
                # keep white text in dark heatmap cells
                if str(current).lower() not in ('white','#ffffff'):
                    artist.set_color(dark)
            # strengthen many regular texts slightly
            fw = artist.get_fontweight()
            if fw in [None, 'normal', 'regular']:
                artist.set_fontweight('medium')
        except Exception:
            pass

def export(fig,path,dpi,a=None):
    if a is not None:
        apply_user_scaling(fig,a)
        apply_text_tone(fig,a)
    fig.savefig(str(path)+'.png',dpi=dpi,bbox_inches='tight',pad_inches=.12)
    fig.savefig(str(path)+'.pdf',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def draw_four(D,xy,out,P,a):
    fig=plt.figure(figsize=(15.6,10.8),facecolor='white')
    grid=fig.add_gridspec(2,2,left=.073,right=.982,bottom=.082,top=.94,
                          hspace=.34,wspace=.25,height_ratios=[1,1])
    fig_a(fig,grid[0,0],D,P);fig_b(fig,grid[0,1],D,P)
    fig_c(fig,grid[1,0],D,P);fig_d(fig,grid[1,1],D,xy,P)
    export(fig,out/'Figure2_4panel',a.dpi,a)


def draw_five(D,xy,out,P,a):
    fig=plt.figure(figsize=(16.2,18.0),facecolor='white')
    grid=fig.add_gridspec(3,2,left=.070,right=.983,bottom=.055,top=.97,
                          hspace=.25,wspace=.24,height_ratios=[.92,.90,1.32])
    fig_a(fig,grid[0,0],D,P);fig_b(fig,grid[0,1],D,P)
    fig_c(fig,grid[1,0],D,P);fig_d(fig,grid[1,1],D,xy,P)
    fig_e(fig,grid[2,:],D,P,a.top_combinations,True)
    export(fig,out/'Figure2_5panel',a.dpi,a)


def draw_upset(D,out,P,a):
    fig=plt.figure(figsize=(16.2,6.8),facecolor='white')
    grid=fig.add_gridspec(1,1,left=.058,right=.989,top=.925,bottom=.10)
    fig_e(fig,grid[0],D,P,top_n=20,letter=False)
    export(fig,out/'Figure2_upset',a.dpi,a)


def main():
    a=cli()
    updates=a.updated_labels
    if updates is not None and str(updates).strip().lower() in ('none','null'):
        updates=None
    a.outdir.mkdir(parents=True,exist_ok=True)
    D=base.compute_all(base.load_data(a.data,updates),a.outdir)
    # If a matching previous cache exists, reuse it, otherwise compute the embedding.
    cache=a.outdir/'fingerprint_tsne_cache.npz'
    if a.tsne_cache.exists() and not cache.exists():
        shutil.copyfile(a.tsne_cache,cache)
    conf=SimpleNamespace(fp_bits=a.fp_bits,tsne_perplexity=a.tsne_perplexity,
                         tsne_iterations=a.tsne_iterations,tsne_seed=a.tsne_seed,
                         force_tsne=False,outdir=a.outdir)
    xy=base.get_tsne(D['df'],conf)
    names=(['pastel_plum','teal_coral'] if a.palette=='both' else
           ['vivid_plum','cobalt_sunset'] if a.palette=='saturated' else
           list(PALETTES) if a.palette=='all' else [a.palette])
    for name in names:
        P=PALETTES[name]
        configure(P)
        out=a.outdir/name;out.mkdir(parents=True,exist_ok=True)
        print('[PLOT]',name,flush=True)
        draw_four(D,xy,out,P,a)
        draw_five(D,xy,out,P,a)
        draw_upset(D,out,P,a)
    settings={'data':str(a.data),'updated_labels':str(updates) if updates else None,
              'palettes':names,'top_n':a.top_combinations,'n_molecules':len(D['df']),
              'font_scale':a.font_scale,'figure_scale':a.figure_scale,'line_scale':a.line_scale,'marker_scale':a.marker_scale,
              'text_tone':a.text_tone,
              'color_palettes':{k:PALETTES[k] for k in names},
              'heatmap':'row-normalized intersection, off-diagonal plotted on 0-30% scale; diagonal masked',
              'UpSet':'exclusive intersections','tsne':'PCA50->tSNE on Morgan2-2048'}
    (a.outdir/'nature_plot_metadata.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[DONE]',a.outdir,flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone ME-BBB Figure 6 plotting (V3.5; KRAS Compound 7 -> 8).

This is a PLOTTING-ONLY script. It reads six verified rows in
Figure6_case_data_v35.csv and uses RDKit for precise molecular depictions.
No model training, TabPFN inference, or BBB prediction is performed.

Dependencies:
    pip install numpy matplotlib pillow rdkit

Usage:
    python plot_Figure6_V35_7to8.py \
        --input Figure6_case_data_v35.csv \
        --out_dir Figure6_V35_7to8 \
        --dpi 450

Outputs:
    Figure6_V35_7to8.png
    Figure6_V35_7to8.pdf
    Figure6_V35_7to8.svg

Colors and formatting chosen for scientific publishing (teal + coral).
Highlighted (peach) atoms are outside one RDKit maximum common substructure;
MCS matching is a graphical annotation, NOT a validated medicinal chemistry SAR.
"""
from __future__ import annotations

import argparse
import csv
from io import BytesIO
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdFMCS, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# Consistent manuscript palette
INK = '#273C4A'
MUTED = '#617A86'
TEAL = '#237E90'
CORAL = '#D36F62'
PEACH = (0.94, 0.70, 0.60)
DIVIDER = '#D7E3E8'
BG = '#FFFFFF'

SERIES = (
    ('TEAD', 'a', 'TEAD inhibitors | RSC Med. Chem. (2026)', 'MSC-4071', 'MSC-4070'),
    ('EGFR', 'b', 'EGFR inhibitors | US patent application (2025)', '14', '15'),
    ('KRAS G12C', 'c', 'KRAS G12C inhibitors | J. Med. Chem. (2026)', '7', '8'),
)
FP = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=False
)


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 11.5,
        'text.color': INK,
        'axes.labelcolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'figure.facecolor': BG,
        'savefig.facecolor': BG,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def read_cases(path: Path):
    expected = {'series', 'compound_id', 'smiles', 'mdck_mdr1_er',
                'mdr1_score_v35', 'bbb_probability_v35'}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {sorted(expected - set(reader.fieldnames or []))}')
        raw = list(reader)
    lookup = {}
    for r in raw:
        k = (r['series'].strip(), r['compound_id'].strip())
        if k in lookup:
            raise ValueError(f'Duplicate compound {k}')
        mol = Chem.MolFromSmiles(r['smiles'])
        if mol is None:
            raise ValueError(f'Invalid SMILES for {k}')
        row = {
            **r,
            'mol': mol,
            'er': float(r['mdck_mdr1_er']),
            'mdr': float(r['mdr1_score_v35']),
            'bbb': float(r['bbb_probability_v35']),
        }
        if row['er'] <= 0 or not (0 <= row['mdr'] <= 1 and 0 <= row['bbb'] <= 1):
            raise ValueError(f'Invalid experimental ER / predicted probability for {k}')
        lookup[k] = row
    for series, _, _, high, low in SERIES:
        for cid in (high, low):
            if (series, cid) not in lookup:
                raise ValueError(f'Missing required case: {series}, {cid}')
    return lookup


def tanimoto(a, b):
    return DataStructs.TanimotoSimilarity(FP.GetFingerprint(a), FP.GetFingerprint(b))


def find_noncommon_atoms(mols):
    """Peach highlight of atoms absent from a largest common substructure match."""
    result = rdFMCS.FindMCS(
        mols, timeout=15, ringMatchesRingOnly=True,
        completeRingsOnly=True, matchChiralTag=False,
    )
    if result.canceled or not result.smartsString:
        return [[], []]
    q = Chem.MolFromSmarts(result.smartsString)
    if q is None:
        return [[], []]
    highlights = []
    for mol in mols:
        matched = mol.GetSubstructMatch(q)
        highlights.append([i for i in range(mol.GetNumAtoms()) if i not in matched])
    return highlights


def pair_structure_images(mol_left, mol_right, size=(1150, 700)):
    mols = [Chem.Mol(mol_left), Chem.Mol(mol_right)]
    rdDepictor.Compute2DCoords(mols[0])
    # Align MCS where possible; fall back to a conventional RDKit depiction.
    try:
        rdDepictor.GenerateDepictionMatching2DStructure(mols[1], mols[0], acceptFailure=True)
    except (RuntimeError, ValueError):
        rdDepictor.Compute2DCoords(mols[1])
    highlights = find_noncommon_atoms(mols)
    imgs = []
    for mol, noncommon in zip(mols, highlights):
        draw = rdMolDraw2D.MolDraw2DCairo(*size)
        opts = draw.drawOptions()
        opts.padding = 0.050
        opts.bondLineWidth = 2.0
        opts.highlightBondWidthMultiplier = 5
        opts.useBWAtomPalette = False
        draw.DrawMolecule(
            mol,
            highlightAtoms=noncommon,
            highlightAtomColors={i: PEACH for i in noncommon},
            highlightAtomRadii={i: 0.22 for i in noncommon},
        )
        draw.FinishDrawing()
        imgs.append(Image.open(BytesIO(draw.GetDrawingText())).convert('RGB'))
    return imgs


def add_molecule_axes(fig, image, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(image, interpolation='antialiased', aspect='equal')
    ax.set_axis_off()


def make_figure(rows, output: Path, dpi: int, formats: tuple[str, ...]):
    setup_style()
    fig = plt.figure(figsize=(16.0, 10.0), facecolor='white')
    row_step = 0.318
    top0 = 0.972
    for i, (series, letter, title, from_id, to_id) in enumerate(SERIES):
        a, b = rows[(series, from_id)], rows[(series, to_id)]
        top = top0 - i * row_step
        if i:
            fig.add_artist(Line2D([0.025, 0.974], [top + 0.020, top + 0.020],
                                  transform=fig.transFigure, color=DIVIDER, lw=1.05))
        fig.text(0.030, top + 0.006, letter, fontsize=19, weight='bold', va='top')
        fig.text(0.068, top + 0.002, title,
                 fontsize=13.7, weight='bold', va='top', color=INK)
        similarity = tanimoto(a['mol'], b['mol'])
        fig.text(0.068, top - 0.031, f'ECFP4 Tanimoto = {similarity:.3f}',
                 fontsize=10.5, va='top', color=MUTED, weight='medium')

        left_img, right_img = pair_structure_images(a['mol'], b['mol'])
        y_img = top - 0.246
        h_img = 0.195
        left_x, right_x, mol_w = 0.047, 0.284, 0.211
        add_molecule_axes(fig, left_img, left_x, y_img, mol_w, h_img)
        add_molecule_axes(fig, right_img, right_x, y_img, mol_w, h_img)
        fig.text(0.266, y_img + h_img / 2, '→', fontsize=22,
                 ha='center', va='center', color=INK)
        from_name = from_id if series == 'TEAD' else f'Compound {from_id}'
        to_name = to_id if series == 'TEAD' else f'Compound {to_id}'
        fig.text(left_x + mol_w / 2, y_img - 0.009, from_name,
                 fontsize=11.6, weight='bold', ha='center', va='top', color=CORAL)
        fig.text(right_x + mol_w / 2, y_img - 0.009, to_name,
                 fontsize=11.6, weight='bold', ha='center', va='top', color=TEAL)

        # Numeric grid keeps the old/new columns aligned across all three pairs.
        ax = fig.add_axes([0.532, top - 0.252, 0.425, 0.205])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 3)
        ax.axis('off')
        er_fold = a['er'] / b['er']
        metrics = [
            ('MDCK–MDR1 ER', f"{a['er']:g}", f"{b['er']:g}", f'{er_fold:.1f}× lower'),
            ('MDR1 expert score', f"{a['mdr']:.3f}", f"{b['mdr']:.3f}", f"Δ = {b['mdr'] - a['mdr']:+.3f}"),
            ('ME-BBB predicted P(BBB+)', f"{a['bbb']:.3f}", f"{b['bbb']:.3f}", f"Δ = {b['bbb'] - a['bbb']:+.3f}"),
        ]
        for j, (name, before, after, delta) in enumerate(metrics):
            y = 2.53 - j * 1.00
            ax.text(0.005, y, name, fontsize=11.4, fontweight='bold', color=INK,
                    ha='left', va='center')
            ax.text(0.564, y, before, fontsize=13.4, fontweight='bold', color=CORAL,
                    ha='right', va='center')
            ax.text(0.612, y, '→', fontsize=13.5, color=INK, ha='center', va='center')
            ax.text(0.660, y, after, fontsize=13.4, fontweight='bold', color=TEAL,
                    ha='left', va='center')
            ax.text(0.990, y, delta, fontsize=10.25, color=MUTED, ha='right', va='center')
            if j < 2:
                ax.axhline(y - 0.5, color=DIVIDER, linewidth=0.75, alpha=.7)

    output.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        target = output / f'Figure6_V35_7to8.{ext}'
        fig.savefig(target, dpi=dpi, bbox_inches='tight', pad_inches=0.10, facecolor='white')
        print('Saved', target)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path,
                   default=REPO_ROOT/'data'/'derived'/'figure6'/'case_data_v35.csv')
    p.add_argument('--out_dir', type=Path,
                   default=REPO_ROOT/'outputs'/'figures'/'figure6')
    p.add_argument('--dpi', type=int, default=450)
    p.add_argument('--formats', nargs='+', choices=['png','pdf','svg'],
                   default=['png','pdf','svg'])
    args = p.parse_args()
    if args.dpi < 120:
        p.error('--dpi should be >=120')
    rows = read_cases(args.input)
    make_figure(rows, args.out_dir, args.dpi, tuple(args.formats))
    print(f'Figure 6 completed. RDKit {rdBase.rdkitVersion}. KRAS pair fixed at Compound 7 -> 8.')


if __name__ == '__main__':
    main()

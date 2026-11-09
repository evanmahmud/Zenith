#!/usr/bin/env python3
"""Recover and verify the mapping between the BraTS 2020 HDF5 slice export and the original NIfTI volumes.

The HDF5 export (files named volume_<k>_slice_<s>.h5, each holding an `image` array of shape (240, 240, 4) and a
`mask` array of shape (240, 240, 3)) does not document which NIfTI patient a volume comes from, the orientation of
the slices, the order of the image channels or the meaning of the mask channels. This script recovers all four from
the data and checks every patient:

1. Patient correspondence. The identity volume_k = BraTS20_Training_k is tested on 8 random patients. If it fails,
   volumes are matched to patients by their per-slice tumour-area signatures (Hungarian assignment).
2. Geometry and mask channels. For the sampled patients the 16 candidate orientations (8 in-plane symmetries of the
   square, with and without reversing the slice order) are searched. The 3 mask channels are assigned one to one to
   either the BraTS labels {1, 2, 4} or the nested regions (ET, TC, WT), whichever reproduces the NIfTI labels best.
   Every patient is then checked with the common orientation, and the 16 orientations are searched again for any
   patient that does not fit. A patient is accepted only if its HDF5 masks reproduce the NIfTI labels (Dice >= 0.99).
3. Image channels. Channels are assigned one to one to the modalities on Pearson correlations computed per axial
   slice inside the brain, so any per-slice or per-volume rescaling of the HDF5 intensities does not matter. The
   order is decided from the evidence of all patients pooled; a single patient keeps its own order only if every
   matched correlation is at least 0.9 and its own match beats the pooled order by at least 0.2.

Outputs
    --out     JSON with the full mapping. It has the same structure as index/h5_nii_alignment_v2.json written by the
              notebook, so it can be dropped into that folder to skip the check there.
    --report  short Markdown summary.
    --csv     one row per patient.

Example
    python recover_h5_mapping.py \
        --nii-root ".../MICCAI_BraTS2020_TrainingData" \
        --h5-root ".../BraTS2020_training_data/content/data" \
        --out h5_nii_alignment_v2.json --report h5_mapping_report.md --csv h5_mapping_per_patient.csv --workers 4

To use the mapping elsewhere, load the JSON and call `canonical_slice` on the arrays of one HDF5 file, and
`nifti_z` to find the matching NIfTI slice index.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment

MODALITIES = ('flair', 't1', 't1ce', 't2')          # canonical channel order used by the notebook
REGIONS = ('ET', 'TC', 'WT')
LABEL_NAMES = ('label 1 (necrotic and non-enhancing core)', 'label 2 (oedema)', 'label 4 (enhancing tumour)')
INPLANE_NAMES = {0: 'identity', 1: 'flip axis 0', 2: 'flip axis 1', 3: 'flip both axes (180 degree rotation)',
                 4: 'transpose', 5: 'transpose, then flip axis 0', 6: 'transpose, then flip axis 1',
                 7: 'transpose, then flip both axes'}
DICE_MIN = 0.99

# ----------------------------------------------------------------------------------------------------------------
# file discovery
# ----------------------------------------------------------------------------------------------------------------
_MOD_RE = re.compile(r'_(flair|t1ce|t1|t2)\.nii(\.gz)?$', re.I)


def resolve_patient_files(pdir):
    """Modality and label files of one patient folder. Handles BraTS20_Training_355, whose label file is named
    W39_1998.09.19_Segm.nii in some mirrors."""
    pdir = Path(pdir)
    files = {}
    niis = [f for f in pdir.iterdir() if f.name.lower().endswith(('.nii', '.nii.gz'))]
    for f in niis:
        m = _MOD_RE.search(f.name)
        if m:
            files[m.group(1).lower()] = str(f)
    segs = [f for f in niis if re.search(r'_seg\.nii(\.gz)?$', f.name, re.I)]
    if not segs:
        segs = [f for f in niis if re.search(r'segm?', f.name, re.I) and not _MOD_RE.search(f.name)]
    if segs:
        files['seg'] = str(segs[0])
    return files


def discover_nii_patients(root):
    out = {}
    for pdir in sorted(Path(root).iterdir()):
        m = re.match(r'BraTS20_(Training|Validation)_(\d{3})$', pdir.name)
        if pdir.is_dir() and m:
            out[int(m.group(2))] = resolve_patient_files(pdir)
    return out


def discover_h5_slices(root):
    out = defaultdict(dict)
    with os.scandir(root) as it:
        for e in it:
            m = re.match(r'volume_(\d+)_slice_(\d+)\.h5$', e.name)
            if m:
                out[int(m.group(1))][int(m.group(2))] = e.path
    return dict(out)


# ----------------------------------------------------------------------------------------------------------------
# array helpers
# ----------------------------------------------------------------------------------------------------------------
def load_nii(path, dtype=np.float32):
    return np.asarray(nib.load(str(path)).dataobj, dtype=dtype)


def load_seg(path):
    seg = load_nii(path).round().astype(np.uint8)
    seg[seg == 3] = 4                                  # some mirrors use 3 for the enhancing tumour
    return seg


def labels_to_regions(lab):
    """BraTS labels {0, 1, 2, 4} -> (3, ...) boolean regions ET, TC, WT."""
    et = lab == 4
    tc = et | (lab == 1)
    wt = lab > 0
    return np.stack([et, tc, wt]).astype(bool)


def _h5_keys(f):
    keys = list(f.keys())
    img_k = 'image' if 'image' in keys else max(keys, key=lambda k: f[k].shape[-1] == 4)
    msk_k = 'mask' if 'mask' in keys else [k for k in keys if k != img_k][0]
    return img_k, msk_k


def read_h5_slice(path):
    with h5py.File(path, 'r') as f:
        ik, mk = _h5_keys(f)
        return np.asarray(f[ik], np.float32), np.asarray(f[mk], np.float32)


def load_h5_stack(slices):
    z = max(slices) + 1
    img = np.zeros((240, 240, z, 4), np.float32)
    msk = np.zeros((240, 240, z, 3), np.float32)
    for s, p in slices.items():
        img[:, :, s], msk[:, :, s] = read_h5_slice(p)
    return img, msk


def apply_inplane(a, code):
    """One of the 8 symmetries of the square on axes (0, 1): bit 4 = transpose, bit 1 = flip axis 0,
    bit 2 = flip axis 1 (applied in this order)."""
    if code & 4:
        a = np.swapaxes(a, 0, 1)
    if code & 1:
        a = a[::-1]
    if code & 2:
        a = a[:, ::-1]
    return a


def dice_np(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def modality_similarity(ti, vols, brain, min_vox=500):
    """(HDF5 channel x NIfTI modality) Pearson correlation inside the brain, per axial slice, averaged over slices."""
    S, n = np.zeros((ti.shape[-1], len(vols))), 0
    for z in range(brain.shape[2]):
        b = brain[:, :, z]
        if b.sum() < min_vox:
            continue
        a = ti[:, :, z][b].astype(np.float64)
        v = np.stack([vol[:, :, z][b] for vol in vols], 1).astype(np.float64)
        a = (a - a.mean(0)) / (a.std(0) + 1e-8)
        v = (v - v.mean(0)) / (v.std(0) + 1e-8)
        S += a.T @ v / len(a)
        n += 1
    return S / max(n, 1)


# ----------------------------------------------------------------------------------------------------------------
# alignment of one volume
# ----------------------------------------------------------------------------------------------------------------
def align_volume(vid, h5_slices, pid, nii_files, code=None, zflip=None):
    """Geometry, mask-channel and modality maps of HDF5 volume `vid` against NIfTI patient `pid`.
    With `code` and `zflip` given only that orientation is scored; otherwise all 16 are searched."""
    img, msk = load_h5_stack(h5_slices)
    seg = load_seg(nii_files['seg'])
    union = msk.max(-1) > 0.5
    combos = [(code, zflip)] if code is not None else [(c, z) for c in range(8) for z in (False, True)]
    best = None
    for c_, z_ in combos:
        m = apply_inplane(union, c_)
        m = m[:, :, ::-1] if z_ else m
        if m.shape == seg.shape:
            d = dice_np(m, seg > 0)
            if best is None or d > best[0]:
                best = (d, c_, z_)
    if best is None:
        return dict(vid=vid, pid=pid, dice=0.0, code=-1, zflip=False, mask_kind='none', mask_score=0.0,
                    mask_map=[0, 1, 2], mod_map=[0, 1, 2, 3], mod_r=np.zeros((4, 4)).tolist(), mod_min_r=0.0)
    d, code, zflip = best

    def T(a):
        a = apply_inplane(a, code)
        return a[:, :, ::-1] if zflip else a

    ref = {'labels': [seg == 1, seg == 2, seg == 4], 'regions': list(labels_to_regions(seg))}
    best_kind = None
    for kind, targets in ref.items():
        md = np.array([[dice_np(T(msk[..., c] > 0.5), t) for t in targets] for c in range(3)])
        _, mp = linear_sum_assignment(-md)
        score = float(md[range(3), mp].min())
        if best_kind is None or score > best_kind[1]:
            best_kind = (kind, score, mp.tolist())
    vols = [load_nii(nii_files[m]) for m in MODALITIES]
    brain = (np.stack(vols) != 0).any(0)
    S = modality_similarity(T(img), vols, brain)
    _, mod_map = linear_sum_assignment(-S)
    return dict(vid=int(vid), pid=int(pid), dice=float(d), code=int(code), zflip=bool(zflip), mask_kind=best_kind[0],
                mask_score=best_kind[1], mask_map=best_kind[2], mod_map=mod_map.tolist(), mod_r=S.round(4).tolist(),
                mod_min_r=float(S[range(4), mod_map].min()))


def _align_item(item):
    return align_volume(*item)


def _verify_item(item):
    """Check one patient with the common orientation; search all 16 orientations only if it does not fit."""
    vid, h5_slices, pid, nii_files, code, zflip = item
    out = align_volume(vid, h5_slices, pid, nii_files, code, zflip)
    if out['dice'] < DICE_MIN or out['mask_score'] < DICE_MIN:
        out = align_volume(vid, h5_slices, pid, nii_files)
    return out


def _h5_tumour_area(slices):
    area = np.zeros(155)
    for s, p in slices.items():
        if s < 155:
            _, msk = read_h5_slice(p)
            area[s] = int((msk > 0.5).any(-1).sum())
    return area


def _nii_tumour_area(seg_path):
    return (load_seg(seg_path) > 0).sum((0, 1)).astype(float)


def match_by_signature(h5_files, nii_files, pool_map):
    """Hungarian matching of per-slice tumour-area signatures (independent of the in-plane orientation)."""
    vids, pids = sorted(h5_files), sorted(nii_files)
    A = np.array(pool_map(_h5_tumour_area, [h5_files[v] for v in vids]))
    B = np.array(pool_map(_nii_tumour_area, [nii_files[p]['seg'] for p in pids]))
    cost = np.minimum(np.abs(A[:, None] - B[None]).sum(-1), np.abs(A[:, None] - B[None, :, ::-1]).sum(-1))
    r, c = linear_sum_assignment(cost)
    return {vids[i]: pids[j] for i, j in zip(r, c)}


def choose_modality_maps(per, exclude=()):
    """Pooled one-to-one channel order; a patient keeps its own order only if its own evidence is decisive."""
    ok = [o for o in per.values() if o['pid'] not in exclude]
    S = np.sum([np.array(o['mod_r']) for o in ok], 0)
    _, pooled = linear_sum_assignment(-S)
    final = {}
    for o in ok:
        R = np.array(o['mod_r'])
        own = np.array(o['mod_map'])
        decisive = (R[range(4), own].min() >= 0.9 and R[range(4), own].sum() - R[range(4), pooled].sum() >= 0.2)
        final[o['vid']] = own.tolist() if decisive else pooled.tolist()
    return pooled.tolist(), final


# ----------------------------------------------------------------------------------------------------------------
# using the mapping
# ----------------------------------------------------------------------------------------------------------------
def load_mapping(path):
    M = json.loads(Path(path).read_text())
    M['vid2pid'] = {int(k): int(v) for k, v in M['vid2pid'].items()}
    M['per_volume'] = {int(k): v for k, v in M['per_volume'].items()}
    return M


def canonical_slice(img, msk, entry):
    """Map the arrays of one HDF5 file to the NIfTI frame.

    img (240, 240, 4) and msk (240, 240, 3) are the arrays stored in the file; entry = mapping['per_volume'][vid].
    Returns the image as (4, 240, 240) in the order FLAIR, T1, T1ce, T2 (intensities unchanged) and the masks as
    (3, 240, 240) booleans ET, TC, WT, both in the in-plane orientation of the NIfTI arrays."""
    inv_mod = np.argsort(entry.get('mod_map_used', entry['mod_map']))
    inv_mask = np.argsort(entry['mask_map'])
    im = apply_inplane(img, entry['code'])[..., inv_mod]
    mk = apply_inplane(msk > 0.5, entry['code'])[..., inv_mask]
    if entry['mask_kind'] == 'regions':
        reg = mk.transpose(2, 0, 1).astype(bool)
    else:
        lab = np.zeros(mk.shape[:2], np.uint8)
        for j, l in enumerate((1, 2, 4)):
            lab[mk[..., j]] = l
        reg = labels_to_regions(lab)
    return np.ascontiguousarray(im.transpose(2, 0, 1)), np.ascontiguousarray(reg)


def nifti_z(s, entry, n_z=155):
    """NIfTI axial index of HDF5 slice s."""
    return n_z - 1 - s if entry['zflip'] else s


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------
def mapping_report(M):
    pv = list(M['per_volume'].values())
    geo = Counter((o['code'], o['zflip']) for o in pv)
    used = Counter(tuple(o.get('mod_map_used', o['mod_map'])) for o in pv if o['pid'] not in M['excluded'])
    names = LABEL_NAMES if M['mask_kind'] == 'labels' else tuple(f'region {g}' for g in REGIONS)
    dice = np.array([o['dice'] for o in pv]); ms = np.array([o['mask_score'] for o in pv])
    mr = np.array([o['mod_min_r'] for o in pv])
    L = ['# HDF5 slice export: recovered mapping to the NIfTI volumes', '',
         f'* Patients checked: {len(pv)}; excluded: {M["excluded"] or "none"}.',
         f'* Patient correspondence: {M["how"]}.',
         f'* In-plane transform applied to every HDF5 slice array (axes 0, 1): code {M["code"]} = '
         f'{INPLANE_NAMES[M["code"]]}; slice order reversed along z: {"yes" if M["zflip"] else "no"} '
         f'(HDF5 slice s is NIfTI z = {"154 - s" if M["zflip"] else "s"}).',
         f'* Transforms found per patient (code, z flip): {dict((f"{c}, {z}", n) for (c, z), n in geo.items())}.',
         '* Image channels (one-to-one assignment on per-slice Pearson correlations inside the brain): '
         + ', '.join(f'channel {c} = {MODALITIES[m].upper()}' for c, m in enumerate(M['mod_map'])) + '.',
         f'* Channel orders used per patient: {dict((str(k), v) for k, v in used.items())}.',
         f'* Mask channels hold {M["mask_kind"]}: '
         + ', '.join(f'channel {c} = {names[l]}' for c, l in enumerate(M['mask_map'])) + '.',
         f'* Verification of every patient: Dice between the HDF5 tumour mask and the NIfTI label > 0 is min '
         f'{dice.min():.4f}, median {np.median(dice):.4f}; worst per-channel mask Dice min {ms.min():.4f}.',
         f'* Matched modality correlation (worst channel per patient): min {mr.min():.3f}, median {np.median(mr):.3f}.',
         f'* Acceptance rule: a patient is used only if its HDF5 masks reproduce the NIfTI labels with Dice >= {DICE_MIN}.']
    return '\n'.join(L) + '\n'


def write_csv(M, path):
    cols = ['vid', 'pid', 'excluded', 'code', 'zflip', 'dice', 'mask_kind', 'mask_score', 'mask_map', 'mod_map',
            'mod_map_used', 'mod_min_r']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for vid in sorted(M['per_volume']):
            o = M['per_volume'][vid]
            w.writerow([o['vid'], o['pid'], o['pid'] in M['excluded'], o['code'], o['zflip'], f"{o['dice']:.4f}",
                        o['mask_kind'], f"{o['mask_score']:.4f}", ' '.join(map(str, o['mask_map'])),
                        ' '.join(map(str, o['mod_map'])), ' '.join(map(str, o.get('mod_map_used', []))),
                        f"{o['mod_min_r']:.4f}"])


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------
def recover(nii_root, h5_root, workers=4, seed=42, n_check=8, limit=None):
    t0 = time.time()
    nii = discover_nii_patients(nii_root)
    needed = set(MODALITIES) | {'seg'}
    missing = {p: sorted(needed - set(f)) for p, f in nii.items() if not needed <= set(f)}
    if missing:
        raise SystemExit(f'incomplete NIfTI patients: {missing}')
    h5 = discover_h5_slices(h5_root)
    print(f'{len(nii)} NIfTI patients, {len(h5)} HDF5 volumes, {sum(len(s) for s in h5.values())} slices')
    if limit:
        keep = sorted(set(h5) & set(nii))[:limit]
        h5 = {v: h5[v] for v in keep}
        nii = {p: nii[p] for p in keep}

    pool = Pool(workers) if workers > 0 else None

    def pmap(fn, items):
        return pool.map(fn, items, chunksize=1) if pool else [fn(i) for i in items]

    rng = np.random.default_rng(seed)
    common = sorted(set(h5) & set(nii))
    sample = sorted(rng.choice(common, min(n_check, len(common)), replace=False).tolist())
    checks = pmap(_align_item, [(v, h5[v], v, nii[v]) for v in sample])
    if np.median([c['dice'] for c in checks]) >= DICE_MIN:
        vid2pid, how = {v: v for v in common}, 'identity (volume_k == BraTS20_Training_k)'
    else:
        print('identity correspondence rejected; matching volumes to patients by tumour-area signatures')
        vid2pid, how = match_by_signature(h5, nii, pmap), 'tumour-area signature matching'
        sample = sorted(rng.choice(sorted(vid2pid), min(n_check, len(vid2pid)), replace=False).tolist())
        checks = pmap(_align_item, [(v, h5[v], vid2pid[v], nii[vid2pid[v]]) for v in sample])
    good = [c for c in checks if c['dice'] >= DICE_MIN and c['mask_score'] >= DICE_MIN]
    if len(good) <= len(checks) // 2:
        raise SystemExit(f'HDF5 and NIfTI masks disagree for most sampled patients: {checks}')
    code0, zflip0 = Counter((c['code'], c['zflip']) for c in good).most_common(1)[0][0]
    print(f'common orientation: code {code0} ({INPLANE_NAMES[code0]}), z reversed: {zflip0}; checking every patient')
    per = {o['vid']: o for o in pmap(_verify_item, [(v, h5[v], p, nii[p], code0, zflip0) for v, p in sorted(vid2pid.items())])}
    if pool:
        pool.close()
    excluded = sorted({o['pid'] for o in per.values() if o['dice'] < DICE_MIN or o['mask_score'] < DICE_MIN}
                      | (set(nii) - set(vid2pid.values())))
    pooled, final = choose_modality_maps(per, excluded)
    for vid, mm in final.items():
        per[vid]['mod_map_used'] = mm
    M = dict(how=how, vid2pid={int(k): int(v) for k, v in vid2pid.items()}, checks=checks, code=int(code0),
             zflip=bool(zflip0), mod_map=pooled, excluded=[int(p) for p in excluded], per_volume=per,
             mask_kind=Counter(o['mask_kind'] for o in per.values()).most_common(1)[0][0],
             mask_map=list(Counter(tuple(o['mask_map']) for o in per.values()).most_common(1)[0][0]))
    print(f'done in {time.time() - t0:.0f} s')
    return M


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--nii-root', required=True, help='folder with the BraTS20_Training_xxx patient folders')
    ap.add_argument('--h5-root', required=True, help='folder with the volume_<k>_slice_<s>.h5 files')
    ap.add_argument('--out', default='h5_nii_alignment_v2.json', help='mapping JSON (same structure as the notebook cache)')
    ap.add_argument('--report', default='h5_mapping_report.md', help='Markdown summary')
    ap.add_argument('--csv', default='h5_mapping_per_patient.csv', help='one row per patient')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1), help='processes (0 = no pool)')
    ap.add_argument('--seed', type=int, default=42, help='seed for the 8 patients used to find the common orientation')
    ap.add_argument('--limit', type=int, default=None, help='check only the first N patients (quick test)')
    a = ap.parse_args(argv)
    M = recover(a.nii_root, a.h5_root, a.workers, a.seed, limit=a.limit)
    Path(a.out).write_text(json.dumps(M, indent=1, default=int))
    txt = mapping_report(M)
    Path(a.report).write_text(txt)
    write_csv(M, a.csv)
    print(txt)
    if M['excluded']:
        print(f'WARNING: {len(M["excluded"])} patient(s) failed the check and are listed as excluded: {M["excluded"]}')
    print(f'wrote {a.out}, {a.report}, {a.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

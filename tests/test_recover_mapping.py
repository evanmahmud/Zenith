"""Test of the standalone recover_h5_mapping.py on synthetic data with a known hidden layout.

usage: python test_recover_mapping.py <path to recover_h5_mapping.py> <synthetic_root> <labels|regions> [notebook_mapping.json]

<synthetic_root> is made by make_synthetic.py (labels: in-plane transpose, channels t1, t1ce, t2, flair, masks
NCR, ED, ET; regions: flip of axis 0, reversed slice order, channels t2, flair, t1ce, t1, masks WT, ET, TC).
The script must find that layout, every canonical slice must reproduce the NIfTI arrays, and, if the mapping file
written by the notebook is given, both mappings must be identical.
"""
import importlib.util, json, sys
import numpy as np

script, root, mode = sys.argv[1:4]
spec = importlib.util.spec_from_file_location('recover', script); R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
nii_root = f'{root}/nii/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
h5_root = f'{root}/h5/BraTS2020_training_data/content/data'
M = R.recover(nii_root, h5_root, workers=0)
M = json.loads(json.dumps(M, default=int))
M = dict(M, vid2pid={int(k): int(v) for k, v in M['vid2pid'].items()}, per_volume={int(k): v for k, v in M['per_volume'].items()})
expected = {'labels': dict(code=4, zflip=False, mod_map=[1, 2, 3, 0], mask_kind='labels', mask_map=[0, 1, 2]),
            'regions': dict(code=1, zflip=True, mod_map=[3, 0, 2, 1], mask_kind='regions', mask_map=[2, 0, 1])}[mode]
found = {k: M[k] for k in expected}
print('expected', expected); print('found   ', found)
assert found == expected and not M['excluded'], 'wrong mapping'
nii, h5 = R.discover_nii_patients(nii_root), R.discover_h5_slices(h5_root)
worst_mask, worst_r = 1.0, 1.0
for vid, e in M['per_volume'].items():
    pid = M['vid2pid'][vid]
    vols = [R.load_nii(nii[pid][m]) for m in R.MODALITIES]; seg = R.load_seg(nii[pid]['seg'])
    for s in range(20, 140, 15):
        img, msk = R.read_h5_slice(h5[vid][s])
        im, reg = R.canonical_slice(img, msk, e); z = R.nifti_z(s, e)
        worst_mask = min(worst_mask, float((reg == R.labels_to_regions(seg[:, :, z])).mean()))
        b = (np.stack([v[:, :, z] for v in vols]) != 0).any(0)
        if b.sum() > 500:
            worst_r = min(worst_r, min(np.corrcoef(im[m][b], vols[m][:, :, z][b])[0, 1] for m in range(4)))
print(f'worst pixel agreement of the masks {worst_mask:.4f} | worst image correlation {worst_r:.4f}')
assert worst_mask == 1.0 and worst_r > 0.999, 'canonical slices do not reproduce the NIfTI arrays'
if len(sys.argv) > 4:
    A = json.load(open(sys.argv[4]))
    A['per_volume'] = {int(k): v for k, v in A['per_volume'].items()}
    same = all(A[k] == M[k] for k in ('how', 'code', 'zflip', 'mod_map', 'mask_kind', 'mask_map', 'excluded')) and \
        all(A['per_volume'][k] == M['per_volume'][k] for k in A['per_volume'])
    print('identical to the notebook mapping:', same)
    assert same, 'the script and the notebook disagree'
print('PASSED')

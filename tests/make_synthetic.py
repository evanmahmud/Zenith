"""Synthetic BraTS-like data in BOTH Kaggle layouts, for an end-to-end smoke test of the notebook.

NIfTI: <root>/nii/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_XXX/*_{flair,t1,t1ce,t2,seg}.nii
       (patient 355's label file is named W39_1998.09.19_Segm.nii, as in some mirrors)
HDF5 : <root>/h5/BraTS2020_training_data/content/data/volume_K_slice_S.h5  ('image' (240,240,4), 'mask' (240,240,3))
       with a hidden geometry the notebook must recover: in-plane transpose, channel order (t1, t1ce, t2, flair),
       a different intensity scaling, and mask channels (NCR, ED, ET).
"""
import sys, os
from pathlib import Path
import numpy as np, nibabel as nib, h5py, pandas as pd

# usage: python make_synthetic.py <root> [labels|regions] [n_patients (default 7; use 14 for the full smoke test)]
#   labels  (default): HDF5 masks hold labels (NCR, ED, ET); hidden in-plane transpose; channels (t1, t1ce, t2, flair)
#   regions          : HDF5 masks hold nested regions (WT, ET, TC); hidden flip + z-flip; channels (t2, flair, t1ce, t1);
#                      intensities shifted so the background is non-zero
root = Path(sys.argv[1]); MODE = sys.argv[2] if len(sys.argv) > 2 else 'labels'
N_PAT = int(sys.argv[3]) if len(sys.argv) > 3 else 7          # 14 patients give 2 validation patients per repeat,
pids = [1, 2, 3, 4, 5, 6, 355][:N_PAT] if N_PAT <= 7 else list(range(1, N_PAT)) + [355]   # so the held-out check runs
nii_root = root / 'nii/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
h5_root = root / 'h5/BraTS2020_training_data/content/data'
nii_root.mkdir(parents=True, exist_ok=True); h5_root.mkdir(parents=True, exist_ok=True)
X, Y, Z = 240, 240, 155
gx, gy, gz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij')

def smooth_noise(rng, scale=8):
    small = rng.normal(size=(X // scale + 2, Y // scale + 2, Z // scale + 2))
    from scipy.ndimage import zoom
    return zoom(small, scale, order=1)[:X, :Y, :Z]

for pid in pids:
    rng = np.random.default_rng(pid)
    c = np.array([120, 125, 75]) + rng.integers(-6, 7, 3); r = np.array([68, 84, 58]) + rng.integers(-5, 6, 3)
    brain = ((gx - c[0]) / r[0]) ** 2 + ((gy - c[1]) / r[1]) ** 2 + ((gz - c[2]) / r[2]) ** 2 <= 1
    tc_ = c + rng.integers(-25, 26, 3) * np.array([1, 1, 0.6])
    d = np.sqrt((gx - tc_[0]) ** 2 + ((gy - tc_[1]) / 1.2) ** 2 + (gz - tc_[2]) ** 2) + 2 * smooth_noise(rng, 16)
    seg = np.zeros((X, Y, Z), np.uint8)
    seg[(d < 20) & brain] = 2; seg[(d < 11) & brain] = 4; seg[(d < 7) & brain] = 1
    tissue = smooth_noise(rng, 8)
    base = {'flair': 300, 't1': 500, 't1ce': 450, 't2': 400}
    contrast = {'flair': {1: 150, 2: 350, 4: 250}, 't1': {1: -250, 2: -120, 4: -60},
                't1ce': {1: -150, 2: -40, 4: 500}, 't2': {1: 400, 2: 300, 4: 200}}
    vols = {}
    for m in base:
        v = base[m] + 60 * tissue + rng.normal(0, 25, (X, Y, Z))
        for lab, dv in contrast[m].items():
            v[seg == lab] += dv
        vols[m] = np.where(brain, np.clip(v, 1, None), 0).astype(np.int16)
    pdir = nii_root / f'BraTS20_Training_{pid:03d}'; pdir.mkdir(exist_ok=True)
    aff = np.diag([-1.0, -1.0, 1.0, 1.0])
    for m, v in vols.items():
        nib.save(nib.Nifti1Image(v, aff), str(pdir / f'BraTS20_Training_{pid:03d}_{m}.nii'))
    seg_name = 'W39_1998.09.19_Segm.nii' if pid == 355 else f'BraTS20_Training_{pid:03d}_seg.nii'
    nib.save(nib.Nifti1Image(seg, aff), str(pdir / seg_name))
    # HDF5 slices with a hidden geometry
    for s in range(Z):
        if MODE == 'regions':
            z = Z - 1 - s
            img = np.stack([vols[m][:, :, z].astype(np.float32) for m in ('t2', 'flair', 't1ce', 't1')], -1)[::-1]
            img = (img - 200.0) / 300.0
            lab = seg[:, :, z][::-1]
            msk = np.stack([lab > 0, lab == 4, (lab == 1) | (lab == 4)], -1).astype(np.uint8)
        else:
            img = np.stack([vols[m][:, :, s].astype(np.float32) / 1000.0 for m in ('t1', 't1ce', 't2', 'flair')], -1).transpose(1, 0, 2)
            lab = seg[:, :, s].T
            msk = np.stack([lab == 1, lab == 2, lab == 4], -1).astype(np.uint8)
        with h5py.File(h5_root / f'volume_{pid}_slice_{s}.h5', 'w') as f:
            f.create_dataset('image', data=img, compression='gzip', compression_opts=1)
            f.create_dataset('mask', data=msk, compression='gzip', compression_opts=1)
    print('patient', pid, 'tumour voxels', int((seg > 0).sum()), flush=True)

pd.DataFrame(dict(Brats20ID=[f'BraTS20_Training_{p:03d}' for p in pids], Age=np.linspace(40, 70, len(pids)).round(1),
                  Survival_days=np.arange(len(pids)) * 100 + 150, Extent_of_Resection='GTR')).to_csv(h5_root / 'survival_info.csv', index=False)
print('done')

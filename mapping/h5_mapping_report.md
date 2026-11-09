# HDF5 slice export: recovered mapping to the NIfTI volumes

* Patients checked: 369; excluded: none.
* Patient correspondence: identity (volume_k == BraTS20_Training_k).
* In-plane transform applied to every HDF5 slice array (axes 0, 1): code 0 = identity; slice order reversed along z: no (HDF5 slice s is NIfTI z = s).
* Transforms found per patient (code, z flip): {'0, False': 369}.
* Image channels (one-to-one assignment on per-slice Pearson correlations inside the brain): channel 0 = FLAIR, channel 1 = T1, channel 2 = T1CE, channel 3 = T2.
* Channel orders used per patient: {'(0, 1, 2, 3)': 369}.
* Mask channels hold labels: channel 0 = label 1 (necrotic and non-enhancing core), channel 1 = label 2 (oedema), channel 2 = label 4 (enhancing tumour).
* Verification of every patient: Dice between the HDF5 tumour mask and the NIfTI label > 0 is min 1.0000, median 1.0000; worst per-channel mask Dice min 1.0000.
* Matched modality correlation (worst channel per patient): min 0.936, median 1.000.
* Acceptance rule: a patient is used only if its HDF5 masks reproduce the NIfTI labels with Dice >= 0.99.

The same procedure is available as a standalone script, `recover_h5_mapping.py`.

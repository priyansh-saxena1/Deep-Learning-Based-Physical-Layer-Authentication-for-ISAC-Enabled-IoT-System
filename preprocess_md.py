"""
Build a micro-Doppler dataset from the DISC `micro_doppler_stft` signatures.

WHY THIS EXISTS
---------------
`data_preprocessing.py` treats the raw CIR array as an image. Per the DISC dataset
description, that array is (n_beam_patterns, n_packets, n_range_bins) -- so axis 1 is
TIME (packets at 0.27 ms), not a feature axis. The existing pipeline then

  * zero-pads / truncates the TIME axis to D_target = 31264 (median length ~18000,
    minimum 64), leaving 39.3% of every tensor as padding, and
  * block-averages 4 CONSECUTIVE PACKETS together.

Averaging across packets low-pass filters the phase evolution over time, which is
precisely the micro-Doppler signal that carries human activity information in mmWave
CIR. Measured consequence: HANDS vs SITTING separability 52.83% (chance) on the padded
CIR, versus 86.85% on micro-Doppler.

The DISC authors ship the intended representation in `micro_doppler_stft/` (STFT,
64-sample Hanning window, 32 overlap, dB, normalised to [0,1] -- Section 2 of the
dataset description). This script uses those signatures directly.

The paper's LABEL_MAPPING is used unchanged. Subject and activity are also stored so
the split can be audited.

Output: data/preprocessed_dataset/preprocessed_md.h5
"""
import os
import json
import numpy as np
import h5py
from scipy.ndimage import zoom

from config import LABEL_MAPPING, OUTPUT_DIR

MD_DIR = os.path.join("data", "micro_doppler_stft")
OUT = os.path.join(OUTPUT_DIR, "preprocessed_md.h5")

T_OUT = 64      # time frames after resampling
F_OUT = 64      # Doppler bins after padding 59 -> 64 (model needs divisibility by 8)
MIN_FRAMES = 8


def build():
    files = sorted(f for f in os.listdir(MD_DIR) if f.endswith(".npy"))
    print("found %d micro-Doppler signatures" % len(files))

    data, labels, subjects, activities, names = [], [], [], [], []
    skipped = []
    nan_frames_dropped = 0

    for fn in files:
        a = np.load(os.path.join(MD_DIR, fn)).astype(np.float32)

        # Drop NaN time-frames. 125/433 files contain a few all-NaN frames
        # (0.076% of all elements); one file is entirely NaN.
        bad = np.isnan(a).any(axis=1)
        nan_frames_dropped += int(bad.sum())
        a = a[~bad]
        if a.shape[0] < MIN_FRAMES:
            skipped.append(fn)
            continue

        # Resample the TIME axis to a fixed length. This keeps the full recording
        # rather than zero-padding it, and never mixes adjacent packets the way the
        # 4x4 block average did.
        a = zoom(a, (T_OUT / a.shape[0], 1.0), order=1)
        a = a[:T_OUT] if a.shape[0] >= T_OUT else np.pad(a, ((0, T_OUT - a.shape[0]), (0, 0)))

        # Pad the Doppler axis 59 -> 64 symmetrically. The outermost Doppler bins are
        # the highest velocities and carry almost no energy, so padding there is
        # preferable to interpolating (which would distort velocity calibration).
        pad = F_OUT - a.shape[1]
        if pad > 0:
            lo = pad // 2
            a = np.pad(a, ((0, 0), (lo, pad - lo)))

        base = os.path.splitext(fn)[0]
        act = base.split("_")[0]
        lab = next((LABEL_MAPPING[k] for k in LABEL_MAPPING if base.startswith(k)), -1)
        if lab == -1:
            skipped.append(fn)
            continue

        data.append(a[..., None])                     # (T, F, 1)
        labels.append(lab)
        subjects.append(int(base.split("_")[1]))
        activities.append(act)
        names.append(base)

    X = np.asarray(data, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(subjects, dtype=np.int32)
    print("built %s  (%.1f MB)" % (X.shape, X.nbytes / 1e6))
    print("NaN time-frames dropped: %d ; files skipped: %d %s"
          % (nan_frames_dropped, len(skipped), skipped))
    assert not np.isnan(X).any(), "NaNs survived preprocessing"

    import collections
    print("labels    :", dict(sorted(collections.Counter(y.tolist()).items())))
    print("activities:", dict(sorted(collections.Counter(activities).items())))
    print("subjects  :", dict(sorted(collections.Counter(s.tolist()).items())))

    with h5py.File(OUT, "w") as hf:
        hf.create_dataset("data", data=X, compression="gzip", chunks=(1,) + X.shape[1:])
        hf.create_dataset("labels", data=y)
        hf.create_dataset("subjects", data=s)
        hf.create_dataset("activities", data=np.array(activities, dtype="S10"))
        hf.create_dataset("names", data=np.array(names, dtype="S40"))
        hf.attrs["source"] = "DISC micro_doppler_stft (STFT W=64, overlap 32, dB, [0,1])"
        hf.attrs["time_frames"] = T_OUT
        hf.attrs["doppler_bins_padded_to"] = F_OUT
        hf.attrs["label_mapping"] = json.dumps(LABEL_MAPPING)
    print("wrote", OUT)


if __name__ == "__main__":
    build()

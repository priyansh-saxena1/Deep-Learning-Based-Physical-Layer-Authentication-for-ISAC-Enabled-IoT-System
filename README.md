# Deep Learning-Based Physical Layer Authentication for ISAC-Enabled IoT Systems

Code for the physical layer authentication (PLA) experiments on 60 GHz mmWave
channel measurements. A convolutional denoising autoencoder (CDAE) compresses a
micro-Doppler spectrogram into a 128-d latent vector, and a cosine-distance weighted
KNN decides whether the sample comes from the authorised activity group.

The measurements come from the public DISC dataset (Pegoraro et al.). **The data is
not redistributed here**; see [`data/README.md`](data/README.md) for where to get it
and how to build the model input.

## Layout

| path | what it is |
|---|---|
| `preprocess_md.py` | builds `preprocessed_md.h5` from the DISC micro-Doppler `.npy` files |
| `run_ablation_md.py` | trains the CDAE and runs every ablation row for one seed/split |
| `run_table3_md.py` | classifier comparison (Table III) from the saved runs |
| `make_figures.py` | figures and the LaTeX rows for Table 4, from the result JSONs only |
| `model.py`, `utils.py`, `config.py` | CDAE definition, feature extraction, metrics, label mapping |
| `data_preprocessing.py`, `run_ablation.py`, `run_ablation_fixed.py`, `run_ablation_extended.py` | the earlier raw-CIR pipeline, kept for reference |
| `results/` | JSON output of the runs behind the reported numbers |
| `logs/` | stdout of the five-split and four-seed runs behind Table 4 |

## Setup

The runs were done with TensorFlow 2.10 (the last release with native Windows GPU
support), CUDA 11.8 and cuDNN 8.8.

```bash
conda env create -f environment.yml
conda activate pla
```

Exact versions from the training machine are in `requirements-lock.txt`.

## Running

After placing the DISC files as described in `data/README.md`:

```bash
python preprocess_md.py
python run_ablation_md.py --seed 42 --split-seed 0 --epochs 200 --tag R_sp0_sd42
```

Table 4 is the mean and standard deviation over five stratified splits:

```bash
for s in 0 1 2 3 4; do
  python run_ablation_md.py --seed 42 --split-seed $s --epochs 200 --tag R_sp${s}_sd42
done
python run_table3_md.py
python make_figures.py
```

Each run writes `results/ablation_results_<tag>.json` with the metrics and confusion
matrix of every row, the k picked on validation, and the final CDAE loss. Always use a
different `--tag` per run; two batches sharing a tag overwrite each other.

On a headless machine or notebook kernel, set `MPLBACKEND=Agg` first.

`make_figures.py` only reads the JSON files, so you can regenerate the figures from the
committed `results/` without the dataset.

## Results

Mean ± sd over 5 splits, 200 epochs, label mapping `{HANDS, RUNNING} -> 0`,
`{WALKING, SITTING} -> 1`:

| Method | ACC | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Raw micro-Doppler + KNN | 87.82 ± 1.6 | 90.37 ± 1.4 | 91.86 ± 2.5 | 91.08 ± 1.2 | 20.71 ± 3.5 |
| AE + KNN | 85.75 ± 2.0 | 92.71 ± 1.5 | 85.76 ± 3.1 | 89.06 ± 1.7 | 14.29 ± 3.2 |
| CDAE + KNN | 88.51 ± 2.2 | 94.67 ± 2.3 | 88.14 ± 4.2 | 91.19 ± 1.9 | 10.71 ± 5.1 |
| CDAE + LR | 89.66 ± 2.3 | 90.65 ± 2.1 | 94.58 ± 3.3 | 92.52 ± 1.7 | 20.71 ± 5.2 |
| CDAE + SVM | 92.18 ± 3.1 | 94.91 ± 2.0 | 93.56 ± 4.7 | 94.15 ± 2.5 | 10.71 ± 4.5 |
| CDAE + RF | 87.82 ± 2.0 | 88.64 ± 2.6 | 94.24 ± 2.8 | 91.30 ± 1.4 | 25.71 ± 6.5 |
| **CDAE + Weighted KNN** | **93.56 ± 2.9** | **95.59 ± 2.2** | **94.92 ± 3.0** | **95.23 ± 2.1** | **9.29 ± 4.8** |

Things worth knowing before you quote these:

* The test set has only 28 negatives, so FPR moves in steps of 3.57 points. A single
  false positive is 3.57%.
* GPU training is not bit-reproducible. `UpSampling2D`'s gradient has no deterministic
  kernel in TF 2.10, and two runs with the same seed and split gave 91.95% and 87.36%.
  Report the mean over repeats rather than a single run. `--deterministic` forces CPU
  and op determinism if you need an exact repeat of your own run.
* Spread across seeds with the split fixed is about 1.2 points; across splits it is
  about 2.9.
* Tuning the KNN decision threshold on the validation set made FPR worse and unstable
  (only 69 validation samples), so the fixed threshold of 0.45 is used throughout.

## Input representation

The CIR array in the DISC `.mat` files is `(beam patterns, packets, range bins)`, so
axis 1 is time (one packet every 0.27 ms). The first pipeline in this repo
(`data_preprocessing.py`) padded that axis and averaged neighbouring packets, which
removes the micro-Doppler signal the classifier depends on. The current pipeline uses
the micro-Doppler STFT signatures that ship with DISC instead, resampled to 64 frames
and padded from 59 to 64 Doppler bins. `preprocess_md.py` documents each step.

## Citation and licence

Code is released under the MIT licence (see `LICENSE`). If you use the data, cite the
DISC authors:

> Pegoraro, Saucedo, Lacruz, Rossi, Widmer. *DISC: a Dataset for Integrated Sensing and
> Communication in mmWave Systems.* IEEE DataPort, doi:10.21227/2gm7-9z72.

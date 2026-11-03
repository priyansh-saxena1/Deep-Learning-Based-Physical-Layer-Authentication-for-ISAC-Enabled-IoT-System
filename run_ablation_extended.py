import os
os.environ.setdefault("SKLEARN_PAIRWISE_DIST_CHUNK_SIZE", "32")
os.environ.setdefault("OMP_NUM_THREADS", "4")
"""
ADDITIVE extension to the paper's method. The paper's pipeline is left intact and is
run as the baseline; the additions are reported as EXTRA rows.

ADDITION 1 - padding-masked reconstruction loss.
  39.3% of every preprocessed tensor is zero padding (181/433 samples exceed 50%
  padding; one exceeds 99%). Plain MSE is therefore dominated by reconstructing zeros,
  which is how the CDAE reaches loss 0.0036 while learning a latent that does not
  separate the classes. The masked loss computes error ONLY over genuine (non-padded)
  delay rows. The mask is derived from y_true itself -- padded rows are exactly zero --
  so the data pipeline, architecture, optimizer, epochs and noise factor are unchanged.
  This is an addition, not a modification of the paper's method.

DIAGNOSTIC - label grouping.
  Each representation is additionally scored against the physically coherent grouping
  {HANDS,SITTING} vs {RUNNING,WALKING}. This is reported as ANALYSIS ONLY. The paper's
  LABEL_MAPPING is NOT changed anywhere in this script.

Protocol for all rows: stratified split (28/59 test preserved), k chosen on a
validation split carved from train, StandardScaler fit on train, float32.
"""
import matplotlib
matplotlib.use("Agg")

import argparse, json, time, collections
import numpy as np
import h5py
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from model import psnr_metric, create_cdae_model
from utils import (extract_features, add_noise_to_image, weighted_knn,
                   run_standard_knn_metrics, run_weighted_knn_metrics,
                   train_logistic_regression_metrics, train_svm_metrics,
                   train_random_forest_metrics)
from config import DATASET_PATH, MODEL_HYPERPARAMETERS

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--split-seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=None)
args = ap.parse_args()

t0 = time.time()
def log(m): print("[%8.1fs] %s" % (time.time() - t0, m), flush=True)

g = tf.config.list_physical_devices("GPU")
if g:
    tf.config.experimental.set_memory_growth(g[0], True)
from tensorflow.keras.mixed_precision import Policy, set_global_policy
set_global_policy(Policy("float32"))
tf.keras.utils.set_random_seed(args.seed)
log("float32 seed=%s split_seed=%s GPU=%s" % (args.seed, args.split_seed, g))

# ------------------------------------------------------------------ data + metadata
LM = {"HANDS": 0, "RUNNING": 0, "WALKING": 1, "SITTING": 1}   # the paper's mapping, unchanged
dims = json.load(open(os.path.join("data", "preprocessed_dataset", "cir_dimensions.json")))
c0 = c1 = 0
tr_a, te_a = [], []
for key in dims:
    fn = key.split("/")[-1]
    lab = next((LM[k] for k in LM if fn.startswith(k)), -1)
    if lab == -1:
        continue
    act = fn.split("_")[0]
    if lab == 0 and c0 < 28:
        te_a.append(act); c0 += 1
    elif lab == 1 and c1 < 59:
        te_a.append(act); c1 += 1
    else:
        tr_a.append(act)
acts = np.array(tr_a + te_a)

log("decompressing HDF5 into RAM")
with h5py.File(DATASET_PATH, "r") as hf:
    X = np.concatenate([hf["train/data"][:], hf["test/data"][:]], axis=0)
    y = np.concatenate([hf["train/labels"][:], hf["test/labels"][:]], axis=0)
T, R = X.shape[1], X.shape[2]
assert (np.isin(acts, ["HANDS", "RUNNING"]).astype(int) == (y == 0).astype(int)).all()

y_alt = np.isin(acts, ["RUNNING", "WALKING"]).astype(int)   # diagnostic grouping only

rng = np.random.RandomState(args.split_seed)
te_idx = np.sort(np.concatenate([rng.choice(np.where(y == l)[0], n, replace=False)
                                 for l, n in [(0, 28), (1, 59)]]))
tr_idx = np.sort(np.setdiff1d(np.arange(len(y)), te_idx))
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.2, stratify=y[tr_idx],
                                  random_state=args.split_seed, shuffle=True)
Xtr, Xva, Xte = X[tr_idx], X[va_idx], X[te_idx]
ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]
atr, ava, ate = y_alt[tr_idx], y_alt[va_idx], y_alt[te_idx]
del X
log("train=%d val=%d test=%d" % (len(tr_idx), len(va_idx), len(te_idx)))

pad = 1.0 - (np.abs(Xtr).sum(axis=(2, 3)) > 0).sum(axis=1) / T
log("padding fraction in train: mean=%.1f%% max=%.1f%%" % (pad.mean() * 100, pad.max() * 100))

SPLITS = {"train": Xtr, "test": Xte}
bs = MODEL_HYPERPARAMETERS["batch_size"]
epochs = args.epochs or MODEL_HYPERPARAMETERS["epochs"]
nf = MODEL_HYPERPARAMETERS["noise_factor"]


def gen(split, noise):
    d = SPLITS[split]
    for i in range(0, d.shape[0], bs):
        xb = d[i:i + bs]
        yield np.array([add_noise_to_image(x, noise) for x in xb]), xb


def ds(split, noise):
    sig = (tf.TensorSpec((bs, T, R, 2), tf.float32), tf.TensorSpec((bs, T, R, 2), tf.float32))
    return (tf.data.Dataset.from_generator(lambda: gen(split, noise), output_signature=sig)
            .map(lambda a, b: (tf.cast(a, tf.float32), tf.cast(b, tf.float32)))
            .repeat().prefetch(tf.data.AUTOTUNE))


def masked_mse(y_true, y_pred):
    """ADDITION 1: error only over genuine (non-padded) delay rows."""
    keep = tf.cast(tf.reduce_sum(tf.abs(y_true), axis=[2, 3], keepdims=True) > 0, tf.float32)
    se = tf.square(y_pred - y_true) * keep
    return tf.reduce_sum(se) / (tf.reduce_sum(keep) * tf.cast(R * 2, tf.float32) + 1e-8)


def train_ae(noise, loss_fn, tag):
    log("--- training %s ---" % tag)
    t = time.time()
    m = create_cdae_model(T, R)
    m.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5), loss=loss_fn)
    m.fit(ds("train", noise), validation_data=ds("test", noise),
          steps_per_epoch=len(Xtr) // bs, validation_steps=len(Xte) // bs,
          epochs=epochs, verbose=2)
    log("--- %s done in %.1fs ---" % (tag, time.time() - t))
    enc = Model(inputs=m.input, outputs=m.get_layer("feature_vector").output)
    return (extract_features(enc, Xtr, 1), extract_features(enc, Xva, 1), extract_features(enc, Xte, 1))


mse = tf.keras.losses.MeanSquaredError()
F_cdae = train_ae(nf, mse, "CDAE (paper, plain MSE)")
F_mask = train_ae(nf, masked_mse, "CDAE-masked (ADDITION 1)")
F_ae = train_ae(0.0, mse, "AE (paper, plain MSE)")
F_raw = (Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1), Xte.reshape(len(Xte), -1))

K_GRID = list(range(3, 30, 2))


def scaled(F):
    s = StandardScaler().fit(F[0])
    return s.transform(F[0]), s.transform(F[1]), s.transform(F[2])


def pick_k_std(tr, ytr_, va, yva_):
    best, bk = -1, K_GRID[0]
    for k in K_GRID:
        a = (KNeighborsClassifier(n_neighbors=k).fit(tr, ytr_).predict(va) == yva_).mean()
        if a > best: best, bk = a, k
    return bk


def pick_k_w(tr, ytr_, va, yva_):
    best, bk = -1, K_GRID[0]
    for k in K_GRID:
        _, p = weighted_knn(tr, ytr_, va, k, metric="cosine")
        a = ((p >= 0.45).astype(int) == yva_).mean()
        if a > best: best, bk = a, k
    return bk


def evaluate(F, labels, tag_rows):
    tr, va, te = scaled(F)
    L_tr, L_va, L_te = labels
    out = {}
    if "knn" in tag_rows:
        out["knn"] = run_standard_knn_metrics(tr, L_tr, te, L_te, k=pick_k_std(tr, L_tr, va, L_va))
    if "wknn" in tag_rows:
        out["wknn"] = run_weighted_knn_metrics(tr, L_tr, te, L_te,
                                               k=pick_k_w(tr, L_tr, va, L_va), metric="cosine")
    if "lr" in tag_rows:
        out["lr"] = train_logistic_regression_metrics(tr, L_tr, te, L_te)
        out["svm"] = train_svm_metrics(tr, L_tr, te, L_te)
        out["rf"] = train_random_forest_metrics(tr, L_tr, te, L_te)
    return out


PAPER = (ytr, yva, yte)
ALT = (atr, ava, ate)

log("evaluating on the PAPER'S labels")
res = {
    "Raw CIR + KNN":              evaluate(F_raw, PAPER, ["knn"])["knn"],
    "AE + KNN":                   evaluate(F_ae, PAPER, ["knn"])["knn"],
}
c = evaluate(F_cdae, PAPER, ["knn", "wknn", "lr"])
res["CDAE + KNN"] = c["knn"]
res["CDAE + LR"] = c["lr"]
res["CDAE + SVM"] = c["svm"]
res["CDAE + RF"] = c["rf"]
res["CDAE + Weighted KNN"] = c["wknn"]
m = evaluate(F_mask, PAPER, ["knn", "wknn", "lr"])
res["CDAE-masked + KNN  (ADDED)"] = m["knn"]
res["CDAE-masked + LR  (ADDED)"] = m["lr"]
res["CDAE-masked + Weighted KNN  (ADDED)"] = m["wknn"]

ORDER = ["Raw CIR + KNN", "AE + KNN", "CDAE + KNN", "CDAE + LR", "CDAE + SVM",
         "CDAE + RF", "CDAE + Weighted KNN",
         "CDAE-masked + KNN  (ADDED)", "CDAE-masked + LR  (ADDED)",
         "CDAE-masked + Weighted KNN  (ADDED)"]
print("\n" + "=" * 86)
print("TABLE 4 + ADDITIONS  --  PAPER'S LABEL MAPPING (unchanged)")
print("=" * 86)
print("%-38s%8s%11s%9s%8s%8s" % ("Method", "ACC", "Precision", "Recall", "F1", "FPR"))
print("-" * 86)
for n in ORDER:
    r = res[n]
    print("%-38s%8.2f%11.2f%9.2f%8.2f%8.2f" % (n, r[0], r[1], r[2], r[3], r[4]))
print("=" * 86)

log("DIAGNOSTIC: same representations, coherent grouping {HANDS,SITTING} vs {RUNNING,WALKING}")
alt = {
    "CDAE + Weighted KNN": evaluate(F_cdae, ALT, ["wknn"])["wknn"],
    "CDAE-masked + Weighted KNN": evaluate(F_mask, ALT, ["wknn"])["wknn"],
    "CDAE + LR": evaluate(F_cdae, ALT, ["lr"])["lr"],
    "CDAE-masked + LR": evaluate(F_mask, ALT, ["lr"])["lr"],
}
print("\n" + "=" * 86)
print("DIAGNOSTIC ONLY -- coherent grouping. The paper's LABEL_MAPPING is NOT changed.")
print("=" * 86)
print("%-38s%8s%11s%9s%8s%8s" % ("Method", "ACC", "Precision", "Recall", "F1", "FPR"))
print("-" * 86)
for n, r in alt.items():
    print("%-38s%8.2f%11.2f%9.2f%8.2f%8.2f" % (n, r[0], r[1], r[2], r[3], r[4]))
print("=" * 86)

json.dump({"paper_labels": {k: [float(x) for x in v] for k, v in res.items()},
           "coherent_grouping_diagnostic": {k: [float(x) for x in v] for k, v in alt.items()},
           "seed": args.seed, "split_seed": args.split_seed, "epochs": epochs,
           "padding_fraction_mean": float(pad.mean()), "runtime_sec": time.time() - t0},
          open("ablation_results_extended.json", "w"), indent=2)
log("wrote ablation_results_extended.json")

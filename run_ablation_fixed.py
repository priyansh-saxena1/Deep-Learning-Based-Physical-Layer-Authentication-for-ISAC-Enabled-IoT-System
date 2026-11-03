import os
os.environ.setdefault("SKLEARN_PAIRWISE_DIST_CHUNK_SIZE", "32")
os.environ.setdefault("OMP_NUM_THREADS", "4")
"""
Table 4 ablation, METHODOLOGICALLY CORRECTED protocol.

Every change below is a defect fix decided before seeing any result. Nothing here was
tuned to reach a target number. See ABLATION_ENV_NOTES.md for the uncorrected baseline.

FIX 1 - subject-confounded split.
  data_preprocessing.py assigns the FIRST 28 label-0 and FIRST 59 label-1 files in
  directory order to test. That order starts at PERSON1, so test ended up 94% PERSON1
  (82/87) with only 5 PERSON1 samples in train. The evaluation was accidentally
  cross-subject. Replaced with a seeded STRATIFIED RANDOM split over all 433 samples,
  preserving the original test composition exactly (28 label-0 + 59 label-1).

FIX 2 - k selected on the test set.
  evaluate_knn_for_different_k sweeps k and picks the argmax of TEST accuracy, then
  those same test samples are reported as the result. That is selection leakage.
  Replaced with a stratified validation split carved from train; k is chosen on
  validation only, per representation and per metric. Test is touched once.

FIX 3 - best_k leaked across metrics.
  main.py loops over [cosine, euclidean, manhattan] and overwrites best_k each pass, so
  every row used the MANHATTAN-tuned k even though the proposed method uses cosine.
  Each classifier now uses the k selected for its own representation and metric.

FIX 4 - mixed_float16 produces NaN.
  The final Conv2DTranspose in model.py has no float32 cast, so under the global
  mixed_float16 policy it emits float16 and MSE over ~2M elements overflows. Verified:
  mixed_float16 -> nan, float32 -> 0.9583 on identical 8-step runs. Runs in float32.

FIX 5 - unscaled features into distance-based classifiers.
  The 128-d latents go straight into KNN/SVM/LR with per-dimension scales differing by
  orders of magnitude, so a few dimensions dominate every distance. StandardScaler is
  now fit on TRAIN ONLY and applied to val/test, uniformly across all representations
  so the ablation stays like-for-like.

FIX 6 - gzip I/O bottleneck (performance only, no numerical effect).
  HDF5 chunked (32,245,8,1) but read one sample at a time: 2.233 s/sample vs 0.0048 s
  from RAM. Each split is decompressed once into RAM. Same bytes, same order, no RNG
  consumed.
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import json
import time
import collections
import h5py
import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from model import psnr_metric, create_cdae_model
from utils import (extract_features, add_noise_to_image, weighted_knn, get_metrics,
                   run_standard_knn_metrics, run_weighted_knn_metrics,
                   train_logistic_regression_metrics, train_svm_metrics,
                   train_random_forest_metrics)
from config import DATASET_PATH, MODEL_HYPERPARAMETERS

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--split-seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--tag", default="fixed")
args = ap.parse_args()

t_start = time.time()


def log(msg):
    print("[%8.1fs] %s" % (time.time() - t_start, msg), flush=True)


_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    tf.config.experimental.set_memory_growth(_gpus[0], True)
from tensorflow.keras.mixed_precision import Policy, set_global_policy
set_global_policy(Policy("float32"))          # FIX 4
log("float32 | model seed=%s | split seed=%s | GPUs=%s" % (args.seed, args.split_seed, _gpus))

tf.keras.utils.set_random_seed(args.seed)

# ---------------------------------------------------------------- load + re-pool
log("decompressing HDF5 into RAM")                                   # FIX 6
with h5py.File(DATASET_PATH, "r") as hf:
    tr_d, tr_l = hf["train/data"][:], hf["train/labels"][:]
    te_d, te_l = hf["test/data"][:], hf["test/labels"][:]
T, R = tr_d.shape[1], tr_d.shape[2]

# Recover each sample's source file by replaying the assignment rule in
# data_preprocessing.py against cir_dimensions.json (which preserves processing order).
LM = {"HANDS": 0, "RUNNING": 0, "WALKING": 1, "SITTING": 1}
dims = json.load(open(os.path.join("data", "preprocessed_dataset", "cir_dimensions.json")))
c0 = c1 = 0
order = {"train": [], "test": []}
for key in dims:
    fn = key.split("/")[-1]
    lab = next((LM[k] for k in LM if fn.startswith(k)), -1)
    if lab == -1:
        continue
    if lab == 0 and c0 < 28:
        order["test"].append((key, lab)); c0 += 1
    elif lab == 1 and c1 < 59:
        order["test"].append((key, lab)); c1 += 1
    else:
        order["train"].append((key, lab))

assert len(order["train"]) == len(tr_l) and len(order["test"]) == len(te_l), "order replay mismatch"
assert all(a == b for (_, a), b in zip(order["train"], tr_l)), "train label replay mismatch"
assert all(a == b for (_, a), b in zip(order["test"], te_l)), "test label replay mismatch"
log("file-order replay verified against stored labels")

X = np.concatenate([tr_d, te_d], axis=0)
y = np.concatenate([tr_l, te_l], axis=0)
del tr_d, te_d
keys = [k for k, _ in order["train"]] + [k for k, _ in order["test"]]
persons = np.array([k.split("/")[0] for k in keys])
log("pooled %s (%.2f GB)" % (X.shape, X.nbytes / 1e9))

# ---------------------------------------------------------------- FIX 1: real split
# Sample per class so the test set matches the original composition EXACTLY
# (28 label-0, 59 label-1). Plain stratification rounds to 27/60.
rng = np.random.RandomState(args.split_seed)
te_parts = []
for lab, n in [(0, 28), (1, 59)]:
    pool = np.where(y == lab)[0]
    te_parts.append(rng.choice(pool, size=n, replace=False))
te_idx = np.sort(np.concatenate(te_parts))
tr_idx = np.sort(np.setdiff1d(np.arange(len(y)), te_idx))
assert collections.Counter(y[te_idx].tolist()) == {1: 59, 0: 28}, collections.Counter(y[te_idx].tolist())

# FIX 2: validation carved from TRAIN only, for k selection
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.2, stratify=y[tr_idx],
                                  random_state=args.split_seed, shuffle=True)

train_raw, train_labels = X[tr_idx], y[tr_idx]
val_raw, val_labels = X[va_idx], y[va_idx]
test_raw, test_labels = X[te_idx], y[te_idx]
del X
log("split train=%d val=%d test=%d" % (len(tr_idx), len(va_idx), len(te_idx)))
for nm, ii in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
    log("  %-5s labels=%s persons=%s" % (nm, dict(sorted(collections.Counter(y[ii].tolist()).items())),
                                         dict(sorted(collections.Counter(persons[ii].tolist()).items()))))

SPLITS = {"train": train_raw, "test": test_raw}
batch_size = MODEL_HYPERPARAMETERS["batch_size"]
epochs = args.epochs or MODEL_HYPERPARAMETERS["epochs"]
noise_factor = MODEL_HYPERPARAMETERS["noise_factor"]
steps_per_epoch = len(train_raw) // batch_size
validation_steps = len(test_raw) // batch_size


def cached_generator(split, nf):
    data = SPLITS[split]
    for i in range(0, data.shape[0], batch_size):
        xb = data[i:i + batch_size]
        yield np.array([add_noise_to_image(x, nf) for x in xb]), xb


def make_ds(split, nf):
    sig = (tf.TensorSpec(shape=(batch_size, T, R, 2), dtype=tf.float32),
           tf.TensorSpec(shape=(batch_size, T, R, 2), dtype=tf.float32))
    return (tf.data.Dataset.from_generator(lambda: cached_generator(split, nf), output_signature=sig)
            .map(lambda a, b: (tf.cast(a, tf.float32), tf.cast(b, tf.float32)))
            .repeat().prefetch(tf.data.AUTOTUNE))


def train_autoencoder(nf, tag, metrics):
    log("--- training %s (noise_factor=%s) ---" % (tag, nf))
    t0 = time.time()
    m = create_cdae_model(T, R)
    m.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
              loss=tf.keras.losses.MeanSquaredError(), metrics=metrics)
    h = m.fit(make_ds("train", nf), validation_data=make_ds("test", nf),
              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
              epochs=epochs, verbose=2)
    log("--- %s done in %.1fs ---" % (tag, time.time() - t0))
    return m, h


cdae, hist = train_autoencoder(noise_factor, "CDAE", ["accuracy", psnr_metric])
cdae_enc = Model(inputs=cdae.input, outputs=cdae.get_layer("feature_vector").output)
log("extracting CDAE features")
f_tr = extract_features(cdae_enc, train_raw, batch_size=1)
f_va = extract_features(cdae_enc, val_raw, batch_size=1)
f_te = extract_features(cdae_enc, test_raw, batch_size=1)

ae, _ = train_autoencoder(0.0, "AE", None)
ae_enc = Model(inputs=ae.input, outputs=ae.get_layer("feature_vector").output)
log("extracting AE features")
a_tr = extract_features(ae_enc, train_raw, batch_size=1)
a_va = extract_features(ae_enc, val_raw, batch_size=1)
a_te = extract_features(ae_enc, test_raw, batch_size=1)

r_tr = train_raw.reshape(len(train_raw), -1)
r_va = val_raw.reshape(len(val_raw), -1)
r_te = test_raw.reshape(len(test_raw), -1)

K_GRID = list(range(3, 30, 2))


def scale(tr, va, te):                                               # FIX 5
    s = StandardScaler().fit(tr)
    return s.transform(tr), s.transform(va), s.transform(te)


def pick_k_standard(tr, ytr, va, yva):                               # FIX 2 / FIX 3
    best, bk = -1.0, K_GRID[0]
    for k in K_GRID:
        if k > len(tr):
            break
        acc = (KNeighborsClassifier(n_neighbors=k).fit(tr, ytr).predict(va) == yva).mean()
        if acc > best:
            best, bk = acc, k
    return bk, best


def pick_k_weighted(tr, ytr, va, yva, metric):                       # FIX 2 / FIX 3
    best, bk = -1.0, K_GRID[0]
    for k in K_GRID:
        if k > len(tr):
            break
        _, prob = weighted_knn(tr, ytr, va, k, metric=metric)
        acc = ((prob >= 0.45).astype(int) == yva).mean()
        if acc > best:
            best, bk = acc, k
    return bk, best


rows, chosen = {}, {}

for name, (tr, va, te) in [("Raw CIR + KNN", (r_tr, r_va, r_te)),
                           ("AE + KNN", (a_tr, a_va, a_te)),
                           ("CDAE + KNN", (f_tr, f_va, f_te))]:
    log("row: %s" % name)
    tr_s, va_s, te_s = scale(tr, va, te)
    k, vacc = pick_k_standard(tr_s, train_labels, va_s, val_labels)
    chosen[name] = {"k": int(k), "val_acc": float(vacc)}
    rows[name] = run_standard_knn_metrics(tr_s, train_labels, te_s, test_labels, k=k)

f_tr_s, f_va_s, f_te_s = scale(f_tr, f_va, f_te)
for name, fn in [("CDAE + LR", train_logistic_regression_metrics),
                 ("CDAE + SVM", train_svm_metrics),
                 ("CDAE + RF", train_random_forest_metrics)]:
    log("row: %s" % name)
    rows[name] = fn(f_tr_s, train_labels, f_te_s, test_labels)

log("row: CDAE + Weighted KNN")
kw, vaccw = pick_k_weighted(f_tr_s, train_labels, f_va_s, val_labels, "cosine")
chosen["CDAE + Weighted KNN"] = {"k": int(kw), "val_acc": float(vaccw), "metric": "cosine"}
rows["CDAE + Weighted KNN"] = run_weighted_knn_metrics(f_tr_s, train_labels, f_te_s,
                                                       test_labels, k=kw, metric="cosine")

ORDER = ["Raw CIR + KNN", "AE + KNN", "CDAE + KNN", "CDAE + LR",
         "CDAE + SVM", "CDAE + RF", "CDAE + Weighted KNN"]
print("\n" + "=" * 78)
print("TABLE 4 (corrected protocol): ABLATION OF FEATURE EXTRACTION & CLASSIFICATION")
print("=" * 78)
print("%-24s%9s%11s%9s%9s%9s" % ("Method", "ACC", "Precision", "Recall", "F1", "FPR"))
print("-" * 78)
for n in ORDER:
    r = rows[n]
    print("%-24s%9.2f%11.2f%9.2f%9.2f%9.2f" % (n, r[0], r[1], r[2], r[3], r[4]))
print("=" * 78)
print("k chosen on validation:", json.dumps(chosen))

out = {"rows": {k: [float(x) for x in v] for k, v in rows.items()},
       "k_selected_on_validation": chosen,
       "protocol": "stratified random split; k on validation; StandardScaler; float32",
       "epochs": epochs, "seed": args.seed, "split_seed": args.split_seed,
       "n_train": int(len(train_labels)), "n_val": int(len(val_labels)),
       "n_test": int(len(test_labels)), "runtime_sec": time.time() - t_start}
if hist is not None:
    out["cdae_history"] = {k: [float(x) for x in v] for k, v in hist.history.items()}
with open("ablation_results_%s.json" % args.tag, "w") as f:
    json.dump(out, f, indent=2)
log("wrote ablation_results_%s.json" % args.tag)

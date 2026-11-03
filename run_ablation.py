import os
os.environ.setdefault("SKLEARN_PAIRWISE_DIST_CHUNK_SIZE", "32")
os.environ.setdefault("OMP_NUM_THREADS", "4")
"""
Driver for Table 4 (ablation study) ONLY. Existing project files are not modified.

Deviations from main.py, each forced, each numerically neutral except where noted:
  * preprocess_data() is skipped (raw .mat files unavailable); T, R come from the HDF5.
  * matplotlib uses Agg so plt.show() does not block.
  * The Table III block (PCA + LR/RF/SVM/CNN) is not run; it is not part of Table 4.
  * Each split is decompressed into RAM once instead of re-reading gzip chunks per
    step. The HDF5 is chunked (32,245,8,1) but read one sample at a time, costing
    2.23 s/sample vs 0.005 s from RAM. Same bytes, same order, consumes no RNG, so
    the seeded noise stream is identical. Purely an I/O change.
  * precision policy is float32, NOT mixed_float16.  *** THIS ONE AFFECTS NUMBERS ***
    Under mixed_float16 (what main.py sets) the final Conv2DTranspose emits float16
    and the MSE over 2M elements overflows -> loss = nan. Verified on an identical
    8-step run: mixed_float16 -> nan, float32 -> 0.9583. main.py as written cannot
    produce Table 4 at all.

The best_k selection loop is reproduced exactly as main.py has it, including the fact
that best_k ends up being the value tuned on 'manhattan' (the loop overwrites it).
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import json
import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.models import Model

from model import psnr_metric, create_cdae_model
from utils import (extract_features, evaluate_knn_for_different_k,
                   run_standard_knn_metrics, run_weighted_knn_metrics,
                   train_logistic_regression_metrics, train_svm_metrics,
                   train_random_forest_metrics, add_noise_to_image)
from config import DATASET_PATH, MODEL_HYPERPARAMETERS

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--policy", default="float32", choices=["float32", "mixed_float16"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=None)
args = ap.parse_args()

t_start = time.time()


def log(msg):
    print("[%8.1fs] %s" % (time.time() - t_start, msg), flush=True)


_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    tf.config.experimental.set_memory_growth(_gpus[0], True)
from tensorflow.keras.mixed_precision import Policy, set_global_policy
set_global_policy(Policy(args.policy))
log("precision=%s seed=%s GPUs=%s" % (args.policy, args.seed, _gpus))

if args.seed is not None:
    tf.keras.utils.set_random_seed(args.seed)

log("decompressing HDF5 into RAM (one pass)")
with h5py.File(DATASET_PATH, "r") as hf:
    train_raw = hf["train/data"][:]
    train_labels = hf["train/labels"][:]
    test_raw = hf["test/data"][:]
    test_labels = hf["test/labels"][:]
T, R = train_raw.shape[1], train_raw.shape[2]
log("train=%s (%.2f GB) test=%s" % (train_raw.shape, train_raw.nbytes / 1e9, test_raw.shape))

if args.smoke:
    train_raw, train_labels = train_raw[:16], train_labels[:16]
    test_raw, test_labels = test_raw[:8], test_labels[:8]

SPLITS = {"train": train_raw, "test": test_raw}

batch_size = MODEL_HYPERPARAMETERS["batch_size"]
epochs = args.epochs or MODEL_HYPERPARAMETERS["epochs"]
noise_factor = MODEL_HYPERPARAMETERS["noise_factor"]
steps_per_epoch = len(train_raw) // batch_size
validation_steps = len(test_raw) // batch_size
if args.smoke:
    epochs, steps_per_epoch, validation_steps = 1, 4, 2
log("batch=%d epochs=%d steps=%d val_steps=%d noise=%s"
    % (batch_size, epochs, steps_per_epoch, validation_steps, noise_factor))


def cached_generator(split, nf):
    """Mirrors utils.load_h5_data_generator, reading from RAM instead of gzip."""
    data = SPLITS[split]
    for i in range(0, data.shape[0], batch_size):
        x_batch = data[i:i + batch_size]
        noisy = np.array([add_noise_to_image(x, nf) for x in x_batch])
        if x_batch.shape[0] < batch_size:
            pad = (batch_size - x_batch.shape[0], T, R, 2)
            x_batch = np.concatenate([x_batch, np.zeros(pad)], axis=0)
            noisy = np.concatenate([noisy, np.zeros(pad)], axis=0)
        yield noisy, x_batch


def make_ds(split, nf):
    sig = (tf.TensorSpec(shape=(batch_size, T, R, 2), dtype=tf.float32),
           tf.TensorSpec(shape=(batch_size, T, R, 2), dtype=tf.float32))
    return (tf.data.Dataset.from_generator(lambda: cached_generator(split, nf),
                                           output_signature=sig)
            .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
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
cdae.save("cdae_model_ablation.keras")
cdae_encoder = Model(inputs=cdae.input, outputs=cdae.get_layer("feature_vector").output)

log("extracting CDAE features")
train_features = extract_features(cdae_encoder, train_raw, batch_size=1)
test_features = extract_features(cdae_encoder, test_raw, batch_size=1)
np.savez("train_features_ablation.npz", features=train_features, labels=train_labels)
np.savez("test_features_ablation.npz", features=test_features, labels=test_labels)
log("features train=%s test=%s" % (train_features.shape, test_features.shape))

k_values = [3] if args.smoke else list(range(3, 30, 2))
per_metric_k = {}
for metric in ["cosine", "euclidean", "manhattan"]:
    best_k = evaluate_knn_for_different_k(train_features, train_labels,
                                          test_features, test_labels, k_values, metric)
    per_metric_k[metric] = int(best_k)
log("best_k per metric=%s -> carried into Table 4: %s" % (per_metric_k, best_k))

ae, _ = train_autoencoder(0.0, "AE", None)
ae_encoder = Model(inputs=ae.input, outputs=ae.get_layer("feature_vector").output)
log("extracting AE features")
ae_train_features = extract_features(ae_encoder, train_raw, batch_size=1)
ae_test_features = extract_features(ae_encoder, test_raw, batch_size=1)

rows = {}
log("row: Raw CIR + KNN")
rows["Raw CIR + KNN"] = run_standard_knn_metrics(
    train_raw.reshape(train_raw.shape[0], -1), train_labels,
    test_raw.reshape(test_raw.shape[0], -1), test_labels, k=best_k)
log("row: AE + KNN")
rows["AE + KNN"] = run_standard_knn_metrics(ae_train_features, train_labels,
                                            ae_test_features, test_labels, k=best_k)
log("row: CDAE + KNN")
rows["CDAE + KNN"] = run_standard_knn_metrics(train_features, train_labels,
                                              test_features, test_labels, k=best_k)
log("row: CDAE + LR")
rows["CDAE + LR"] = train_logistic_regression_metrics(train_features, train_labels,
                                                      test_features, test_labels)
log("row: CDAE + SVM")
rows["CDAE + SVM"] = train_svm_metrics(train_features, train_labels,
                                       test_features, test_labels)
log("row: CDAE + RF")
rows["CDAE + RF"] = train_random_forest_metrics(train_features, train_labels,
                                                test_features, test_labels)
log("row: CDAE + Weighted KNN")
rows["CDAE + Weighted KNN"] = run_weighted_knn_metrics(train_features, train_labels,
                                                       test_features, test_labels,
                                                       k=best_k, metric="cosine")

print("\n" + "=" * 78)
print("TABLE 4: ABLATION STUDY OF FEATURE EXTRACTION & CLASSIFICATION METHODS")
print("=" * 78)
print("%-24s%9s%11s%9s%9s%9s" % ("Method", "ACC", "Precision", "Recall", "F1", "FPR"))
print("-" * 78)
for name, r in rows.items():
    print("%-24s%9.2f%11.2f%9.2f%9.2f%9.2f" % (name, r[0], r[1], r[2], r[3], r[4]))
print("=" * 78)

out = {"rows": {k: [float(x) for x in v] for k, v in rows.items()},
       "best_k_per_metric": per_metric_k, "best_k_used": int(best_k),
       "epochs": epochs, "batch_size": batch_size, "noise_factor": noise_factor,
       "T": int(T), "R": int(R), "policy": args.policy, "seed": args.seed,
       "train_n": int(len(train_labels)), "test_n": int(len(test_labels)),
       "smoke": args.smoke, "runtime_sec": time.time() - t_start}
if hist is not None:
    out["cdae_history"] = {k: [float(x) for x in v] for k, v in hist.history.items()}
with open("ablation_results.json", "w") as f:
    json.dump(out, f, indent=2)
log("wrote ablation_results.json")

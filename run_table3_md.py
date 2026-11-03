"""
Regenerate manuscript Table III (classifier comparison) on the micro-Doppler input.

The paper's Table III baselines (LR, SVM, RF, BPNN/MLP) are trained on PCA-reduced
input features -- mirroring utils.load_and_preprocess_data(apply_pca=True,
variance_threshold=0.95) -- while the proposed row uses CDAE + Weighted KNN. The same
structure is reproduced here on micro-Doppler, using the identical splits as the
ablation so the two tables are consistent.

The proposed row is read from the ablation run JSONs rather than retrained, guaranteeing
Table III and Table 4 report the same numbers for the same configuration.

    python run_table3_md.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, collections
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from utils import get_metrics
from config import OUTPUT_DIR

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SPLIT_SEEDS = [0, 1, 2, 3, 4]
PROPOSED = "CDAE + Weighted KNN"

with h5py.File(os.path.join(OUTPUT_DIR, "preprocessed_md.h5"), "r") as hf:
    X, y = hf["data"][:], hf["labels"][:]
X = X.reshape(len(X), -1)
print("input:", X.shape, dict(sorted(collections.Counter(y.tolist()).items())))


def split_of(ss):
    """Identical split construction to run_ablation_md.py."""
    rng = np.random.RandomState(ss)
    te = np.sort(np.concatenate([rng.choice(np.where(y == l)[0], n, replace=False)
                                 for l, n in [(0, 28), (1, 59)]]))
    tr = np.sort(np.setdiff1d(np.arange(len(y)), te))
    tr, va = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=ss, shuffle=True)
    return tr, va, te


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "BPNN-Based Approach (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64),
                                                   max_iter=2000, random_state=42),
    }


acc = collections.defaultdict(list)
cms = collections.defaultdict(list)
for ss in SPLIT_SEEDS:
    tr, va, te = split_of(ss)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    pca = PCA(n_components=0.95, svd_solver="full", random_state=0).fit(Xtr)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    for name, m in models().items():
        m.fit(Ptr, y[tr])
        pred = m.predict(Pte)
        acc[name].append(get_metrics(y[te], pred))
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        cms[name].append({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    print("  split %d done (PCA kept %d comps)" % (ss, pca.n_components_))

# proposed row, taken from the ablation runs so both tables agree exactly
prop = []
for ss in SPLIT_SEEDS:
    p = os.path.join(BASE, "ablation_results_R_sp%d_sd42.json" % ss)
    if os.path.exists(p):
        prop.append(json.load(open(p))["rows"][PROPOSED])
if prop:
    acc[PROPOSED] = prop

ORDER = ["Logistic Regression", "SVM", "Random Forest", "BPNN-Based Approach (MLP)", PROPOSED]
MET = ["ACC", "Precision", "Recall", "F1", "FPR"]

print("\n" + "=" * 92)
print("TABLE III -- classifier comparison, micro-Doppler input (mean +- sd over %d splits)"
      % len(SPLIT_SEEDS))
print("=" * 92)
print("%-30s%s" % ("Classifier", "".join("%13s" % m for m in MET)))
print("-" * 92)
out = {}
for n in ORDER:
    if n not in acc:
        continue
    a = np.array(acc[n])
    out[n] = {"mean": a.mean(0).tolist(), "sd": a.std(0).tolist()}
    print("%-30s%s" % (n, "".join("%8.2f+-%-4.1f" % (m, s) for m, s in zip(a.mean(0), a.std(0)))))
print("=" * 92)

# Specificity is reported in the manuscript's Table III; it is 100 - FPR.
print("\nSpecificity (100 - FPR):")
for n in ORDER:
    if n in out:
        print("   %-30s %.2f" % (n, 100 - out[n]["mean"][4]))

with open(os.path.join(BASE, "table3_md.json"), "w") as f:
    json.dump({"rows": out, "split_seeds": SPLIT_SEEDS,
               "note": "baselines: StandardScaler + PCA(0.95) on micro-Doppler; "
                       "proposed row copied from ablation runs"}, f, indent=2)
print("\nwrote table3_md.json")

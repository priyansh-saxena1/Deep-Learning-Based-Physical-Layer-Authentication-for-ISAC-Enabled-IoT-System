"""
Regenerate every manuscript figure from the saved run JSONs.

Pure function of `ablation_results_R_*.json` (+ `table3_md.json` if present) -- no
training, no randomness, so it is exactly reproducible. Run after the ablation runs:

    python make_figures.py

Writes PDFs (vector, LaTeX-friendly) to figures/.
"""
import os, json, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.autolayout": True, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

SPLIT_RUNS = [os.path.join(BASE, "ablation_results_R_sp%d_sd42.json" % s) for s in range(5)]
ORDER = ["Raw micro-Doppler + KNN", "AE + KNN", "CDAE + KNN", "CDAE + LR",
         "CDAE + SVM", "CDAE + RF", "CDAE + Weighted KNN"]
PROPOSED = "CDAE + Weighted KNN"


def load(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            out.append(json.load(open(p)))
        else:
            print("  missing:", os.path.basename(p))
    return out


runs = load(SPLIT_RUNS)
if not runs:
    raise SystemExit("no split runs found; run run_ablation_md.py first")
print("loaded %d split runs" % len(runs))
ref = runs[0]                       # split_seed 0 is the reference run for curves
saved = []


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig); saved.append(name)
    print("  wrote", name)


# ---------------------------------------------------------------- training curves
def curve(keys, ylabel, title, name, pct=False):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, lab, style in keys:
        vals = np.array(ref["cdae_history"][key], dtype=float)
        if pct:
            vals = vals * 100
        ax.plot(np.arange(1, len(vals) + 1), vals, style, label=lab, linewidth=1.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False)
    save(fig, name)


curve([("loss", "Training", "-"), ("val_loss", "Validation", "--")],
      "MSE loss", "CDAE reconstruction loss", "fig_loss.pdf")
curve([("accuracy", "Training", "-"), ("val_accuracy", "Validation", "--")],
      "Reconstruction accuracy (%)", "CDAE accuracy", "fig_accuracy.pdf", pct=True)
curve([("psnr_metric", "Training", "-"), ("val_psnr_metric", "Validation", "--")],
      "PSNR (dB)", "CDAE reconstruction PSNR", "fig_psnr.pdf")

# ---------------------------------------------------------------- confusion matrices
SHORT = {"Raw micro-Doppler + KNN": "raw_knn", "AE + KNN": "ae_knn", "CDAE + KNN": "cdae_knn",
         "CDAE + LR": "lr", "CDAE + SVM": "svm", "CDAE + RF": "rf",
         "CDAE + Weighted KNN": "cdae_wknn"}
for name in ORDER:
    c = ref["confusion_matrices"][name]
    M = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(M, cmap="Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=15,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(name, fontsize=9)
    save(fig, "fig_cm_%s.pdf" % SHORT[name])

# ---------------------------------------------------------------- ROC curves
fig, ax = plt.subplots(figsize=(5.2, 4.2))
yt = np.array(ref["y_test"])
aucs = {}
for name in ORDER:
    if name not in ref["roc_scores"]:
        continue
    fpr, tpr, _ = roc_curve(yt, np.array(ref["roc_scores"][name]))
    a = auc(fpr, tpr); aucs[name] = a
    ax.plot(fpr, tpr, linewidth=2.0 if name == PROPOSED else 1.1,
            label="%s (AUC %.3f)" % (name, a))
ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (split 0)"); ax.legend(fontsize=7, frameon=False, loc="lower right")
save(fig, "fig_roc_all.pdf")

# ---------------------------------------------------------------- ablation bars
means = {n: np.array([r["rows"][n] for r in runs]).mean(0) for n in ORDER}
sds = {n: np.array([r["rows"][n] for r in runs]).std(0) for n in ORDER}
fig, ax = plt.subplots(figsize=(7.4, 3.8))
xs = np.arange(len(ORDER))
vals = [means[n][0] for n in ORDER]
errs = [sds[n][0] for n in ORDER]
cols = ["#4C78A8"] * len(ORDER); cols[ORDER.index(PROPOSED)] = "#E45756"
ax.bar(xs, vals, yerr=errs, capsize=4, color=cols)
ax.axhline(89.66, color="grey", linestyle="--", linewidth=1, label="published 89.66%")
ax.set_xticks(xs); ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in ORDER], fontsize=7)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(60, 100)
ax.set_title("Ablation: accuracy (mean $\\pm$ sd over %d splits)" % len(runs))
ax.legend(frameon=False, fontsize=8)
save(fig, "fig_ablation_acc.pdf")

# ---------------------------------------------------------------- distance metrics
if "distance_metric_sweep" in ref:
    mets = ["cosine", "euclidean", "manhattan"]
    acc = np.array([[r["distance_metric_sweep"][m]["metrics"][0] for m in mets] for r in runs])
    fpr_ = np.array([[r["distance_metric_sweep"][m]["metrics"][4] for m in mets] for r in runs])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    w = 0.35; xs = np.arange(len(mets))
    ax.bar(xs - w / 2, acc.mean(0), w, yerr=acc.std(0), capsize=4, label="Accuracy")
    ax.bar(xs + w / 2, fpr_.mean(0), w, yerr=fpr_.std(0), capsize=4, label="FPR")
    ax.set_xticks(xs); ax.set_xticklabels([m.capitalize() for m in mets])
    ax.set_ylabel("%"); ax.set_title("Weighted KNN distance metrics")
    ax.legend(frameon=False)
    save(fig, "fig_distance_metrics.pdf")
    print("\n  distance metric means (ACC / FPR):")
    for i, m in enumerate(mets):
        print("    %-10s %.2f / %.2f" % (m, acc.mean(0)[i], fpr_.mean(0)[i]))

# ---------------------------------------------------------------- Table III figure
t3p = os.path.join(BASE, "table3_md.json")
if os.path.exists(t3p):
    t3 = json.load(open(t3p))
    names = list(t3["rows"].keys())
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xs = np.arange(len(names))
    a = [t3["rows"][n]["mean"][0] for n in names]
    e = [t3["rows"][n]["sd"][0] for n in names]
    c = ["#4C78A8"] * len(names)
    if PROPOSED in names: c[names.index(PROPOSED)] = "#E45756"
    ax.bar(xs, a, yerr=e, capsize=4, color=c)
    ax.set_xticks(xs); ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(50, 100)
    ax.set_title("Classifier comparison (mean $\\pm$ sd)")
    save(fig, "fig_classifier_comparison.pdf")

# ---------------------------------------------------------------- LaTeX tables
def fmt(m, s): return "%.2f $\\pm$ %.1f" % (m, s)


with open(os.path.join(OUT, "table4_generated.tex"), "w") as f:
    f.write("%% auto-generated by make_figures.py -- do not edit by hand\n")
    for n in ORDER:
        m, s = means[n], sds[n]
        cells = " & ".join(fmt(m[i], s[i]) for i in range(5))
        bold = "\\textbf{%s}" % n if n == PROPOSED else n
        f.write("%s & %s \\\\\n" % (bold, cells))
saved.append("table4_generated.tex")
print("  wrote table4_generated.tex")

summary = {"n_splits": len(runs),
           "table4_mean": {n: means[n].tolist() for n in ORDER},
           "table4_sd": {n: sds[n].tolist() for n in ORDER},
           "auc_split0": aucs,
           "figures": saved}
json.dump(summary, open(os.path.join(BASE, "figures_summary.json"), "w"), indent=2)
print("\n%d artefacts in %s" % (len(saved), OUT))

"""
Step 10 (Section 24 and required visualizations 2, 3 and 14): graphical performance analysis.

  10_accuracy_curves.eps     24.1  epoch versus training and validation accuracy (baseline and attention models)
  10_loss_curves.eps         24.2  epoch versus training and validation cross entropy
  10_bar_accuracy.eps        24.3  attention mechanism versus test accuracy (with 95% bootstrap interval)
  10_bar_f1.eps              24.4  attention mechanism versus macro F1 (with 95% bootstrap interval)
  10_accuracy_vs_params.eps  24.5  total parameters versus test accuracy
  10_f1_vs_inference.eps     24.6  inference time per image versus macro F1
  10_position_ablation.eps   Experiment 2: accuracy and F1 per position
Models that are not trained yet are skipped with a message.
"""
import numpy as np

import analysis_utils as au
import config
import experiments as E
import plot_style
import visualize as viz

plt = viz.plt
config.make_dirs()


def save(fig, stem):
    fig.tight_layout()
    print("saved", plot_style.save_fig(fig, config.plot_path(stem)))
    plt.close(fig)


def label_of(n):
    return E.EXPERIMENTS[n]["label"]


# ---- 24.1 and 24.2 curves -----------------------------------------------------------------------
curve_runs = au.available(E.CURVE_RUNS)
if "A0" not in curve_runs:
    raise SystemExit("Train the baseline (03_train_attention_type.py) first.")
for key, ylabel, stem in (("accuracy", "Accuracy", "10_accuracy_curves"), ("loss", "Cross-entropy loss", "10_loss_curves")):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split, title in ((axes[0], key, "Training"), (axes[1], "val_" + key, "Validation")):
        for n in curve_runs:
            h = au.load_history(n)
            ep = np.arange(1, len(h[split]) + 1)
            ax.plot(ep, h[split], color=viz.color_for(n), lw=1.6, label=label_of(n))
        ax.axvline(config.PHASE1_EPOCHS + 0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("Epoch"); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8)
    axes[0].text(config.PHASE1_EPOCHS + 0.6, axes[0].get_ylim()[0], " fine tuning", fontsize=8, va="bottom", color="gray")
    save(fig, stem)

# ---- 24.3 and 24.4 bars ------------------------------------------------------------------------
bars = [(lab, n) for lab, n in E.BAR_GRAPH_RUNS if au.is_done(n)]
ci = {}
for _, n in bars:
    y, p = au.test_predictions(n)
    ci[n] = au.bootstrap_ci(y, p)
for metric, ci_key, ylabel, stem in (("accuracy", "accuracy_ci", "Test accuracy", "10_bar_accuracy"),
                                     ("f1", "f1_ci", "Test macro F1-score", "10_bar_f1")):
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = np.array([au.load_metrics(n)[metric] for _, n in bars])
    lo = np.array([vals[i] - ci[n][ci_key][0] for i, (_, n) in enumerate(bars)])
    hi = np.array([ci[n][ci_key][1] - vals[i] for i, (_, n) in enumerate(bars)])
    b = ax.bar(range(len(bars)), vals, yerr=[lo, hi], capsize=3, color=[viz.color_for(n) for _, n in bars])
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([lab for lab, _ in bars], rotation=25, ha="right")
    ax.set_xlabel("Attention mechanism"); ax.set_ylabel(ylabel)
    ax.set_ylim(max(0, (vals - lo).min() - 0.05), min(1.02, (vals + hi).max() + 0.04))
    ax.grid(alpha=0.3, axis="y")
    save(fig, stem)

# ---- 24.5 and 24.6 trade-off scatters --------------------------------------------------------------
runs_all = [n for n in E.real_runs() if au.is_done(n)]
for xk, yk, xlabel, ylabel, stem, scale in (
        ("total_params", "accuracy", "Total model parameters (millions)", "Test accuracy", "10_accuracy_vs_params", 1e6),
        ("ms_per_image", "f1", "Inference time per image (ms)", "Test macro F1-score", "10_f1_vs_inference", 1.0)):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in runs_all:
        m = au.load_metrics(n)
        ax.scatter(m[xk] / scale, m[yk], s=55, color=viz.color_for(n), zorder=3)
        ax.annotate(n, (m[xk] / scale, m[yk]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=viz.color_for(n), label=f"{n}: {label_of(n)}") for n in runs_all]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="best")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    save(fig, stem)

# ---- position ablation summary ----------------------------------------------------------------------
pos = [n for n in E.GROUPS["position"] if au.is_done(n)]
if pos:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, lab in ((axes[0], "accuracy", "Test accuracy"), (axes[1], "f1", "Test macro F1-score")):
        vals = [au.load_metrics(n)[metric] for n in pos]
        b = ax.bar(range(len(pos)), vals, color=[f"C{i}" for i in range(len(pos))])
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(au.load_metrics("A0")[metric], color="k", ls="--", lw=1, label="Baseline")
        ax.set_xticks(range(len(pos))); ax.set_xticklabels([f"P{E.EXPERIMENTS[n]['label'][-1]}" for n in pos])
        ax.set_xlabel("Attention position (Channel + Spatial)"); ax.set_ylabel(lab)
        ax.set_ylim(max(0, min(vals + [au.load_metrics('A0')[metric]]) - 0.05), 1.02)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    save(fig, "10_position_ablation")

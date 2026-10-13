"""
Step 11 (Section 25 and required visualization 4): confusion matrices of the baseline and the selected attention model,
with per class precision, recall and F1 and the largest confusions.

  11_confusion_matrices.eps
  results/11_per_class_metrics.csv / .tex      per class precision, recall, F1 of both models and the change
  results/11_top_confusions.csv                the most frequent confusions of both models
"""
import numpy as np

import analysis_utils as au
import config
import plot_style
import report_utils as ru
import visualize as viz

plt = viz.plt
config.make_dirs()
sel = au.selected_model()
print("Selected attention model:", sel)
names = {"A0": "Baseline ResNet50", sel: f"{sel}: {au.load_metrics(sel)['label']}"}

cms, preds = {}, {}
for n in names:
    y, p = au.test_predictions(n)
    cms[n], preds[n] = au.confusion(y, p), p
vmax = max(c.max() for c in cms.values())
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
for ax, (n, title) in zip(axes, names.items()):
    im = ax.imshow(cms[n], cmap="Blues", vmin=0, vmax=vmax)
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            ax.text(j, i, cms[n][i, j], ha="center", va="center", fontsize=11,
                    color="white" if cms[n][i, j] > vmax / 2 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(viz.CLASS_TITLES, rotation=35, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(viz.CLASS_TITLES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"{title}\naccuracy {100 * np.trace(cms[n]) / cms[n].sum():.2f}%", fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
print("saved", plot_style.save_fig(fig, config.plot_path("11_confusion_matrices")))

mb_, ms_ = au.classification_metrics(y, preds["A0"]), au.classification_metrics(y, preds[sel])
header = ["Class", "Baseline P", "Baseline R", "Baseline F1", "Attention P", "Attention R", "Attention F1", "Delta recall", "Delta F1"]
rows = []
for i, c in enumerate(config.CLASS_NAMES):
    rows.append([c, *[ru.pct(mb_[k][i]) for k in ("per_class_precision", "per_class_recall", "per_class_f1")],
                 *[ru.pct(ms_[k][i]) for k in ("per_class_precision", "per_class_recall", "per_class_f1")],
                 f"{100 * (ms_['per_class_recall'][i] - mb_['per_class_recall'][i]):+.2f}",
                 f"{100 * (ms_['per_class_f1'][i] - mb_['per_class_f1'][i]):+.2f}"])
ru.write_csv(config.result_path("11_per_class_metrics.csv"), header, rows)
ru.write_latex_table(config.result_path("11_per_class_metrics.tex"), header, rows,
                     f"Per class precision (P), recall (R) and F1 (%) of the baseline and {sel}.", "tab:per_class", True)
for r in rows:
    print(r)

conf_rows = []
for n in names:
    cm = cms[n].copy()
    np.fill_diagonal(cm, 0)
    for k in np.argsort(cm.ravel())[::-1][:5]:
        i, j = divmod(int(k), config.NUM_CLASSES)
        if cm[i, j] > 0:
            conf_rows.append([n, config.CLASS_NAMES[i], config.CLASS_NAMES[j], int(cm[i, j])])
ru.write_csv(config.result_path("11_top_confusions.csv"), ["Model", "True class", "Predicted as", "Count"], conf_rows)
for r in conf_rows:
    print(r)

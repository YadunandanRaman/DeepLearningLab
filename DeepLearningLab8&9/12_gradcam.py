"""
Step 12 (Section 27 and required visualizations 11 and 12): Grad-CAM, baseline versus the selected attention model.

Layout (Section 27): Input | Baseline Grad-CAM | Attention model Grad-CAM, for 3 correctly and 2 incorrectly
classified test images. Grad-CAM is computed on conv5_block3_out for the predicted class.

  12_gradcam_baseline_vs_attention.eps
  results/12_gradcam_statistics.csv     compactness of the maps over the whole test set (area above 0.5,
                                        normalised entropy, distance of the peak from the image centre)
  results/12_gradcam_examples.csv       which images were shown
"""
import numpy as np

import analysis_utils as au
import config
import data_utils
import gradcam
import plot_style
import report_utils as ru
import train_utils
import visualize as viz

plt = viz.plt
config.make_dirs()
d = data_utils.load_split()
x, y = d["x_test"], d["y_test"]
sel = au.selected_model()
print("Selected attention model:", sel)
base, att = train_utils.load_trained("A0"), train_utils.load_trained(sel)

preds = {"A0": au.test_predictions("A0")[1], sel: au.test_predictions(sel)[1]}
correct, wrong = viz.pick_examples(y, preds, ["A0", sel])
# if the attention model never errs, show images the baseline gets wrong; otherwise images the attention model gets wrong
if not wrong:
    _, wrong = viz.pick_examples(y, preds, ["A0"])
chosen = correct + wrong

cam_b, pred_b, prob_b, _ = gradcam.gradcam(base, x[chosen])
cam_a, pred_a, prob_a, _ = gradcam.gradcam(att, x[chosen])
fig, axes = plt.subplots(len(chosen), 3, figsize=(8.4, 2.9 * len(chosen)))
rows = []
for r, i in enumerate(chosen):
    kind = "correct" if r < len(correct) else "incorrect"
    viz.show(axes[r, 0], x[i], f"Input: {config.CLASS_NAMES[y[i]]} ({kind} example)")
    viz.show(axes[r, 1], viz.overlay(x[i], cam_b[r]), f"Baseline: {config.CLASS_NAMES[pred_b[r]]} ({prob_b[r].max():.2f})")
    viz.show(axes[r, 2], viz.overlay(x[i], cam_a[r]), f"{sel}: {config.CLASS_NAMES[pred_a[r]]} ({prob_a[r].max():.2f})")
    rows.append([int(i), config.CLASS_NAMES[y[i]], kind, config.CLASS_NAMES[pred_b[r]], config.CLASS_NAMES[pred_a[r]]])
fig.tight_layout()
print("saved", plot_style.save_fig(fig, config.plot_path("12_gradcam_baseline_vs_attention")))
ru.write_csv(config.result_path("12_gradcam_examples.csv"),
             ["test index", "true class", "kind", "baseline prediction", f"{sel} prediction"], rows)

# compactness statistics over the whole test set
sb, sa = gradcam.cam_statistics(gradcam.gradcam(base, x)[0]), gradcam.cam_statistics(gradcam.gradcam(att, x)[0])
header = ["Statistic", "Baseline", sel]
stat_rows = [[k, f"{sb[k].mean():.4f}", f"{sa[k].mean():.4f}"] for k in sb]
ru.write_csv(config.result_path("12_gradcam_statistics.csv"), header, stat_rows)
for r in stat_rows:
    print(r)

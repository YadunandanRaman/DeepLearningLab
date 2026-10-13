"""
Step 15 (Sections 30, 32 and 36): qualitative comparison across attention mechanisms on the SAME images:
3 correctly classified and 2 incorrectly classified test images.

  15_qualitative_comparison.eps     Section 30 layout:
        Input | Baseline Grad-CAM | Channel Grad-CAM | Spatial attention map | Channel+Spatial Grad-CAM | Self-attention response
  15_mechanism_comparison.eps       Section 32 / 36 layout (Grad-CAM of every mechanism):
        Input | Baseline | SE | ECA | CBAM | Self-Attention | Coordinate
  results/15_qualitative_examples.csv   the images used and each model's prediction

Titles of wrongly classified predictions are printed in red. Missing models are left out with a message.
"""
import numpy as np

import analysis_utils as au
import config
import data_utils
import experiments as E
import gradcam
import model_builder as mb
import plot_style
import report_utils as ru
import train_utils
import visualize as viz

plt = viz.plt
config.make_dirs()
d = data_utils.load_split()
x, y = d["x_test"], d["y_test"]
short = {"A0": "Baseline", "A1": "Channel (SE)", "A2": "Spatial", "A3": "Channel+Spatial", "A4": "Self-Attention",
         "A5": "Combined", "M1": "ECA", "M2": "CBAM", "M3": "Coordinate"}


def internal_map(run, key):
    model = train_utils.load_trained(run)
    pos = E.resolve(run)[1]["positions"][0]
    layer, io = model.get_layer(f"att_P{pos}"), mb.attention_io_model(model, pos)

    def f(img):
        before, _ = io(data_utils.preprocess_array(img[None]), training=False)
        return np.asarray(layer.explain(before)[1][key][0], dtype=np.float32)
    return f


def color(ok):
    return "black" if ok else "red"


def figure(columns, runs_for_examples, stem, title_note=""):
    """columns: list of (kind, run) with kind in {'input', 'gradcam', 'internal:<key>'}."""
    runs = [r for r in runs_for_examples if au.is_done(r)]
    preds = {r: au.test_predictions(r)[1] for r in runs}
    correct, wrong = viz.pick_examples(y, preds, runs)
    chosen = correct + wrong
    if not chosen:
        print("no examples found for", stem)
        return
    cols = [c for c in columns if c[0] == "input" or au.is_done(c[1])]
    for c in columns:
        if c not in cols:
            print(f"[skip] {c[1]} not trained: column left out of {stem}")
    fig, axes = plt.subplots(len(chosen), len(cols), figsize=(2.35 * len(cols), 2.5 * len(chosen)))
    rows = []
    for r, i in enumerate(chosen):
        kind = "correct" if r < len(correct) else "incorrect"
        row = [int(i), config.CLASS_NAMES[y[i]], kind]
        for c, (ck, run) in enumerate(cols):
            ax = axes[r, c]
            if ck == "input":
                viz.show(ax, x[i], f"{config.CLASS_NAMES[y[i]]} ({kind})", fontsize=8)
            elif ck == "gradcam":
                cam, pred, prob, _ = gradcam.gradcam(train_utils.load_trained(run), x[i:i + 1])
                viz.show(ax, viz.overlay(x[i], cam[0]), f"{short[run]}: {config.CLASS_NAMES[pred[0]]}", fontsize=8)
                ax.title.set_color(color(pred[0] == y[i]))
                row.append(config.CLASS_NAMES[pred[0]])
            else:
                key = ck.split(":", 1)[1]
                m = internal_map(run, key)(x[i])
                viz.show(ax, viz.overlay(x[i], viz.resize_map(m)), f"{short[run]} map" if "spatial" in key else
                         f"{short[run]} response", fontsize=8)
                row.append("")
        rows.append(row)
    fig.tight_layout()
    print("saved", plot_style.save_fig(fig, config.plot_path(stem)))
    plt.close(fig)
    header = ["test index", "true class", "kind"] + [f"{short[run]} prediction" for ck, run in cols if ck == "gradcam"]
    body = [[v for j, v in enumerate(row) if j < 3 or v != ""] for row in rows]
    ru.write_csv(config.result_path(f"{stem}_examples.csv"), header, body)


figure([("input", None), ("gradcam", "A0"), ("gradcam", "A1"), ("internal:spatial/spatial_map", "A2"),
        ("gradcam", "A3"), ("internal:self/response", "A4")],
       ["A0", "A1", "A2", "A3", "A4"], "15_qualitative_comparison")

figure([("input", None), ("gradcam", "A0"), ("gradcam", "A1"), ("gradcam", "M1"), ("gradcam", "M2"),
        ("gradcam", "A4"), ("gradcam", "M3")],
       ["A0", "A1", "M1", "M2", "A4", "M3"], "15_mechanism_comparison")

"""
Step 14 (Section 29 and required visualization 13): spatial attention maps of the Channel+Spatial model at P2, P3 and P4
for the SAME test images (B2, B3 = A3 and B4).

  14_position_maps_heatmap.eps    Input | M_s at P2 | M_s at P3 | M_s at P4
  14_position_maps_overlay.eps    the same maps overlaid on the input
  results/14_position_map_statistics.csv
"""
import numpy as np

import analysis_utils as au
import config
import data_utils
import model_builder as mb
import plot_style
import report_utils as ru
import train_utils
import visualize as viz

plt = viz.plt
config.make_dirs()
d = data_utils.load_split()
x, y = d["x_test"], d["y_test"]
runs = {2: "B2", 3: "B3", 4: "B4"}
missing = [r for r in runs.values() if not au.is_done(r)]
if missing:
    raise SystemExit(f"Train {missing} first (04_train_attention_position.py).")

parts = {}
for pos, run in runs.items():
    model = train_utils.load_trained(run)
    parts[pos] = (model.get_layer(f"att_P{pos}"), mb.attention_io_model(model, pos))


def spatial_map(pos, img):
    layer, io = parts[pos]
    before, _ = io(data_utils.preprocess_array(img[None]), training=False)
    _, info = layer.explain(before)
    return np.asarray(info["spatial/spatial_map"][0], dtype=np.float32)


rng = np.random.RandomState(config.SEED)
idx = [int(rng.choice(np.where(y == c)[0])) for c in range(config.NUM_CLASSES)][:4]
for mode, stem in (("heat", "14_position_maps_heatmap"), ("overlay", "14_position_maps_overlay")):
    fig, axes = plt.subplots(len(idx), 4, figsize=(10.5, 2.7 * len(idx)))
    for r, i in enumerate(idx):
        viz.show(axes[r, 0], x[i], f"Input ({config.CLASS_NAMES[y[i]]})")
        for c, pos in enumerate(runs, start=1):
            m = spatial_map(pos, x[i])
            heat = viz.resize_map(m)
            title = f"$M_s^{{P_{pos}}}$ ({m.shape[0]}x{m.shape[1]})"
            if mode == "heat":
                viz.show(axes[r, c], heat, title, "jet")
            else:
                viz.show(axes[r, c], viz.overlay(x[i], heat), title)
    fig.tight_layout()
    print("saved", plot_style.save_fig(fig, config.plot_path(stem)))
    plt.close(fig)

# numbers that support the visual description: how concentrated and how contrasted the maps are
rows = []
sample = x[::max(1, len(x) // 100)]
for pos in runs:
    ms = np.stack([spatial_map(pos, im) for im in sample])
    hn = (ms - ms.min((1, 2), keepdims=True)) / (np.ptp(ms, (1, 2), keepdims=True) + 1e-8)
    rows.append([f"P{pos}", f"{ms.shape[1]}x{ms.shape[2]}", f"{ms.min():.3f}", f"{ms.max():.3f}", f"{ms.mean():.3f}",
                 f"{np.ptp(ms, (1, 2)).mean():.3f}", f"{(hn > 0.5).mean():.3f}"])
    print(rows[-1])
ru.write_csv(config.result_path("14_position_map_statistics.csv"),
             ["Position", "Map size", "Min", "Max", "Mean", "Mean range per image", "Area above 0.5 (normalised)"], rows)

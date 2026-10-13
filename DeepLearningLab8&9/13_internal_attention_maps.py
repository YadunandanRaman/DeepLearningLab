"""
Step 13 (Section 28 and required visualizations 5 to 10): INTERNAL attention, read from the forward pass of the
attention modules (not Grad-CAM).

  13_feature_maps_before_after.eps   Input | feature map before attention | after channel | after channel+spatial (A3)
  13_channel_weights.eps             28.1  top-K channel weights of one image, and the same channels for all five classes (A1)
  13_spatial_attention.eps           28.2  Input | spatial attention map | attention overlay (A2)
  13_self_attention.eps              28.3  query maps, mean, maximum and aggregated response of self-attention (A4)
  results/13_channel_weight_statistics.csv, 13_spatial_map_statistics.csv, 13_self_attention_statistics.csv

Feature maps are shown as the mean over channels of the H x W x C tensor entering / leaving the attention module.
"""
import numpy as np
import tensorflow as tf

import analysis_utils as au
import attention
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
rng = np.random.RandomState(config.SEED)
per_class = [int(rng.choice(np.where(y == c)[0])) for c in range(config.NUM_CLASSES)]   # one test image per class


def save(fig, stem):
    fig.tight_layout()
    print("saved", plot_style.save_fig(fig, config.plot_path(stem)))
    plt.close(fig)


def io_for(name, pos=3):
    model = train_utils.load_trained(name)
    return model.get_layer(f"att_P{pos}"), mb.attention_io_model(model, pos)


def batched_explain(name, pos=3, n=None):
    """Run explain() over the test set; returns dict of stacked arrays (feature maps are not stored)."""
    layer, io = io_for(name, pos)
    out = {}
    xs = x if n is None else x[:n]
    for i in range(0, len(xs), 32):
        before, _ = io(data_utils.preprocess_array(xs[i:i + 32]), training=False)
        _, info = layer.explain(before)
        for k, v in info.items():
            out.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------
# feature maps before and after attention (A3: channel then spatial)
# ---------------------------------------------------------------------------------------------------
if au.is_done("A3"):
    layer, io = io_for("A3")
    idx = per_class[:3]
    fig, axes = plt.subplots(len(idx), 4, figsize=(10, 2.7 * len(idx)))
    for r, i in enumerate(idx):
        before, after = io(data_utils.preprocess_array(x[i:i + 1]), training=False)
        mid = before
        stages = []
        for blk in layer.blocks:
            mid, _ = blk.explain(mid)
            stages.append(mid)
        viz.show(axes[r, 0], x[i], f"Input ({config.CLASS_NAMES[y[i]]})")
        viz.show(axes[r, 1], viz.feature_image(before[0].numpy()), "Feature map before attention", "viridis")
        viz.show(axes[r, 2], viz.feature_image(stages[0][0].numpy()), "After channel attention", "viridis")
        viz.show(axes[r, 3], viz.feature_image(after[0].numpy()), "After channel + spatial attention", "viridis")
    print("feature tensor entering / leaving the module:", tuple(before.shape[1:]), "->", tuple(after.shape[1:]))
    save(fig, "13_feature_maps_before_after")
else:
    print("[skip] A3 not trained: feature map figure")

# ---------------------------------------------------------------------------------------------------
# channel attention weights (A1)
# ---------------------------------------------------------------------------------------------------
if au.is_done("A1"):
    layer, io = io_for("A1")
    K = config.TOP_K_CHANNELS
    W = []
    for i in per_class:
        before, _ = io(data_utils.preprocess_array(x[i:i + 1]), training=False)
        _, info = layer.explain(before)
        W.append(np.asarray(info["channel/channel_weights"][0], dtype=np.float32))
    W = np.stack(W)
    top = np.argsort(W[0])[::-1][:K]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(K), W[0][top], color="C0")
    axes[0].set_xticks(range(K)); axes[0].set_xticklabels(top, rotation=90, fontsize=7)
    axes[0].set_xlabel("Channel index"); axes[0].set_ylabel("Channel attention weight $m_c$")
    axes[0].set_title(f"Top-{K} of {W.shape[1]} channels ({config.CLASS_NAMES[y[per_class[0]]]} image)")
    axes[0].grid(alpha=0.3, axis="y")
    for j, i in enumerate(per_class):
        axes[1].plot(range(K), W[j][top], "o-", ms=3, lw=1.2, label=config.CLASS_NAMES[y[i]])
    axes[1].set_xticks(range(K)); axes[1].set_xticklabels(top, rotation=90, fontsize=7)
    axes[1].set_xlabel("Channel index (same channels as left)"); axes[1].set_ylabel("Channel attention weight $m_c$")
    axes[1].set_title("The same channels for one image of each class"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    save(fig, "13_channel_weights")

    allw = batched_explain("A1")["channel/channel_weights"]                      # (N, C) over the whole test set
    mean_w = allw.mean(0)
    top_idx = np.argsort(mean_w)[::-1][:K]
    cls_means = np.stack([allw[y == c].mean(0) for c in range(config.NUM_CLASSES)])
    corr = np.corrcoef(cls_means)
    stats = [["weight range (min, max)", f"{allw.min():.4f}, {allw.max():.4f}"],
             ["mean weight", f"{allw.mean():.4f}"],
             ["fraction of weights below 0.5", f"{(allw < 0.5).mean():.4f}"],
             [f"share of the total weight in the top-{K} channels (per image, mean)",
              f"{np.sort(allw, 1)[:, -K:].sum(1).mean() / allw.sum(1).mean():.4f}"],
             [f"uniform share of K channels ({K}/{allw.shape[1]})", f"{K / allw.shape[1]:.4f}"],
             ["mean over channels of the std across images", f"{allw.std(0).mean():.4f}"],
             ["mean over channels of the std of the class mean weights", f"{cls_means.std(0).mean():.4f}"],
             ["mean off diagonal correlation between class mean weight vectors", f"{(corr.sum() - 5) / 20:.4f}"],
             [f"mean weight of the global top-{K} channels", f"{mean_w[top_idx].mean():.4f}"]]
    ru.write_csv(config.result_path("13_channel_weight_statistics.csv"), ["Statistic", "Value"], stats)
    for s in stats:
        print(s)
else:
    print("[skip] A1 not trained: channel weights")

# ---------------------------------------------------------------------------------------------------
# spatial attention (A2)
# ---------------------------------------------------------------------------------------------------
if au.is_done("A2"):
    layer, io = io_for("A2")
    idx = per_class[:4]
    fig, axes = plt.subplots(len(idx), 3, figsize=(8.4, 2.8 * len(idx)))
    for r, i in enumerate(idx):
        before, _ = io(data_utils.preprocess_array(x[i:i + 1]), training=False)
        _, info = layer.explain(before)
        smap = np.asarray(info["spatial/spatial_map"][0], dtype=np.float32)
        heat = viz.resize_map(smap)
        viz.show(axes[r, 0], x[i], f"Input ({config.CLASS_NAMES[y[i]]})")
        viz.show(axes[r, 1], heat, f"Spatial attention map ({smap.shape[0]}x{smap.shape[1]} to 224x224)", "jet")
        viz.show(axes[r, 2], viz.overlay(x[i], heat), "Attention overlay")
    save(fig, "13_spatial_attention")
    maps = batched_explain("A2")["spatial/spatial_map"]
    hn = (maps - maps.min((1, 2), keepdims=True)) / (np.ptp(maps, (1, 2), keepdims=True) + 1e-8)
    rows = [["raw map minimum", f"{maps.min():.4f}"], ["raw map maximum", f"{maps.max():.4f}"],
            ["raw map mean", f"{maps.mean():.4f}"], ["mean range max-min per image", f"{np.ptp(maps, (1, 2)).mean():.4f}"],
            ["area above 0.5 after per image normalisation", f"{(hn > 0.5).mean():.4f}"]]
    ru.write_csv(config.result_path("13_spatial_map_statistics.csv"), ["Statistic", "Value"], rows)
    for s in rows:
        print(s)
else:
    print("[skip] A2 not trained: spatial attention")

# ---------------------------------------------------------------------------------------------------
# self-attention (A4)
# ---------------------------------------------------------------------------------------------------
if au.is_done("A4"):
    layer, io = io_for("A4")
    sa = layer.get_block("self")
    idx = per_class[:3]
    qs = config.SELF_ATT_QUERIES
    ncol = 1 + len(qs) + 3
    fig, axes = plt.subplots(len(idx), ncol, figsize=(1.9 * ncol, 2.1 * len(idx)))
    cell = config.IMG_SIZE / sa.h
    for r, i in enumerate(idx):
        before, _ = io(data_utils.preprocess_array(x[i:i + 1]), training=False)
        _, info = sa.explain(before)
        qmaps = sa.query_maps(before, qs)[0].numpy()
        viz.show(axes[r, 0], x[i], f"Input ({config.CLASS_NAMES[y[i]]})", fontsize=8)
        for c, (q, qm) in enumerate(zip(qs, qmaps), start=1):
            viz.show(axes[r, c], viz.overlay(x[i], viz.resize_map(qm)), f"Query ({q[0]},{q[1]})", fontsize=8)
            axes[r, c].plot((q[1] + 0.5) * cell, (q[0] + 0.5) * cell, "w*", ms=9, mec="k")
        for c, (key, ttl) in enumerate((("mean_over_queries", "Mean over queries"), ("max_over_queries", "Max over queries"),
                                        ("response", "Aggregated response")), start=1 + len(qs)):
            viz.show(axes[r, c], viz.resize_map(info[key][0].numpy()), ttl, "jet", fontsize=8)
    save(fig, "13_self_attention")
    full = batched_explain("A4")
    uniform_dist = float(sa.dist.numpy().mean())
    rows = [["learned gamma (residual scale)", f"{float(sa.gamma.numpy()):.5f}"],
            ["attention matrix size N x N", f"{sa.n} x {sa.n} = {sa.n * sa.n}"],
            ["mean attention distance (grid cells)", f"{full['self/mean_distance'].mean():.3f}"],
            ["mean distance if attention were uniform (grid cells)", f"{uniform_dist:.3f}"],
            ["mean aggregated response", f"{full['self/response'].mean():.5f}"],
            ["max aggregated response", f"{full['self/response'].max():.5f}"]]
    ru.write_csv(config.result_path("13_self_attention_statistics.csv"), ["Statistic", "Value"], rows)
    for s in rows:
        print(s)
else:
    print("[skip] A4 not trained: self-attention")

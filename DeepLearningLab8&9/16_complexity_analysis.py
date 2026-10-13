"""
Step 16 (Sections 13, 16, 33.2, 37 questions on cost): complexity analysis.

  results/16_complexity.csv / .tex          parameters, extra parameters, analytic multiply accumulates, measured inference time,
                                            peak GPU memory and training time of every run, relative to the baseline
  results/16_self_attention_cost.csv / .tex size of the N x N self-attention matrix at P1 to P4 (N = H x W)

MACs: the ResNet50 backbone count is computed exactly from the convolution shapes of the built model; the attention
cost is the analytic count of attention.macs_component (pooling, MLPs, convolutions, projections, N x N products).
Inference time and memory are measured by train_utils.measure_inference on the test set.
"""
import numpy as np

import analysis_utils as au
import attention
import config
import experiments as E
import model_builder as mb
import report_utils as ru
from tensorflow.keras import layers

config.make_dirs()


def backbone_macs():
    model = mb.build_model(E.resolve("A0")[1], pretrained=False)
    total = 0
    for l in model.layers:
        if isinstance(l, layers.Conv2D):
            h, w, cout = l.output.shape[1:]
            total += h * w * l.kernel_size[0] * l.kernel_size[1] * l.input.shape[-1] * cout
        elif isinstance(l, layers.Dense):
            total += l.input.shape[-1] * l.units
    return int(total)


def attention_macs(spec):
    total = 0
    for p in spec["positions"]:
        info = config.POSITIONS[p]
        for comp in spec["components"]:
            total += attention.macs_component(comp, info["resolution"], info["resolution"], info["channels"],
                                              config.REDUCTION, spec["kernel"])
    return int(total)


BB = backbone_macs()
print(f"ResNet50 backbone: {BB / 1e9:.3f} GMACs per image")
base = au.load_metrics("A0")
header = ["Run", "Model", "Total params", "Extra params", "Extra params (%)", "Attention params", "Attention MMACs",
          "Total GMACs", "Extra MACs (%)", "Inference (ms/img)", "Extra time (%)", "Peak GPU (MB)", "Train time (min)"]
rows = []
for n in E.real_runs():
    m = au.load_metrics(n, required=False)
    if m is None:
        continue
    spec = E.resolve(n)[1]
    a_macs = attention_macs(spec)
    rows.append([n, m["label"], m["total_params"], m["total_params"] - base["total_params"],
                 f"{100 * (m['total_params'] / base['total_params'] - 1):.2f}", m["attention_params"],
                 f"{a_macs / 1e6:.2f}", f"{(BB + a_macs) / 1e9:.3f}", f"{100 * a_macs / BB:.2f}",
                 f"{m['ms_per_image']:.2f}", f"{100 * (m['ms_per_image'] / base['ms_per_image'] - 1):.1f}",
                 "n/a" if m.get("peak_gpu_mb") is None else f"{m['peak_gpu_mb']:.0f}", f"{m['train_seconds'] / 60:.1f}"])
ru.write_csv(config.result_path("16_complexity.csv"), header, rows)
ru.write_latex_table(config.result_path("16_complexity.tex"), header, rows,
                     "Complexity of every model: parameters, multiply accumulates, inference time and training time.",
                     "tab:complexity", True)
for r in rows:
    print(r)

# self-attention matrix size versus resolution (Q13 and Q14)
sh = ["Position", "Resolution", "Channels", "Tokens N", "Matrix entries N^2", "Memory per image (MB, fp32)", "Relative to P3",
      "Self-attention MMACs"]
ref = attention.self_attention_matrix_size(14, 14)["entries"]
srows = []
for pos, info in config.POSITIONS.items():
    r = info["resolution"]
    s = attention.self_attention_matrix_size(r, r)
    srows.append([info["name"], f"{r}x{r}", info["channels"], s["tokens"], s["entries"], f"{s['megabytes_per_image_fp32']:.3f}",
                  f"{s['entries'] / ref:.2f}x", f"{attention.macs_component('self', r, r, info['channels']) / 1e6:.1f}"])
ru.write_csv(config.result_path("16_self_attention_cost.csv"), sh, srows)
ru.write_latex_table(config.result_path("16_self_attention_cost.tex"), sh, srows,
                     "Size of the self-attention matrix at the four insertion positions (it grows with the fourth power of the "
                     "linear resolution).", "tab:selfatt_cost", True)
for r in srows:
    print(r)

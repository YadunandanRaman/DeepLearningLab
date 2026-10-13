"""
Step 2 (Sections 4, 5, 7, 13 and 38.1): architecture summary and the input / output feature dimensions of
every attention module at every position P1 to P4.

For each module (channel, spatial, self, eca, cbam, coord) and each position the script builds the module
on the real feature shape, runs it on a dummy tensor, checks that the output shape equals the input shape
(Section 13), and records the number of parameters and the analytic multiply accumulates.

Outputs
  outputs/results/02_feature_dimensions.csv / .tex
  outputs/results/02_stage_shapes.csv
  outputs/results/02_summary_baseline.txt, 02_summary_channel_spatial_P3.txt
"""
import numpy as np
import tensorflow as tf

import attention
import config
import experiments as E
import model_builder as mb
import report_utils

config.make_dirs()
config.set_seed()

# 1. baseline architecture and stage output shapes (Section 5)
_, base_spec = E.resolve("A0")
base = mb.build_model(base_spec, pretrained=False)
with open(config.result_path("02_summary_baseline.txt"), "w") as f:
    base.summary(print_fn=lambda s, **kw: f.write(s + "\n"), expand_nested=False)
_, a3_spec = E.resolve("A3")
a3 = mb.build_model(a3_spec, pretrained=False)
with open(config.result_path("02_summary_channel_spatial_P3.txt"), "w") as f:
    a3.summary(print_fn=lambda s, **kw: f.write(s + "\n"), expand_nested=False)

stage_rows = []
for stage, shape in mb.stage_shapes(base):
    stage_rows.append([stage, f"{shape[0]}x{shape[1]}", shape[2]])
    print(stage, shape)
report_utils.write_csv(config.result_path("02_stage_shapes.csv"), ["Stage", "Resolution", "Channels"], stage_rows)
p = mb.count_params(base)
print(f"baseline: total {p['total']:,} parameters")

# 2. attention modules at every position (Section 38.1)
header = ["Module", "Position", "Input shape", "Output shape", "Shape preserved", "Parameters", "MACs (M)"]
rows = []
for kind in attention.COMPONENT_NAMES:
    for pos, info in config.POSITIONS.items():
        h = w = info["resolution"]
        c = info["channels"]
        layer = attention.make_component(kind, config.REDUCTION, config.SPATIAL_KERNEL, config.GATE_BIAS_INIT)
        x = tf.zeros((1, h, w, c))
        y = layer(x)
        same = tuple(y.shape) == tuple(x.shape)
        assert same, f"{kind} at P{pos} changed the feature shape: {x.shape} -> {y.shape}"
        n_par = int(sum(int(np.prod(v.shape)) for v in layer.weights))
        macs = attention.macs_component(kind, h, w, c)
        rows.append([kind, info["name"], f"{h}x{w}x{c}", f"{tuple(y.shape)[1]}x{tuple(y.shape)[2]}x{tuple(y.shape)[3]}",
                     "yes" if same else "NO", n_par, f"{macs / 1e6:.2f}"])
        print(rows[-1])
report_utils.write_csv(config.result_path("02_feature_dimensions.csv"), header, rows)
report_utils.write_latex_table(config.result_path("02_feature_dimensions.tex"), header, rows,
                               "Input and output feature dimensions, parameters and multiply accumulates of every "
                               "attention module at every insertion position.", "tab:feature_dimensions", resize=True)

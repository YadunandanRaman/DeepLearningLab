"""
Step 9 (Sections 14 to 19, 22, 23, 33.1, 38): every result table, confidence intervals, significance tests and
an automatically generated evidence file that supports the answers to the Section 37 questions.

Tables (CSV for checking, LaTeX for the report)
  09_exp1_attention_type        Main Experiment 1  (A0 to A5)
  09_exp2_attention_position    Main Experiment 2  (B1 to B4)
  09_exp3_multiple_positions    Main Experiment 3  (P3, P4, P3+P4, P2+P3+P4)
  09_exp4_attention_order       Main Experiment 4  (C->S versus S->C)
  09_exp5_kernel_size           Main Experiment 5  (Section 18 table: kernel, accuracy, macro F1, inference time)
  09_table_19_1_mechanisms      Section 19.1 table
  09_table_23_main_results      Section 23 required main result table
  09_all_runs                   every metric of every run
  09_significance               bootstrap 95% CI and McNemar test against the baseline
  09_observations_section38     the quantities listed in Section 38
  09_evidence.txt               plain language facts for the discussion
Runs that are not trained yet appear as n/a, so the script can be run at any time.
"""
import numpy as np

import analysis_utils as au
import config
import experiments as E
import report_utils as ru

config.make_dirs()
M = {n: au.load_metrics(n, required=False) for n in E.EXPERIMENTS}


def have(n):
    return M[n] is not None


def val(n, key):
    return M[n][key] if have(n) else None


def row_metrics(label, n, extra=()):
    return [label, *extra, ru.pct(val(n, "accuracy")), ru.pct(val(n, "precision")), ru.pct(val(n, "recall")),
            ru.pct(val(n, "f1")), ru.millions(val(n, "total_params")), ru.ms(val(n, "ms_per_image"))]


def emit(stem, header, rows, caption, label, resize=False):
    ru.write_csv(config.result_path(f"{stem}.csv"), header, rows)
    ru.write_latex_table(config.result_path(f"{stem}.tex"), header, rows, caption, label, resize)
    print(f"\n{stem}")
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(str(c) for c in r))


H = ["Accuracy (%)", "Precision (%)", "Recall (%)", "Macro F1 (%)", "Parameters", "Inference (ms/img)"]

# Experiment 1
emit("09_exp1_attention_type", ["Model", *H],
     [row_metrics(f"{n}: {E.EXPERIMENTS[n]['label']}", n) for n in E.GROUPS["type"]],
     "Main Experiment 1: effect of attention type at P3 (test set).", "tab:exp1", True)

# Experiment 2
geom = {1: ("56x56", 256), 2: ("28x28", 512), 3: ("14x14", 1024), 4: ("7x7", 2048)}
emit("09_exp2_attention_position", ["Exp", "Position", "Resolution", "Channels", *H],
     [row_metrics(n, n, (f"P{p}", geom[p][0], geom[p][1])) for n, p in zip(E.GROUPS["position"], (1, 2, 3, 4))],
     "Main Experiment 2: Channel then Spatial attention at different positions (test set).", "tab:exp2", True)

# Experiment 3
emit("09_exp3_multiple_positions", ["Positions", *H],
     [row_metrics(E.EXPERIMENTS[n]["label"], n) for n in E.GROUPS["multi"]],
     "Main Experiment 3: Channel then Spatial attention at one or several positions (test set).", "tab:exp3", True)

# Experiment 4
emit("09_exp4_attention_order", ["Order at P3", *H],
     [row_metrics(E.EXPERIMENTS[n]["label"], n) for n in E.GROUPS["order"]],
     "Main Experiment 4: order of channel and spatial attention at P3 (test set).", "tab:exp4", True)

# Experiment 5 (columns exactly as in Section 18)
emit("09_exp5_kernel_size", ["Spatial kernel", "Accuracy (%)", "Macro F1 (%)", "Inference (ms/img)"],
     [[E.EXPERIMENTS[n]["label"], ru.pct(val(n, "accuracy")), ru.pct(val(n, "f1")), ru.ms(val(n, "ms_per_image"))]
      for n in E.GROUPS["kernel"]],
     "Main Experiment 5: kernel size of spatial attention at P3 (test set).", "tab:exp5")

# Section 19.1
emit("09_table_19_1_mechanisms", ["Attention", *H],
     [row_metrics(lab, n) for lab, n in E.TABLE_19_1],
     "Comparison of different attention mechanisms at P3 (test set).", "tab:mechanisms", True)

# Section 23
emit("09_table_23_main_results", ["Model", "Position", *H],
     [row_metrics(lab, n, (pos,)) for lab, n, pos in E.TABLE_23],
     "Main result table (test set).", "tab:main", True)

# all runs, every metric
real = E.real_runs()
keys = ["accuracy", "precision", "recall", "f1", "test_loss", "best_val_accuracy", "val_loss_at_best",
        "final_train_accuracy", "final_val_accuracy", "final_train_loss", "final_val_loss", "train_val_gap",
        "total_params", "trainable_params_phase1", "trainable_params_phase2", "attention_params",
        "ms_per_image", "ms_std", "peak_gpu_mb", "train_seconds", "best_epoch", "epochs_to_95pct_of_best"]
ru.write_csv(config.result_path("09_all_runs.csv"), ["run", "label", "positions", "components", "kernel", *keys],
             [[n, M[n]["label"], "+".join(f"P{p}" for p in M[n]["positions"]) or "-", ">".join(M[n]["components"]) or "-",
               M[n]["kernel"], *[M[n].get(k) for k in keys]] for n in real if have(n)])

# bootstrap CI and McNemar against the baseline
sig_rows = []
if have("A0"):
    y0, p0 = au.test_predictions("A0")
    for n in real:
        if not have(n):
            continue
        y, p = au.test_predictions(n)
        ci = au.bootstrap_ci(y, p)
        mc = au.mcnemar_exact(y, p0, p) if n != "A0" else {"a_only_correct": 0, "b_only_correct": 0, "p_value": 1.0}
        sig_rows.append([n, M[n]["label"], ru.pct(M[n]["accuracy"]), f"[{ru.pct(ci['accuracy_ci'][0])}, {ru.pct(ci['accuracy_ci'][1])}]",
                         ru.pct(M[n]["f1"]), f"[{ru.pct(ci['f1_ci'][0])}, {ru.pct(ci['f1_ci'][1])}]",
                         mc["a_only_correct"], mc["b_only_correct"], f"{mc['p_value']:.4f}"])
    emit("09_significance", ["Run", "Model", "Accuracy (%)", "Accuracy 95% CI", "Macro F1 (%)", "F1 95% CI",
                             "Baseline only correct", "Model only correct", "McNemar p"],
         sig_rows, "Bootstrap 95% confidence intervals and exact McNemar test against the baseline.", "tab:significance", True)

# Section 38 observations
obs_header = ["Run", "Train acc (%)", "Val acc (%)", "Test acc (%)", "Train loss", "Val loss", "Test loss", "Macro P (%)",
              "Macro R (%)", "Macro F1 (%)", "Trainable params (phase 2)", "Total params", "Inference (ms/img)"]
obs = [[n, ru.pct(M[n]["final_train_accuracy"]), ru.pct(M[n]["best_val_accuracy"]), ru.pct(M[n]["accuracy"]),
        f"{M[n]['final_train_loss']:.4f}", f"{M[n]['val_loss_at_best']:.4f}", f"{M[n]['test_loss']:.4f}",
        ru.pct(M[n]["precision"]), ru.pct(M[n]["recall"]), ru.pct(M[n]["f1"]), M[n]["trainable_params_phase2"],
        M[n]["total_params"], ru.ms(M[n]["ms_per_image"])] for n in real if have(n)]
emit("09_observations_section38", obs_header, obs,
     "Quantities recorded for every run (Section 38; input and output dimensions are in 02_feature_dimensions).",
     "tab:observations", True)

# ---------------------------------------------------------------------------------------------------
# Evidence for the discussion (facts only, with the statistical caveat)
# ---------------------------------------------------------------------------------------------------
L = []
n_test = M["A0"]["n_test"] if have("A0") else None
L.append(f"Test set size: {n_test}. A difference of 1 percentage point is about {0.01 * (n_test or 0):.1f} images; "
         f"check 09_significance.csv before calling any difference real.\n")


def line(n):
    return f"{n} {M[n]['label']}: accuracy {100 * M[n]['accuracy']:.2f}%, macro F1 {100 * M[n]['f1']:.2f}%, " \
           f"{M[n]['total_params'] / 1e6:.3f}M parameters, {M[n]['ms_per_image']:.2f} ms/image"


done = [n for n in real if have(n)]
if done:
    best_acc = max(done, key=lambda n: M[n]["accuracy"])
    best_f1 = max(done, key=lambda n: M[n]["f1"])
    most_par = max(done, key=lambda n: M[n]["total_params"])
    L.append(f"Best test accuracy: {line(best_acc)}")
    L.append(f"Best macro F1: {line(best_f1)}")
    L.append(f"Most parameters: {line(most_par)}")
    L.append("Is the largest model the most accurate? " + ("YES" if most_par == best_acc else "NO") + " (Q23).\n")
if have("A0"):
    L.append("Gain over the baseline (accuracy points / F1 points / extra parameters / extra ms per image):")
    for n in done:
        if n == "A0":
            continue
        L.append(f"  {n:3s} {M[n]['label']:24s} {100 * (M[n]['accuracy'] - M['A0']['accuracy']):+.2f} / "
                 f"{100 * (M[n]['f1'] - M['A0']['f1']):+.2f} / {M[n]['total_params'] - M['A0']['total_params']:+,} / "
                 f"{M[n]['ms_per_image'] - M['A0']['ms_per_image']:+.2f}")
    L.append("")
pos = [n for n in E.GROUPS["position"] if have(n)]
if pos:
    b = max(pos, key=lambda n: M[n]["accuracy"])
    L.append(f"Position ablation (Q15): best position {E.EXPERIMENTS[b]['label']} with accuracy "
             f"{100 * M[b]['accuracy']:.2f}%. Order of accuracy: " +
             ", ".join(f"{E.EXPERIMENTS[n]['label']} {100 * M[n]['accuracy']:.2f}" for n in sorted(pos, key=lambda n: -M[n]["accuracy"])) + "\n")
multi = [n for n in E.GROUPS["multi"] if have(n)]
if len(multi) >= 2:
    single = max([n for n in ("C1", "C2") if have(n)], key=lambda n: M[n]["accuracy"], default=None)
    more = [n for n in ("C3", "C4") if have(n)]
    if single and more:
        for n in more:
            L.append(f"Multiple positions (Q17): {E.EXPERIMENTS[n]['label']} {100 * M[n]['accuracy']:.2f}% versus best single "
                     f"position {E.EXPERIMENTS[single]['label']} {100 * M[single]['accuracy']:.2f}% -> "
                     f"{'improvement' if M[n]['accuracy'] > M[single]['accuracy'] else 'NO improvement'}; extra parameters "
                     f"{M[n]['total_params'] - M[single]['total_params']:+,}, extra time {M[n]['ms_per_image'] - M[single]['ms_per_image']:+.2f} ms")
        L.append("")
if have("D1") and have("D2"):
    L.append(f"Order (Q18): C->S {100 * M['D1']['accuracy']:.2f}% / F1 {100 * M['D1']['f1']:.2f}%, S->C "
             f"{100 * M['D2']['accuracy']:.2f}% / F1 {100 * M['D2']['f1']:.2f}%\n")
ker = [n for n in E.GROUPS["kernel"] if have(n)]
if ker:
    L.append("Kernel size (Q19): " + ", ".join(f"{E.EXPERIMENTS[n]['label']} {100 * M[n]['accuracy']:.2f}%" for n in ker) + "\n")
if have("A0"):
    sel = au.selected_model() if any(have(n) for n in E.SELECTION_CANDIDATES) else None
    if sel:
        pb, ps = au.test_predictions("A0")[1], au.test_predictions(sel)[1]
        y = au.test_predictions("A0")[0]
        rb, rs = au.confusion(y, pb).diagonal() / np.bincount(y, minlength=5), au.confusion(y, ps).diagonal() / np.bincount(y, minlength=5)
        L.append(f"Selected attention model (best validation accuracy among A1 to A5): {sel} ({M[sel]['label']}).")
        L.append("Per class recall change (selected minus baseline, Q27): " +
                 ", ".join(f"{c} {100 * (s - b_):+.1f}" for c, s, b_ in zip(config.CLASS_NAMES, rs, rb)) + "\n")
with open(config.result_path("09_evidence.txt"), "w") as f:
    f.write("\n".join(L))
print("\n" + "\n".join(L))

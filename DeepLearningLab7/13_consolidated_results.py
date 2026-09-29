"""
Step 13: consolidated result tables (Sections 21 and 25 of the lab
manual).

Reads the metric files written by scripts 3 to 6 and builds:

  Section 21  FC AE vs CAE vs denoising CAE (MSE, MAE, SSIM, parameters)
  Section 21  VAE metrics table
  Section 25  all four models with training time

Each table is saved as CSV and as a LaTeX tabular snippet that can be
pasted into the report. Nothing here is typed in by hand: every number
comes from your own run.
"""
import os

import config
from train_utils import load_json, save_table

SOURCES = [
    ("fc_ae_metrics.json", "3_fc_autoencoder.py"),
    ("cae_metrics.json", "4_conv_autoencoder.py"),
    ("denoising_cae_metrics.json", "5_denoising_autoencoder.py"),
    ("vae_metrics.json", "6_train_vae.py"),
]


def write_latex(df, path, formats):
    cols = list(df.columns)
    lines = ["\\begin{tabular}{|l|" + "c|" * (len(cols) - 1) + "}", "\\hline",
             " & ".join(cols) + "\\\\", "\\hline"]
    for _, row in df.iterrows():
        cells = [str(row[c]) if c not in formats else formats[c].format(row[c]) for c in cols]
        lines.append(" & ".join(cells) + "\\\\")
    lines += ["\\hline", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved", path)


def main():
    config.make_dirs()
    fc, cae, dcae, vae = [load_json(config.result_path(f), p) for f, p in SOURCES]

    fmt = {"MSE": "{:.5f}", "MAE": "{:.5f}", "SSIM": "{:.4f}", "Parameters": "{:,}", "Time (s)": "{:.1f}",
           "Value": "{:.5f}"}

    sec21 = save_table([
        {"Model": "Fully Connected AE", "MSE": fc["mse"], "MAE": fc["mae"], "SSIM": fc["ssim"], "Parameters": fc["params"]},
        {"Model": "Convolutional AE", "MSE": cae["mse"], "MAE": cae["mae"], "SSIM": cae["ssim"], "Parameters": cae["params"]},
        {"Model": "Denoising CAE", "MSE": dcae["mse"], "MAE": dcae["mae"], "SSIM": dcae["ssim"], "Parameters": dcae["params"]},
    ], config.result_path("13_section21_reconstruction_models.csv"))

    vae_table = save_table([
        {"VAE Metric": "Reconstruction Loss", "Value": vae["rec_loss"]},
        {"VAE Metric": "KL Loss", "Value": vae["kl_loss"]},
        {"VAE Metric": "Total Loss", "Value": vae["total_loss"]},
        {"VAE Metric": "Test MSE", "Value": vae["mse"]},
        {"VAE Metric": "Test MAE", "Value": vae["mae"]},
        {"VAE Metric": "Mean SSIM", "Value": vae["ssim"]},
    ], config.result_path("13_section21_vae_metrics.csv"))

    sec25 = save_table([
        {"Model": name, "MSE": m["mse"], "MAE": m["mae"], "SSIM": m["ssim"],
         "Parameters": m["params"], "Time (s)": m["train_time_s"]}
        for name, m in (("FC Autoencoder", fc), ("Conv. Autoencoder", cae),
                        ("Denoising CAE", dcae), ("VAE", vae))
    ], config.result_path("13_section25_consolidated.csv"))

    print("\nSection 21: reconstruction models (denoising CAE: "
          f"{dcae['noise_type']} noise, level {dcae['noise_level']}, error measured against the clean image)")
    print(sec21.to_string(index=False))
    print("\nSection 21: VAE")
    print(vae_table.to_string(index=False))
    print("\nSection 25: consolidated results")
    print(sec25.to_string(index=False))

    write_latex(sec21, os.path.join(config.RESULT_DIR, "13_section21_reconstruction_models.tex"), fmt)
    write_latex(vae_table, os.path.join(config.RESULT_DIR, "13_section21_vae_metrics.tex"), {"Value": "{:.4f}"})
    write_latex(sec25, os.path.join(config.RESULT_DIR, "13_section25_consolidated.tex"), fmt)


if __name__ == "__main__":
    main()

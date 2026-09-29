"""
Step 8: reconstruction error distribution and high error image analysis
(Sections 23 and 24 of the lab manual).

For every trained model (FC AE, CAE, denoising CAE, VAE) this computes
the per image error e_i on the test set and produces:

  Plot 9  histogram of e_i (one figure per model, plus an overlay)
  Plot    per image MSE against per image SSIM (error vs visual quality)
  Table   typical error range and spread for each model
  Sec 24  the five test images with the largest error, with reconstructions

The denoising CAE is evaluated the way it was trained: its input is the
noisy test image and its error is measured against the clean image.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import config
from data_utils import add_noise, flatten, load_splits, unflatten
from metrics import error_statistics, per_image_mse, per_image_ssim
from plot_style import setup_fonts, save_fig
from train_utils import require_file, save_json, save_table
from vae import encode_mean, load_vae_parts
from visualize import show_image_rows


def load_model(name, producer):
    path = config.model_path(name)
    require_file(path, producer)
    return keras.models.load_model(path, compile=False)


def collect_reconstructions(x_test):
    bs = config.BATCH_SIZE
    fc = load_model("fc_autoencoder.keras", "3_fc_autoencoder.py")
    cae = load_model("conv_autoencoder.keras", "4_conv_autoencoder.py")
    dcae = load_model("denoising_cae.keras", "5_denoising_autoencoder.py")
    encoder, decoder = load_vae_parts("vae")

    x_noisy = add_noise(x_test, config.MAIN_NOISE_TYPE, config.MAIN_NOISE_LEVEL, "test")
    z_mean, _ = encode_mean(encoder, x_test)

    return {
        "FC Autoencoder": (x_test, unflatten(fc.predict(flatten(x_test), batch_size=bs, verbose=0))),
        "Conv. Autoencoder": (x_test, cae.predict(x_test, batch_size=bs, verbose=0)),
        "Denoising CAE": (x_noisy, dcae.predict(x_noisy, batch_size=bs, verbose=0)),
        "VAE": (x_test, decoder.predict(z_mean, batch_size=bs, verbose=0)),
    }


def slug(name):
    return name.lower().replace(".", "").replace(" ", "_")


def plot_histogram(errors, name, path):
    stats = error_statistics(errors)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(errors, bins=60, color="steelblue", edgecolor="white", linewidth=0.3, label="Test images")
    ax.axvline(stats["mean"], color="crimson", linewidth=1.3, label=f"Mean = {stats['mean']:.4f}")
    ax.axvline(stats["median"], color="darkorange", linestyle="--", linewidth=1.3,
               label=f"Median = {stats['median']:.4f}")
    ax.axvline(stats["p95"], color="black", linestyle=":", linewidth=1.3,
               label=f"95th percentile = {stats['p95']:.4f}")
    ax.set_xlabel("Per image reconstruction error $e_i$ (MSE)")
    ax.set_ylabel("Number of test images")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(title=name, fontsize=8)
    fig.tight_layout()
    print("saved", save_fig(fig, path))
    plt.close(fig)


def main():
    config.make_dirs()
    config.set_seed()
    setup_fonts()

    d = load_splits()
    x_test, y_test = d["x_test"], d["y_test"]
    recon = collect_reconstructions(x_test)

    all_errors, all_ssim = {}, {}
    stat_rows, top_rows = [], []

    for name, (x_in, x_hat) in recon.items():
        errors = per_image_mse(x_test, x_hat)
        ssims = per_image_ssim(x_test, x_hat)
        all_errors[name], all_ssim[name] = errors, ssims

        stats = error_statistics(errors)
        stat_rows.append({"model": name, **stats})
        plot_histogram(errors, name, config.plot_path(f"08_error_histogram_{slug(name)}"))

        # Section 24: five largest errors
        top = np.argsort(errors)[::-1][:config.N_HIGH_ERROR]
        rows, labels = [], []
        show_input = name == "Denoising CAE"
        for rank, i in enumerate(top, start=1):
            row = [x_test[i], x_in[i], x_hat[i]] if show_input else [x_test[i], x_hat[i]]
            rows.append(row)
            labels.append(f"#{rank} digit {y_test[i]}\n$e$ = {errors[i]:.4f}")
            top_rows.append({"model": name, "rank": rank, "test_index": int(i),
                             "digit": int(y_test[i]), "error": float(errors[i]), "ssim": float(ssims[i])})
        cols = ["Original", "Noisy input", "Reconstruction"] if show_input else ["Original", "Reconstruction"]
        show_image_rows(rows, labels, config.plot_path(f"08_high_error_{slug(name)}"), col_titles=cols)

    stat_df = save_table(stat_rows, config.result_path("08_error_statistics.csv"))
    top_df = save_table(top_rows, config.result_path("08_high_error_images.csv"))
    print("\nper image reconstruction error statistics:")
    print(stat_df.to_string(index=False))
    print("\nfive largest errors per model:")
    print(top_df.to_string(index=False))

    # overlay of all four distributions on the same axis (same quantity, same scale)
    upper = max(np.percentile(e, 99.5) for e in all_errors.values())
    bins = np.linspace(0.0, upper, 70)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for name, errors in all_errors.items():
        ax.hist(errors, bins=bins, histtype="step", linewidth=1.4, label=name)
    ax.set_xlabel("Per image reconstruction error $e_i$ (MSE)")
    ax.set_ylabel("Number of test images")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    print("saved", save_fig(fig, config.plot_path("08_error_histogram_all_models")))
    plt.close(fig)

    # error vs visual quality
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    markers = ["o", "s", "^", "D"]
    for (name, errors), mk in zip(all_errors.items(), markers):
        ax.scatter(errors, all_ssim[name], s=5, marker=mk, label=name, rasterized=True)
    ax.set_xlabel("Per image reconstruction error $e_i$ (MSE)")
    ax.set_ylabel("Per image SSIM")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(markerscale=3)
    fig.tight_layout()
    print("saved", save_fig(fig, config.plot_path("08_error_vs_ssim")))
    plt.close(fig)

    corr = {name: float(np.corrcoef(all_errors[name], all_ssim[name])[0, 1]) for name in all_errors}
    mean_err_by_digit = {
        name: {int(k): float(all_errors[name][y_test == k].mean()) for k in range(10)} for name in all_errors
    }
    save_json({"pearson_mse_vs_ssim": corr, "mean_error_by_digit": mean_err_by_digit},
              config.result_path("08_error_quality_relationship.json"))
    print("\ncorrelation between per image MSE and SSIM:")
    for name, r in corr.items():
        print(f"  {name}: {r:.3f}")


if __name__ == "__main__":
    main()

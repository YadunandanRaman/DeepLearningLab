"""
Step 7: exploring the VAE latent space (Sections 18, 19 and 20 of the lab
manual, plus additional exercises 7 and 8).

  Plot 6  test images encoded to (z1, z2), coloured by digit label
  Plot    distribution of each latent coordinate against the N(0, 1) prior
  Plot    decoded latent manifold (a regular grid of z values)
  Plot 7  25 images generated from z ~ N(0, I)
  Ex. 7   a larger 10 x 10 grid of generated images
  Plot 8  interpolation between the latent codes of two test images
  Ex. 8   interpolation between the centres of two digit regions
"""
import numpy as np
import matplotlib.pyplot as plt

import config
from data_utils import load_splits
from plot_style import setup_fonts, save_fig
from train_utils import save_json, save_table
from vae import encode_mean, kl_per_dimension, load_vae_parts, sample_prior
from visualize import plot_latent_scatter, show_image_grid, show_image_rows


def interpolate(z_a, z_b, steps):
    alphas = np.linspace(0.0, 1.0, steps)
    points = np.array([(1 - a) * z_a + a * z_b for a in alphas], dtype="float32")
    return alphas, points


def plot_latent_histograms(z_mean, path_prefix):
    grid = np.linspace(-4, 4, 400)
    pdf = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    for j in range(z_mean.shape[1]):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(z_mean[:, j], bins=60, density=True, histtype="step", linewidth=1.4,
                color="steelblue", label=rf"Encoded $\mu_{j + 1}$ (test set)")
        ax.plot(grid, pdf, color="crimson", linewidth=1.4, label="Prior N(0, 1)")
        ax.set_xlabel(rf"$z_{j + 1}$")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.legend()
        fig.tight_layout()
        print("saved", save_fig(fig, f"{path_prefix}_z{j + 1}"))
        plt.close(fig)


def plot_manifold(decoder, path, n=15, limit=2.5):
    values = np.linspace(-limit, limit, n)
    canvas = np.zeros((28 * n, 28 * n), dtype="float32")
    zs = np.array([[x, y] for y in values[::-1] for x in values], dtype="float32")
    imgs = decoder.predict(zs, batch_size=config.BATCH_SIZE, verbose=0).reshape(-1, 28, 28)
    for k, img in enumerate(imgs):
        r, c = divmod(k, n)
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, extent=[-limit, limit, -limit, limit])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    fig.tight_layout()
    print("saved", save_fig(fig, path))
    plt.close(fig)


def main():
    config.make_dirs()
    config.set_seed()
    setup_fonts()

    d = load_splits()
    x_test, y_test = d["x_test"], d["y_test"]
    encoder, decoder = load_vae_parts("vae")
    latent_dim = decoder.input_shape[-1]
    if latent_dim != 2:
        raise ValueError("this script expects the 2D VAE from 6_train_vae.py")

    z_mean, z_log_var = encode_mean(encoder, x_test)

    # latent statistics for the "characteristics of the latent distribution" inference
    centroids = {int(k): z_mean[y_test == k].mean(axis=0).tolist() for k in range(10)}
    stats = {
        "mu_mean": z_mean.mean(axis=0).tolist(),
        "mu_std": z_mean.std(axis=0).tolist(),
        "mean_sigma": np.exp(0.5 * z_log_var).mean(axis=0).tolist(),
        "kl_per_dimension": kl_per_dimension(z_mean, z_log_var).tolist(),
        "digit_centroids": centroids,
    }
    save_json(stats, config.result_path("07_vae_latent_statistics.json"))
    print("latent statistics:")
    for k, v in stats.items():
        if k != "digit_centroids":
            print(f"  {k}: {np.round(v, 4).tolist()}")

    # Plot 6
    plot_latent_scatter(z_mean, y_test, config.plot_path("07_vae_latent_space"))
    plot_latent_histograms(z_mean, config.plot_path("07_vae_latent_histogram"))
    plot_manifold(decoder, config.plot_path("07_vae_latent_manifold"))

    # Plot 7: 25 random samples from the prior
    generated, _ = sample_prior(decoder, config.N_GENERATED, latent_dim, seed=config.SEED)
    side = int(np.sqrt(config.N_GENERATED))
    show_image_grid(generated, side, side, config.plot_path("07_vae_generated_5x5"))

    # Exercise 7: larger grid
    generated_large, _ = sample_prior(decoder, config.N_GENERATED_LARGE, latent_dim, seed=config.SEED + 7)
    side = int(np.sqrt(config.N_GENERATED_LARGE))
    show_image_grid(generated_large, side, side, config.plot_path("07_vae_generated_10x10"))

    # Plot 8: interpolation between two test images of different digits
    digit_a, digit_b = config.INTERP_DIGITS
    idx_a = int(np.where(y_test == digit_a)[0][0])
    idx_b = int(np.where(y_test == digit_b)[0][0])
    alphas, path_img = interpolate(z_mean[idx_a], z_mean[idx_b], config.INTERP_STEPS)
    decoded_img = decoder.predict(path_img, verbose=0)
    titles = [f"$\\alpha$={a:.1f}" for a in alphas]
    show_image_rows([decoded_img], [f"{digit_a} to {digit_b}"],
                    config.plot_path("07_vae_interpolation_images"), col_titles=titles)
    show_image_rows([[x_test[idx_a], x_test[idx_b]]], ["Endpoints"],
                    config.plot_path("07_vae_interpolation_endpoints"),
                    col_titles=[f"Test image, digit {digit_a}", f"Test image, digit {digit_b}"])

    # Exercise 8: interpolation between two digit regions (their centroids)
    c_a, c_b = config.CENTROID_DIGITS
    _, path_cent = interpolate(np.array(centroids[c_a]), np.array(centroids[c_b]), config.INTERP_STEPS)
    decoded_cent = decoder.predict(path_cent, verbose=0)
    show_image_rows([decoded_cent], [f"Region {c_a} to {c_b}"],
                    config.plot_path("07_vae_interpolation_centroids"), col_titles=titles)

    plot_latent_scatter(
        z_mean, y_test, config.plot_path("07_vae_interpolation_paths"),
        paths=[
            {"points": path_img, "label": f"Image path {digit_a} to {digit_b}"},
            {"points": path_cent, "label": f"Centroid path {c_a} to {c_b}"},
        ],
    )

    save_table(
        [{"path": "image", "alpha": float(a), "z1": float(p[0]), "z2": float(p[1])} for a, p in zip(alphas, path_img)]
        + [{"path": "centroid", "alpha": float(a), "z1": float(p[0]), "z2": float(p[1])} for a, p in zip(alphas, path_cent)],
        config.result_path("07_interpolation_points.csv"),
    )


if __name__ == "__main__":
    main()

"""
Step 12: effect of increasing the KL loss contribution (additional
exercise 6 of the lab manual).

The VAE loss becomes reconstruction + beta * KL. beta = 1 is the standard
VAE from 6_train_vae.py; larger beta pushes q(z|x) harder towards the
N(0, I) prior. Every run uses the 2D VAE and the same training protocol.
"""
import numpy as np

import config
from data_utils import load_splits
from plot_style import setup_fonts
from train_utils import save_table
from vae import evaluate_vae, sample_prior, train_vae
from visualize import plot_curves, plot_latent_scatter, show_image_grid, show_image_rows


def main():
    config.make_dirs()
    setup_fonts()

    d = load_splits()
    x_train, x_val, x_test, y_test = d["x_train"], d["x_val"], d["x_test"], d["y_test"]
    dz = config.VAE_LATENT_DIM

    rows, previews = [], []
    for beta in config.KL_WEIGHTS:
        print(f"\ntraining VAE with KL weight beta = {beta}")
        model, _, train_time = train_vae(x_train, x_val, dz, beta=beta, epochs=config.VAE_EPOCHS)
        m, z_mean, _, x_hat = evaluate_vae(model, x_test)
        rows.append({"beta": beta, **m, "train_time_s": train_time})
        previews.append(x_hat[:config.N_DISPLAY])
        tag = str(beta).replace(".", "p")
        plot_latent_scatter(z_mean, y_test, config.plot_path(f"12_latent_space_beta_{tag}"))
        generated, _ = sample_prior(model.decoder, config.N_GENERATED, dz, seed=config.SEED)
        side = int(np.sqrt(config.N_GENERATED))
        show_image_grid(generated, side, side, config.plot_path(f"12_generated_beta_{tag}"))

    df = save_table(rows, config.result_path("12_kl_weight_study.csv"))
    print("\nKL weight study (test set):")
    print(df[["beta", "rec_loss", "kl_loss", "total_loss", "mse", "ssim"]].to_string(index=False))

    betas = config.KL_WEIGHTS
    for key, label in (("rec_loss", "Test reconstruction loss (BCE per image)"),
                       ("kl_loss", "Test KL divergence"),
                       ("ssim", "Mean SSIM")):
        plot_curves(betas, {"2D VAE": df[key].tolist()}, r"KL weight $\beta$", label,
                    config.plot_path(f"12_beta_vs_{key}"), xticks=betas)

    n = config.N_DISPLAY
    show_image_rows([x_test[:n]] + previews,
                    ["Original"] + [rf"$\beta$ = {b}" for b in betas],
                    config.plot_path("12_reconstructions_by_beta"))


if __name__ == "__main__":
    main()

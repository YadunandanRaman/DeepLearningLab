"""
Step 11: VAE with latent dimension 2 vs 8 (additional exercise 5 of the
lab manual).

Both VAEs are trained here under the same protocol. The 2D latent space
is plotted directly; the 8D latent space is projected to 2D with PCA
for visualization. The average KL per latent dimension shows how many
dimensions each model actually uses.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import config
from data_utils import load_splits
from plot_style import setup_fonts, save_fig
from train_utils import save_json, save_table
from vae import evaluate_vae, kl_per_dimension, sample_prior, train_vae
from visualize import plot_curves, plot_latent_scatter, show_image_grid, show_image_rows


def main():
    config.make_dirs()
    setup_fonts()

    d = load_splits()
    x_train, x_val, x_test, y_test = d["x_train"], d["x_val"], d["x_test"], d["y_test"]

    rows, histories, kl_dims, previews = [], {}, {}, {}
    for dz in config.VAE_COMPARE_LATENT_DIMS:
        print(f"\ntraining VAE with latent dimension {dz}")
        model, history, train_time = train_vae(x_train, x_val, dz, beta=1.0, epochs=config.VAE_EPOCHS)
        m, z_mean, z_log_var, x_hat = evaluate_vae(model, x_test)
        kl_dims[dz] = kl_per_dimension(z_mean, z_log_var)
        rows.append({"latent_dim": dz, **m, "active_dims_kl_above_0.1": int((kl_dims[dz] > 0.1).sum()),
                     "train_time_s": train_time})
        histories[dz] = history
        previews[dz] = x_hat[:config.N_DISPLAY]

        if dz == 2:
            plot_latent_scatter(z_mean, y_test, config.plot_path("11_vae_latent_z2"))
        else:
            pca = PCA(n_components=2, random_state=config.SEED)
            z_2d = pca.fit_transform(z_mean)
            ev = pca.explained_variance_ratio_
            plot_latent_scatter(z_2d, y_test, config.plot_path(f"11_vae_latent_z{dz}_pca"),
                                xlabel=f"PC 1 ({ev[0] * 100:.1f}% variance)",
                                ylabel=f"PC 2 ({ev[1] * 100:.1f}% variance)")

        generated, _ = sample_prior(model.decoder, config.N_GENERATED, dz, seed=config.SEED)
        side = int(np.sqrt(config.N_GENERATED))
        show_image_grid(generated, side, side, config.plot_path(f"11_vae_generated_z{dz}"))

    df = save_table(rows, config.result_path("11_vae_latent_comparison.csv"))
    save_json({str(k): v.tolist() for k, v in kl_dims.items()}, config.result_path("11_vae_kl_per_dimension.json"))
    print("\nVAE latent dimension comparison (test set):")
    print(df.to_string(index=False))

    epochs = np.arange(1, config.VAE_EPOCHS + 1)
    plot_curves(epochs, {f"$d_z$ = {k}": h["val_rec_loss"] for k, h in histories.items()},
                "Epoch", "Validation reconstruction loss (BCE per image)",
                config.plot_path("11_vae_val_reconstruction_loss"))
    plot_curves(epochs, {f"$d_z$ = {k}": h["val_kl_loss"] for k, h in histories.items()},
                "Epoch", "Validation KL divergence",
                config.plot_path("11_vae_val_kl_loss"))

    # average KL per latent dimension
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    dims = config.VAE_COMPARE_LATENT_DIMS
    width = 0.8 / len(dims)
    for i, dz in enumerate(dims):
        pos = np.arange(1, dz + 1) + (i - (len(dims) - 1) / 2) * width
        ax.bar(pos, kl_dims[dz], width, label=f"$d_z$ = {dz}")
    ax.set_xticks(np.arange(1, max(dims) + 1))
    ax.set_xlabel("Latent dimension index")
    ax.set_ylabel("Average KL divergence (test set)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    print("saved", save_fig(fig, config.plot_path("11_vae_kl_per_dimension")))
    plt.close(fig)

    n = config.N_DISPLAY
    show_image_rows([x_test[:n]] + [previews[dz] for dz in dims],
                    ["Original"] + [f"$d_z$ = {dz}" for dz in dims],
                    config.plot_path("11_vae_reconstructions"))


if __name__ == "__main__":
    main()

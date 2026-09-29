"""
Step 6: variational autoencoder with a 2 dimensional latent space
(Sections 15, 16 and 17 of the lab manual).

Trains the VAE, saves its encoder and decoder, and produces:

  Plot    training and validation reconstruction loss (required plot 9)
  Plot    training and validation KL loss, and total loss
  Plot    original vs VAE reconstruction
  Table   reconstruction loss, KL loss, total loss, test MSE, MAE, SSIM

Reconstruction metrics (MSE, MAE, SSIM) decode the mean mu of q(z|x),
so they are directly comparable with the deterministic autoencoders.
"""
import config
from data_utils import load_splits
from metrics import per_image_mse
from plot_style import setup_fonts
from train_utils import save_json
from vae import evaluate_vae, save_vae, train_vae
from visualize import plot_loss_curves, show_image_rows


def main():
    config.make_dirs()
    setup_fonts()

    d = load_splits()
    x_train, x_val, x_test = d["x_train"], d["x_val"], d["x_test"]

    model, history, train_time = train_vae(x_train, x_val, config.VAE_LATENT_DIM, beta=1.0,
                                           epochs=config.VAE_EPOCHS)
    model.encoder.summary()
    model.decoder.summary()
    save_vae(model, "vae")
    save_json(history, config.result_path("06_vae_history.json"))

    metrics, z_mean, z_log_var, x_hat = evaluate_vae(model, x_test)
    metrics.update({
        "model": "VAE",
        "latent_dim": config.VAE_LATENT_DIM,
        "beta": 1.0,
        "train_time_s": float(train_time),
        "epochs": config.VAE_EPOCHS,
        "note": "rec_loss is BCE summed over 784 pixels per image; MSE/MAE/SSIM decode mu",
    })
    save_json(metrics, config.result_path("vae_metrics.json"))

    print("\nVAE, test set")
    print(f"  reconstruction loss : {metrics['rec_loss']:.3f}")
    print(f"  KL loss             : {metrics['kl_loss']:.3f}")
    print(f"  total loss          : {metrics['total_loss']:.3f}")
    print(f"  test MSE            : {metrics['mse']:.6f}")
    print(f"  test MAE            : {metrics['mae']:.6f}")
    print(f"  mean SSIM           : {metrics['ssim']:.4f}")
    print(f"  parameters          : {metrics['params']:,}")
    print(f"  train time          : {train_time:.1f} s")

    plot_loss_curves(history, "rec_loss", "val_rec_loss",
                     "Reconstruction loss (BCE summed per image)",
                     config.plot_path("06_vae_reconstruction_loss"))
    plot_loss_curves(history, "kl_loss", "val_kl_loss", "KL divergence",
                     config.plot_path("06_vae_kl_loss"))
    plot_loss_curves(history, "total_loss", "val_total_loss", "Total VAE loss",
                     config.plot_path("06_vae_total_loss"))

    n = config.N_DISPLAY
    errs = per_image_mse(x_test[:n], x_hat[:n])
    show_image_rows(
        [x_test[:n], x_hat[:n]],
        ["Original", "VAE"],
        config.plot_path("06_vae_original_vs_reconstructed"),
        col_titles=[f"MSE {e:.4f}" for e in errs],
    )


if __name__ == "__main__":
    main()

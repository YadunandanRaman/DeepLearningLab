"""
Step 3: fully connected autoencoder (Sections 7, 8 and 9 of the lab manual).

Trains the 784 -> 128 -> 32 -> 16 -> 32 -> 128 -> 784 autoencoder with the
suggested configuration, then produces:

  Plot 1  original vs reconstructed test images (with the error map)
  Plot 2  training and validation reconstruction loss
  Table   test MSE, MAE and mean SSIM
"""
import numpy as np

import config
from data_utils import flatten, load_splits, unflatten
from metrics import per_image_mse, reconstruction_metrics
from model_builder import build_fc_autoencoder, compile_autoencoder
from plot_style import setup_fonts
from train_utils import convergence_summary, fit_timed, save_json
from visualize import plot_loss_curves, show_image_rows


def main():
    config.make_dirs()
    config.set_seed()
    setup_fonts()

    d = load_splits()
    x_train, x_val, x_test = flatten(d["x_train"]), flatten(d["x_val"]), flatten(d["x_test"])

    model = compile_autoencoder(build_fc_autoencoder())
    model.summary()

    history, train_time = fit_timed(model, x_train, x_train, x_val, x_val, config.EPOCHS, config.BATCH_SIZE)
    model.save(config.model_path("fc_autoencoder.keras"))
    save_json(history, config.result_path("03_fc_history.json"))

    x_hat = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=0)
    metrics = reconstruction_metrics(x_test, x_hat)
    metrics.update({
        "model": "FC Autoencoder",
        "latent_dim": config.FC_LATENT_DIM,
        "params": int(model.count_params()),
        "train_time_s": float(train_time),
        "epochs": config.EPOCHS,
    })
    metrics["convergence"] = convergence_summary(history)
    save_json(metrics, config.result_path("fc_ae_metrics.json"))

    print("\nfully connected autoencoder, test set")
    print(f"  test MSE   : {metrics['mse']:.6f}")
    print(f"  test MAE   : {metrics['mae']:.6f}")
    print(f"  mean SSIM  : {metrics['ssim']:.4f}")
    print(f"  parameters : {metrics['params']:,}")
    print(f"  train time : {train_time:.1f} s")
    c = metrics["convergence"]
    print(f"  best validation loss {c['best_val_loss']:.5f} at epoch {c['best_val_epoch']} of {c['epochs_run']}")
    print(f"  final gap (validation minus training): {c['final_gap_val_minus_train']:.5f}")

    # Plot 1: original | reconstruction | absolute error, first N_DISPLAY test images
    n = config.N_DISPLAY
    orig = d["x_test"][:n]
    rec = unflatten(x_hat[:n])
    err = np.abs(orig - rec)
    errs = per_image_mse(orig, rec)
    show_image_rows(
        [orig, rec, err],
        ["Original", "Reconstruction", "Abs. error"],
        config.plot_path("03_fc_original_vs_reconstructed"),
        col_titles=[f"MSE {e:.4f}" for e in errs],
        row_cmaps=["gray", "gray", "inferno"],
        row_vmax=[1.0, 1.0, float(err.max())],
    )

    # Plot 2: training vs validation loss
    plot_loss_curves(history, "loss", "val_loss", "Reconstruction loss (binary cross entropy)",
                     config.plot_path("03_fc_loss_curves"))


if __name__ == "__main__":
    main()

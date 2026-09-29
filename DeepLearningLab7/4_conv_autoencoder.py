"""
Step 4: convolutional autoencoder and comparison with the fully connected
one (Sections 10 and 11 of the lab manual).

Both models use the same training, validation and test images and the
same optimizer, learning rate, batch size and number of epochs.

  Plot 3  Original -> FC AE reconstruction -> CAE reconstruction
  Table   MSE, MAE, SSIM and parameter count for both models
"""
from tensorflow import keras

import config
from data_utils import flatten, load_splits, unflatten
from metrics import per_image_mse, reconstruction_metrics
from model_builder import build_conv_autoencoder, compile_autoencoder
from plot_style import setup_fonts
from train_utils import convergence_summary, fit_timed, load_json, require_file, save_json, save_table
from visualize import plot_loss_curves, show_image_rows


def main():
    config.make_dirs()
    config.set_seed()
    setup_fonts()

    d = load_splits()
    x_train, x_val, x_test = d["x_train"], d["x_val"], d["x_test"]

    model = compile_autoencoder(build_conv_autoencoder("upsampling"))
    model.summary()

    history, train_time = fit_timed(model, x_train, x_train, x_val, x_val, config.EPOCHS, config.BATCH_SIZE)
    model.save(config.model_path("conv_autoencoder.keras"))
    save_json(history, config.result_path("04_cae_history.json"))

    x_hat_cae = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=0)
    metrics = reconstruction_metrics(x_test, x_hat_cae)
    metrics.update({
        "model": "Conv. Autoencoder",
        "latent_shape": "7 x 7 x 64",
        "params": int(model.count_params()),
        "train_time_s": float(train_time),
        "epochs": config.EPOCHS,
    })
    metrics["convergence"] = convergence_summary(history)
    save_json(metrics, config.result_path("cae_metrics.json"))

    plot_loss_curves(history, "loss", "val_loss", "Reconstruction loss (binary cross entropy)",
                     config.plot_path("04_cae_loss_curves"))

    # comparison with the fully connected autoencoder from step 3
    fc_path = config.model_path("fc_autoencoder.keras")
    require_file(fc_path, "3_fc_autoencoder.py")
    fc_model = keras.models.load_model(fc_path, compile=False)
    fc_metrics = load_json(config.result_path("fc_ae_metrics.json"), "3_fc_autoencoder.py")
    x_hat_fc = unflatten(fc_model.predict(flatten(x_test), batch_size=config.BATCH_SIZE, verbose=0))

    rows = []
    for m in (fc_metrics, metrics):
        rows.append({
            "Model": "Fully Connected AE" if m is fc_metrics else "Convolutional AE",
            "MSE": m["mse"], "MAE": m["mae"], "SSIM": m["ssim"], "Parameters": m["params"],
        })
    df = save_table(rows, config.result_path("04_fc_vs_cae.csv"))
    print("\nfully connected vs convolutional autoencoder (test set):")
    print(df.to_string(index=False))

    # Plot 3
    n = config.N_DISPLAY
    orig = x_test[:n]
    fc_mse = per_image_mse(orig, x_hat_fc[:n])
    cae_mse = per_image_mse(orig, x_hat_cae[:n])
    show_image_rows(
        [orig, x_hat_fc[:n], x_hat_cae[:n]],
        ["Original", "FC AE", "CAE"],
        config.plot_path("04_fc_vs_cae_reconstruction"),
        col_titles=[f"FC {a:.4f}\nCAE {b:.4f}" for a, b in zip(fc_mse, cae_mse)],
    )


if __name__ == "__main__":
    main()

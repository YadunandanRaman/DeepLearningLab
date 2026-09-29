"""
Step 10: UpSampling2D decoder vs Conv2DTranspose decoder (additional
exercise 4 of the lab manual).

Both models share the Section 10 encoder and are trained under the same
protocol. Only the way the decoder doubles the spatial size changes.
"""
import numpy as np

import config
from data_utils import load_splits
from metrics import reconstruction_metrics
from model_builder import build_conv_autoencoder, compile_autoencoder
from plot_style import setup_fonts
from train_utils import fit_timed, save_json, save_table
from visualize import plot_curves, show_image_rows

LABELS = {"upsampling": "UpSampling2D decoder", "transpose": "Conv2DTranspose decoder"}


def main():
    config.make_dirs()
    setup_fonts()

    d = load_splits()
    x_train, x_val, x_test = d["x_train"], d["x_val"], d["x_test"]

    rows, histories, previews = [], {}, {}
    for decoder_type in ("upsampling", "transpose"):
        print(f"\ntraining CAE with {LABELS[decoder_type]}")
        config.set_seed()
        model = compile_autoencoder(build_conv_autoencoder(decoder_type))
        model.summary()
        history, train_time = fit_timed(model, x_train, x_train, x_val, x_val, config.EPOCHS, config.BATCH_SIZE)
        x_hat = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=0)
        m = reconstruction_metrics(x_test, x_hat)
        rows.append({"decoder": LABELS[decoder_type], **m,
                     "params": int(model.count_params()), "train_time_s": train_time})
        histories[decoder_type] = history
        previews[decoder_type] = x_hat[:config.N_DISPLAY]

    df = save_table(rows, config.result_path("10_decoder_comparison.csv"))
    save_json(histories, config.result_path("10_decoder_histories.json"))
    print("\ndecoder comparison (test set):")
    print(df.to_string(index=False))

    epochs = np.arange(1, config.EPOCHS + 1)
    plot_curves(epochs, {LABELS[k]: h["val_loss"] for k, h in histories.items()},
                "Epoch", "Validation reconstruction loss (binary cross entropy)",
                config.plot_path("10_decoder_validation_loss"))
    plot_curves(epochs, {LABELS[k]: h["loss"] for k, h in histories.items()},
                "Epoch", "Training reconstruction loss (binary cross entropy)",
                config.plot_path("10_decoder_training_loss"))

    n = config.N_DISPLAY
    show_image_rows([x_test[:n], previews["upsampling"], previews["transpose"]],
                    ["Original", "UpSampling", "Transpose"],
                    config.plot_path("10_decoder_reconstructions"))


if __name__ == "__main__":
    main()

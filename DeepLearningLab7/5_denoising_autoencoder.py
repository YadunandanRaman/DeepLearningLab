"""
Step 5: denoising convolutional autoencoder (Sections 12, 13 and 14 of
the lab manual, plus additional exercises 2 and 3).

For every corruption setting a fresh copy of the Section 10 CAE is
trained with the noisy image as input and the CLEAN image as target:

  Gaussian noise        sigma in {0.1, 0.2, 0.3}, clipped to [0, 1]
  salt and pepper noise p in {0.05, 0.10, 0.20}

Outputs:
  Plot 4  clean | noisy | denoised, for every setting
  Plot 5  noise level vs MSE, MAE and SSIM (one figure per metric and
          noise type), comparing the noisy input with the denoised output
  Ex. 2   Gaussian vs salt and pepper on the same images
  Ex. 3   each Gaussian model tested on every Gaussian level (SSIM)

The Gaussian sigma = 0.2 model is saved as the "Denoising CAE" used in
the summary tables and the error analysis.
"""
import numpy as np

import config
from data_utils import add_noise, load_splits, noise_label, noise_tag
from metrics import reconstruction_metrics
from model_builder import build_conv_autoencoder, compile_autoencoder
from plot_style import setup_fonts
from train_utils import convergence_summary, fit_timed, save_json, save_table
from visualize import plot_curves, plot_loss_curves, show_image_rows

METRIC_LABELS = {"mse": "MSE", "mae": "MAE", "ssim": "Mean SSIM"}


def main():
    config.make_dirs()
    setup_fonts()

    d = load_splits()
    x_train, x_val, x_test = d["x_train"], d["x_val"], d["x_test"]
    n = config.N_DISPLAY

    runs = [("gaussian", s) for s in config.GAUSSIAN_SIGMAS]
    runs += [("salt_pepper", p) for p in config.SALT_PEPPER_LEVELS]

    rows = []
    display = {}
    gaussian_models = {}

    for kind, level in runs:
        print(f"\ntraining denoising CAE: {noise_label(kind, level)}")
        config.set_seed()
        x_train_n = add_noise(x_train, kind, level, "train")
        x_val_n = add_noise(x_val, kind, level, "val")
        x_test_n = add_noise(x_test, kind, level, "test")

        model = compile_autoencoder(build_conv_autoencoder("upsampling"))
        # noisy input, clean target
        history, train_time = fit_timed(model, x_train_n, x_train, x_val_n, x_val, config.EPOCHS, config.BATCH_SIZE)
        x_hat = model.predict(x_test_n, batch_size=config.BATCH_SIZE, verbose=0)

        noisy_m = reconstruction_metrics(x_test, x_test_n)
        den_m = reconstruction_metrics(x_test, x_hat)
        rows.append({
            "noise_type": kind, "noise_level": level,
            "noisy_mse": noisy_m["mse"], "noisy_mae": noisy_m["mae"], "noisy_ssim": noisy_m["ssim"],
            "denoised_mse": den_m["mse"], "denoised_mae": den_m["mae"], "denoised_ssim": den_m["ssim"],
            "train_time_s": train_time,
        })
        print(f"  noisy input    MSE {noisy_m['mse']:.5f}  MAE {noisy_m['mae']:.5f}  SSIM {noisy_m['ssim']:.4f}")
        print(f"  denoised output MSE {den_m['mse']:.5f}  MAE {den_m['mae']:.5f}  SSIM {den_m['ssim']:.4f}")

        display[(kind, level)] = (x_test_n[:n], x_hat[:n])
        tag = noise_tag(kind, level)

        # Plot 4 for this setting
        show_image_rows(
            [x_test[:n], x_test_n[:n], x_hat[:n]],
            ["Clean", "Noisy", "Denoised"],
            config.plot_path(f"05_clean_noisy_denoised_{tag}"),
        )

        if kind == "gaussian":
            gaussian_models[level] = model

        if kind == config.MAIN_NOISE_TYPE and np.isclose(level, config.MAIN_NOISE_LEVEL):
            model.save(config.model_path("denoising_cae.keras"))
            save_json(history, config.result_path("05_denoising_history.json"))
            plot_loss_curves(history, "loss", "val_loss", "Reconstruction loss (binary cross entropy)",
                             config.plot_path("05_denoising_loss_curves"))
            main_metrics = dict(den_m)
            main_metrics.update({
                "model": "Denoising CAE",
                "noise_type": kind,
                "noise_level": level,
                "params": int(model.count_params()),
                "train_time_s": float(train_time),
                "epochs": config.EPOCHS,
                "noisy_input_metrics": noisy_m,
                "convergence": convergence_summary(history),
            })
            save_json(main_metrics, config.result_path("denoising_cae_metrics.json"))

    df = save_table(rows, config.result_path("05_noise_level_results.csv"))
    print("\nnoise level vs reconstruction quality (test set):")
    print(df.to_string(index=False))

    # Plot 5: separate figure per metric, as the manual asks, per noise type
    for kind, xlabel in (("gaussian", r"Noise standard deviation $\sigma$"),
                         ("salt_pepper", "Corrupted pixel proportion $p$")):
        sub = df[df["noise_type"] == kind]
        levels = sub["noise_level"].tolist()
        for key, label in METRIC_LABELS.items():
            plot_curves(
                levels,
                {"Noisy input": sub[f"noisy_{key}"].tolist(), "Denoised output": sub[f"denoised_{key}"].tolist()},
                xlabel, label,
                config.plot_path(f"05_noise_level_vs_{key}_{kind}"),
                xticks=levels,
            )

    # Exercise 2: Gaussian vs salt and pepper, middle level of each, same images
    g_level = config.GAUSSIAN_SIGMAS[len(config.GAUSSIAN_SIGMAS) // 2]
    sp_level = config.SALT_PEPPER_LEVELS[len(config.SALT_PEPPER_LEVELS) // 2]
    g_noisy, g_den = display[("gaussian", g_level)]
    sp_noisy, sp_den = display[("salt_pepper", sp_level)]
    show_image_rows(
        [x_test[:n], g_noisy, g_den, sp_noisy, sp_den],
        ["Clean", f"Gauss. {g_level}", "Denoised", f"S&P {sp_level}", "Denoised"],
        config.plot_path("05_gaussian_vs_salt_pepper"),
    )

    # Exercise 3: every Gaussian model evaluated on every Gaussian test level
    cross_rows = []
    series = {}
    for train_level, model in gaussian_models.items():
        ssims = []
        for test_level in config.GAUSSIAN_SIGMAS:
            x_test_n = add_noise(x_test, "gaussian", test_level, "test")
            x_hat = model.predict(x_test_n, batch_size=config.BATCH_SIZE, verbose=0)
            m = reconstruction_metrics(x_test, x_hat)
            ssims.append(m["ssim"])
            cross_rows.append({"train_sigma": train_level, "test_sigma": test_level,
                               "mse": m["mse"], "ssim": m["ssim"]})
        series[rf"Trained with $\sigma$ = {train_level}"] = ssims
    cross_df = save_table(cross_rows, config.result_path("05_cross_noise_level_ssim.csv"))
    print("\ncross noise level evaluation (exercise 3):")
    print(cross_df.to_string(index=False))
    plot_curves(
        config.GAUSSIAN_SIGMAS, series,
        r"Test noise standard deviation $\sigma$", "Mean SSIM",
        config.plot_path("05_cross_noise_level_ssim"),
        xticks=config.GAUSSIAN_SIGMAS,
    )


if __name__ == "__main__":
    main()

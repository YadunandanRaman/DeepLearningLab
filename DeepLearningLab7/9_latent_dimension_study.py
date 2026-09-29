"""
Step 9: latent dimension study (Section 27 of the lab manual and
additional exercise 1).

  Section 27  fully connected autoencoder with dz in {2, 8, 16, 32}
  Exercise 1  convolutional autoencoder with a dz vector bottleneck,
              dz in {4, 8, 16, 32}

Every model uses the same data, optimizer, learning rate, batch size and
epochs, so only the latent size changes.

  Plot 10  latent dimension vs reconstruction MSE (and vs SSIM)
  Plot     reconstructions of the same test images for every dz
"""
import config
from data_utils import flatten, load_splits, unflatten
from metrics import reconstruction_metrics
from model_builder import build_conv_autoencoder_bottleneck, build_fc_autoencoder, compile_autoencoder
from plot_style import setup_fonts
from train_utils import fit_timed, save_table
from visualize import plot_curves, show_image_rows


def run_study(family, dims, build_fn, x_train, x_val, x_test, flat):
    rows, previews = [], []
    for dz in dims:
        print(f"\n{family}: latent dimension {dz}")
        config.set_seed()
        model = compile_autoencoder(build_fn(dz))
        xt, xv, xs = (flatten(x_train), flatten(x_val), flatten(x_test)) if flat else (x_train, x_val, x_test)
        _, train_time = fit_timed(model, xt, xt, xv, xv, config.EPOCHS, config.BATCH_SIZE)
        x_hat = model.predict(xs, batch_size=config.BATCH_SIZE, verbose=0)
        if flat:
            x_hat = unflatten(x_hat)
        m = reconstruction_metrics(x_test, x_hat)
        rows.append({
            "model": family, "latent_dim": dz,
            "mse": m["mse"], "mae": m["mae"], "ssim": m["ssim"],
            "params": int(model.count_params()), "train_time_s": train_time,
        })
        previews.append(x_hat[:config.N_DISPLAY])
        print(f"  MSE {m['mse']:.5f}  SSIM {m['ssim']:.4f}  params {model.count_params():,}")
    return rows, previews


def main():
    config.make_dirs()
    setup_fonts()

    d = load_splits()
    x_train, x_val, x_test = d["x_train"], d["x_val"], d["x_test"]

    fc_rows, fc_prev = run_study("FC Autoencoder", config.FC_LATENT_DIMS, build_fc_autoencoder,
                                 x_train, x_val, x_test, flat=True)
    cae_rows, cae_prev = run_study("Conv. Autoencoder", config.CAE_LATENT_DIMS, build_conv_autoencoder_bottleneck,
                                   x_train, x_val, x_test, flat=False)

    fc_df = save_table(fc_rows, config.result_path("09_latent_dim_fc.csv"))
    cae_df = save_table(cae_rows, config.result_path("09_latent_dim_cae.csv"))
    print("\nSection 27, fully connected autoencoder:")
    print(fc_df[["latent_dim", "mse", "ssim", "params"]].to_string(index=False))
    print("\nexercise 1, convolutional autoencoder:")
    print(cae_df[["latent_dim", "mse", "ssim", "params"]].to_string(index=False))

    # Plot 10 (separate figures for MSE and SSIM, since the scales differ)
    for key, label in (("mse", "Test reconstruction MSE"), ("ssim", "Mean SSIM")):
        plot_curves(config.FC_LATENT_DIMS, {"FC Autoencoder": fc_df[key].tolist()},
                    "Latent dimension $d_z$", label,
                    config.plot_path(f"09_latent_dim_vs_{key}_fc"), xticks=config.FC_LATENT_DIMS)
        plot_curves(config.CAE_LATENT_DIMS, {"Conv. Autoencoder": cae_df[key].tolist()},
                    "Latent dimension $d_z$", label,
                    config.plot_path(f"09_latent_dim_vs_{key}_cae"), xticks=config.CAE_LATENT_DIMS)

    n = config.N_DISPLAY
    show_image_rows([x_test[:n]] + fc_prev, ["Original"] + [f"$d_z$ = {dz}" for dz in config.FC_LATENT_DIMS],
                    config.plot_path("09_latent_dim_reconstructions_fc"))
    show_image_rows([x_test[:n]] + cae_prev, ["Original"] + [f"$d_z$ = {dz}" for dz in config.CAE_LATENT_DIMS],
                    config.plot_path("09_latent_dim_reconstructions_cae"))


if __name__ == "__main__":
    main()

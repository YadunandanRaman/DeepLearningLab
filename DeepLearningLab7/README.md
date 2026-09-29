# Autoencoders, Convolutional, Denoising and Variational Autoencoders on MNIST

CS3807 Deep Learning Lab, Experiment 7. An end to end study of
autoencoders implemented with TensorFlow/Keras on MNIST: a fully
connected autoencoder, a convolutional autoencoder (CAE), a denoising
CAE, and a variational autoencoder (VAE) with a 2 dimensional latent
space, followed by error analysis, a latent dimension study and the
additional exercises from the lab manual.

## Files

Shared modules:

- `config.py`: every path, seed and hyperparameter in one place (subset sizes, epochs, noise levels, latent sizes, KL weights)
- `plot_style.py`: loads Times New Roman, saves figures as EPS at 600 DPI
- `data_utils.py`: MNIST loading, the laboratory subset, normalization, saving/loading splits, Gaussian and salt and pepper noise
- `metrics.py`: MSE, MAE, SSIM, per image error $e_i$, error statistics
- `model_builder.py`: FC autoencoder, CAE (UpSampling2D or Conv2DTranspose decoder), CAE with a vector bottleneck, VAE encoder and decoder
- `vae.py`: reparameterization trick, reconstruction and KL losses, the VAE training model, training/evaluation/sampling helpers
- `train_utils.py`: timed training, history and table saving, convergence summary
- `visualize.py`: image grids, loss curves, latent space scatter plots

Numbered scripts (run in order):

- `1_data_exploration.py`: dataset dimensions, sample digits, class distribution (Section 3)
- `2_preprocessing.py`: 10000 train (9000 train + 1000 validation) and 2000 test images, scaling to [0, 1], shapes for FC and conv input (Sections 3, 4)
- `3_fc_autoencoder.py`: 784, 128, 32, 16, 32, 128, 784 autoencoder, Plot 1, Plot 2, metrics (Sections 7, 8, 9)
- `4_conv_autoencoder.py`: Section 10 CAE, FC vs CAE table, Plot 3 (Sections 10, 11)
- `5_denoising_autoencoder.py`: denoising CAE for Gaussian sigma 0.1/0.2/0.3 and salt and pepper p 0.05/0.10/0.20, Plot 4, Plot 5 (Sections 12, 13, 14; exercises 2, 3)
- `6_train_vae.py`: 2D VAE, reconstruction/KL/total loss curves, VAE metrics (Sections 15, 16, 17)
- `7_vae_latent_space.py`: latent space, generation, interpolation (Sections 18, 19, 20; exercises 7, 8)
- `8_error_analysis.py`: error distribution, five highest error images per model (Sections 23, 24)
- `9_latent_dimension_study.py`: FC AE with dz 2/8/16/32 and CAE with dz 4/8/16/32 (Section 27; exercise 1)
- `10_decoder_comparison.py`: UpSampling2D vs Conv2DTranspose decoder (exercise 4)
- `11_vae_latent_comparison.py`: VAE with latent dimension 2 vs 8 (exercise 5)
- `12_kl_weight_study.py`: KL weight beta 1/2/5/10 (exercise 6)
- `13_consolidated_results.py`: Section 21 and Section 25 tables as CSV and LaTeX

## Required plots (Section 22) and where they come from

All figures are saved in `outputs/plots/` as `.eps` at 600 DPI, in Times
New Roman, with axis labels and a legend on every plot that has axes and
more than one series. Image grids have no axes or legend.

| # | Plot | File(s) | Produced by |
|---|------|---------|-------------|
| 1 | Original vs reconstructed | `03_fc_original_vs_reconstructed.eps` | `3_fc_autoencoder.py` |
| 2 | Training/validation loss, basic AE | `03_fc_loss_curves.eps` | `3_fc_autoencoder.py` |
| 3 | FC AE vs CAE reconstruction | `04_fc_vs_cae_reconstruction.eps` | `4_conv_autoencoder.py` |
| 4 | Clean vs noisy vs denoised | `05_clean_noisy_denoised_*.eps` | `5_denoising_autoencoder.py` |
| 5 | Noise level vs MSE/MAE/SSIM | `05_noise_level_vs_{mse,mae,ssim}_*.eps` | `5_denoising_autoencoder.py` |
| 6 | VAE latent space | `07_vae_latent_space.eps` | `7_vae_latent_space.py` |
| 7 | Random VAE samples | `07_vae_generated_5x5.eps` | `7_vae_latent_space.py` |
| 8 | Latent interpolation | `07_vae_interpolation_images.eps`, `07_vae_interpolation_paths.eps` | `7_vae_latent_space.py` |
| 9 | VAE reconstruction loss | `06_vae_reconstruction_loss.eps` | `6_train_vae.py` |
| 10 | Reconstruction error distribution | `08_error_histogram_*.eps` | `8_error_analysis.py` |
| extra | Latent dimension vs MSE (Plot 10 of Section 27) | `09_latent_dim_vs_mse_fc.eps` | `9_latent_dimension_study.py` |

Numerical results (JSON and CSV) land in `outputs/results/`. The tables
for Sections 9, 11, 14, 21, 24, 25 and 27 are all there, and
`13_consolidated_results.py` also writes ready to paste LaTeX
`tabular` snippets (`13_*.tex`).

## Figure font (Colab only, run once)

Times New Roman isn't installed on Colab by default. Run this in a
Colab cell before running any plotting script:

```python
!echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
!sudo apt-get install -y ttf-mscorefonts-installer
!sudo fc-cache -f
```

If you skip this, the scripts still run, `plot_style.py` falls back to a
generic serif font and prints a warning instead of failing.

## How to run

```bash
pip install -r requirements.txt

python 1_data_exploration.py
python 2_preprocessing.py
python 3_fc_autoencoder.py
python 4_conv_autoencoder.py
python 5_denoising_autoencoder.py
python 6_train_vae.py
python 7_vae_latent_space.py
python 8_error_analysis.py
python 9_latent_dimension_study.py
python 10_decoder_comparison.py
python 11_vae_latent_comparison.py
python 12_kl_weight_study.py
python 13_consolidated_results.py
```

In Colab, prefix each line with `!` after `%cd` into the repository
folder. Use a GPU runtime (Runtime > Change runtime type > GPU).

Dependencies between scripts:

- every script from 3 onwards needs `2_preprocessing.py` (it loads the arrays from `outputs/saved_data/`)
- `4_conv_autoencoder.py` needs `3_fc_autoencoder.py` (for the FC vs CAE comparison)
- `7_vae_latent_space.py` needs `6_train_vae.py`
- `8_error_analysis.py` needs scripts 3, 4, 5 and 6 (it loads all four saved models)
- `13_consolidated_results.py` needs scripts 3, 4, 5 and 6
- scripts 9 to 12 train their own models and only need `2_preprocessing.py`

## Experimental protocol

- Same data for every model: 9000 training, 1000 validation and 2000 test images, drawn with a fixed seed.
- Same training setup for every deterministic model: Adam, learning rate 0.001, batch size 128, 20 epochs, binary cross entropy.
- The VAE uses the same optimizer, learning rate and batch size and trains for 30 epochs (`VAE_EPOCHS` in `config.py`), since its objective also has to shape the latent distribution.
- VAE loss: binary cross entropy summed over the 784 pixels plus the KL divergence to N(0, I), both averaged over the batch. VAE MSE, MAE and SSIM decode the mean mu, so they are comparable with the other models.
- The denoising CAE reported in the summary tables is the Gaussian sigma = 0.2 model; its input is the noisy test image and its error is always measured against the clean image.
- Digit labels are used only for colouring plots and for the error analysis, never as a training target.
- Every training run calls `config.set_seed()` first. GPU kernels can still introduce very small run to run differences.

Every setting above lives in `config.py`, so a change there applies to
every script.

## Dataset

MNIST handwritten digits: 60000 training and 10000 test grayscale
images of size 28 x 28, 10 classes. Loaded with
`tensorflow.keras.datasets.mnist.load_data()`, which downloads and caches
it automatically the first time (needs internet access). Source:
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist

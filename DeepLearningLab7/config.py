"""
Central configuration for Experiment 7 (autoencoders on MNIST).

Every script imports its paths, seeds and hyperparameters from here, so
the whole experiment follows one controlled protocol (Section 1 of the
lab manual) and any setting only has to be changed in one place.
"""
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(OUTPUT_DIR, "saved_data")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_model")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

# reproducibility
SEED = 42

# dataset (Section 3: recommended laboratory subset)
N_TRAIN = 10000          # training pool drawn from the MNIST training set
N_TEST = 2000            # test images drawn from the MNIST test set
VAL_FRACTION = 0.1       # 1000 of the 10000 training images are held out for validation
IMG_SHAPE = (28, 28, 1)
FLAT_DIM = 28 * 28

# shared training setup (Section 7 suggested configuration)
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 20

# fully connected autoencoder (Section 7)
FC_LATENT_DIM = 16
# The manual asks for ReLU in every hidden layer, so the latent layer uses
# ReLU by default. Set this to "linear" if the very small latent sizes in
# the latent dimension study (dz = 2) collapse because of dead units.
LATENT_ACTIVATION = "relu"

# denoising autoencoder (Section 13)
GAUSSIAN_SIGMAS = [0.1, 0.2, 0.3]
SALT_PEPPER_LEVELS = [0.05, 0.10, 0.20]
MAIN_NOISE_TYPE = "gaussian"   # the setting reported as "Denoising CAE" in the summary tables
MAIN_NOISE_LEVEL = 0.2

# variational autoencoder (Section 17)
VAE_LATENT_DIM = 2
VAE_EPOCHS = 30

# studies and additional exercises (Sections 27 and 29)
FC_LATENT_DIMS = [2, 8, 16, 32]
CAE_LATENT_DIMS = [4, 8, 16, 32]
VAE_COMPARE_LATENT_DIMS = [2, 8]
KL_WEIGHTS = [1.0, 2.0, 5.0, 10.0]

# visualization settings
N_DISPLAY = 10              # images shown in reconstruction grids (at least 10 per the manual)
N_GENERATED = 25            # VAE samples in the main generation grid (5 x 5)
N_GENERATED_LARGE = 100     # larger VAE sample grid for exercise 7 (10 x 10)
INTERP_STEPS = 11           # alpha = 0, 0.1, ..., 1.0
INTERP_DIGITS = (3, 8)      # digits of the two test images used for interpolation
CENTROID_DIGITS = (0, 1)    # digit regions used for the centroid interpolation (exercise 8)
N_HIGH_ERROR = 5            # Section 24


def make_dirs():
    for d in (DATA_DIR, MODEL_DIR, PLOT_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)


def set_seed(seed=SEED):
    """Seeds python, numpy and tensorflow so every run starts identically."""
    import numpy as np
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def plot_path(name):
    return os.path.join(PLOT_DIR, name)


def result_path(name):
    return os.path.join(RESULT_DIR, name)


def model_path(name):
    return os.path.join(MODEL_DIR, name)

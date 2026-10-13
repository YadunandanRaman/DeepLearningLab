"""
Central configuration for Experiments 8 and 9 (attention in a pretrained ResNet50).

Every script imports its paths, seed and hyperparameters from here, so all
models are trained under one controlled protocol (Section 34 of the lab
manual). Only the factor under study (attention type, position, order,
kernel size, mechanism) may change between runs.

Environment variables (so no command line flags are needed):
  OUTPUT_ROOT=/path   folder that receives outputs/ (default: Google Drive CS3807_Exp8_9 if Drive is mounted, else next to the code)
  QUICK=1        tiny smoke test: 1+1 epochs on a small data subset, outputs go to outputs_quick/
  SYNTHETIC=1    dry run of the whole pipeline on random images without any download (implies QUICK)
  FORCE=1        retrain runs even if their results already exist (also repeats the hyperparameter search)
  MIXED=0        switch off mixed precision (default: on whenever a GPU is present)
  USE_TUNED=0    ignore the result of the hyperparameter search and use the defaults below
  ONLY=A1,A3     run only the listed experiment ids (prefix match on the experiment name)
"""
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SYNTHETIC = os.environ.get("SYNTHETIC", "0") == "1"   # developer dry run: random images, random weights, no downloads
QUICK = os.environ.get("QUICK", "0") == "1" or SYNTHETIC
PRETRAINED = not SYNTHETIC
MIXED_PRECISION = os.environ.get("MIXED", "auto")      # auto = on when a GPU is present
FORCE = os.environ.get("FORCE", "0") == "1"
ONLY = [s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()]



def _output_root():
    """Where results are written. OUTPUT_ROOT env var wins; on Colab with Google Drive mounted the results go to
    Drive (so a disconnect loses nothing); otherwise they go next to the code."""
    root = os.environ.get("OUTPUT_ROOT")
    if root:
        return root
    drive = "/content/drive/MyDrive"
    if os.path.isdir(drive):
        return os.path.join(drive, "CS3807_Exp8_9")
    return BASE_DIR


OUTPUT_ROOT = _output_root()
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "outputs_quick" if QUICK else "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")      # big, re-creatable image cache stays on the fast local disk
DATA_DIR = os.path.join(OUTPUT_DIR, "saved_data")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_model")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
RUN_DIR = os.path.join(RESULT_DIR, "runs")

# reproducibility
SEED = 42

# dataset (Section 3)
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
SPLIT = (0.70, 0.15, 0.15)           # train : validation : test
QUICK_SUBSET = (40, 20, 20) if SYNTHETIC else (320, 96, 96)   # images per split when QUICK=1

# augmentation, identical for every model (Section 3.1)
AUG_FLIP = "horizontal"
AUG_ROTATION = 0.03                  # fraction of 2*pi, about 11 degrees
AUG_ZOOM = 0.10
AUG_TRANSLATION = 0.10
AUG_CONTRAST = 0.10

# transfer learning protocol (Section 20), identical for every model
BATCH_SIZE = 64                      # default for a 24 GB GPU; the hyperparameter search (02b) may change it
DROPOUT = 0.3
HEAD_LR = 1e-3                       # phase 1: attention and classifier layers, backbone frozen
FT_LR = 1e-4                         # phase 2: fine tuning, cosine decayed
FT_LR_END_FRACTION = 0.01
PHASE1_EPOCHS = 1 if QUICK else 8
PHASE2_EPOCHS = 1 if QUICK else 7
UNFREEZE_STAGES = (5,)               # conv5_x is unfrozen in phase 2 (BatchNorm layers stay frozen)

# hyperparameter search on the baseline (02b_hyperparameter_search.py): 5 combinations x 3 epochs = 15 epochs.
# Each trial = HP_PHASE1_EPOCHS epochs with the backbone frozen + HP_PHASE2_EPOCHS epoch of fine tuning of conv5_x.
# The learning rates scale with the batch size; the best VALIDATION accuracy wins (the test set is never used).
HP_PHASE1_EPOCHS = 1 if QUICK else 2
HP_PHASE2_EPOCHS = 1
HP_CANDIDATES = [
    {"id": "H1", "batch_size": 64,  "head_lr": 1e-3, "ft_lr": 1e-4, "dropout": 0.3},   # reference
    {"id": "H2", "batch_size": 64,  "head_lr": 3e-4, "ft_lr": 3e-5, "dropout": 0.3},   # smaller learning rates
    {"id": "H3", "batch_size": 64,  "head_lr": 3e-3, "ft_lr": 3e-4, "dropout": 0.3},   # larger learning rates
    {"id": "H4", "batch_size": 128, "head_lr": 3e-3, "ft_lr": 3e-4, "dropout": 0.5},   # larger batch, more dropout
    {"id": "H5", "batch_size": 32,  "head_lr": 1e-3, "ft_lr": 1e-4, "dropout": 0.5},   # smaller batch, more dropout
]
HP_FILE = os.path.join(RESULT_DIR, "hp_search", "best_hparams.json")

# attention module settings
REDUCTION = 16                       # channel attention reduction ratio r (Section 9)
SPATIAL_KERNEL = 7                   # default spatial attention kernel (Section 10)
GATE_BIAS_INIT = 3.0                 # attention gates start close to identity (sigmoid(3) = 0.95)
GATE_KERNEL_STD = 0.01               # the kernels that produce a gate start small, so the initial gate is about sigmoid(bias)
SELF_ATT_QK_DIV = 8                  # query/key width = C / 8
SELF_ATT_V_DIV = 2                   # value width = C / 2, projected back to C

# stage and position definitions (Section 7)
POSITIONS = {
    1: {"name": "P1", "after": "conv2_x", "resolution": 56, "channels": 256},
    2: {"name": "P2", "after": "conv3_x", "resolution": 28, "channels": 512},
    3: {"name": "P3", "after": "conv4_x", "resolution": 14, "channels": 1024},
    4: {"name": "P4", "after": "conv5_x", "resolution": 7, "channels": 2048},
}
GRADCAM_LAYER = "conv5_block3_out"   # last ResNet50 feature map, 7 x 7 x 2048

# evaluation
INFER_BATCH = 64
PREDICT_BATCH = 128                  # batch size for prediction passes (inference only, no effect on results)
INFER_REPEATS = 3
BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95

# visualization
TOP_K_CHANNELS = 20
N_QUALITATIVE_CORRECT = 3
N_QUALITATIVE_WRONG = 2
OVERLAY_ALPHA = 0.45
SELF_ATT_QUERIES = [(3, 3), (3, 10), (10, 3), (10, 10)]   # query positions on the 14 x 14 grid

# optional manual override of the automatically selected attention model
# (the default selection is the best validation accuracy among A1 to A5)
SELECTED_OVERRIDE = None


HP_SOURCE = "defaults of config.py"


def _apply_tuned_hparams():
    """Use the winner of the hyperparameter search for every run (same values for all models)."""
    global HEAD_LR, FT_LR, DROPOUT, BATCH_SIZE, HP_SOURCE
    if os.environ.get("USE_TUNED", "1") == "1" and os.path.exists(HP_FILE):
        import json
        with open(HP_FILE) as f:
            best = json.load(f)["best"]
        HEAD_LR, FT_LR, DROPOUT, BATCH_SIZE = best["head_lr"], best["ft_lr"], best["dropout"], best["batch_size"]
        HP_SOURCE = f"hyperparameter search ({best['id']})"


_apply_tuned_hparams()


def hparams():
    return {"batch_size": BATCH_SIZE, "head_lr": HEAD_LR, "ft_lr": FT_LR, "dropout": DROPOUT}


_RUNTIME_READY = False


def setup_runtime():
    """GPU memory growth and mixed precision (float16 compute, float32 weights) for speed on a large GPU.
    Called before every model is built (keras.backend.clear_session() resets the policy, so it is set again each time)."""
    global _RUNTIME_READY
    import tensorflow as tf
    from tensorflow import keras

    gpus = tf.config.list_physical_devices("GPU")
    use = (MIXED_PRECISION == "1") or (MIXED_PRECISION == "auto" and len(gpus) > 0)
    keras.mixed_precision.set_global_policy("mixed_float16" if use else "float32")
    if _RUNTIME_READY:
        return
    _RUNTIME_READY = True
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print(f"[config] GPUs: {len(gpus)}, precision policy: {keras.mixed_precision.global_policy().name}, "
          f"hyperparameters from {HP_SOURCE}: {hparams()}")


_ANNOUNCED = False


def make_dirs():
    global _ANNOUNCED
    for d in (DATA_DIR, MODEL_DIR, PLOT_DIR, RESULT_DIR, RUN_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(f"[config] results are saved in: {OUTPUT_DIR}")
        if os.path.isdir("/content") and not OUTPUT_DIR.startswith("/content/drive"):
            print("[config] WARNING: Google Drive is not mounted, results are on the temporary Colab disk and are lost "
                  "if the runtime disconnects. Mount Drive first (see README).")


def set_seed(seed=SEED):
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


def run_dir(name):
    return os.path.join(RUN_DIR, name)

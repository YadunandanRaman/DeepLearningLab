# CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning, and Cross Validation on Oxford-IIIT Pet

CS3807 Deep Learning Lab, Experiment 5. A single architecture,
MobileNetV2, studied across every stage of the training pipeline:
weight initialization, regularization, Batch Normalization,
optimizers, hyperparameter tuning, transfer learning and fine tuning,
K fold cross validation, and a final held out test evaluation.

## Files

- `plot_style.py`: shared plotting setup, loads Times New Roman, saves figures as EPS at 600 DPI
- `data_loader.py`: loads the Oxford-IIIT Pet dataset via `tensorflow_datasets`, resizes to 224 x 224 x 3, and produces a genuine training/validation/test split, with the test set never touched outside `9_final_evaluation.py`
- `model_builder.py`: builds a MobileNetV2 transfer learning model with configurable initialization, regularization, and freeze state; also the fine tuning helpers used in Section 10
- `1_data_exploration.py`: loads the dataset, shows 10 sample images, prints dimensions, plots class distribution across all 37 breeds
- `2_weight_initialization.py`: Section 5, compares Zero, Random, Xavier, and He initialization (Plots 1 and 2)
- `3_regularization.py`: Section 6, compares no regularization, L2, Dropout, and Batch Normalization (Plots 3 and 4)
- `4_batch_norm_comparison.py`: Section 7, a focused with/without Batch Normalization comparison (Plot 5); the numerical Batch Normalization example itself is pure arithmetic and needs no code
- `5_optimizer_comparison.py`: Section 8, compares SGD, Momentum, RMSProp, and Adam (Plots 6 and 7, plus a summary table)
- `6_hyperparameter_tuning.py`: Section 9, sweeps learning rate, batch size, and dropout rate one at a time (Plots 8, 9, 10)
- `7_transfer_learning_finetuning.py`: Section 10, Case A (feature extraction) followed by Case B (fine tuning), plotted on one continuous epoch timeline (Plots 11 and 12)
- `8_cross_validation.py`: Section 11, 5 fold cross validation on 4 candidate configurations (Plot 13)
- `9_final_evaluation.py`: Section 12, retrains the winning configuration on the complete training pool and evaluates once on the untouched test set (Plots 14 and 15)

Scripts 2 through 8 are independent of each other and can run in any
order, each loads its own data and trains its own models from scratch.
`9_final_evaluation.py` is the only script with a soft dependency: if
`8_cross_validation.py` has already been run and its results are
sitting in `outputs/results/cross_validation.csv`, the final evaluation
script pulls the matching configuration's mean accuracy and standard
deviation directly from that file rather than asking you to retype
them. If that file is not there yet, those two fields are simply left
blank rather than guessed.

## Two things you need to fill in based on your own results

This experiment is explicitly designed around choosing configurations
based on what earlier sections actually show, so two scripts ship with
placeholder choices that are reasonable defaults, not claimed answers:

- `8_cross_validation.py`'s `CONFIGS` dictionary holds 4 representative
  candidates (a baseline, He initialization with Dropout, L2, and a
  lower learning rate with Dropout). Section 11 asks you to select
  "3 to 4 promising configurations" based on Sections 5 through 10's
  results, so replace these with whichever configurations actually
  performed best once you have run those sections for real.
- `9_final_evaluation.py`'s `BEST_CONFIG_NAME` and `BEST_CONFIG` must
  match whichever configuration actually won in `8_cross_validation.py`.
  Update both together if your cross validation results point to a
  different configuration than the current default (`He + Dropout`).

## A note on weight initialization

Section 5's four strategies are applied only to the new classifier
head's Dense layers, not to the pretrained MobileNetV2 backbone. The
backbone's weights always come from ImageNet regardless of this
setting, since using anything else would defeat the point of transfer
learning entirely. Zero initialization is deliberately allowed to
produce the pathology Discussion Question 3 asks about directly: with
every weight in a layer starting at exactly zero, every neuron in that
layer computes an identical output and receives an identical gradient,
so they never differentiate from each other and the layer never
actually learns to use more than one effective neuron's worth of
capacity. This was verified directly, not just described: after
building the zero initialized model, `model.get_weights()` on both new
Dense layers confirmed every value is exactly 0.0, while He
initialization on the same layers produced the expected properly
scaled random spread.

## A note on fine tuning and the model save/reload

`7_transfer_learning_finetuning.py` builds its model with the
Functional API rather than Sequential and retrieves the nested
MobileNetV2 backbone after saving and reloading by finding the one
layer that is itself a Keras `Model` instance (`get_base_model` in
`model_builder.py`), rather than by its auto generated name, which is
Keras version dependent. This exact pattern, save, reload, retrieve the
nested backbone by type, unfreeze its last block, and continue
training, was tested end to end before being used here, since an
earlier project in this course hit a real bug in exactly this spot
with a Sequential model.

## A note on `classification_report` and macro averaging

`9_final_evaluation.py` passes `labels=list(range(37))` explicitly to
`classification_report`, `confusion_matrix`, and the macro averaged
precision/recall/F1 calls, rather than letting scikit-learn infer the
label set from whatever classes happen to appear in a given batch. With
the real test set (37 balanced breeds, about 99 images each) every
class will appear, so this makes no visible difference, but it was
caught as a real, reproducible crash during testing with a smaller
sample where one breed had zero images in a batch:
`classification_report` raises `ValueError: Number of classes ... does
not match size of target_names` in exactly that case. Pinning the label
space explicitly avoids the crash and, just as importantly, keeps the
macro average genuinely over all 37 breeds even if that situation ever
recurs, rather than silently averaging over fewer classes.

## Figure font (Colab only, run once)

Times New Roman isn't installed on Colab by default. Run this in a
Colab cell before running any plotting script:

```python
!echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
!sudo apt-get install -y ttf-mscorefonts-installer
!sudo fc-cache -f
```

If you skip this, the scripts still run, `plot_style.py` falls back to
a generic serif font and prints a warning instead of failing.

## Known issue: protobuf / tensorflow_datasets version mismatch

If `1_data_exploration.py`, or any other script, fails with
`AttributeError: module 'tensorflow_datasets' has no attribute 'load'`,
look one screen up in the same traceback for a line like:

```
google.protobuf.runtime_version.VersionError: Detected incompatible Protobuf Gencode/Runtime versions when loading tensorflow_metadata/proto/v0/anomalies.proto: gencode 6.31.1 runtime 5.29.6. Runtime version cannot be older than the linked gencode version.
```

That is the real error. `tensorflow_datasets` depends on
`tensorflow_metadata`, which ships code generated against a newer
version of the `protobuf` library than the one already active in
Colab's preinstalled environment. Protobuf's own compatibility rule is
one directional, a newer runtime can read older generated code, but not
the other way around, so `tensorflow_datasets` fails partway through
its own import, leaving the `tensorflow_datasets` module technically
importable but missing the pieces (like `load`) that never finished
wiring themselves up, which is why the second error looks unrelated to
protobuf at first glance.

`requirements.txt` already pins `protobuf>=6.31.0`, but exactly like
the SciKeras issue in Experiment 2, Colab's base image may have already
imported an older protobuf into memory before `pip install -r requirements.txt`
runs in a given session, so installing the newer version to disk is not
enough on its own. Run this instead, then restart the runtime before
running any script:

```python
!pip install --upgrade tensorflow-datasets tensorflow-metadata protobuf -q
```

Then **Runtime > Restart session** in Colab, and run the scripts again
from `1_data_exploration.py`. This was verified to work as a compatible
set: `tensorflow-datasets==4.9.10`, `tensorflow-metadata==1.21.0`,
`protobuf==7.34.1`, if the upgrade above lands on different exact
versions and the error persists, pinning to these three explicitly is
the fallback.

## How to run

```bash
pip install -r requirements.txt

python 1_data_exploration.py
python 2_weight_initialization.py
python 3_regularization.py
python 4_batch_norm_comparison.py
python 5_optimizer_comparison.py
python 6_hyperparameter_tuning.py
python 7_transfer_learning_finetuning.py
python 8_cross_validation.py
python 9_final_evaluation.py
```

**Use a GPU runtime** (Runtime > Change runtime type > GPU in Colab).
This experiment trains a great many MobileNetV2 based models: 4 for
initialization, 4 for regularization, 2 for Batch Normalization, 4 for
optimizers, 8 across the three hyperparameter sweeps, 2 for transfer
learning and fine tuning, 20 for cross validation (4 configurations
times 5 folds), and 1 final retrain, all at the manual's specified
224 x 224 resolution, which is larger, and therefore slower per image,
than the 96 x 96 or 32 x 32 used in earlier experiments in this course.
`8_cross_validation.py` in particular is the most expensive single
script, since it trains 20 separate models.

## About the Oxford-IIIT Pet dataset

37 cat and dog breeds, roughly 200 images per breed, loaded via
`tensorflow_datasets.load("oxford_iiit_pet")`, which downloads and
caches the dataset automatically the first time it runs (needs internet
access, and is a different mechanism from the `tensorflow.keras.datasets`
loaders used for CIFAR-10 and Fashion-MNIST in earlier experiments,
since this dataset is not one of Keras's built in ones). The official
split provides only training (3,680 images) and test (3,669 images)
sets; `data_loader.py` carves a validation set out of the training
split itself (80/20), so all three of training, validation, and test
genuinely exist as separate pools, exactly as Section 3 requires, and
the test set is never touched until `9_final_evaluation.py`.

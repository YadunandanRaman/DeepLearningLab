# MLP for Fashion-MNIST Classification

CS3807 Deep Learning Lab, Experiment 2. Multi-layer perceptron implemented
with TensorFlow/Keras, trained on Fashion-MNIST, with automated
hyperparameter optimization using RandomizedSearchCV and SciKeras.

## Files

- `plot_style.py` - shared plotting setup: loads Times New Roman, saves figures as EPS at 600 DPI
- `model_builder.py` - the baseline architecture (Task 3) and the flexible architecture used only by the hyperparameter search (Section 7)
- `1_data_exploration.py` - load Fashion-MNIST, print dimensions, sample images, class distribution (Task 1)
- `2_preprocessing.py` - flatten, normalize, one-hot encode, prints tensor shapes before/after (Task 2)
- `3_model_construction.py` - builds the baseline MLP and prints `model.summary()` (Task 3)
- `4_train_model.py` - compiles and trains the baseline model, saves it, plots 4 training curves (Task 4)
- `5_evaluate.py` - accuracy/precision/recall/F1/confusion matrix/classification report on the baseline model (Task 5)
- `6_hyperparameter_search.py` - RandomizedSearchCV over the Section 7 search space, retrains the best config on the full training set, compares against the baseline (Section 7)
- `7_xor_gate.py` - builds a multi-layer perceptron to solve the XOR gate using backpropagation and plots the decision boundary

All figures are saved as `.eps` at 600 DPI, in Times New Roman, with axis
labels and a legend on every plot, matching the 9 mandatory plots listed
in Section 8 of the lab manual:


| # | Plot | Produced by |
|---|------|-------------|
| 1 | Sample Images | `1_data_exploration.py` |
| 2 | Class Distribution | `1_data_exploration.py` |
| 3 | Training Accuracy vs Epoch | `4_train_model.py` |
| 4 | Validation Accuracy vs Epoch | `4_train_model.py` |
| 5 | Training Loss vs Epoch | `4_train_model.py` |
| 6 | Validation Loss vs Epoch | `4_train_model.py` |
| 7 | Confusion Matrix | `5_evaluate.py` |
| 8 | Hyperparameter Search Results | `6_hyperparameter_search.py` |
| 9 | Best Model Accuracy Comparison | `6_hyperparameter_search.py` |

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

## Known issue: SciKeras / scikit-learn version mismatch

If `6_hyperparameter_search.py` fails with
`AttributeError: 'super' object has no attribute '__sklearn_tags__'`,
that's a compatibility break between scikit-learn 1.6+ (which rewrote its
internal tags system) and SciKeras 0.13.0 (the latest release, which
hasn't caught up yet). `requirements.txt` already pins
`scikit-learn<1.6` to avoid this, but Colab ships with a newer
scikit-learn preinstalled, so a plain `pip install -r requirements.txt`
in an existing Colab session may not be enough, since scikit-learn is
usually already imported in memory. Run this instead, then restart the
runtime before running any script:

```python
!pip install -r requirements.txt --force-reinstall --no-deps -q
!pip install "scikit-learn<1.6" --force-reinstall -q
```

Then **Runtime > Restart session** in Colab, and run the scripts again
from `1_data_exploration.py`.

## How to run

```bash
pip install -r requirements.txt

python 1_data_exploration.py
python 2_preprocessing.py
python 3_model_construction.py
python 4_train_model.py
python 5_evaluate.py
python 6_hyperparameter_search.py
python 7_xor_gate.py
```

Scripts 3 onwards need `2_preprocessing.py` to have run first, since they
load the flattened/normalized arrays from `outputs/saved_model/`.
`5_evaluate.py` needs `4_train_model.py` to have run first (it loads the
saved baseline model). `6_hyperparameter_search.py` needs both
`2_preprocessing.py` and `4_train_model.py` to have run first, since it
compares against the saved baseline model and its recorded training time.

**Use a GPU runtime if you can** (Runtime > Change runtime type > GPU in
Colab). The baseline model is small and trains fine on CPU, but the
hyperparameter search trains up to `N_ITER x CV_FOLDS = 50` models, some
with up to 3 hidden layers of 256 neurons for up to 30 epochs, which is
meaningfully faster on a GPU.

## About the hyperparameter search

The full search space in Section 7 of the manual has up to 3,888 possible
combinations (3 layer counts x 4 neuron counts x 3 learning rates x 4
batch sizes x 3 epoch counts x 3 optimizers x 3 activations x 3 dropout
rates). Trying all of them with 5-fold cross validation on the full
60,000 image training set is not realistic on typical classroom hardware,
which is exactly why the manual recommends RandomizedSearchCV over
GridSearchCV.

`6_hyperparameter_search.py` therefore:

1. Samples `N_ITER = 10` random combinations from the search space (adjust
   this constant at the top of the script if you have more time or
   compute available).
2. Runs the 5-fold cross validation search on a random subsample of
   `SEARCH_SAMPLE_SIZE = 6000` training images rather than the full
   60,000, to keep the search itself fast.
3. Retrains the best combination found on the **full** 60,000 image
   training set, so the shortcuts above only affect how the search is
   scored, not the final delivered model.
4. Evaluates that retrained model on the untouched test set and compares
   it against the baseline model from `4_train_model.py`.

`outputs/results/hyperparameter_search_cv_results.csv` has the full
scikit-learn `cv_results_` table, and the script prints the mean CV
accuracy grouped by each hyperparameter's value, which is useful evidence
for the "which hyperparameter had the greatest impact" discussion
question in the manual.

## Dataset

Fashion-MNIST: 60,000 training images, 10,000 test images, 10 classes,
28 x 28 grayscale. Loaded directly via
`tensorflow.keras.datasets.fashion_mnist.load_data()`, which downloads
and caches the dataset automatically the first time it runs (needs
internet access).

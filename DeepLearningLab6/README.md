# RNN, LSTM, and GRU for Sequence Learning and Video Understanding

CS3807 Deep Learning Lab, Experiment 6. Sequence classification on the
UCI HAR dataset (RNN vs LSTM vs GRU), video action recognition with a
frozen CNN feature extractor feeding an LSTM or GRU, and a synthetic
sequence to sequence reversal task with an encoder decoder LSTM.

## Files

- `plot_style.py`: shared plotting setup, Times New Roman, EPS at 600 DPI
- `data_loader.py`: extracts the UCI HAR dataset from your local `UCI_Human.zip`, loads its raw inertial signals (not the precomputed 561 feature vectors), windows them into `(N, 128, 9)`, and produces a genuine train/validation/test split
- `model_builder.py`: the shared RNN/LSTM/GRU classifier, the CNN feature extractor, the CNN-LSTM/CNN-GRU video classifier, and the seq2seq encoder decoder
- `1_data_exploration.py`: prints dataset shapes and class counts, plots representative sensor signals from 3 activities (Plot 1)
- `2_bptt_numerical.py`: Section 8's numerical exercise, computed by formula and cross checked against an actual Keras `SimpleRNN` layer
- `3_train_rnn.py` / `4_train_lstm.py` / `5_train_gru.py`: train each recurrent classifier under identical conditions (Plots 2, 3 for each)
- `6_evaluate_and_compare.py`: evaluates all three on the test set, one confusion matrix per model (Plot 4), and a comparison bar chart (Plot 5)
- `7_sequence_length_study.py`: retrains all three at sequence lengths 32, 64, and 128 (Plot 6)
- `8_video_understanding.py`: extracts your local `UCF101.rar`, samples frames from 5 selected classes, extracts frozen MobileNetV2 features, trains a CNN-LSTM or CNN-GRU classifier (Plots 7, 8, 9)
- `9_seq2seq.py`: the synthetic sequence reversal task, trains the encoder decoder with teacher forcing, evaluates with real greedy inference (no teacher forcing)
- `10_consolidated_results.py`: merges every script's results into Section 25's final tables

Run `1` through `7`, then `9`, then `10` in order; each of `3` through
`7` trains its own models from scratch. `8` is independent of the rest.

## Before running anything: mount Google Drive and set two paths

Both datasets are read from your Google Drive, so mount it first in a
Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then set these two constants to wherever your files actually are:

- `data_loader.py`'s `ZIP_PATH`, currently `/content/drive/MyDrive/UCI_Human.zip`
- `8_video_understanding.py`'s `RAR_PATH`, currently `/content/drive/MyDrive/LatentWorldModel_/UCF101.rar`, the folder name here was truncated in the screenshot this path was taken from, double check it matches your actual Drive folder name exactly.

`8_video_understanding.py` also needs the `unrar` tool, not installed
on a fresh Colab runtime by default:

```python
!apt-get install -y unrar
```

## The UCI HAR dataset: raw signals, not the 561 feature vectors

The dataset ships both a precomputed 561 feature version and the raw
sensor signals the manual specifically asks for. `data_loader.py`
reads only the raw `Inertial Signals` files (9 of them: 3 axes of body
acceleration, 3 of gyroscope, 3 of total acceleration), each with one
row per window and one column per of the 128 time steps in that
window, and stacks them into `(N, 128, 9)`.

`ZIP_PATH` is extracted once into a cached `uci_har_extracted` folder
next to the zip itself; later runs reuse it rather than extracting again.
If your zip's internal folder layout differs slightly from the
standard release, `_extract` searches the extracted archive for any
folder containing both a `train` and `test` subfolder, so it should
still be found automatically.

## Train, validation, and test: why it is not a plain 70/15/15 split

The dataset's own train and test partitions are subject disjoint, no
person appears in both, which is what makes the test set a genuine
measure of generalizing to new people rather than new windows from
people the model has already partly seen. Rather than discard that
design for a fresh random 70/15/15 cut across everyone, the official
test split is kept exactly as given, and validation is carved out of
the official training split instead (15 percent), so validation stays
subject disjoint from training the same way, and the untouched test
set is not touched by that split either. This lands close to, but not
exactly, 70/15/15 (the official test split is somewhat larger than 15
percent on its own), documented here since it is a deliberate choice,
not an oversight.

## Video understanding: extracting UCF101 and picking classes

`UCF101.rar` is the complete archive, all 101 classes, so
`8_video_understanding.py` extracts the whole thing once (there is no
simple, reliable way to extract only specific class subfolders from a
`.rar` without first knowing its exact internal path, which varies by
release) and caches the result in `ucf101_extracted` next to the
archive; later runs reuse it rather than extracting again. Only 5 classes
are actually used for training, set in `SELECTED_CLASSES`:

```python
SELECTED_CLASSES = ["Basketball", "Biking", "WalkingWithDog", "TennisSwing", "JumpRope"]
```

One correction from the lab manual's own suggested list: it names
"Running" as a possible class, but UCF101 has no class by that exact
name, so this was swapped for JumpRope, and "Walking" was corrected to
UCF101's actual class name, WalkingWithDog. Change `SELECTED_CLASSES`
to any other real UCF101 class names if you would prefer a different
5 (or 3 to 5, per the manual).

`MAX_VIDEOS_PER_CLASS` (default 10) caps how many videos per class are
actually processed, keeping this fast, since the point of this section
is to see the CNN plus recurrent pipeline work, not to train a
competitive action recognition model on the full dataset.

This script was verified end to end against synthetic placeholder
videos and a simulated extracted archive before being handed off:
frame sampling, the CNN feature extractor's output dimension (1280,
from MobileNetV2), the exact `(10, 1280)` tensor shape handed to the
recurrent classifier, training, and a genuine model produced confidence
value were all confirmed working. Only the actual accuracy numbers need
your real UCF101 videos and a working `unrar` install.

## Sequence to sequence: teacher forcing versus real inference

`9_seq2seq.py` trains the decoder with teacher forcing, at each step
its input is the true previous target token, not its own prediction,
the standard way to train these models (Discussion Question 22).
Evaluation, however, uses genuine greedy inference: the decoder only
ever sees its own previously generated token, exactly how the model
would actually be used once there is no ground truth to peek at. The
two are meaningfully different, and using teacher forcing at evaluation
time would overstate accuracy, so the evaluation loop in this script
never does that.

## Figure font (Colab only, run once)

Times New Roman is not installed on Colab by default:

```python
!echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
!sudo apt-get install -y ttf-mscorefonts-installer
!sudo fc-cache -f
```

If you skip this, the scripts still run, `plot_style.py` falls back to
a generic serif font and prints a warning instead of failing.

## How to run

```bash
pip install -r requirements.txt

python 1_data_exploration.py
python 2_bptt_numerical.py
python 3_train_rnn.py
python 4_train_lstm.py
python 5_train_gru.py
python 6_evaluate_and_compare.py
python 7_sequence_length_study.py
python 8_video_understanding.py
python 9_seq2seq.py
python 10_consolidated_results.py
```

**Use a GPU runtime** (Runtime > Change runtime type > GPU in Colab).
Scripts 3 through 7 alone train 3 (main models) + 9 (sequence length
study) = 12 recurrent models on the HAR dataset, and script 8 extracts
CNN features from every sampled frame of every video before training
its own classifier.

## Datasets

UCI HAR: 10,299 windows total across 30 subjects, 6 activities
(WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING,
LAYING), each window 128 time steps across 9 sensor channels.

UCF101: 101 action classes; this experiment uses 5 of them
(`SELECTED_CLASSES` in `8_video_understanding.py`), up to
`MAX_VIDEOS_PER_CLASS` videos each.

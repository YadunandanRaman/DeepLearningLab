# CNN for CIFAR-10 Image Classification

CS3807 Deep Learning Lab, Experiment 3, updated to match the full lab
manual (kernel size study, stride and padding study, feature map
visualization, pooling comparison, and a two block CNN trained exactly
as specified).

## Files

- `plot_style.py`: shared plotting setup, loads Times New Roman, saves figures as EPS at 600 DPI
- `data_loader.py`: shared CIFAR-10 loader, reads a local extracted copy if you have one, otherwise falls back to downloading via Keras
- `model_builder.py`: the CNN architecture from Task 6, with a switch for max versus average pooling used by Task 5
- `1_data_exploration.py`: load CIFAR-10, sample images, class distribution, dimensions (Task 1)
- `2_3_kernel_stride_study.py`: kernel size comparison and stride/padding study, using real Conv2D shape inference rather than hand arithmetic (Tasks 2 and 3)
- `3_train_model.py`: builds and trains the CNN (Task 6)
- `4_evaluate.py`: accuracy/precision/recall/F1/confusion matrix/classification report (Task 7)
- `5_feature_maps.py`: visualizes at least 8 feature maps from the first convolutional layer of the trained model (Task 4)
- `6_pooling_comparison.py`: confirms max and average pooling produce identical output sizes, then trains both variants to compare accuracy (Task 5)
- `7_filter_count_comparison.py`: for comparison of the effect of accuracy on number of filter size(additional task)

The numbers in these filenames are the order to run them in, which is
not quite the same as the task order in the lab manual, since Task 4
(feature maps) and Task 7 (evaluation) both need Task 6's trained model
to already exist. Running the files in the numeric order their names
give you is always safe.

## Run order

```bash
pip install -r requirements.txt

python 1_data_exploration.py
python 2_3_kernel_stride_study.py
python 3_train_model.py
python 4_evaluate.py
python 5_feature_maps.py
python 6_pooling_comparison.py
python 7_filter_count_comparison.py
```

`4_evaluate.py` and `5_feature_maps.py` both need `3_train_model.py` to
have run first, since they load the saved trained model.
`2_3_kernel_stride_study.py` and `6_pooling_comparison.py`'s output
size check need only the dataset, they build their own layers.

**Use a GPU runtime if you can** (Runtime > Change runtime type > GPU
in Colab). `6_pooling_comparison.py` trains two full CNNs on the
complete 50,000 image training set to give a fair accuracy comparison,
on top of the main model trained in `3_train_model.py`.

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

## Dataset

CIFAR-10: 50,000 training images, 10,000 test images, 10 classes,
32 x 32 colour images. `data_loader.py` is the single place the
dataset path lives, see the comment at the top of that file for how to
point it at a local copy instead of downloading one.

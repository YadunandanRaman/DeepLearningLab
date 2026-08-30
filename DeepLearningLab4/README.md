# Transfer Learning on CIFAR-10

CS3807 Deep Learning Lab, Experiment 4. Comparative study of deep CNN
architectures using transfer learning, implemented with TensorFlow/Keras
and trained on CIFAR-10.

## Files

- `plot_style.py`: shared plotting setup, loads Times New Roman, saves figures as EPS at 600 DPI
- `data_loader.py`: shared CIFAR-10 loader, reads a local extracted copy if you have one, otherwise falls back to downloading via Keras
- `model_builder.py`: builds a transfer learning model on top of a pretrained backbone (MobileNetV2, VGG16, or ResNet50), and the freeze and fine tuning logic used across the other scripts
- `1_data_exploration.py`: loads CIFAR-10, prints dimensions, shows 10 sample images (Task 1)
- `2_transfer_learning_model.py`: builds the frozen base transfer learning model, prints its summary and a trainable versus frozen parameter breakdown (Task 2)
- `3_train_model.py`: trains the frozen base model, saves it, plots 4 training curves (Task 3)
- `4_fine_tune.py`: unfreezes the base model's last block and continues training, comparing accuracy before and after (Task 4)
- `5_evaluate.py`: accuracy/precision/recall/F1/confusion matrix/classification report/misclassified images on the fine tuned model (Task 5)
- `6_hyperparameter_study.py`: trains 8 configurations, each changing one hyperparameter (learning rate, batch size, epochs, optimizer, dense units, frozen layers) away from the Task 3 baseline (Section 16)
- `7_architecture_comparison.py`: trains MobileNetV2, VGG16, ResNet50, and InceptionV3 (standing in for GoogleNet) under matched conditions via Keras, and AlexNet via PyTorch/torchvision, since Keras has never shipped a pretrained AlexNet. LeNet-5 is left out entirely, it has no pretrained weights and was designed for single channel images, not CIFAR-10's 3 channel colour images (Section 18.2)

All figures are saved as `.eps` at 600 DPI, in Times New Roman, with
axis labels and a legend on every plot that has one.

| # | Plot | Produced by |
|---|------|-------------|
| 1 | Sample Images | `1_data_exploration.py` |
| 2 | Training Accuracy | `3_train_model.py` |
| 3 | Validation Accuracy | `3_train_model.py` |
| 4 | Training Loss | `3_train_model.py` |
| 5 | Validation Loss | `3_train_model.py` |
| 6 | Confusion Matrix | `5_evaluate.py` |
| 7 | Misclassified Images (Optional) | `5_evaluate.py` |
| 8 | Fine Tuning Training Accuracy | `4_fine_tune.py` |
| 9 | Fine Tuning Validation Accuracy | `4_fine_tune.py` |
| 10 | Fine Tuning Before/After Comparison | `4_fine_tune.py` |
| 11 | Hyperparameter Study Comparison | `6_hyperparameter_study.py` |
| 12 | Architecture Comparison | `7_architecture_comparison.py` |

Plots 1 through 7 are the mandatory plots listed in the lab manual.
Plots 8 through 12 go beyond that list, since Task 4's before and after
comparison, the hyperparameter study, and the architecture comparison
are all central results of this experiment that are worth seeing as
figures rather than only as printed numbers.

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

## How to run

```bash
pip install -r requirements.txt

python 1_data_exploration.py
python 2_transfer_learning_model.py
python 3_train_model.py
python 4_fine_tune.py
python 5_evaluate.py
python 6_hyperparameter_study.py
python 7_architecture_comparison.py
```

**Use a GPU runtime** (Runtime > Change runtime type > GPU in Colab).
This experiment trains transfer learning heads on top of large
pretrained backbones repeatedly, once in Task 3, once in Task 4, eight
times in the hyperparameter study, and three times in the architecture
comparison, and VGG16 and ResNet50 in particular are meaningfully
slower per step than a CNN trained from scratch.

## About the hyperparameter study and architecture comparison

Both scripts train on a random subsample of the training set for a
small number of epochs, rather than the full 50,000 images to full
convergence, specifically to keep these comparisons fast. This is
enough to see the direction and rough size of an effect, not to fully
optimize any single configuration or declare one architecture
definitively best. `SUBSAMPLE_SIZE` and `EPOCHS` sit as constants near
the top of each script if you want to scale them up.

## Dataset

CIFAR-10: 50,000 training images, 10,000 test images, 10 classes,
32 x 32 colour images. `data_loader.py` is the single place the
dataset path lives, see the comment at the top of that file for how to
point it at a local copy instead of downloading one.

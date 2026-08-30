import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

from data_loader import load_data
from model_builder import build_transfer_model, compile_model
from plot_style import setup_fonts, save_fig

# torch and torchvision are imported before tensorflow and Keras
# above, not just for tidiness. Importing torchvision after tensorflow
# has already been imported in the same process reliably segfaults,
# both libraries bundle their own native image and math libraries, and
# whichever loads its versions of shared symbols first determines
# which one survives. Importing torch and torchvision first avoids
# this entirely. This was found by testing, not assumed, so please
# keep this import order if you edit this file.

setup_fonts()

# LeNet-5 was designed for single channel, 28x28 grayscale digit images
# and predates ImageNet scale pretraining entirely, it was never trained
# on anything resembling CIFAR-10's 3 channel colour images. Unlike the
# five architectures below, there is no pretrained LeNet-5 to load and
# adapt, so it is left out of this comparison rather than measured on
# data it was never suited for.
#
# All five architectures actually compared here use real ImageNet
# pretrained weights. MobileNetV2, VGG16, ResNet50, and InceptionV3 come
# from tensorflow.keras.applications; AlexNet does not, Keras has never
# shipped a pretrained AlexNet, so it is loaded from torchvision instead,
# the standard source for it, and used as a frozen feature extractor
# (see the AlexNet section below for exactly how it is combined with the
# rest of this otherwise Keras based experiment). InceptionV3 stands in
# for "GoogleNet": Keras has no true 2014 Inception v1, InceptionV3 is
# the closest architecture actually available with ImageNet weights, see
# model_builder.py for the same note.
#
# Every architecture below trains on the complete 50,000 image training
# set, the same as the rest of this experiment, rather than a
# subsample. This takes meaningfully longer than a subsample would,
# roughly 20 minutes total for the four Keras architectures plus
# AlexNet on a Colab GPU, since even with the backbone frozen, every
# training image still needs a full forward pass through it every
# epoch, and VGG16 and ResNet50 in particular are considerably heavier
# per image than the small CNN trained from scratch in Experiment 3.
KERAS_ARCHITECTURES = ["mobilenetv2", "vgg16", "resnet50", "inceptionv3"]
EPOCHS = 6
BATCH_SIZE = 32
LEARNING_RATE = 0.001

(X_train, y_train), (X_test, y_test) = load_data()
y_train = y_train.flatten()
y_test_labels = y_test.flatten()
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train_onehot = to_categorical(y_train, num_classes=10)

measured_results = []

# ------------------------------------------------------------------
# MobileNetV2, VGG16, ResNet50, InceptionV3
# ------------------------------------------------------------------
for architecture in KERAS_ARCHITECTURES:
    print(f"\ntraining architecture: {architecture}")
    model, base_model = build_transfer_model(architecture=architecture, dense_units=128, freeze_base=True)
    compile_model(model, optimizer_name="adam", learning_rate=LEARNING_RATE)

    total_params = model.count_params()

    start = time.time()
    model.fit(
        X_train, y_train_onehot,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    elapsed = time.time() - start

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_accuracy = accuracy_score(y_test_labels, y_pred)

    print(f"{architecture}: test accuracy={test_accuracy:.4f}, params={total_params:,}, time={elapsed:.2f}s")
    measured_results.append({
        "Model": architecture,
        "Parameters": total_params,
        "Accuracy (%)": round(100 * test_accuracy, 2),
        "Training Time (s)": round(elapsed, 2),
        "Source": "measured in this experiment (Keras, ImageNet weights)",
    })

# ------------------------------------------------------------------
# AlexNet, via torchvision, used as a frozen feature extractor
#
# PyTorch and Keras do not share a training graph, so AlexNet cannot be
# dropped into build_transfer_model the way the four architectures
# above are. Instead, its pretrained convolutional layers (features)
# plus average pool are used to extract a fixed 9216 dimensional
# feature vector from every CIFAR-10 image, exactly once, with no
# gradients, matching the "frozen base" setup Task 2 specifies. A small
# Keras head, the same shape as the other four use, is then trained on
# these fixed features.
#
# This is mathematically equivalent to including the frozen backbone
# directly in the training graph: since its weights never change
# either way, a head trained for EPOCHS epochs on precomputed features
# reaches the same result as one trained for EPOCHS epochs against the
# same frozen backbone recomputed every batch, just without paying to
# recompute it. Every other setting, epochs, batch size, optimizer,
# head architecture, and the training subsample itself, is kept
# identical to the four Keras architectures above for a fair
# comparison.
# ------------------------------------------------------------------
print("\ntraining architecture: alexnet")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

alexnet = tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1)
alexnet_extractor = nn.Sequential(alexnet.features, alexnet.avgpool, nn.Flatten()).to(device)
alexnet_extractor.eval()

# only the convolutional backbone actually used is counted, not
# AlexNet's original 1000 class ImageNet classifier, which is discarded
# entirely and replaced by the new Keras head below, matching how the
# four Keras architectures above only count their own backbone plus new
# head as well
alexnet_backbone_params = sum(p.numel() for p in alexnet.features.parameters())

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def extract_alexnet_features(images, batch_size=256):
    """
    images: numpy array (N, 32, 32, 3), pixel values 0 to 255.
    Returns: numpy array (N, 9216) of frozen AlexNet features, resizing
    each image to AlexNet's expected 224 x 224 and normalizing with the
    standard ImageNet per channel mean and standard deviation first.
    """
    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
            tensor = tensor.to(device)
            tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            features = alexnet_extractor(tensor)
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


start = time.time()
train_features = extract_alexnet_features(X_train)
test_features = extract_alexnet_features(X_test)
feature_extraction_time = time.time() - start
print(f"AlexNet feature extraction time: {feature_extraction_time:.2f}s "
      f"for {len(X_train) + len(X_test)} images")

alexnet_head = Sequential([
    Input(shape=(9216,)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax"),
])
alexnet_head.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
alexnet_head_params = alexnet_head.count_params()

start = time.time()
alexnet_head.fit(
    train_features, y_train_onehot,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)
alexnet_head_train_time = time.time() - start
alexnet_total_time = feature_extraction_time + alexnet_head_train_time

y_pred_alexnet = np.argmax(alexnet_head.predict(test_features, verbose=0), axis=1)
alexnet_accuracy = accuracy_score(y_test_labels, y_pred_alexnet)

print(f"alexnet: test accuracy={alexnet_accuracy:.4f}, "
      f"backbone params={alexnet_backbone_params:,}, head params={alexnet_head_params:,}, "
      f"total time={alexnet_total_time:.2f}s "
      f"({feature_extraction_time:.2f}s extraction + {alexnet_head_train_time:.2f}s head training)")

measured_results.append({
    "Model": "alexnet",
    "Parameters": alexnet_backbone_params + alexnet_head_params,
    "Accuracy (%)": round(100 * alexnet_accuracy, 2),
    "Training Time (s)": round(alexnet_total_time, 2),
    "Source": "measured in this experiment (PyTorch/torchvision, ImageNet weights)",
})

# ------------------------------------------------------------------
comparison_df = pd.DataFrame(measured_results)
comparison_df.to_csv("outputs/results/architecture_comparison.csv", index=False)
print("\narchitecture comparison:")
print(comparison_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.bar(comparison_df["Model"], comparison_df["Accuracy (%)"], color="steelblue", label="Test Accuracy")
ax.set_xlabel("Architecture")
ax.set_ylabel("Test Accuracy (%)")
ax.legend()
fig.tight_layout()
save_fig(fig, "outputs/plots/architecture_comparison")
plt.close(fig)

print("\nsaved architecture comparison plot and results table")
print("all five architectures use real ImageNet pretrained weights and are measured directly:")
print("MobileNetV2, VGG16, ResNet50, and InceptionV3 via Keras, AlexNet via PyTorch/torchvision,")
print("since Keras itself has never shipped a pretrained AlexNet. LeNet-5 is not included, since")
print("it has no pretrained weights and was never designed for 3 channel colour images.")
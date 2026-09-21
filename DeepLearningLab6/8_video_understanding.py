import os
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model_builder import build_cnn_feature_extractor, build_video_classifier
from plot_style import setup_fonts, save_fig

setup_fonts()

# Extracting the .rar needs the unrar tool, not installed on a fresh
# Colab runtime by default. Run this once in a Colab cell before this
# script:
#
#   !apt-get install -y unrar
#
# Set this to wherever your archive actually is once Google Drive is
# mounted (from google.colab import drive; drive.mount('/content/drive')),
# this is the one place the dataset path needs to be set.
RAR_PATH = "/content/drive/MyDrive/LatentWorldModel_/UCF101.rar"

# The manual's suggested class list includes "Running", which is not an
# actual UCF101 class name (there is no plain "Running" class in the
# dataset), it has been swapped for JumpRope, another visually distinct,
# dynamic action, keeping Basketball, Biking, and TennisSwing as given
# and using UCF101's actual name WalkingWithDog rather than "Walking".
SELECTED_CLASSES = ["Basketball", "Biking", "WalkingWithDog", "TennisSwing", "JumpRope"]
MAX_VIDEOS_PER_CLASS = 10  # keeps this experiment's purpose (see the CNN + recurrent pipeline work) fast

NUM_FRAMES = 10
IMAGE_SIZE = (224, 224)
UNITS = 32
RECURRENT_TYPE = "lstm"  # or "gru", Additional Exercise 6 asks for both
EPOCHS = 15
BATCH_SIZE = 8
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def _find_videos_root(extract_dir):
    for root, dirs, _ in os.walk(extract_dir):
        if any(d in SELECTED_CLASSES for d in dirs):
            return root
    return None


def _extract_ucf101(rar_path):
    """
    Extracts the full archive (UCF101 has no simple, reliable way to
    extract only specific class subfolders without first knowing the
    archive's exact internal path, which varies by release), then
    finds whichever folder inside it directly contains the selected
    class subfolders. This only happens once, cached for later runs.
    """
    extract_dir = os.path.join(os.path.dirname(rar_path), "ucf101_extracted")

    videos_root = _find_videos_root(extract_dir) if os.path.isdir(extract_dir) else None

    if videos_root is None:
        print(f"extracting {rar_path}, this is the full archive (all 101 classes), "
              f"only needs to happen once, and later runs reuse it")
        os.makedirs(extract_dir, exist_ok=True)
        result = subprocess.run(["unrar", "x", "-y", rar_path, extract_dir], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "unrar failed, is it installed? run '!apt-get install -y unrar' in a Colab cell first.\n"
                f"unrar output:\n{result.stdout}\n{result.stderr}"
            )
        videos_root = _find_videos_root(extract_dir)

    if videos_root is None:
        raise FileNotFoundError(
            f"could not find any of {SELECTED_CLASSES} as subfolders under {extract_dir} "
            f"after extraction, check the archive's actual class folder names"
        )
    return videos_root


def sample_frames(video_path, num_frames=NUM_FRAMES):
    """
    Samples num_frames frames uniformly across a video's full length,
    resized to 224 x 224 x 3, in raw 0 to 255 pixel values, matching
    what the frozen CNN's own preprocessing (applied inside
    build_cnn_feature_extractor) expects.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"could not read frame count from {video_path}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else np.zeros((*IMAGE_SIZE, 3), dtype="uint8")
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, IMAGE_SIZE)
        frames.append(frame)
    cap.release()
    return np.stack(frames).astype("float32")


videos_root = _extract_ucf101(RAR_PATH)

class_names = [c for c in SELECTED_CLASSES if os.path.isdir(os.path.join(videos_root, c))]
missing = [c for c in SELECTED_CLASSES if c not in class_names]
if missing:
    print(f"warning: {missing} not found under {videos_root}, continuing with {class_names}")

video_paths, labels = [], []
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(videos_root, class_name)
    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv")))
    for fname in files[:MAX_VIDEOS_PER_CLASS]:
        video_paths.append(os.path.join(class_dir, fname))
        labels.append(class_idx)

print(f"found {len(video_paths)} videos across {len(class_names)} classes: {class_names}")

print("\nsampling frames and saving Plot 7 (sample frames from one video)")
sample_video_frames = sample_frames(video_paths[0])
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, frame in zip(axes.ravel(), sample_video_frames):
    ax.imshow(frame.astype("uint8"))
    ax.axis("off")
fig.suptitle(f"10 Sampled Frames: {class_names[labels[0]]}", fontsize=12)
fig.tight_layout()
save_fig(fig, "outputs/plots/video_sample_frames")
plt.close(fig)

print("\nbuilding the frozen CNN feature extractor and extracting features for every video")
print("this runs once per video; the CNN itself is never trained, exactly as Section 20 specifies")
cnn_extractor, feature_dim = build_cnn_feature_extractor()
print(f"CNN feature dimension: D = {feature_dim}")
print(f"tensor shape supplied to the recurrent network per video: ({NUM_FRAMES}, {feature_dim})")

all_features = []
for i, path in enumerate(video_paths):
    frames = sample_frames(path)
    features = cnn_extractor.predict(frames, verbose=0)
    all_features.append(features)
    if (i + 1) % 5 == 0 or (i + 1) == len(video_paths):
        print(f"  extracted features for {i + 1} of {len(video_paths)} videos")

X = np.stack(all_features)
y = np.array(labels)
print(f"\nfull feature dataset shape: {X.shape}")

rng = np.random.default_rng(42)
perm = rng.permutation(len(X))
n_test = max(1, int(len(X) * TEST_FRACTION))
n_val = max(1, int(len(X) * VAL_FRACTION))
test_idx = perm[:n_test]
val_idx = perm[n_test:n_test + n_val]
train_idx = perm[n_test + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]
print(f"train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

model = build_video_classifier(
    recurrent_type=RECURRENT_TYPE, num_frames=NUM_FRAMES, feature_dim=feature_dim,
    num_classes=len(class_names), units=UNITS,
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
total_params = model.count_params()
print(f"\n{RECURRENT_TYPE} video classifier parameters: {total_params:,}")

start = time.time()
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
)
training_time = time.time() - start

model.save(f"outputs/saved_model/video_{RECURRENT_TYPE}_model.keras")

out_dir = "outputs/plots"
epochs_range = range(1, EPOCHS + 1)

# Plot 8: training and validation curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(epochs_range, history.history["accuracy"], color="green", label="Training Accuracy")
axes[0].plot(epochs_range, history.history["val_accuracy"], color="orange", label="Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[1].plot(epochs_range, history.history["loss"], color="red", label="Training Loss")
axes[1].plot(epochs_range, history.history["val_loss"], color="purple", label="Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/video_training_curves")
plt.close(fig)

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0)
print(f"\ntest accuracy={accuracy:.4f}, macro precision={precision:.4f}, "
      f"macro recall={recall:.4f}, macro F1={f1:.4f}, training time={training_time:.2f}s")

# example prediction with a real, model produced confidence value
example_probs = model.predict(X_test[:1], verbose=0)[0]
predicted_class = class_names[int(np.argmax(example_probs))]
actual_class = class_names[int(y_test[0])]
confidence = float(np.max(example_probs))
print(f"\nPredicted class : {predicted_class}")
print(f"Actual class    : {actual_class}")
print(f"Confidence      : {confidence:.2f}")

pd.DataFrame([{
    "model": f"CNN-{RECURRENT_TYPE.upper()}",
    "accuracy": accuracy, "macro_precision": precision, "macro_recall": recall,
    "macro_f1": f1, "parameters": total_params, "training_time_seconds": training_time,
}]).to_csv("outputs/results/video_model_metrics.csv", index=False)

# Plot 9: confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, cbar_kws={"label": "Count"}, ax=ax)
ax.set_xlabel("Predicted Action")
ax.set_ylabel("Actual Action")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
save_fig(fig, f"{out_dir}/video_confusion_matrix")
plt.close(fig)

print(f"\nsaved video_sample_frames, video_training_curves, and video_confusion_matrix plots to {out_dir}")

import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_har_data, CLASS_NAMES, SEQUENCE_LENGTH, NUM_CHANNELS
from plot_style import setup_fonts, save_fig

setup_fonts()

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_har_data()

print("Input tensor shape:")
print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Testing    : {X_test.shape}")
print()
print(f"Number of classes: {len(CLASS_NAMES)}")
print(f"Number of features per time step: {NUM_CHANNELS}")
print(f"Sequence length: {SEQUENCE_LENGTH}")

print("\nclass distribution (training):")
for class_idx, name in enumerate(CLASS_NAMES):
    count = int((y_train == class_idx).sum())
    print(f"  {name}: {count}")

# Plot 1: sensor signal versus time, for 3 representative activities
# and 3 representative channels each. WALKING (dynamic, periodic),
# SITTING (static, one posture), and LAYING (static, a different
# posture) are chosen specifically to contrast a moving activity
# against two different stationary ones.
CHANNELS_TO_SHOW = ["body_acc_x", "body_acc_y", "body_acc_z"]
CHANNEL_INDICES = [0, 1, 2]  # matches SIGNAL_NAMES order in data_loader.py
ACTIVITIES_TO_SHOW = ["WALKING", "SITTING", "LAYING"]

fig, axes = plt.subplots(len(ACTIVITIES_TO_SHOW), 1, figsize=(9, 9), sharex=True)
time_steps = np.arange(1, SEQUENCE_LENGTH + 1)

for ax, activity_name in zip(axes, ACTIVITIES_TO_SHOW):
    class_idx = CLASS_NAMES.index(activity_name)
    sample_idx = np.where(y_train == class_idx)[0][0]
    sample = X_train[sample_idx]

    for channel_idx, channel_name in zip(CHANNEL_INDICES, CHANNELS_TO_SHOW):
        ax.plot(time_steps, sample[:, channel_idx], label=channel_name)
    ax.set_ylabel("Sensor Value")
    ax.set_title(activity_name, fontsize=11)
    ax.legend(fontsize=8)

axes[-1].set_xlabel("Time Step")
fig.tight_layout()
save_fig(fig, "outputs/plots/temporal_signals")
plt.close(fig)

print("\nsaved temporal_signals plot to outputs/plots")

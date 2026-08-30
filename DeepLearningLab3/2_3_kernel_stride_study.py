import pandas as pd
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

INPUT_SHAPE = (32, 32, 3)
FILTERS = 16

# ------------------------------------------------------------------
# Task 2: kernel size comparison
# ------------------------------------------------------------------
# Stride 1 and valid padding are held fixed here, so the only thing
# changing between the three rows is kernel size, isolating its effect
# on feature map size the way Task 2 asks for. Stride and padding are
# then varied on their own in Task 3 below.
print("Task 2: kernel size comparison (stride 1, valid padding, 32x32x3 input)")

kernel_results = []
for kernel_size in [3, 5, 7]:
    inputs = Input(shape=INPUT_SHAPE)
    outputs = Conv2D(FILTERS, kernel_size=kernel_size, strides=1, padding="valid")(inputs)
    feature_map_size = outputs.shape[1]
    print(f"  kernel {kernel_size}x{kernel_size}: feature map {feature_map_size}x{feature_map_size}")
    kernel_results.append({
        "kernel_size": f"{kernel_size}x{kernel_size}",
        "feature_map_size": f"{feature_map_size}x{feature_map_size}",
    })

pd.DataFrame(kernel_results).to_csv("outputs/results/task2_kernel_size_study.csv", index=False)

# ------------------------------------------------------------------
# Task 3: stride and padding study
# ------------------------------------------------------------------
print("\nTask 3: stride and padding study (kernel 3x3, 32x32x3 input)")

stride_padding_configs = [
    (1, "valid"),
    (2, "valid"),
    (1, "same"),
    (2, "same"),
]

stride_padding_results = []
for stride, padding in stride_padding_configs:
    inputs = Input(shape=INPUT_SHAPE)
    outputs = Conv2D(FILTERS, kernel_size=3, strides=stride, padding=padding)(inputs)
    feature_map_size = outputs.shape[1]
    print(f"  stride {stride}, padding {padding}: feature map {feature_map_size}x{feature_map_size}")
    stride_padding_results.append({
        "stride": stride,
        "padding": padding,
        "feature_map_size": f"{feature_map_size}x{feature_map_size}",
    })

pd.DataFrame(stride_padding_results).to_csv("outputs/results/task3_stride_padding_study.csv", index=False)

print("\nsaved Task 2 and Task 3 result tables to outputs/results")

from model_builder import build_transfer_model

# MobileNetV2 is used as the primary architecture for this experiment,
# since it is by far the smallest and fastest of the four models listed
# in Task 2 (about 3.5 million parameters against VGG16's 138 million
# and ResNet50's 25.6 million), which matters here because this same
# architecture gets trained again in Task 3, fine tuned in Task 4, and
# retrained several more times over in the hyperparameter study.
# Section 6 of this experiment (architecture_comparison) trains VGG16
# and ResNet50 as well, to actually compare architectures rather than
# assuming which one is best. Change ARCHITECTURE below, and the same
# choice will be picked up by scripts 3, 4, and 5.
ARCHITECTURE = "mobilenetv2"

model, base_model = build_transfer_model(architecture=ARCHITECTURE, dense_units=128, freeze_base=True)

model.summary()

total_params = model.count_params()
trainable_params = sum(w.shape.num_elements() for w in model.trainable_weights)
frozen_params = total_params - trainable_params

print(f"\narchitecture: {ARCHITECTURE}")
print(f"total parameters:     {total_params:,}")
print(f"trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f} percent)")
print(f"frozen parameters:    {frozen_params:,} ({100 * frozen_params / total_params:.1f} percent)")
print("\nwith the base frozen, only the new Global Average Pooling head, the Dense ReLU")
print("layer, and the Softmax output layer are trainable; the pretrained convolutional")
print("base contributes its features but is not updated during Task 3 training")

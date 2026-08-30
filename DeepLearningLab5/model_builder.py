import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.initializers import Zeros, RandomNormal, GlorotUniform, HeNormal

IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = 37

# Section 5's four initialization strategies, applied to the new
# classifier head's Dense layers. The pretrained backbone's own weights
# always come from ImageNet regardless of this setting, since that is
# what makes this transfer learning rather than training from scratch;
# initialization strategy only has anything to compare on the layers
# that actually start untrained.
INITIALIZERS = {
    "zero": Zeros(),
    "random": RandomNormal(mean=0.0, stddev=0.05, seed=42),
    "xavier": GlorotUniform(seed=42),
    "he": HeNormal(seed=42),
}


def build_model(
    initializer="xavier",
    regularization=None,
    dropout_rate=0.5,
    l2_lambda=0.001,
    dense_units=128,
    freeze_base=True,
):
    """
    Builds a MobileNetV2 transfer learning model: a frozen (or later,
    partially unfrozen) ImageNet pretrained backbone, Global Average
    Pooling, and a new classifier head. Every study in this experiment
    that changes one thing about the new head, its weight
    initialization (Section 5), its regularization (Section 6), its
    width, goes through this one function, holding everything else
    fixed.

    regularization is one of None, "l2", "dropout", or "batchnorm",
    matching Section 6's four way comparison (no regularization is
    None). Only one is ever applied at a time, since the manual asks
    for a "no regularization" baseline and each technique in isolation,
    not combinations of them.

    Preprocessing (MobileNetV2's own required rescale to -1 to 1) is
    applied inside the model graph, so this expects raw pixel values in
    the 0 to 255 range as input, exactly what data_loader.py returns.

    Returns both the full model and a reference to the base model
    itself, since Section 10's fine tuning needs to unfreeze specific
    layers of the base after this function returns.
    """
    if regularization not in (None, "l2", "dropout", "batchnorm"):
        raise ValueError(f"unknown regularization {regularization!r}")

    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=IMAGE_SHAPE)
    base_model.trainable = not freeze_base

    kernel_init = INITIALIZERS[initializer]
    kernel_regularizer = l2(l2_lambda) if regularization == "l2" else None

    inputs = Input(shape=IMAGE_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=(False if freeze_base else None))
    x = GlobalAveragePooling2D()(x)

    x = Dense(dense_units, kernel_initializer=kernel_init, kernel_regularizer=kernel_regularizer)(x)
    if regularization == "batchnorm":
        # Batch Normalization is applied between the Dense layer and its
        # activation, the now standard placement, and normalizes exactly
        # the way Section 7's numerical example describes, per feature
        # batch mean and variance, then a learnable scale (gamma) and
        # shift (beta).
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if regularization == "dropout":
        x = Dropout(dropout_rate)(x)

    outputs = Dense(NUM_CLASSES, activation="softmax", kernel_initializer=kernel_init)(x)

    model = Model(inputs=inputs, outputs=outputs, name="pets_mobilenetv2")
    return model, base_model


def compile_model(model, optimizer_name="adam", learning_rate=0.001):
    """
    optimizer_name is one of "sgd", "momentum", "rmsprop", or "adam",
    matching Section 8's four way optimizer comparison. "momentum" is
    SGD with a momentum term added, the standard way to turn plain SGD
    into SGD with Momentum without a separate Keras class for it.

    sparse_categorical_crossentropy is used throughout this project
    rather than categorical_crossentropy, since data_loader.py keeps
    labels as plain integers (tfds' native format for this dataset)
    rather than one hot vectors.
    """
    if optimizer_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == "momentum":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"unknown optimizer {optimizer_name!r}")

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def get_base_model(full_model):
    """
    Retrieves the nested MobileNetV2 base model from a model built by
    build_model, by finding the one layer that is itself a Model
    instance, rather than by its auto generated name (something like
    "mobilenetv2_1.00_224"), which is Keras version dependent and is not
    known in a script that only has the saved .keras file, not the
    original base_model object build_model returned when the model was
    first constructed.
    """
    candidates = [layer for layer in full_model.layers if isinstance(layer, Model)]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one nested Model layer (the pretrained base), found {len(candidates)}"
        )
    return candidates[0]


def unfreeze_last_block(base_model):
    """
    Unfreezes only MobileNetV2's last inverted residual block, block_16
    onward, following Section 10 Case B's instruction to unfreeze
    selected upper layers rather than the whole backbone. Every layer
    from the first one named like "block_16" onward is set trainable,
    and everything before it, including the initial stem convolution
    and blocks 1 through 15, stays frozen.
    """
    base_model.trainable = True
    found = False
    for layer in base_model.layers:
        if "block_16" in layer.name:
            found = True
        layer.trainable = found

    unfrozen = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"unfroze {unfrozen} of {len(base_model.layers)} layers in the base model, "
          f"from the first layer named like 'block_16' onward")

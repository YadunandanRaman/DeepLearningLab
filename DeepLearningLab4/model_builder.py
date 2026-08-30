from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Resizing, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess

CIFAR_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

# CIFAR-10 images are only 32 x 32, well below the resolution these
# architectures were designed for. Resizing up to 96 x 96 inside the
# model itself, rather than precomputing and storing a resized copy of
# the whole dataset, keeps every array passed to fit() at native
# CIFAR-10 resolution (small and fast to load), while still giving the
# pretrained backbone a reasonable amount of spatial detail to work
# with. 224 x 224, the resolution these models were originally trained
# at, would look sharper still, but at a training time cost this
# experiment does not need to pay to make its point.
RESIZE_SHAPE = (96, 96)

ARCHITECTURES = {
    "mobilenetv2": {
        "builder": MobileNetV2,
        "preprocess": mobilenetv2_preprocess,
        # the name every layer in this architecture's final block shares
        "last_block_name": "block_16",
    },
    "vgg16": {
        "builder": VGG16,
        "preprocess": vgg16_preprocess,
        "last_block_name": "block5",
    },
    "resnet50": {
        "builder": ResNet50,
        "preprocess": resnet50_preprocess,
        "last_block_name": "conv5",
    },
    # tensorflow.keras.applications has no true 2014 GoogleNet/Inception v1.
    # InceptionV3 is the closest architecture Keras actually ships with
    # ImageNet weights, and is the standard replacement for "GoogleNet" in
    # this kind of comparison, it keeps the same core idea, Inception
    # modules combining multiple filter sizes in parallel, described in
    # Section 9, at a later, more refined revision.
    "inceptionv3": {
        "builder": InceptionV3,
        "preprocess": inceptionv3_preprocess,
        # Unlike the other three architectures, InceptionV3's individual
        # conv/batch norm/activation layers inside a block do not share a
        # common name prefix, only the block's final concatenate layer
        # does ("mixed10"), so a substring match only catches that one
        # layer, not the roughly 30 layers that feed into it. "mixed9" is
        # the previous block's concatenate layer, so unfreezing strictly
        # after it, rather than matching a substring, correctly captures
        # the whole final block instead.
        "last_block_name": "mixed9",
        "match_mode": "after_exact",
    },
}


def build_transfer_model(architecture="mobilenetv2", dense_units=128, freeze_base=True):
    """
    Builds a transfer learning model on top of a pretrained ImageNet
    backbone. Resizing and this architecture's own preprocessing both
    happen inside the model graph, ahead of the frozen (or partially
    frozen) base, a Global Average Pooling layer, a Dense ReLU layer,
    and a Softmax output layer, following Task 2 of the lab manual.

    Because preprocessing lives inside the model, this expects raw,
    unnormalized pixel values in the 0 to 255 range as input, exactly
    what data_loader.load_data() returns. Every pretrained backbone
    here has its own required preprocessing (MobileNetV2 scales to
    negative 1 to 1, VGG16 and ResNet50 subtract per channel ImageNet
    means), so a single generic 0 to 1 rescale applied before this
    model would not match what any of them expect and would hurt
    accuracy rather than help it.

    Returns both the full model and a reference to the base model
    itself, since Task 4 (fine tuning) needs to unfreeze specific
    layers of the base after this function returns.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}, choose from {list(ARCHITECTURES)}")

    cfg = ARCHITECTURES[architecture]
    base_model = cfg["builder"](include_top=False, weights="imagenet", input_shape=(*RESIZE_SHAPE, 3))
    base_model.trainable = not freeze_base

    inputs = Input(shape=CIFAR_SHAPE)
    x = Resizing(*RESIZE_SHAPE)(inputs)
    x = cfg["preprocess"](x)
    x = base_model(x, training=(False if freeze_base else None))
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation="relu")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"{architecture}_transfer")
    return model, base_model


def compile_model(model, optimizer_name="adam", learning_rate=0.001):
    if optimizer_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def get_base_model(full_model):
    """
    Retrieves the nested pretrained base model from a model built by
    build_transfer_model, by finding the one layer that is itself a
    Model instance, rather than by its auto generated name (something
    like "mobilenetv2_1.00_96"), which is architecture and Keras
    version dependent and is not known in a script that only has the
    saved .keras file, not the original base_model object build_transfer_model
    returned when the model was first constructed.
    """
    candidates = [layer for layer in full_model.layers if isinstance(layer, Model)]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one nested Model layer (the pretrained base), found {len(candidates)}"
        )
    return candidates[0]


def unfreeze_last_block(base_model, architecture="mobilenetv2"):
    """
    Unfreezes only the base model's last convolutional block, following
    Task 4's instruction to fine tune the last block rather than the
    whole backbone. Which layers make up that last block, and how to
    find them, is architecture specific, tracked in ARCHITECTURES above.

    Most architectures here name every layer in a block with a shared
    prefix (MobileNetV2's "block_16", VGG16's "block5", ResNet50's
    "conv5"), so every layer from the first one whose name contains
    that marker onward is set trainable ("prefix" mode, the default).

    InceptionV3 does not share a prefix across a block's internal
    layers, only its final concatenate layer is distinctly named, so
    for it "last_block_name" is instead the previous block's boundary
    layer, and everything strictly after that layer's index is set
    trainable ("after_exact" mode).
    """
    cfg = ARCHITECTURES[architecture]
    marker = cfg["last_block_name"]
    match_mode = cfg.get("match_mode", "prefix")

    base_model.trainable = True

    if match_mode == "after_exact":
        names = [layer.name for layer in base_model.layers]
        boundary_idx = names.index(marker)
        for i, layer in enumerate(base_model.layers):
            layer.trainable = i > boundary_idx
    else:
        found = False
        for layer in base_model.layers:
            if marker in layer.name:
                found = True
            layer.trainable = found

    unfrozen = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"unfroze {unfrozen} of {len(base_model.layers)} layers in the base model "
          f"({match_mode} match on {marker!r})")
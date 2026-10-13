"""
ResNet50 with attention inserted after chosen stages (Sections 4, 7, 8 and 13).

The backbone is a re-implementation of keras.applications.ResNet50 with the SAME layer names, so the
ImageNet weights are copied across by layer name. A flat functional graph makes it simple to insert an
AttentionStack after conv2_x / conv3_x / conv4_x / conv5_x (P1..P4), to freeze stages, and to read any
intermediate tensor for the feature map, attention map and Grad-CAM analysis.

Head (identical for every model): GlobalAveragePooling -> Dropout -> Dense(5, logits) -> softmax.
"""
from tensorflow import keras
from tensorflow.keras import initializers, layers

import attention
import config

STAGE_BLOCKS = {2: 3, 3: 4, 4: 6, 5: 3}
STAGE_FILTERS = {2: 64, 3: 128, 4: 256, 5: 512}
STAGE_STRIDE = {2: 1, 3: 2, 4: 2, 5: 2}
STAGE_OUT = {2: "conv2_block3_out", 3: "conv3_block4_out", 4: "conv4_block6_out", 5: "conv5_block3_out"}
BN_EPS = 1.001e-5


def _block(x, filters, stride, conv_shortcut, name):
    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(x)
        shortcut = layers.BatchNormalization(epsilon=BN_EPS, name=name + "_0_bn")(shortcut)
    else:
        shortcut = x
    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name=name + "_1_bn")(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=name + "_2_conv")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name=name + "_2_bn")(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)
    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name=name + "_3_bn")(x)
    x = layers.Add(name=name + "_add")([shortcut, x])
    return layers.Activation("relu", name=name + "_out")(x)


def build_model(spec, pretrained=True, dropout=None):
    """spec is a resolved entry of experiments.EXPERIMENTS. Returns a functional keras.Model."""
    config.setup_runtime()                               # precision policy must be set before the layers are created
    dropout = config.DROPOUT if dropout is None else dropout
    positions = tuple(spec["positions"])
    inp = keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3), name="image")
    x = layers.ZeroPadding2D(3, name="conv1_pad")(inp)
    x = layers.Conv2D(64, 7, strides=2, name="conv1_conv")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.ZeroPadding2D(1, name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    for stage in (2, 3, 4, 5):
        f = STAGE_FILTERS[stage]
        for b in range(1, STAGE_BLOCKS[stage] + 1):
            x = _block(x, f, STAGE_STRIDE[stage] if b == 1 else 1, conv_shortcut=(b == 1),
                       name=f"conv{stage}_block{b}")
        pos = stage - 1                                    # conv2_x -> P1, ..., conv5_x -> P4
        if pos in positions:
            x = attention.AttentionStack(spec["components"], config.REDUCTION, spec["kernel"],
                                         config.GATE_BIAS_INIT, name=f"att_P{pos}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, seed=config.SEED, name="dropout")(x)
    logits = layers.Dense(config.NUM_CLASSES, kernel_initializer=initializers.GlorotUniform(seed=config.SEED),
                          dtype="float32", name="logits")(x)              # logits and softmax stay in float32
    probs = layers.Activation("softmax", dtype="float32", name="probs")(logits)
    model = keras.Model(inp, probs, name="resnet50_" + ("baseline" if not positions else
                                                      "att_" + "".join(f"P{p}" for p in positions)))
    if pretrained:
        load_imagenet_weights(model)
    return model


def load_imagenet_weights(model, weights="imagenet"):
    """Copy the pretrained ResNet50 weights into the matching layers (same names)."""
    ref = keras.applications.ResNet50(weights=weights, include_top=False, input_shape=(config.IMG_SIZE,) * 2 + (3,))
    n = 0
    for layer in ref.layers:
        if layer.weights:
            model.get_layer(layer.name).set_weights(layer.get_weights())
            n += 1
    return n


# --------------------------------------------------------------------------------------------------
# Freezing (Section 20)
# --------------------------------------------------------------------------------------------------

def _is_backbone(layer):
    return layer.name.startswith("conv") and not layer.name.startswith("att_")


def set_phase(model, phase):
    """Phase 1: backbone frozen, attention modules and classifier train.
    Phase 2: the stages in config.UNFREEZE_STAGES are also unfrozen; BatchNorm layers always stay frozen."""
    for layer in model.layers:
        if _is_backbone(layer):
            stage = int(layer.name[4]) if layer.name[4].isdigit() else 0
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = (phase == 2 and stage in config.UNFREEZE_STAGES)
        elif layer.name.startswith("att_") or layer.name == "logits":
            layer.trainable = True
    return model


def count_params(model):
    def n(ws):
        return int(sum(int(w.shape.num_elements()) for w in ws))
    total = n(model.weights)
    trainable = n(model.trainable_weights)
    att = n([w for l in model.layers if l.name.startswith("att_") for w in l.weights])
    return {"total": total, "trainable": trainable, "attention": att}


# --------------------------------------------------------------------------------------------------
# Handles for the analysis scripts
# --------------------------------------------------------------------------------------------------

def attention_layers(model):
    return {int(l.name.split("P")[-1]): l for l in model.layers if l.name.startswith("att_P")}


def attention_io_model(model, pos):
    """Model returning (tensor entering the attention stack, tensor leaving it) for position P<pos>."""
    layer = model.get_layer(f"att_P{pos}")
    return keras.Model(model.input, [layer.input, layer.output])


def stage_shapes(model):
    """Output shapes of the four stages (and of the attention stacks) for the dimension table."""
    rows = []
    for stage in (2, 3, 4, 5):
        rows.append((f"conv{stage}_x", tuple(model.get_layer(STAGE_OUT[stage]).output.shape[1:])))
    return rows

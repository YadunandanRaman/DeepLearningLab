from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


def build_cnn(pooling_type="max", num_filters=None):
    """
    Task 6's architecture exactly:

    Input -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax

    32 and 64 filters, a 3x3 kernel with same padding, and a 64 unit
    Dense layer are used, since the manual specifies the layer sequence
    but not these particular sizes; they are a standard, small starting
    point for a two block CNN on 32 x 32 images.

    pooling_type switches between MaxPooling2D and AveragePooling2D,
    used by Task 5 to compare the two strategies with everything else
    held fixed.

    num_filters, if given, overrides both convolutional layers to use
    that same filter count instead of the baseline's 32 then 64, used
    by the Additional Exercise 5 filter count comparison to isolate
    filter count as a single, clean variable rather than changing two
    layers by different, confounded amounts.

    Built with the Functional API rather than Sequential, since Task 4
    needs to read the output of the named conv1 layer after the model
    is reloaded from disk, and a Functional model's `.input` attribute
    survives a save and reload reliably, where a Sequential model's
    does not in every Keras version.
    """
    pooling_layer = MaxPooling2D if pooling_type == "max" else AveragePooling2D
    filters1 = num_filters if num_filters is not None else 32
    filters2 = num_filters if num_filters is not None else 64

    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(filters1, kernel_size=3, padding="same", activation="relu", name="conv1")(inputs)
    x = pooling_layer(pool_size=2, name="pool1")(x)
    x = Conv2D(filters2, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
    x = pooling_layer(pool_size=2, name="pool2")(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs, name="cifar_cnn")

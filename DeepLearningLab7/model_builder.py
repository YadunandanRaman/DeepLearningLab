"""
All architectures used in the experiment, built with the Keras Functional
API so every model saves and reloads cleanly and its named layers can be
inspected later.
"""
from tensorflow import keras
from tensorflow.keras import layers

import config


def compile_autoencoder(model, learning_rate=config.LEARNING_RATE):
    """Adam + binary cross entropy, the Section 7 configuration shared by every deterministic model."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
    )
    return model


def build_fc_autoencoder(latent_dim=config.FC_LATENT_DIM, latent_activation=config.LATENT_ACTIVATION):
    """
    Section 7 fully connected autoencoder:

    Input 784 -> Dense 128 -> Dense 32 -> Latent dz -> Dense 32 -> Dense 128 -> Output 784

    ReLU in the hidden layers, sigmoid at the output so reconstructions
    stay in [0, 1] like the normalized input pixels.
    """
    inputs = keras.Input(shape=(config.FLAT_DIM,), name="input")
    x = layers.Dense(128, activation="relu", name="enc_dense_128")(inputs)
    x = layers.Dense(32, activation="relu", name="enc_dense_32")(x)
    z = layers.Dense(latent_dim, activation=latent_activation, name="latent")(x)
    x = layers.Dense(32, activation="relu", name="dec_dense_32")(z)
    x = layers.Dense(128, activation="relu", name="dec_dense_128")(x)
    outputs = layers.Dense(config.FLAT_DIM, activation="sigmoid", name="output")(x)
    return keras.Model(inputs, outputs, name=f"fc_autoencoder_z{latent_dim}")


def _conv_encoder(inputs):
    x = layers.Conv2D(32, 3, activation="relu", padding="same", name="enc_conv_32")(inputs)
    x = layers.MaxPooling2D(2, padding="same", name="enc_pool_1")(x)          # 14 x 14 x 32
    x = layers.Conv2D(64, 3, activation="relu", padding="same", name="enc_conv_64")(x)
    x = layers.MaxPooling2D(2, padding="same", name="enc_pool_2")(x)          # 7 x 7 x 64
    return x


def _conv_decoder(x, decoder_type):
    if decoder_type == "upsampling":
        x = layers.UpSampling2D(2, name="dec_upsample_1")(x)                  # 14 x 14
        x = layers.Conv2D(32, 3, activation="relu", padding="same", name="dec_conv_32")(x)
        x = layers.UpSampling2D(2, name="dec_upsample_2")(x)                  # 28 x 28
        return layers.Conv2D(1, 3, activation="sigmoid", padding="same", name="output")(x)
    if decoder_type == "transpose":
        # the same two doubling stages, learned by transposed convolutions
        x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same",
                                   name="dec_convT_32")(x)                     # 14 x 14
        return layers.Conv2DTranspose(1, 3, strides=2, activation="sigmoid", padding="same",
                                      name="output")(x)                       # 28 x 28
    raise ValueError(f"unknown decoder type: {decoder_type}")


def build_conv_autoencoder(decoder_type="upsampling"):
    """
    Section 10 convolutional autoencoder, exactly the listed layers:

    Input(28,28,1)
    Conv2D(32, 3, relu, same) -> MaxPooling2D(2)
    Conv2D(64, 3, relu, same) -> MaxPooling2D(2)
    Conv2D(64, 3, relu, same)          <- latent feature map, 7 x 7 x 64
    UpSampling2D(2) -> Conv2D(32, 3, relu, same)
    UpSampling2D(2) -> Conv2D(1, 3, sigmoid, same)

    decoder_type="transpose" swaps the UpSampling2D + Conv2D stages for
    Conv2DTranspose layers (additional exercise 4).
    """
    inputs = keras.Input(shape=config.IMG_SHAPE, name="input")
    x = _conv_encoder(inputs)
    latent = layers.Conv2D(64, 3, activation="relu", padding="same", name="latent")(x)
    outputs = _conv_decoder(latent, decoder_type)
    return keras.Model(inputs, outputs, name=f"conv_autoencoder_{decoder_type}")


def build_conv_autoencoder_bottleneck(latent_dim, latent_activation=config.LATENT_ACTIVATION):
    """
    Convolutional autoencoder with a vector bottleneck of size latent_dim,
    so the latent size can be varied directly (additional exercise 1).
    The convolutional encoder and decoder match build_conv_autoencoder().
    """
    inputs = keras.Input(shape=config.IMG_SHAPE, name="input")
    x = _conv_encoder(inputs)
    x = layers.Conv2D(64, 3, activation="relu", padding="same", name="enc_conv_64b")(x)
    x = layers.Flatten(name="flatten")(x)
    z = layers.Dense(latent_dim, activation=latent_activation, name="latent")(x)
    x = layers.Dense(7 * 7 * 64, activation="relu", name="dec_dense")(z)
    x = layers.Reshape((7, 7, 64), name="dec_reshape")(x)
    outputs = _conv_decoder(x, "upsampling")
    return keras.Model(inputs, outputs, name=f"conv_autoencoder_z{latent_dim}")


def build_vae_encoder(latent_dim=config.VAE_LATENT_DIM):
    """
    Section 17 VAE encoder: convolution layers -> flatten -> (mu, log sigma^2).

    Sampling is kept out of the encoder (it happens inside the VAE model),
    so the encoder is a plain functional model that saves and reloads
    without any custom layers.
    """
    inputs = keras.Input(shape=config.IMG_SHAPE, name="input")
    x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same", name="enc_conv_32")(inputs)  # 14 x 14
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same", name="enc_conv_64")(x)       # 7 x 7
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(16, activation="relu", name="enc_dense_16")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    return keras.Model(inputs, [z_mean, z_log_var], name=f"vae_encoder_z{latent_dim}")


def build_vae_decoder(latent_dim=config.VAE_LATENT_DIM):
    """Section 17 VAE decoder: z -> dense -> transposed convolutions -> 28 x 28 x 1."""
    z = keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(7 * 7 * 64, activation="relu", name="dec_dense")(z)
    x = layers.Reshape((7, 7, 64), name="dec_reshape")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same", name="dec_convT_64")(x)  # 14 x 14
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same", name="dec_convT_32")(x)  # 28 x 28
    outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name="output")(x)
    return keras.Model(z, outputs, name=f"vae_decoder_z{latent_dim}")

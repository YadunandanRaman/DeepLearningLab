"""
Variational autoencoder (Sections 15 to 17 of the lab manual).

  reparameterization:  z = mu + sigma * eps,  eps ~ N(0, I),  sigma = exp(0.5 * log sigma^2)
  reconstruction loss: binary cross entropy, summed over the 784 pixels, averaged over the batch
  KL divergence:       -0.5 * sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2), averaged over the batch
  total loss:          reconstruction + beta * KL   (beta = 1 is the standard VAE)
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from metrics import reconstruction_metrics
from model_builder import build_vae_decoder, build_vae_encoder
from train_utils import fit_timed


def sample_latent(z_mean, z_log_var):
    """The reparameterization trick: randomness moves into eps, so gradients reach mu and log sigma^2."""
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def reconstruction_loss(x, x_hat):
    bce = keras.losses.binary_crossentropy(x, x_hat)          # (batch, 28, 28)
    return tf.reduce_mean(tf.reduce_sum(bce, axis=(1, 2)))


def kl_loss(z_mean, z_log_var):
    kl = -0.5 * (1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return tf.reduce_mean(tf.reduce_sum(kl, axis=1))


def kl_per_dimension(z_mean, z_log_var):
    """Average KL contribution of every latent dimension over a dataset (numpy)."""
    kl = -0.5 * (1.0 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
    return kl.mean(axis=0)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = float(beta)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.rec_loss_tracker = keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.rec_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs, training=False):
        z_mean, z_log_var = self.encoder(inputs, training=training)
        z = sample_latent(z_mean, z_log_var)
        return self.decoder(z, training=training)

    @staticmethod
    def _inputs(data):
        # fit() and evaluate() are given (x, x); the target is the input itself
        if isinstance(data, (tuple, list)):
            return data[0]
        return data

    def _compute_losses(self, x, training):
        z_mean, z_log_var = self.encoder(x, training=training)
        z = sample_latent(z_mean, z_log_var)
        x_hat = self.decoder(z, training=training)
        rec = reconstruction_loss(x, x_hat)
        kl = kl_loss(z_mean, z_log_var)
        return rec, kl, rec + self.beta * kl

    def _update(self, rec, kl, total):
        self.total_loss_tracker.update_state(total)
        self.rec_loss_tracker.update_state(rec)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        x = self._inputs(data)
        with tf.GradientTape() as tape:
            rec, kl, total = self._compute_losses(x, training=True)
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self._update(rec, kl, total)

    def test_step(self, data):
        x = self._inputs(data)
        rec, kl, total = self._compute_losses(x, training=False)
        return self._update(rec, kl, total)


def build_vae(latent_dim=config.VAE_LATENT_DIM, beta=1.0):
    encoder = build_vae_encoder(latent_dim)
    decoder = build_vae_decoder(latent_dim)
    model = VAE(encoder, decoder, beta=beta, name=f"vae_z{latent_dim}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))
    return model


def train_vae(x_train, x_val, latent_dim=config.VAE_LATENT_DIM, beta=1.0, epochs=config.VAE_EPOCHS):
    config.set_seed()
    model = build_vae(latent_dim, beta)
    history, train_time = fit_timed(model, x_train, x_train, x_val, x_val, epochs, config.BATCH_SIZE)
    return model, history, train_time


def encode_mean(encoder, x):
    z_mean, z_log_var = encoder.predict(x, batch_size=config.BATCH_SIZE, verbose=0)
    return z_mean, z_log_var


def evaluate_vae(model, x_test):
    """
    Loss terms are the VAE objective on the test set (with sampling).
    MSE, MAE and SSIM decode the mean mu of q(z|x), which is the
    deterministic reconstruction and is directly comparable with the
    other autoencoders.
    """
    losses = model.evaluate(x_test, x_test, batch_size=config.BATCH_SIZE, verbose=0, return_dict=True)
    z_mean, z_log_var = encode_mean(model.encoder, x_test)
    x_hat = model.decoder.predict(z_mean, batch_size=config.BATCH_SIZE, verbose=0)
    result = {
        "rec_loss": float(losses["rec_loss"]),
        "kl_loss": float(losses["kl_loss"]),
        "total_loss": float(losses["total_loss"]),
    }
    result.update(reconstruction_metrics(x_test, x_hat))
    result["params"] = int(model.encoder.count_params() + model.decoder.count_params())
    return result, z_mean, z_log_var, x_hat


def save_vae(model, tag):
    model.encoder.save(config.model_path(f"{tag}_encoder.keras"))
    model.decoder.save(config.model_path(f"{tag}_decoder.keras"))


def load_vae_parts(tag):
    enc_path = config.model_path(f"{tag}_encoder.keras")
    dec_path = config.model_path(f"{tag}_decoder.keras")
    for p in (enc_path, dec_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run 6_train_vae.py first.")
    encoder = keras.models.load_model(enc_path, compile=False)
    decoder = keras.models.load_model(dec_path, compile=False)
    return encoder, decoder


def sample_prior(decoder, n, latent_dim, seed=config.SEED):
    """Draws z ~ N(0, I) and decodes it (Section 19)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, latent_dim)).astype("float32")
    return decoder.predict(z, batch_size=config.BATCH_SIZE, verbose=0), z

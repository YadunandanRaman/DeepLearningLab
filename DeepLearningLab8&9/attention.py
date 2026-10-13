"""
Attention modules (Sections 9 to 12 and 19 of the lab manual), written as Keras layers.

Every module keeps the feature tensor shape unchanged (H x W x C in, H x W x C out),
as required by Section 13. Every module also has an explain() method that runs the
same computation eagerly and returns the internal attention quantities (channel
weights, spatial maps, self attention responses) used for the qualitative analysis
in Sections 26 to 30. These are the INTERNAL attention weights, computed in the
forward pass, which is different from the post hoc Grad-CAM explanation.

Modules
  ChannelAttention     Section 9: GAP, two layer bottleneck MLP (ratio r), sigmoid. This is the
                       Squeeze and Excitation (SE) block, so "Channel / SE" in Section 19.1
                       is this module.
  SpatialAttention     Section 10: channel mean and max, concatenation, k x k convolution, sigmoid
  SelfAttention        Section 12: Q, K, V projections, softmax(QK^T / sqrt(d)), residual with a
                       learnable scale gamma
  ECAAttention         Efficient Channel Attention: GAP, 1D convolution across channels, sigmoid
  CBAMAttention        Convolutional Block Attention Module: channel attention (average and max
                       pooling through a shared MLP) followed by spatial attention
  CoordinateAttention  Coordinate Attention: direction aware pooling along height and width
  AttentionStack       applies an ordered list of the modules above (for example channel then
                       spatial, or spatial then channel)

Gate initialisation: the final bias of every attention gate is initialised to a positive
constant (config.GATE_BIAS_INIT) and the kernel that produces the gate starts small
(config.GATE_KERNEL_STD), so every gate starts close to sigmoid(bias) = 0.95 and the pretrained
features are barely disturbed at the start of training. This is the same for every
model, so it does not affect the fairness of the comparisons.
"""
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, layers

import config


def _small():
    """Small random kernel for the layer that produces a gate (keeps every gate near sigmoid(bias) at the start)."""
    return initializers.RandomNormal(stddev=config.GATE_KERNEL_STD)


class ChannelAttention(layers.Layer):
    kind = "channel"

    def __init__(self, reduction=config.REDUCTION, gate_bias=config.GATE_BIAS_INIT, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.gate_bias = gate_bias

    def build(self, input_shape):
        c = int(input_shape[-1])
        hidden = max(c // self.reduction, 1)
        self.fc1 = layers.Dense(hidden, activation="relu", name="fc1")
        self.fc2 = layers.Dense(c, kernel_initializer=_small(),
                                bias_initializer=initializers.Constant(self.gate_bias), name="fc2")
        self.fc1.build((None, c))
        self.fc2.build((None, hidden))
        super().build(input_shape)

    def channel_weights(self, x):
        z = tf.reduce_mean(x, axis=[1, 2])                      # global average pooling, (B, C)
        return tf.sigmoid(self.fc2(self.fc1(z)))                # M_c(F), (B, C)

    def call(self, x, training=None):
        return x * self.channel_weights(x)[:, None, None, :]

    def explain(self, x):
        w = self.channel_weights(x)
        return x * w[:, None, None, :], {"channel_weights": w}

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction": self.reduction, "gate_bias": self.gate_bias})
        return cfg


class SpatialAttention(layers.Layer):
    kind = "spatial"

    def __init__(self, kernel_size=config.SPATIAL_KERNEL, gate_bias=config.GATE_BIAS_INIT, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gate_bias = gate_bias

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding="same", kernel_initializer=_small(),
                                  bias_initializer=initializers.Constant(self.gate_bias), name="conv")
        self.conv.build((None, input_shape[1], input_shape[2], 2))
        super().build(input_shape)

    def spatial_map(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)         # F_avg, (B, H, W, 1)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)           # F_max, (B, H, W, 1)
        return tf.sigmoid(self.conv(tf.concat([avg, mx], axis=-1)))   # M_s(F), (B, H, W, 1)

    def call(self, x, training=None):
        return x * self.spatial_map(x)

    def explain(self, x):
        m = self.spatial_map(x)
        return x * m, {"spatial_map": m[..., 0]}

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size, "gate_bias": self.gate_bias})
        return cfg


class SelfAttention(layers.Layer):
    """
    Lightweight self attention over the H*W spatial positions. Queries and keys use C/8
    channels, values use C/2 channels followed by a projection back to C. The output is
    F + gamma * reshape(Y) with gamma a learnable scalar initialised to 0 (Section 12).
    """
    kind = "self"

    def __init__(self, qk_div=config.SELF_ATT_QK_DIV, v_div=config.SELF_ATT_V_DIV, **kwargs):
        super().__init__(**kwargs)
        self.qk_div = qk_div
        self.v_div = v_div

    def build(self, input_shape):
        self.h, self.w, c = int(input_shape[1]), int(input_shape[2]), int(input_shape[3])
        self.n = self.h * self.w
        self.d = max(c // self.qk_div, 8)
        dv = max(c // self.v_div, 8)
        self.wq = layers.Dense(self.d, use_bias=False, name="query")
        self.wk = layers.Dense(self.d, use_bias=False, name="key")
        self.wv = layers.Dense(dv, use_bias=False, name="value")
        self.wo = layers.Dense(c, use_bias=False, name="out")
        for layer, width in ((self.wq, c), (self.wk, c), (self.wv, c), (self.wo, dv)):
            layer.build((None, self.n, width))
        self.gamma = self.add_weight(name="gamma", shape=(), initializer="zeros", trainable=True)
        # pairwise distances between grid positions, used by explain()
        yy, xx = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing="ij")
        pos = np.stack([yy.ravel(), xx.ravel()], axis=1).astype("float32")
        dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        self.dist = tf.constant(dist)
        super().build(input_shape)

    def _attend(self, x):
        b = tf.shape(x)[0]
        c = x.shape[-1]
        tokens = tf.reshape(x, (b, self.n, c))                                  # X, (B, N, C)
        # the N x N part is computed in float32 (softmax and the products are not safe in float16)
        q = tf.cast(self.wq(tokens), tf.float32)
        k = tf.cast(self.wk(tokens), tf.float32)
        v = tf.cast(self.wv(tokens), tf.float32)
        logits = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.d)          # QK^T / sqrt(d), (B, N, N)
        a = tf.nn.softmax(logits, axis=-1)                                      # attention matrix A
        y = self.wo(tf.matmul(a, v))                                            # Y = A V, projected to C
        return a, tf.reshape(y, (b, self.h, self.w, c))

    def call(self, x, training=None):
        _, y = self._attend(x)
        return x + tf.cast(self.gamma, y.dtype) * y

    def explain(self, x):
        a, y = self._attend(x)
        b = tf.shape(x)[0]
        center = (self.h // 2) * self.w + self.w // 2
        shape = (b, self.h, self.w)
        info = {
            "query_center": tf.reshape(a[:, center, :], shape),                         # attention from the centre query
            "mean_over_queries": tf.reshape(tf.reduce_mean(a, axis=1), shape),          # mean attention received
            "max_over_queries": tf.reshape(tf.reduce_max(a, axis=1), shape),            # maximum attention received
            "response": tf.norm(tf.cast(self.gamma, tf.float32) * tf.cast(y, tf.float32), axis=-1),   # aggregated response
            "mean_distance": tf.reduce_mean(tf.reduce_sum(a * self.dist[None], axis=-1), axis=-1),   # (B,)
            "gamma": tf.fill((b,), self.gamma),
        }
        return x + tf.cast(self.gamma, y.dtype) * y, info

    def query_maps(self, x, positions):
        """Attention maps of selected query positions [(row, col), ...], shape (B, len(positions), H, W)."""
        a, _ = self._attend(x)
        idx = [r * self.w + c for r, c in positions]
        maps = tf.gather(a, idx, axis=1)
        return tf.reshape(maps, (tf.shape(x)[0], len(idx), self.h, self.w))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"qk_div": self.qk_div, "v_div": self.v_div})
        return cfg


class ECAAttention(layers.Layer):
    kind = "eca"

    def __init__(self, gamma=2, b=1, gate_bias=config.GATE_BIAS_INIT, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b
        self.gate_bias = gate_bias

    def build(self, input_shape):
        c = int(input_shape[-1])
        t = int(abs((math.log2(c) + self.b) / self.gamma))
        self.k = t if t % 2 == 1 else t + 1                     # adaptive odd kernel size
        self.conv = layers.Conv1D(1, self.k, padding="same", kernel_initializer=_small(),
                                  bias_initializer=initializers.Constant(self.gate_bias), name="conv1d")
        self.conv.build((None, c, 1))
        super().build(input_shape)

    def channel_weights(self, x):
        z = tf.reduce_mean(x, axis=[1, 2])                      # (B, C)
        return tf.sigmoid(self.conv(z[..., None])[..., 0])      # local cross channel interaction, no dimension reduction

    def call(self, x, training=None):
        return x * self.channel_weights(x)[:, None, None, :]

    def explain(self, x):
        w = self.channel_weights(x)
        return x * w[:, None, None, :], {"channel_weights": w}

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "b": self.b, "gate_bias": self.gate_bias})
        return cfg


class CBAMAttention(layers.Layer):
    kind = "cbam"

    def __init__(self, reduction=config.REDUCTION, kernel_size=config.SPATIAL_KERNEL,
                 gate_bias=config.GATE_BIAS_INIT, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.gate_bias = gate_bias

    def build(self, input_shape):
        c = int(input_shape[-1])
        hidden = max(c // self.reduction, 1)
        self.fc1 = layers.Dense(hidden, activation="relu", name="shared_fc1")
        self.fc2 = layers.Dense(c, use_bias=False, kernel_initializer=_small(), name="shared_fc2")
        self.fc1.build((None, c))
        self.fc2.build((None, hidden))
        self.gate = self.add_weight(name="channel_gate_bias", shape=(c,),
                                    initializer=initializers.Constant(self.gate_bias), trainable=True)
        self.spatial = SpatialAttention(self.kernel_size, self.gate_bias, name="cbam_spatial")
        self.spatial.build(input_shape)
        super().build(input_shape)

    def channel_weights(self, x):
        avg = tf.reduce_mean(x, axis=[1, 2])
        mx = tf.reduce_max(x, axis=[1, 2])
        mlp = lambda t: self.fc2(self.fc1(t))                   # shared MLP for both descriptors
        s = mlp(avg) + mlp(mx)
        return tf.sigmoid(s + tf.cast(self.gate, s.dtype))

    def call(self, x, training=None):
        x = x * self.channel_weights(x)[:, None, None, :]
        return self.spatial(x)

    def explain(self, x):
        w = self.channel_weights(x)
        xc = x * w[:, None, None, :]
        y, info = self.spatial.explain(xc)
        info = dict(info)
        info["channel_weights"] = w
        return y, info

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction": self.reduction, "kernel_size": self.kernel_size, "gate_bias": self.gate_bias})
        return cfg


class CoordinateAttention(layers.Layer):
    kind = "coord"

    def __init__(self, reduction=32, gate_bias=config.GATE_BIAS_INIT, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.gate_bias = gate_bias

    def build(self, input_shape):
        self.h, self.w, c = int(input_shape[1]), int(input_shape[2]), int(input_shape[3])
        mip = max(8, c // self.reduction)
        self.conv1 = layers.Conv2D(mip, 1, name="reduce")
        self.bn = layers.BatchNormalization(name="bn")
        self.conv_h = layers.Conv2D(c, 1, kernel_initializer=_small(),
                                    bias_initializer=initializers.Constant(self.gate_bias), name="to_h")
        self.conv_w = layers.Conv2D(c, 1, kernel_initializer=_small(),
                                    bias_initializer=initializers.Constant(self.gate_bias), name="to_w")
        self.conv1.build((None, self.h + self.w, 1, c))
        self.bn.build((None, self.h + self.w, 1, mip))
        self.conv_h.build((None, self.h, 1, mip))
        self.conv_w.build((None, 1, self.w, mip))
        super().build(input_shape)

    def gates(self, x, training=None):
        x_h = tf.reduce_mean(x, axis=2, keepdims=True)                          # pool along width, (B, H, 1, C)
        x_w = tf.transpose(tf.reduce_mean(x, axis=1, keepdims=True), [0, 2, 1, 3])   # (B, W, 1, C)
        y = tf.concat([x_h, x_w], axis=1)                                       # (B, H + W, 1, C)
        y = self.bn(self.conv1(y), training=training)
        y = y * tf.nn.relu6(y + 3.0) / 6.0                                      # h-swish
        y_h, y_w = tf.split(y, [self.h, self.w], axis=1)
        y_w = tf.transpose(y_w, [0, 2, 1, 3])                                   # (B, 1, W, mip)
        a_h = tf.sigmoid(self.conv_h(y_h))                                      # (B, H, 1, C)
        a_w = tf.sigmoid(self.conv_w(y_w))                                      # (B, 1, W, C)
        return a_h, a_w

    def call(self, x, training=None):
        a_h, a_w = self.gates(x, training)
        return x * a_h * a_w

    def explain(self, x):
        a_h, a_w = self.gates(x, training=False)
        gate = a_h * a_w                                                        # (B, H, W, C)
        info = {"spatial_map": tf.reduce_mean(gate, axis=-1),
                "attention_h": tf.reduce_mean(a_h, axis=[2, 3]),
                "attention_w": tf.reduce_mean(a_w, axis=[1, 3])}
        return x * gate, info

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction": self.reduction, "gate_bias": self.gate_bias})
        return cfg


COMPONENT_NAMES = ("channel", "spatial", "self", "eca", "cbam", "coord")


def make_component(kind, reduction, spatial_kernel, gate_bias):
    if kind == "channel":
        return ChannelAttention(reduction, gate_bias, name="channel")
    if kind == "spatial":
        return SpatialAttention(spatial_kernel, gate_bias, name="spatial")
    if kind == "self":
        return SelfAttention(name="self")
    if kind == "eca":
        return ECAAttention(gate_bias=gate_bias, name="eca")
    if kind == "cbam":
        return CBAMAttention(reduction, spatial_kernel, gate_bias, name="cbam")
    if kind == "coord":
        return CoordinateAttention(gate_bias=gate_bias, name="coord")
    raise ValueError(f"unknown attention component: {kind}")


class AttentionStack(layers.Layer):
    """Ordered list of attention components applied one after another, for example ("channel", "spatial")."""

    def __init__(self, components, reduction=config.REDUCTION, spatial_kernel=config.SPATIAL_KERNEL,
                 gate_bias=config.GATE_BIAS_INIT, **kwargs):
        super().__init__(**kwargs)
        self.components_spec = tuple(components)
        self.reduction = reduction
        self.spatial_kernel = spatial_kernel
        self.gate_bias = gate_bias
        self.blocks = [make_component(c, reduction, spatial_kernel, gate_bias) for c in self.components_spec]

    def build(self, input_shape):
        for block in self.blocks:
            block.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def explain(self, x):
        """Eager run returning (output, info). Keys of info are 'kind/quantity', for example 'spatial/spatial_map'."""
        info = {}
        for block in self.blocks:
            x, block_info = block.explain(x)
            for key, value in block_info.items():
                info[f"{block.kind}/{key}"] = value
        return x, info

    def get_block(self, kind):
        for block in self.blocks:
            if block.kind == kind:
                return block
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"components": list(self.components_spec), "reduction": self.reduction,
                    "spatial_kernel": self.spatial_kernel, "gate_bias": self.gate_bias})
        return cfg


# ------------------------------------------------------------------------------------------
# Analytic cost model (multiply accumulate operations) used by 14_complexity_analysis.py
# ------------------------------------------------------------------------------------------

def macs_component(kind, h, w, c, reduction=config.REDUCTION, spatial_kernel=config.SPATIAL_KERNEL):
    """Approximate multiply accumulates of one attention component for one image."""
    hw = h * w
    if kind == "channel":
        hid = max(c // reduction, 1)
        return hw * c + 2 * c * hid + hw * c                 # pooling, MLP, rescaling
    if kind == "spatial":
        return 2 * hw * c + 2 * spatial_kernel ** 2 * hw + hw * c
    if kind == "self":
        d, dv = max(c // config.SELF_ATT_QK_DIV, 8), max(c // config.SELF_ATT_V_DIV, 8)
        n = hw
        proj = n * c * (2 * d + dv) + n * dv * c
        mats = n * n * d + n * n * dv
        return proj + mats
    if kind == "eca":
        t = int(abs((math.log2(c) + 1) / 2))
        k = t if t % 2 == 1 else t + 1
        return hw * c + c * k + hw * c
    if kind == "cbam":
        hid = max(c // reduction, 1)
        return 2 * hw * c + 4 * c * hid + hw * c + macs_component("spatial", h, w, c, reduction, spatial_kernel)
    if kind == "coord":
        mip = max(8, c // 32)
        return 2 * hw * c + (h + w) * c * mip + (h + w) * mip * c + 2 * hw * c
    raise ValueError(kind)


def self_attention_matrix_size(h, w):
    n = h * w
    return {"tokens": n, "entries": n * n, "megabytes_per_image_fp32": n * n * 4 / 1e6}

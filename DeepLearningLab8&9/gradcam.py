"""
Grad-CAM (Section 27) for any model of this project.

  alpha_k^c = 1/(HW) sum_ij dy^c / dA^k_ij          (global average of the gradients)
  L^c       = ReLU( sum_k alpha_k^c A^k )           (then resized to 224 x 224 and scaled to [0, 1])

A is the last ResNet50 feature map (config.GRADCAM_LAYER = conv5_block3_out, 7 x 7 x 2048) and y^c is the
pre softmax class score (the 'logits' layer). The same layer is used for every model so that the maps
are comparable. This is a post hoc explanation, different from the internal attention weights.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
import data_utils

_GRAD_MODELS = {}


def _grad_model(model):
    key = id(model)
    if key not in _GRAD_MODELS:
        _GRAD_MODELS[key] = keras.Model(model.inputs, [model.get_layer(config.GRADCAM_LAYER).output,
                                                       model.get_layer("logits").output])
    return _GRAD_MODELS[key]


def gradcam(model, x_uint8, class_idx=None, batch=16):
    """Returns (cams [N,224,224] in [0,1], predicted class [N], probabilities [N,5], explained class [N])."""
    gm = _grad_model(model)
    cams, preds, probs, classes = [], [], [], []
    for i in range(0, len(x_uint8), batch):
        x = tf.constant(data_utils.preprocess_array(x_uint8[i:i + batch]))
        with tf.GradientTape() as tape:
            fmap, logits = gm(x, training=False)
            pred = tf.argmax(logits, axis=-1)
            cls = pred if class_idx is None else tf.constant(np.asarray(class_idx)[i:i + batch], dtype=tf.int64)
            # sum of the per image class scores; the factor 1024 only keeps small float16 gradients from underflowing
            # (the map is rescaled to [0, 1] afterwards, so it has no effect on the result)
            score = tf.reduce_sum(tf.gather(logits, cls, batch_dims=1)) * 1024.0
        grads = tf.cast(tape.gradient(score, fmap), tf.float32)                # (B, 7, 7, 2048)
        fmap = tf.cast(fmap, tf.float32)
        alpha = tf.reduce_mean(grads, axis=(1, 2))                             # (B, 2048)
        cam = tf.nn.relu(tf.reduce_sum(fmap * alpha[:, None, None, :], axis=-1))
        cam = tf.image.resize(cam[..., None], (config.IMG_SIZE, config.IMG_SIZE), method="bilinear")[..., 0].numpy()
        mn, mx = cam.min((1, 2), keepdims=True), cam.max((1, 2), keepdims=True)
        cams.append((cam - mn) / (mx - mn + 1e-8))
        preds.append(pred.numpy())
        probs.append(tf.nn.softmax(logits).numpy())
        classes.append(np.asarray(cls))
    return np.concatenate(cams), np.concatenate(preds), np.concatenate(probs), np.concatenate(classes)


def cam_statistics(cams, threshold=0.5):
    """Compactness numbers of Grad-CAM maps: area above the threshold, entropy, distance of the peak from the centre."""
    n = len(cams)
    area = (cams > threshold).reshape(n, -1).mean(1)
    p = cams.reshape(n, -1) + 1e-12
    p = p / p.sum(1, keepdims=True)
    entropy = -(p * np.log(p)).sum(1) / np.log(p.shape[1])                  # 1 = uniformly spread
    peak = np.array([np.unravel_index(c.argmax(), c.shape) for c in cams], dtype=float)
    centre = (config.IMG_SIZE - 1) / 2.0
    dist = np.sqrt(((peak - centre) ** 2).sum(1)) / (config.IMG_SIZE / 2.0)
    return {"area_above_0.5": area, "normalised_entropy": entropy, "peak_distance_from_centre": dist}

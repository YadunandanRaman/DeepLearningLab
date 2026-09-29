"""
Reconstruction metrics from Section 9 and Section 23 of the lab manual:
MSE, MAE, SSIM, and the per image reconstruction error e_i.
"""
import numpy as np
from skimage.metrics import structural_similarity


def _as_images(x):
    return np.asarray(x, dtype=np.float32).reshape(len(x), 28, 28)


def per_image_mse(x, x_hat):
    """e_i = (1 / D) * sum_j (x_ij - x_hat_ij)^2 for every image i."""
    a, b = _as_images(x), _as_images(x_hat)
    return np.mean((a - b) ** 2, axis=(1, 2))


def per_image_ssim(x, x_hat):
    a, b = _as_images(x), _as_images(x_hat)
    return np.array(
        [structural_similarity(a[i], b[i], data_range=1.0) for i in range(len(a))],
        dtype=np.float64,
    )


def reconstruction_metrics(x, x_hat):
    a, b = _as_images(x), _as_images(x_hat)
    return {
        "mse": float(np.mean((a - b) ** 2)),
        "mae": float(np.mean(np.abs(a - b))),
        "ssim": float(np.mean(per_image_ssim(a, b))),
    }


def error_statistics(errors):
    errors = np.asarray(errors)
    mean, std = float(errors.mean()), float(errors.std())
    return {
        "mean": mean,
        "median": float(np.median(errors)),
        "std": std,
        "min": float(errors.min()),
        "p5": float(np.percentile(errors, 5)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(errors.max()),
        "fraction_above_mean_plus_3std": float(np.mean(errors > mean + 3 * std)),
    }

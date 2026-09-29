"""Shared training, timing and result saving helpers."""
import json
import os
import time

import numpy as np
import pandas as pd


def fit_timed(model, x, y, x_val, y_val, epochs, batch_size, verbose=2):
    start = time.perf_counter()
    history = model.fit(
        x, y,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start
    return history_to_dict(history), elapsed


def history_to_dict(history):
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


def _to_builtin(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"cannot serialize {type(obj)}")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def load_json(path, producer=None):
    if not os.path.exists(path):
        hint = f" Run {producer} first." if producer else ""
        raise FileNotFoundError(f"{path} not found.{hint}")
    with open(path) as f:
        return json.load(f)


def save_table(rows, path):
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def require_file(path, producer):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run {producer} first.")


def convergence_summary(history, loss_key="loss", val_key="val_loss"):
    """Numbers that support the 'does it converge, does it overfit' inference."""
    train, val = np.array(history[loss_key]), np.array(history[val_key])
    best_epoch = int(np.argmin(val)) + 1
    return {
        "final_train_loss": float(train[-1]),
        "final_val_loss": float(val[-1]),
        "best_val_loss": float(val.min()),
        "best_val_epoch": best_epoch,
        "epochs_run": len(val),
        "final_gap_val_minus_train": float(val[-1] - train[-1]),
        "val_loss_rise_after_best": float(val[-1] - val.min()),
    }

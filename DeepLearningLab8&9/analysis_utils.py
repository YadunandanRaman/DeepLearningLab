"""Metrics, confidence intervals, significance tests and loaders for finished runs."""
import os

import numpy as np

import config
import experiments as E
import report_utils


# --------------------------------------------------------------------------------------------------
# Classification metrics (Section 22)
# --------------------------------------------------------------------------------------------------

def classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    labels = list(range(config.NUM_CLASSES))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p), "recall": float(r),
            "f1": float(f), "per_class_precision": pc.tolist(), "per_class_recall": rc.tolist(),
            "per_class_f1": fc.tolist(), "per_class_support": sc.tolist()}


def confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))


def bootstrap_ci(y_true, y_pred, n=config.BOOTSTRAP_SAMPLES, conf=config.CONFIDENCE, seed=config.SEED):
    """Percentile bootstrap confidence intervals of accuracy and macro F1 over the test set."""
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs, f1s = [], []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        m = classification_metrics(y_true[idx], y_pred[idx])
        accs.append(m["accuracy"])
        f1s.append(m["f1"])
    lo, hi = (1 - conf) / 2 * 100, (1 + conf) / 2 * 100
    return {"accuracy_ci": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
            "f1_ci": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))]}


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact McNemar test on paired correctness. b = A right and B wrong, c = A wrong and B right."""
    from scipy.stats import binomtest

    a_ok, b_ok = np.asarray(pred_a) == np.asarray(y_true), np.asarray(pred_b) == np.asarray(y_true)
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


# --------------------------------------------------------------------------------------------------
# Loading finished runs (aliases are followed, so B3 and A3 are the same files)
# --------------------------------------------------------------------------------------------------

def is_done(name):
    real, _ = E.resolve(name)
    return os.path.exists(os.path.join(config.run_dir(real), "metrics.json"))


def load_metrics(name, required=True):
    real, _ = E.resolve(name)
    path = os.path.join(config.run_dir(real), "metrics.json")
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Run {name} ({real}) has not been trained yet: expected {path}. "
                                    f"Run the training script of its group first (see README).")
        return None
    return report_utils.read_json(path)


def load_history(name):
    real, _ = E.resolve(name)
    return report_utils.read_json(os.path.join(config.run_dir(real), "history.json"))


def load_outputs(name):
    """Dict with y_val, probs_val, y_test, probs_test of a finished run."""
    real, _ = E.resolve(name)
    z = np.load(os.path.join(config.run_dir(real), "outputs.npz"))
    return {k: z[k] for k in z.files}


def test_predictions(name):
    o = load_outputs(name)
    return o["y_test"], o["probs_test"].argmax(1)


def selected_model():
    """The attention model used for the confusion matrix and the Grad-CAM comparison:
    the best validation accuracy among A1 to A5 (never the test set), unless overridden in config."""
    if config.SELECTED_OVERRIDE:
        return config.SELECTED_OVERRIDE
    best, best_v = None, -1.0
    for n in E.SELECTION_CANDIDATES:
        m = load_metrics(n, required=False)
        if m is not None and m["best_val_accuracy"] > best_v:
            best, best_v = n, m["best_val_accuracy"]
    if best is None:
        raise FileNotFoundError("No attention model (A1 to A5) has been trained yet.")
    return best


def available(names):
    return [n for n in names if is_done(n)]

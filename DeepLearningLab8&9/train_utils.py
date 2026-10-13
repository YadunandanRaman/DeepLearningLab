"""
Training and evaluation of one run (Sections 20, 21, 22, 34).

Protocol (identical for every model):
  phase 1  backbone frozen, attention modules and classifier train  (Adam, head_lr, PHASE1_EPOCHS)
  phase 2  conv5_x also trains, BatchNorm frozen                    (Adam, ft_lr cosine decay, PHASE2_EPOCHS)
  loss     categorical cross entropy
  the weights of the best validation accuracy epoch of phase 2 are kept (validation only, never the test set)

Speed (large GPU): mixed precision, and augmentation + ImageNet preprocessing run on the GPU inside the training
model, so the input pipeline only has to deliver uint8 batches.
"""
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import analysis_utils as au
import config
import data_utils
import experiments as E
import model_builder as mb
import report_utils

_MODELS = {}

IMAGENET_BGR_MEAN = (103.939, 116.779, 123.68)


class ImageNetPreprocess(keras.layers.Layer):
    """ResNet50 preprocessing on the GPU: RGB -> BGR and ImageNet mean subtraction (same as
    keras.applications.resnet50.preprocess_input, which is used for the numpy prediction path)."""

    def call(self, x):
        return x[..., ::-1] - tf.constant(IMAGENET_BGR_MEAN, tf.float32)


def make_training_model(base):
    """Wrap the model for training: uint8 images -> augmentation (training only) -> preprocessing -> model.
    The wrapper has no weights of its own, the trained weights are those of `base`."""
    inp = keras.Input((config.IMG_SIZE, config.IMG_SIZE, 3), dtype="uint8", name="raw_image")
    x = keras.layers.Rescaling(1.0, dtype="float32", name="to_float")(inp)
    x = data_utils.augmenter()(x)
    x = ImageNetPreprocess(dtype="float32", name="imagenet_preprocess")(x)
    return keras.Model(inp, base(x), name="training_wrapper")


class SaveBest(keras.callbacks.Callback):
    """Save the weights of `base` whenever the validation accuracy improves (the first best wins ties)."""

    def __init__(self, base, path):
        super().__init__()
        self.base, self.path, self.best = base, path, -np.inf

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_accuracy", -np.inf)
        if v > self.best:
            self.best = v
            self.base.save_weights(self.path)


def _compile(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])


def fit_two_phase(spec, hp, epochs1, epochs2, ckpt_path=None):
    """Identical two phase transfer learning protocol for every model (Section 20).
    hp = {batch_size, head_lr, ft_lr, dropout}. Returns (model, history, params_phase1, params_phase2, seconds).
    If ckpt_path is given, the best validation epoch of phase 2 is saved there and loaded back at the end."""
    d = data_utils.load_split()
    train_ds = data_utils.make_dataset(d["x_train"], d["y_train"], True, hp["batch_size"])
    val_ds = data_utils.make_dataset(d["x_val"], d["y_val"], False, hp["batch_size"])
    config.set_seed()
    model = mb.build_model(spec, pretrained=config.PRETRAINED, dropout=hp["dropout"])
    wrapper = make_training_model(model)
    t0 = time.time()

    mb.set_phase(model, 1)
    p1 = mb.count_params(model)
    _compile(wrapper, hp["head_lr"])
    h1 = wrapper.fit(train_ds, validation_data=val_ds, epochs=epochs1, shuffle=False, verbose=2).history

    mb.set_phase(model, 2)
    p2 = mb.count_params(model)
    steps = len(train_ds) * epochs2
    _compile(wrapper, keras.optimizers.schedules.CosineDecay(hp["ft_lr"], steps, alpha=config.FT_LR_END_FRACTION))
    callbacks = [SaveBest(model, ckpt_path)] if ckpt_path else []
    h2 = wrapper.fit(train_ds, validation_data=val_ds, epochs=epochs2, callbacks=callbacks, shuffle=False,
                     verbose=2).history
    seconds = time.time() - t0
    if ckpt_path:
        model.load_weights(ckpt_path)                                     # best validation epoch of phase 2
    hist = {k: [float(v) for v in h1[k] + h2[k]] for k in h1}
    hist["phase1_epochs"] = epochs1
    return model, hist, p1, p2, seconds


def measure_inference(model, x_uint8):
    """Average forward pass time per image in milliseconds (batches of INFER_BATCH on preloaded tensors)."""
    xp = data_utils.preprocess_array(x_uint8)
    batches = [tf.constant(xp[i:i + config.INFER_BATCH]) for i in range(0, len(xp), config.INFER_BATCH)]
    fwd = tf.function(lambda b: model(b, training=False))
    for b in batches[:2]:
        fwd(b).numpy()                                                    # warm up (graph tracing, cuDNN autotune)
    peak = None
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass
    times = []
    for _ in range(config.INFER_REPEATS):
        t0 = time.perf_counter()
        probs = np.concatenate([fwd(b).numpy() for b in batches])
        times.append((time.perf_counter() - t0) / len(xp) * 1000.0)
    if gpus:
        try:
            peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6
        except Exception:
            peak = None
    return {"ms_per_image": float(np.median(times)), "ms_std": float(np.std(times)), "peak_gpu_mb": peak}, probs


def predict_probs(model, x_uint8):
    return model.predict(data_utils.preprocess_array(x_uint8), batch_size=config.PREDICT_BATCH, verbose=0)


def convergence_stats(hist):
    va = np.array(hist["val_accuracy"])
    ta = np.array(hist["accuracy"])
    best = float(va.max())
    first95 = int(np.argmax(va >= 0.95 * best)) + 1
    return {"best_epoch": int(va.argmax()) + 1, "epochs_to_95pct_of_best": first95,
            "final_train_accuracy": float(ta[-1]), "final_val_accuracy": float(va[-1]),
            "final_train_loss": float(hist["loss"][-1]), "final_val_loss": float(hist["val_loss"][-1]),
            "train_val_gap": float(ta[-1] - va[-1]),
            "val_acc_std_last_5": float(va[-5:].std())}


def train_and_evaluate(name):
    """Train and evaluate one real run and store history, metrics, predictions and weights."""
    real, spec = E.resolve(name)
    out_dir = config.run_dir(real)
    os.makedirs(out_dir, exist_ok=True)
    config.make_dirs()
    config.setup_runtime()
    print(f"\n===== {E.describe(real)}")

    d = data_utils.load_split()
    hp = config.hparams()
    ckpt = config.model_path(f"{real}.weights.h5")
    model, hist, p1, p2, train_seconds = fit_two_phase(spec, hp, config.PHASE1_EPOCHS, config.PHASE2_EPOCHS, ckpt)
    best_val_acc = float(max(hist["val_accuracy"][config.PHASE1_EPOCHS:]))

    timing, _ = measure_inference(model, d["x_test"])
    probs_val = predict_probs(model, d["x_val"])
    probs_test = predict_probs(model, d["x_test"])
    ce = lambda p, y: float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
    m = au.classification_metrics(d["y_test"], probs_test.argmax(1))
    m.update({
        "name": real, "label": spec.get("label", real), "positions": list(spec["positions"]),
        "components": list(spec["components"]), "kernel": spec["kernel"],
        "test_accuracy": m["accuracy"], "test_loss": ce(probs_test, d["y_test"]),
        "val_accuracy_at_best": float(np.mean(probs_val.argmax(1) == d["y_val"])),
        "best_val_accuracy": best_val_acc, "val_loss_at_best": ce(probs_val, d["y_val"]),
        "total_params": p2["total"], "trainable_params_phase1": p1["trainable"],
        "trainable_params_phase2": p2["trainable"], "attention_params": p2["attention"],
        "train_seconds": float(train_seconds), "n_test": int(len(d["y_test"])), "hparams": hp,
        "precision_policy": keras.mixed_precision.global_policy().name,
    })
    m.update(timing)
    m.update(convergence_stats(hist))

    report_utils.write_json(os.path.join(out_dir, "history.json"), hist)
    report_utils.write_json(os.path.join(out_dir, "metrics.json"), m)
    np.savez(os.path.join(out_dir, "outputs.npz"), y_val=d["y_val"], probs_val=probs_val,
             y_test=d["y_test"], probs_test=probs_test)
    print(f"  test accuracy {m['accuracy']:.4f}  macro F1 {m['f1']:.4f}  params {m['total_params']:,} "
          f"(attention {m['attention_params']:,})  {m['ms_per_image']:.2f} ms/image  {train_seconds / 60:.1f} min")
    keras.backend.clear_session()
    return m


def run_group(group):
    """Train every run of a group that has not been trained yet (FORCE=1 retrains)."""
    names = E.real_runs(E.GROUPS[group])
    if not names:
        print("Nothing to run for this group (ONLY filter?).")
    for r in names:
        if au.is_done(r) and not config.FORCE:
            print(f"[skip] {r} already trained (set FORCE=1 to retrain)")
            continue
        train_and_evaluate(r)
    aliases = [n for n in E.GROUPS[group] if "alias" in E.EXPERIMENTS[n]]
    for n in aliases:
        print(f"[alias] {n} is the same model as {E.resolve(n)[0]} ({E.EXPERIMENTS[n]['label']})")


def load_trained(name):
    """Rebuild a run's architecture and load its trained weights (cached)."""
    real, spec = E.resolve(name)
    if real not in _MODELS:
        path = config.model_path(f"{real}.weights.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained weights for {real}: {path}. Train it first (see README).")
        model = mb.build_model(spec, pretrained=False)
        model.load_weights(path)
        _MODELS[real] = model
    return _MODELS[real]

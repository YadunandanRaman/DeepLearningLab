"""
Step 2b: small hyperparameter search on the BASELINE ResNet50 (instead of assuming the training hyperparameters).

  5 combinations x 3 epochs = 15 epochs in total (config.HP_CANDIDATES).
  Every trial uses the same two phase protocol as the real runs, shortened to 3 epochs:
      epochs 1 and 2: backbone frozen, classifier trains (learning rate head_lr)
      epoch 3       : conv5_x is also fine tuned (learning rate ft_lr, cosine decayed)
  The combination with the best VALIDATION accuracy at the end of epoch 3 wins (ties: lower validation loss).
  The test set is never used.

The winner (batch size, head learning rate, fine tuning learning rate, dropout) is stored in
  outputs/results/hp_search/best_hparams.json
and is then used by EVERY later run, baseline and attention models alike, so the comparison stays fair.
Only the baseline is tuned; attention models inherit the same values (no model gets its own tuning).

  results/hp_search/02b_hp_search.csv / .tex     the five trials
FORCE=1 repeats the search. If runs were already trained with other hyperparameters, retrain them with FORCE=1.
"""
import glob
import os

import config
import report_utils as ru
import train_utils
import experiments as E

config.make_dirs()
config.setup_runtime()
if os.path.exists(config.HP_FILE) and not config.FORCE:
    print("[skip] hyperparameters already tuned:", ru.read_json(config.HP_FILE)["best"],
          "(FORCE=1 repeats the search)")
    raise SystemExit(0)

_, spec = E.resolve("A0")
trials = []
for cand in config.HP_CANDIDATES:
    hp = {k: cand[k] for k in ("batch_size", "head_lr", "ft_lr", "dropout")}
    print(f"\n===== trial {cand['id']}: {hp}")
    model, hist, _, _, secs = train_utils.fit_two_phase(spec, hp, config.HP_PHASE1_EPOCHS, config.HP_PHASE2_EPOCHS)
    t = {**cand, "val_accuracy": hist["val_accuracy"][-1], "val_loss": hist["val_loss"][-1],
         "train_accuracy": hist["accuracy"][-1], "train_loss": hist["loss"][-1], "seconds": secs}
    trials.append(t)
    print(f"  validation accuracy {t['val_accuracy']:.4f}, validation loss {t['val_loss']:.4f}, {secs:.0f} s")
    train_utils.keras.backend.clear_session()

best = max(trials, key=lambda t: (round(t["val_accuracy"], 6), -t["val_loss"]))
header = ["Trial", "Batch size", "Head LR", "Fine-tune LR", "Dropout", "Train acc (%)", "Val acc (%)", "Val loss", "Time (s)", "Selected"]
rows = [[t["id"], t["batch_size"], f"{t['head_lr']:g}", f"{t['ft_lr']:g}", t["dropout"], ru.pct(t["train_accuracy"]),
         ru.pct(t["val_accuracy"]), f"{t['val_loss']:.4f}", f"{t['seconds']:.0f}", "yes" if t is best else ""] for t in trials]
out = os.path.join(config.RESULT_DIR, "hp_search")
ru.write_csv(os.path.join(out, "02b_hp_search.csv"), header, rows)
ru.write_latex_table(os.path.join(out, "02b_hp_search.tex"), header, rows,
                     f"Hyperparameter search on the baseline ({config.HP_PHASE1_EPOCHS}+{config.HP_PHASE2_EPOCHS} epochs per trial, "
                     "selection by validation accuracy).", "tab:hp_search", True)
ru.write_json(config.HP_FILE, {"best": {k: best[k] for k in ("id", "batch_size", "head_lr", "ft_lr", "dropout")},
                               "trials": trials, "epochs_per_trial": config.HP_PHASE1_EPOCHS + config.HP_PHASE2_EPOCHS})
print("\nSelected:", {k: best[k] for k in ("id", "batch_size", "head_lr", "ft_lr", "dropout")})
for r in rows:
    print(r)
stale = glob.glob(os.path.join(config.RUN_DIR, "*", "metrics.json"))
if stale:
    print(f"\nWARNING: {len(stale)} run(s) already exist and were trained with other hyperparameters. "
          f"Use FORCE=1 python run_all.py to retrain them with the selected values.")

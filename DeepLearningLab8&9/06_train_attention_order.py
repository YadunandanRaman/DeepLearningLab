"""
Step 06 : Main Experiment 4 (Section 17): Channel then Spatial (D1, reuses A3) versus Spatial then Channel (D2) at P3.

Every run uses the identical protocol (Section 34): same split, preprocessing, augmentation, optimizer, schedule,
epochs, batch size, backbone and classifier. Only the factor under study changes.
Runs that already exist are skipped (FORCE=1 retrains; ONLY=B2,B4 trains a subset).

Outputs per run in outputs/results/runs/<run>/: history.json, metrics.json, outputs.npz
and the trained weights in outputs/saved_model/<run>.weights.h5
"""
import train_utils

train_utils.run_group("order")

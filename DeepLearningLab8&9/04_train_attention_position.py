"""
Step 04 : Main Experiment 2 (Section 15): position ablation, Channel then Spatial at P1 to P4 (B1 to B4; B3 reuses A3).

Every run uses the identical protocol (Section 34): same split, preprocessing, augmentation, optimizer, schedule,
epochs, batch size, backbone and classifier. Only the factor under study changes.
Runs that already exist are skipped (FORCE=1 retrains; ONLY=B2,B4 trains a subset).

Outputs per run in outputs/results/runs/<run>/: history.json, metrics.json, outputs.npz
and the trained weights in outputs/saved_model/<run>.weights.h5
"""
import train_utils

train_utils.run_group("position")

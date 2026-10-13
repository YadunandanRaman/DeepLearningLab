"""
Step 05 : Main Experiment 3 (Section 16): P3, P4, P3+P4 and the extended P2+P3+P4 (C1 and C2 reuse A3 and B4).

Every run uses the identical protocol (Section 34): same split, preprocessing, augmentation, optimizer, schedule,
epochs, batch size, backbone and classifier. Only the factor under study changes.
Runs that already exist are skipped (FORCE=1 retrains; ONLY=B2,B4 trains a subset).

Outputs per run in outputs/results/runs/<run>/: history.json, metrics.json, outputs.npz
and the trained weights in outputs/saved_model/<run>.weights.h5
"""
import train_utils

train_utils.run_group("multi")

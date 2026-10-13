"""
Runs the whole experiment in order: python run_all.py

  QUICK=1 python run_all.py          quick check (1+1 epochs, small subset, separate outputs_quick/ folder)
  FORCE=1 python run_all.py          repeat the hyperparameter search and retrain every run
Finished runs are skipped automatically, so the script can be restarted after an interruption.
"""
import os
import subprocess
import sys
import time

STEPS = [
    "01_prepare_data.py", "02_model_summary.py", "02b_hyperparameter_search.py",
    "03_train_attention_type.py", "04_train_attention_position.py", "05_train_multiple_positions.py",
    "06_train_attention_order.py", "07_train_kernel_size.py", "08_train_other_mechanisms.py",
    "09_results_tables.py", "10_performance_plots.py", "11_confusion_matrices.py", "12_gradcam.py",
    "13_internal_attention_maps.py", "14_position_maps.py", "15_qualitative_comparison.py", "16_complexity_analysis.py",
]
here = os.path.dirname(os.path.abspath(__file__))
t_all = time.time()
for step in STEPS:
    print(f"\n{'#' * 100}\n# {step}\n{'#' * 100}", flush=True)
    t0 = time.time()
    code = subprocess.call([sys.executable, os.path.join(here, step)], cwd=here)
    print(f"[{step}] exit code {code}, {(time.time() - t0) / 60:.1f} min", flush=True)
    if code != 0:
        sys.exit(f"Stopped: {step} failed (see the error above). Fix it and run python run_all.py again; "
                 f"finished runs are not repeated.")
print(f"\nAll steps finished in {(time.time() - t_all) / 60:.1f} min.")

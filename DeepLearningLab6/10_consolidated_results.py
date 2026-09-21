import os

import pandas as pd

results_dir = "outputs/results"

print("=== Table: RNN / LSTM / GRU / CNN-LSTM(or GRU) ===")
rows = []

comparison_path = os.path.join(results_dir, "model_comparison.csv")
if os.path.isfile(comparison_path):
    seq_df = pd.read_csv(comparison_path)
    for _, row in seq_df.iterrows():
        rows.append({
            "Model": row["model"],
            "Accuracy": row["accuracy"],
            "Precision": row["macro_precision"],
            "Recall": row["macro_recall"],
            "F1": row["macro_f1"],
            "Parameters": row["parameters"],
        })
else:
    print(f"warning: {comparison_path} not found, run 6_evaluate_and_compare.py first")

video_path = os.path.join(results_dir, "video_model_metrics.csv")
if os.path.isfile(video_path):
    video_df = pd.read_csv(video_path)
    for _, row in video_df.iterrows():
        rows.append({
            "Model": row["model"],
            "Accuracy": row["accuracy"],
            "Precision": row["macro_precision"],
            "Recall": row["macro_recall"],
            "F1": row["macro_f1"],
            "Parameters": row["parameters"],
        })
else:
    print(f"warning: {video_path} not found, run 8_video_understanding.py first")

if rows:
    final_table = pd.DataFrame(rows)
    final_table.to_csv(os.path.join(results_dir, "consolidated_results.csv"), index=False)
    print(final_table.to_string(index=False))
else:
    print("no results found yet, run the training and evaluation scripts first")

print("\n=== Table: Sequence to Sequence Task ===")
seq2seq_path = os.path.join(results_dir, "seq2seq_metrics.csv")
if os.path.isfile(seq2seq_path):
    seq2seq_df = pd.read_csv(seq2seq_path)
    print(seq2seq_df.to_string(index=False))
else:
    print(f"warning: {seq2seq_path} not found, run 9_seq2seq.py first")

print("\nsaved outputs/results/consolidated_results.csv")

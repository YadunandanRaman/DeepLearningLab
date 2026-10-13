# Experiments 8 and 9: Effect of Attention Type and Attention Position in a Pretrained ResNet50

CS3807 Deep Learning Laboratory. Dataset: `tf_flowers` (5 classes). Backbone: ImageNet pretrained ResNet50.
Everything in the lab manual (Sections 1 to 42) that can be produced by code is produced by one of the numbered scripts below.

## 1. Run on Colab (results are saved to Google Drive)

```python
# cell 1: mount Google Drive FIRST (all results, weights and figures are then written to
#         MyDrive/CS3807_Exp8_9/outputs, so nothing is lost if the runtime disconnects)
from google.colab import drive
drive.mount('/content/drive')

# cell 2: unzip the project (upload resnet50-attention.zip to the Colab files panel) and go into it
!unzip -q -o resnet50-attention.zip
%cd resnet50-attention

# cell 3: Times New Roman for the EPS figures (once per runtime)
!sudo apt-get update -qq
!echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
!sudo apt-get install -y ttf-mscorefonts-installer
!sudo fc-cache -f

# cell 4: packages (Colab already has TensorFlow, do not reinstall it)
!pip install -q scipy scikit-learn matplotlib pandas

# cell 5: run everything (hyperparameter search, 17 trained models, all tables and figures), or run the scripts one by one
!python run_all.py
```

Use a GPU runtime (Runtime, Change runtime type, choose the largest GPU available; the code is written for a 24 GB card). The first line printed by every script tells you where results go
(`[config] results are saved in: /content/drive/MyDrive/CS3807_Exp8_9/outputs`); a warning is printed if Drive is not mounted.

If the runtime disconnects: reconnect, run cells 1 to 4 again and run `!python run_all.py`. Every finished run is stored on Drive
(`results/runs/<run>/metrics.json` marks it as done) and is skipped, so you only lose the run that was training at that moment
(a few minutes). The downloaded images are cached on the local disk (`cache/`) and are re-created in about a minute.
The split (`split_indices.json`) is stored on Drive, so the same split is reused after a restart.

To use another folder: `import os; os.environ['OUTPUT_ROOT'] = '/content/drive/MyDrive/my_folder'` before running the scripts.
Without Drive the results go to the `outputs/` folder next to the code.

## 2. Scripts, in running order

| Script | Manual sections | What it does |
|---|---|---|
| `01_prepare_data.py` | 3, 3.1, 3.2, 31(1) | downloads tf_flowers, resizes to 224x224, stratified 70/15/15 split stored once, class counts, sample images of all five classes |
| `02_model_summary.py` | 4, 5, 7, 13, 38(1) | ResNet50 stage shapes, model summaries, input and output feature dimensions, parameters and MACs of every attention module at P1 to P4 (asserts that the shape is preserved) |
| `02b_hyperparameter_search.py` | 20, 34 | small hyperparameter search on the baseline: 5 combinations x 3 epochs, selected by validation accuracy; the winner is used by every run |
| `03_train_attention_type.py` | 14 | Experiment 1: A0 baseline, A1 channel, A2 spatial, A3 channel+spatial, A4 self-attention, A5 combined, all at P3 |
| `04_train_attention_position.py` | 15 | Experiment 2: channel+spatial at P1, P2, P3, P4 (B1 to B4, B3 is A3) |
| `05_train_multiple_positions.py` | 16 | Experiment 3: P3, P4, P3+P4 and the extended P2+P3+P4 |
| `06_train_attention_order.py` | 17 | Experiment 4: Channel to Spatial (A3) versus Spatial to Channel at P3 |
| `07_train_kernel_size.py` | 18 | Experiment 5: spatial attention kernel 3, 5, 7 at P3 (spatial attention alone, 7 is A2) |
| `08_train_other_mechanisms.py` | 19, 19.1, 36 | ECA, CBAM and Coordinate Attention at P3 (SE is A1, self-attention is A4) |
| `09_results_tables.py` | 14 to 19, 22, 23, 33.1, 37, 38 | all result tables (CSV and LaTeX), bootstrap confidence intervals, McNemar tests, evidence file for the discussion |
| `10_performance_plots.py` | 24.1 to 24.6, 31(2,3,14) | accuracy curves, loss curves, bar graphs of accuracy and macro F1, accuracy versus parameters, F1 versus inference time, position ablation |
| `11_confusion_matrices.py` | 25, 31(4) | confusion matrices of the baseline and the selected attention model, per class metrics, top confusions |
| `12_gradcam.py` | 27, 27.1, 31(11,12) | Grad-CAM, baseline versus selected attention model, 3 correct and 2 incorrect images, compactness statistics |
| `13_internal_attention_maps.py` | 28, 28.1 to 28.3, 31(5 to 10) | feature maps before and after attention, channel weights (top-K), spatial attention map and overlay, self-attention responses |
| `14_position_maps.py` | 29, 31(13) | spatial attention maps at P2, P3, P4 for the same images |
| `15_qualitative_comparison.py` | 30, 30.1, 32, 36 | side by side comparison of the mechanisms on 3 correct and 2 incorrect images |
| `16_complexity_analysis.py` | 13, 16, 33.2 | parameters, MACs, inference time, memory, training time; size of the self-attention matrix at each position |
| `run_all.py` | | runs 01 to 16 in order |

Shared modules: `config.py` (all hyperparameters and paths), `experiments.py` (registry of every run),
`data_utils.py`, `attention.py` (channel, spatial, self, ECA, CBAM, coordinate attention), `model_builder.py`,
`train_utils.py`, `analysis_utils.py`, `gradcam.py`, `visualize.py`, `report_utils.py`, `plot_style.py`.

Environment variables (instead of command line flags):
`QUICK=1` quick check on a small subset with 1+1 epochs (separate `outputs_quick/` folder),
`FORCE=1` repeat the hyperparameter search and retrain finished runs, `MIXED=0` switch off mixed precision,
`USE_TUNED=0` ignore the search result and use the defaults of `config.py`, `ONLY=B2,B4` train only runs whose id starts with one of the given prefixes.
(`SYNTHETIC=1` is a developer dry run on random images without any download.)

## 3. Section 31: required visualizations and where they are

| # | Required visualization | File in `outputs/plots/` (on Drive) |
|---|---|---|
| 1 | sample images from all five classes | `01_sample_images.eps` |
| 2 | training and validation accuracy curves | `10_accuracy_curves.eps` |
| 3 | training and validation loss curves | `10_loss_curves.eps` |
| 4 | confusion matrix | `11_confusion_matrices.eps` |
| 5, 10 | feature map before and after attention | `13_feature_maps_before_after.eps` |
| 6 | channel attention weights | `13_channel_weights.eps` |
| 7, 8 | spatial attention heatmap and overlay | `13_spatial_attention.eps` |
| 9 | self-attention response | `13_self_attention.eps` |
| 11, 12 | baseline Grad-CAM and Grad-CAM of an attention model | `12_gradcam_baseline_vs_attention.eps` |
| 13 | attention maps at P2, P3, P4 | `14_position_maps_heatmap.eps`, `14_position_maps_overlay.eps` |
| 14 | graphical comparison of the attention mechanisms | `10_bar_accuracy.eps`, `10_bar_f1.eps`, `10_accuracy_vs_params.eps`, `10_f1_vs_inference.eps` |

Also produced for the sections that ask for them: `10_position_ablation.eps` (Experiment 2),
`15_qualitative_comparison.eps` (Section 30 layout), `15_mechanism_comparison.eps` (Section 32 layout).
All figures are EPS at 600 DPI in Times New Roman with labelled axes and legends. Heatmap overlays are blended in numpy
because EPS has no transparency.

## 4. Tables in `outputs/results/`

`09_table_19_1_mechanisms` (Section 19.1), `09_table_23_main_results` (Section 23), `09_exp1_attention_type`,
`09_exp2_attention_position`, `09_exp3_multiple_positions`, `09_exp4_attention_order`, `09_exp5_kernel_size` (Section 18 table),
`09_observations_section38` (Section 38 quantities), `09_significance` (95% bootstrap intervals and McNemar p values),
`11_per_class_metrics`, `16_complexity`, `16_self_attention_cost`, `02_feature_dimensions`, `01_split_counts`.
Every table exists as `.csv` (to check the numbers) and `.tex` (to paste into the report).
`09_evidence.txt` lists the facts that answer the Section 37 questions about performance (best model, gain per position,
whether more positions help, order effect, kernel effect, per class change), including the test set size so that
differences of about one percentage point are not over interpreted.

## 5. Protocol (Section 34) and design decisions

* One fixed stratified split (`outputs/saved_data/split_indices.json`) is used by every run. The test set is never used for any choice.
* Hyperparameters are not assumed: `02b_hyperparameter_search.py` tries 5 combinations (batch size 32/64/128, head learning
  rate 3e-4 to 3e-3, fine tuning learning rate 3e-5 to 3e-4, dropout 0.3/0.5) for 3 epochs each (2 epochs with the backbone frozen,
  1 epoch with conv5_x unfrozen) on the baseline, picks the best VALIDATION accuracy, and stores it in
  `results/hp_search/best_hparams.json`. Every later run, baseline and attention models alike, uses these values
  (only the baseline is tuned, so no attention model gets an advantage; 3 epoch trials favour fast learners, which is a known
  limitation of a short search). The candidates are listed in `config.HP_CANDIDATES`.
* Identical for every model: preprocessing (ImageNet), augmentation (flip, rotation, zoom, translation, contrast), batch size,
  Adam, categorical cross entropy, seed 42, epochs (8 + 7), learning rates, dropout, GAP + Dropout + Dense(5) head with the same initializer.
* Transfer learning: phase 1 trains only the attention modules and the classifier (backbone frozen, head learning rate);
  phase 2 also unfreezes conv5_x (BatchNorm layers stay frozen, cosine decayed fine tuning learning rate). The weights of the best validation
  accuracy epoch of phase 2 are kept. The baseline follows exactly the same two phases.
* Attention modules keep the tensor shape unchanged. All gates start close to identity (final bias 3.0 and a small kernel, so every gate starts at about 0.95), and
  the self-attention residual scale gamma starts at 0, so the pretrained features are not disturbed at the start for any model.
* Channel attention uses reduction r = 16, which is the SE block (the "Channel / SE" row of Section 19.1 is A1).
  Spatial attention uses a 7x7 convolution on the channel mean and max maps. Self-attention uses query and key width C/8, value width C/2.
* Runs that are identical are trained once and reused (B3 = A3 = C1 = D1, B4 = C2, E7 = A2).
* "Selected attention model" (confusion matrix, Grad-CAM): the model of A1 to A5 with the best validation accuracy.
  Change it with `SELECTED_OVERRIDE` in `config.py`.
* Experiment 5 uses spatial attention alone at P3. Section 23 uses ECA and CBAM as Additional Attention-1 and 2, and
  Coordinate Attention is the "Additional Mechanism" of Section 19.1 (all three at P3).
* Grad-CAM always uses `conv5_block3_out` (7x7x2048) and the pre softmax score of the predicted class, so the maps are comparable
  between models. Internal attention (channel weights, spatial maps, self-attention) is read from the forward pass and is not Grad-CAM.
* Inference time: average forward pass per image in batches of 32 on tensors already on the device, median of 3 passes after warm up.
  Compare times only within one runtime, because they change with the GPU type.
* Speed on a large GPU (24 GB VRAM, 50 GB RAM): mixed precision (float16 compute, float32 weights, float32 softmax and
  self-attention matrix) is switched on automatically when a GPU is present; random augmentation and ImageNet preprocessing run
  on the GPU inside the training model, so the input pipeline only delivers uint8 batches and the CPU is not the bottleneck;
  the whole dataset (about 0.55 GB as uint8) is held in RAM; the default batch size is 64 and the search also tries 128.
  GPU memory growth is enabled. The in-graph preprocessing was checked to give exactly the same predictions as the numpy path.
  Inference times are therefore measured with mixed precision (`MIXED=0` gives float32 times); compare them only within one run
  of the pipeline on the same GPU.
* Expected time: a few minutes per run on a 24 GB card (the self-attention models are the slowest), so the whole pipeline is a
  matter of roughly one to two hours; the exact time depends on the GPU model. The search itself is 15 epochs.

## 6. Folder layout

```
<OUTPUT_ROOT>/outputs/        (Drive: MyDrive/CS3807_Exp8_9/outputs)
  saved_data/    split_indices.json (the cached images are in cache/ next to the code)
  saved_model/   <run>.weights.h5
  results/       tables, statistics, evidence file;  runs/<run>/{history.json, metrics.json, outputs.npz}
  plots/         all figures (EPS, 600 DPI)
```

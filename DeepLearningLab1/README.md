# Single Layer Perceptron - Banknote Authentication

CS3807 Deep Learning Lab, Experiment 1. Perceptron implemented from
scratch with numpy, trained on the UCI banknote authentication
dataset.

## Files

- `plot_style.py` - shared plotting setup: loads Times New Roman, saves figures as EPS at 600 DPI
- `perceptron.py` - the Perceptron class 
- `1_data_exploration.py` - load data, check shape/missing values/stats 
- `2_eda.py` - histograms, correlation heatmap, scatter plot, boxplots 
- `3_preprocessing.py` - normalize features, 80/20 split, saves arrays to `outputs/saved_model/` 
- `4_train_perceptron.py` - trains the perceptron, saves weights + training plots 
- `5_evaluate.py` - accuracy/precision/recall/F1/confusion matrix on test set 
- `6_learning_rate_comparison.py` - compares lr = 0.001, 0.01, 0.1 
- `7_sklearn_comparison.py` - compares against sklearn's Perceptron 
- `8_decision_boundary.py` - plots the decision boundary using variance and skewness only 
- `9_normalization_comparison.py` - trains with and without normalization on the same split, plots both training error curves side by side 
- `10_logic_gates.py` - Perceptron learning algorithm for AND, OR, and NOT logic gates


## Figure font (Colab only, run once)

Times New Roman isn't installed on Colab by default. Run this in a
Colab cell before running `2_eda.py` or any other plotting script:

```python
!echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
!sudo apt-get update
!sudo apt-get install -y ttf-mscorefonts-installer
!sudo fc-cache -f
```

If you skip this, `plot_style.py` falls back to a generic serif font
and prints a warning instead of failing.

## How to run

Run them in order:

```bash
pip install -r requirements.txt

python 1_data_exploration.py
python 2_eda.py
python 3_preprocessing.py
python 4_train_perceptron.py
python 5_evaluate.py
python 6_learning_rate_comparison.py
python 7_sklearn_comparison.py
python 8_decision_boundary.py
python 9_normalization_comparison.py
python 10_logic_gates.py
```


Plots land in `outputs/plots/`, saved arrays in `outputs/saved_model/`.

## Dataset

`data/banknote_authentication.csv` - UCI banknote authentication
dataset, 1372 rows, 4 features (variance, skewness, curtosis,
entropy), binary class (0 = authentic, 1 = forged). No missing
values. Raw original file also included as
`data/data_banknote_authentication_raw.txt`.

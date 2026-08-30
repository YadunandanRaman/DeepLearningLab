import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from plot_style import setup_fonts, save_fig

setup_fonts()

df = pd.read_csv("data/banknote_authentication.csv")
out_dir = "outputs/plots"

features = ["variance", "skewness", "curtosis", "entropy"]
class_colors = {0: "green", 1: "red"}

# 1. Histograms
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), features):
    sns.histplot(data=df, x=col, hue="class", kde=True, palette=class_colors, ax=ax)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
fig.tight_layout()
save_fig(fig, f"{out_dir}/histograms")
plt.close(fig)

# 2. Correlation Heatmap
custom_cmap = LinearSegmentedColormap.from_list("red_green_red", ["red", "green", "red"])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[features].corr(), annot=True, cmap=custom_cmap, vmin=-1, vmax=1, ax=ax)
fig.tight_layout()
save_fig(fig, f"{out_dir}/correlation_heatmap")
plt.close(fig)

# 3. Scatter Plot (Variance vs Skewness)
fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(data=df, x="variance", y="skewness", hue="class", palette=class_colors, ax=ax)
ax.set_xlabel("Variance")
ax.set_ylabel("Skewness")
fig.tight_layout()
save_fig(fig, f"{out_dir}/scatter_plot")
plt.close(fig)

# 4. Boxplots
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(data=df, x="class", y=col, hue="class", palette=class_colors, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Class")
    ax.set_ylabel(col)
fig.tight_layout()
save_fig(fig, f"{out_dir}/boxplots")
plt.close(fig)

print(f"saved 4 plots to {out_dir}/ in EPS format at 600 DPI with Times New Roman font")

"""
Perceptron learning algorithm for AND, OR, and NOT logic gates.

Only the Python standard library plus matplotlib is used. Matplotlib is
the standard plotting tool in Python, and since there is no plotting
module in the standard library itself, it is the only dependency
outside the standard library used here. No numpy and no scikit-learn:
the perceptron math is done with plain lists.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import math
import os

OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Times New Roman on Colab comes from the msttcorefonts package.
# Run this in a Colab cell once, before running this script, to install it:
#
#   !echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
#   !sudo apt-get install -y ttf-mscorefonts-installer
#   !sudo fc-cache -f
#
FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
FIG_FORMAT = "eps"
FIG_DPI = 600


def setup_fonts():
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = "Times New Roman"
    except FileNotFoundError:
        print("Times New Roman not found at", FONT_PATH)
        print("Run the msttcorefonts install cell in Colab first (see comment above).")
        print("Falling back to a generic serif font for now.")
        plt.rcParams["font.family"] = "serif"


setup_fonts()


class Perceptron:
    """A simple perceptron trained with the classic perceptron learning rule:
    w_i <- w_i + lr * (target - prediction) * x_i
    b   <- b   + lr * (target - prediction)
    """

    def __init__(self, num_inputs, learning_rate=0.2):
        self.weights = [0.0] * num_inputs
        self.bias = 0.0
        self.lr = learning_rate
        # history stores a snapshot after every weight update, plus the
        # initial (untrained) state as entry 0
        self.history = []
        self._record("initial weights (before training)")

    def _record(self, note):
        self.history.append({
            "weights": self.weights.copy(),
            "bias": self.bias,
            "note": note,
        })

    def predict(self, x):
        activation = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if activation >= 0 else 0

    def train(self, X, y, max_epochs=50):
        for epoch in range(1, max_epochs + 1):
            made_update = False
            for x, target in zip(X, y):
                pred = self.predict(x)
                error = target - pred
                if error != 0:
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * x[i]
                    self.bias += self.lr * error
                    made_update = True
                    note = (f"epoch {epoch}, input {x}, target={target}, "
                            f"predicted={pred}, error={error}")
                    self._record(note)
            if not made_update:
                break
        return self.history


def print_history(name, history):
    print(f"\n{'=' * 60}")
    print(f"{name} GATE: weights after each update")
    print(f"{'=' * 60}")
    for i, h in enumerate(history):
        w_str = ", ".join(f"w{j+1}={w:.2f}" for j, w in enumerate(h["weights"]))
        print(f"  step {i}: {w_str}, bias={h['bias']:.2f}   ({h['note']})")


def _class_legend_handles(boundary_label="Decision Boundary"):
    """Shared legend handles used by both the 2D and 1D gate plots."""
    return [
        mlines.Line2D([], [], color="tab:blue", marker="o", linestyle="None",
                      markersize=10, markeredgecolor="black", label="Output = 1"),
        mlines.Line2D([], [], color="tab:red", marker="s", linestyle="None",
                      markersize=10, markeredgecolor="black", label="Output = 0"),
        mlines.Line2D([], [], color="black", linewidth=2, label=boundary_label),
    ]


def plot_2d_gate_history(name, X, y, history, filename):
    """Grid of subplots, one per update, showing the 2D decision boundary."""
    n = len(history)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, h in enumerate(history):
        ax = axes[idx]
        w1, w2 = h["weights"]
        b = h["bias"]

        # plot the data points
        for x, target in zip(X, y):
            color = "tab:blue" if target == 1 else "tab:red"
            marker = "o" if target == 1 else "s"
            ax.scatter(x[0], x[1], c=color, marker=marker, s=140,
                       edgecolors="black", zorder=3)

        # decision boundary: w1*x1 + w2*x2 + b = 0
        xs = [-0.5, 1.5]
        if abs(w2) > 1e-9:
            ys = [-(w1 * xv + b) / w2 for xv in xs]
            ax.plot(xs, ys, "k-", linewidth=2)
        elif abs(w1) > 1e-9:
            x_vert = -b / w1
            ax.axvline(x_vert, color="black", linewidth=2)
        # if w1 == w2 == 0, no line to draw yet, since no update has happened

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"step {idx}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}",
                     fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    # hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{name} Gate: Decision Boundary After Each Weight Update",
                 fontsize=14, y=1.02)
    fig.legend(handles=_class_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=True)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_1d_gate_history(name, X, y, history, filename):
    """Grid of subplots for the NOT gate (single input, plotted on a line)."""
    n = len(history)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, h in enumerate(history):
        ax = axes[idx]
        w1 = h["weights"][0]
        b = h["bias"]

        for x, target in zip(X, y):
            color = "tab:blue" if target == 1 else "tab:red"
            marker = "o" if target == 1 else "s"
            ax.scatter(x[0], 0, c=color, marker=marker, s=180,
                       edgecolors="black", zorder=3)
            ax.annotate(f"x={x[0]}, y={target}", (x[0], 0),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=8)

        if abs(w1) > 1e-9:
            boundary = -b / w1
            ax.axvline(boundary, color="black", linewidth=2)

        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xlabel("x")
        ax.set_title(f"step {idx}: w1={w1:.2f}, b={b:.2f}", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{name} Gate: Decision Boundary After Each Weight Update",
                 fontsize=14, y=1.02)
    fig.legend(handles=_class_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=True)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    # ---- AND gate ----
    X_and = [(0, 0), (0, 1), (1, 0), (1, 1)]
    y_and = [0, 0, 0, 1]
    p_and = Perceptron(num_inputs=2, learning_rate=0.2)
    hist_and = p_and.train(X_and, y_and)
    print_history("AND", hist_and)
    plot_2d_gate_history("AND", X_and, y_and, hist_and, f"and_gate_updates.{FIG_FORMAT}")

    # ---- OR gate ----
    X_or = [(0, 0), (0, 1), (1, 0), (1, 1)]
    y_or = [0, 1, 1, 1]
    p_or = Perceptron(num_inputs=2, learning_rate=0.2)
    hist_or = p_or.train(X_or, y_or)
    print_history("OR", hist_or)
    plot_2d_gate_history("OR", X_or, y_or, hist_or, f"or_gate_updates.{FIG_FORMAT}")

    # ---- NOT gate ----
    X_not = [(0,), (1,)]
    y_not = [1, 0]
    p_not = Perceptron(num_inputs=1, learning_rate=0.2)
    hist_not = p_not.train(X_not, y_not)
    print_history("NOT", hist_not)
    plot_1d_gate_history("NOT", X_not, y_not, hist_not, f"not_gate_updates.{FIG_FORMAT}")

    print("\nAll plots saved to:", OUTPUT_DIR)

"""
Multi layer perceptron trained with backpropagation for the XOR gate.

A single layer perceptron cannot solve XOR, since XOR is not linearly
separable: no single straight line can separate its two output classes.
This script adds one hidden layer with a nonlinear (sigmoid) activation,
which is the standard minimal architecture that can solve XOR.

Only the Python standard library plus matplotlib is used. Matplotlib is
the standard plotting tool in Python, and since there is no plotting
module in the standard library itself, it is the only dependency
outside the standard library used here. No numpy and no scikit-learn:
the network's forward pass, backpropagation, and weight updates are all
done with plain lists and the math module.
"""

import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines

OUTPUT_DIR = "/content/outputs"
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


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(a):
    # a is already sigmoid(z), so the derivative with respect to z is a*(1-a)
    return a * (1.0 - a)


def binary_cross_entropy(output, target):
    # clipped to avoid log(0) when the network is extremely confident
    eps = 1e-12
    output = min(max(output, eps), 1.0 - eps)
    return -(target * math.log(output) + (1.0 - target) * math.log(1.0 - output))


class MultiLayerPerceptron:
    """
    A minimal 2 input, N hidden, 1 output network, trained with full
    batch gradient descent and binary cross entropy loss.

    Binary cross entropy is the natural loss for a sigmoid output, since
    a sigmoid represents a Bernoulli probability and cross entropy is
    exactly the negative log likelihood of that distribution. It also
    has a practical advantage over mean squared error: for a sigmoid
    output, the gradient of cross entropy with respect to the pre
    activation z simplifies to (output - target), with the sigmoid's
    own derivative cancelling out completely. Mean squared error does
    not get this cancellation, so its gradient carries an extra
    output * (1 - output) factor that shrinks toward zero whenever the
    network is confidently wrong, slowing learning exactly when it is
    needed most. Cross entropy does not have this problem.
    """

    def __init__(self, num_inputs, num_hidden, learning_rate=5.0, seed=42):
        random.seed(seed)
        self.lr = learning_rate
        self.num_hidden = num_hidden

        self.w_hidden = [[random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_hidden)]
        self.b_hidden = [random.uniform(-1, 1) for _ in range(num_hidden)]
        self.w_output = [random.uniform(-1, 1) for _ in range(num_hidden)]
        self.b_output = random.uniform(-1, 1)

        # history stores a snapshot at regular checkpoints during training,
        # plus the initial (untrained) state as entry 0
        self.history = []
        self._record(epoch=0, loss=None, note="initial weights (before training)")

    def _record(self, epoch, loss, note):
        self.history.append({
            "epoch": epoch,
            "w_hidden": [row.copy() for row in self.w_hidden],
            "b_hidden": self.b_hidden.copy(),
            "w_output": self.w_output.copy(),
            "b_output": self.b_output,
            "loss": loss,
            "note": note,
        })

    def forward(self, x):
        hidden_out = [
            sigmoid(sum(w * xi for w, xi in zip(self.w_hidden[j], x)) + self.b_hidden[j])
            for j in range(self.num_hidden)
        ]
        z_out = sum(w * h for w, h in zip(self.w_output, hidden_out)) + self.b_output
        return hidden_out, sigmoid(z_out)

    def predict(self, x):
        _, output = self.forward(x)
        return 1 if output >= 0.5 else 0

    def train(self, X, y, max_epochs=400, snapshot_every=25):
        """
        Trains with full batch gradient descent: every epoch, gradients
        are accumulated over all training samples and applied as one
        weight update. That single update per epoch is treated as "an
        update" for the purposes of this script.

        A single layer perceptron only updates its weights when it
        misclassifies a sample, so AND, OR, and NOT needed only a
        handful of updates in total. Gradient descent instead changes
        every weight by a small amount every single epoch, and this
        network needs several hundred epochs to converge, so plotting
        every epoch individually would produce an impractically large
        grid of nearly identical panels. Weights are therefore
        checkpointed and plotted every snapshot_every epochs, plus the
        initial and final state, while every single epoch's weights are
        still printed to the console in full below.
        """
        m = len(X)
        num_inputs = len(X[0])

        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            grad_w_hidden = [[0.0] * num_inputs for _ in range(self.num_hidden)]
            grad_b_hidden = [0.0] * self.num_hidden
            grad_w_output = [0.0] * self.num_hidden
            grad_b_output = 0.0

            for x, target in zip(X, y):
                hidden_out, output = self.forward(x)
                total_loss += binary_cross_entropy(output, target)

                # for a sigmoid output trained with binary cross entropy,
                # the gradient with respect to the pre activation simplifies
                # to (output - target), the sigmoid derivative cancels out
                delta_output = output - target
                for j in range(self.num_hidden):
                    grad_w_output[j] += delta_output * hidden_out[j]
                grad_b_output += delta_output

                for j in range(self.num_hidden):
                    delta_hidden = delta_output * self.w_output[j] * sigmoid_derivative(hidden_out[j])
                    for i in range(num_inputs):
                        grad_w_hidden[j][i] += delta_hidden * x[i]
                    grad_b_hidden[j] += delta_hidden

            for j in range(self.num_hidden):
                for i in range(num_inputs):
                    self.w_hidden[j][i] -= self.lr * grad_w_hidden[j][i] / m
                self.b_hidden[j] -= self.lr * grad_b_hidden[j] / m
                self.w_output[j] -= self.lr * grad_w_output[j] / m
            self.b_output -= self.lr * grad_b_output / m

            mean_loss = total_loss / m
            print(f"epoch {epoch}: binary cross entropy={mean_loss:.5f}")

            if epoch % snapshot_every == 0 or epoch == max_epochs:
                self._record(epoch, mean_loss,
                              f"epoch {epoch}, binary cross entropy={mean_loss:.5f}")

        return self.history


def print_history(name, history):
    print(f"\n{'=' * 70}")
    print(f"{name} GATE: weights at each recorded checkpoint")
    print(f"{'=' * 70}")
    for h in history:
        hidden_str = " | ".join(
            f"h{j + 1}: w={[round(w, 3) for w in row]}, b={h['b_hidden'][j]:.3f}"
            for j, row in enumerate(h["w_hidden"])
        )
        output_str = f"output: w={[round(w, 3) for w in h['w_output']]}, b={h['b_output']:.3f}"
        loss_str = f"loss={h['loss']:.5f}" if h["loss"] is not None else "loss not yet computed"
        print(f"  epoch {h['epoch']:4d}   {hidden_str}   {output_str}   {loss_str}")


def _class_legend_handles():
    return [
        mlines.Line2D([], [], color="tab:blue", marker="o", linestyle="None",
                      markersize=10, markeredgecolor="black", label="Output = 1"),
        mlines.Line2D([], [], color="tab:red", marker="s", linestyle="None",
                      markersize=10, markeredgecolor="black", label="Output = 0"),
        mlines.Line2D([], [], color="black", linewidth=2, label="Decision Boundary"),
    ]


def compute_output_grid(w_hidden, b_hidden, w_output, b_output, x_range, y_range, resolution=80):
    """Evaluates the network across a grid of points, used to draw the
    curved decision boundary a hidden layer makes possible."""
    xs = [x_range[0] + i * (x_range[1] - x_range[0]) / (resolution - 1) for i in range(resolution)]
    ys = [y_range[0] + i * (y_range[1] - y_range[0]) / (resolution - 1) for i in range(resolution)]
    Z = []
    for yv in ys:
        row = []
        for xv in xs:
            hidden_out = [
                sigmoid(w_hidden[j][0] * xv + w_hidden[j][1] * yv + b_hidden[j])
                for j in range(len(w_hidden))
            ]
            z_out = sum(w * h for w, h in zip(w_output, hidden_out)) + b_output
            row.append(sigmoid(z_out))
        Z.append(row)
    return xs, ys, Z


def plot_xor_decision_boundaries(X, y, history, filename):
    n = len(history)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, h in enumerate(history):
        ax = axes[idx]
        xs, ys, Z = compute_output_grid(h["w_hidden"], h["b_hidden"], h["w_output"], h["b_output"],
                                         x_range=(-0.5, 1.5), y_range=(-0.5, 1.5))

        ax.contourf(xs, ys, Z, levels=[0.0, 0.5, 1.0], colors=["#f4cccc", "#cfe2f3"], alpha=0.7)
        ax.contour(xs, ys, Z, levels=[0.5], colors="black", linewidths=2)

        for x, target in zip(X, y):
            color = "tab:blue" if target == 1 else "tab:red"
            marker = "o" if target == 1 else "s"
            ax.scatter(x[0], x[1], c=color, marker=marker, s=140,
                       edgecolors="black", zorder=3)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        loss_str = f"{h['loss']:.4f}" if h["loss"] is not None else "not yet trained"
        ax.set_title(f"epoch {h['epoch']}, loss {loss_str}", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("XOR Gate: Decision Boundary After Each Recorded Weight Update",
                 fontsize=14, y=1.02)
    fig.legend(handles=_class_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    X_xor = [(0, 0), (0, 1), (1, 0), (1, 1)]
    y_xor = [0, 1, 1, 0]

    mlp = MultiLayerPerceptron(num_inputs=2, num_hidden=2, learning_rate=5.0, seed=42)
    history = mlp.train(X_xor, y_xor, max_epochs=400, snapshot_every=25)

    print_history("XOR", history)

    print("\nfinal predictions:")
    for x, target in zip(X_xor, y_xor):
        _, output = mlp.forward(x)
        predicted = mlp.predict(x)
        print(f"  input={x} target={target} output={output:.4f} predicted={predicted}")

    plot_xor_decision_boundaries(X_xor, y_xor, history, f"xor_gate_updates.{FIG_FORMAT}")

    print("\nAll plots saved to:", OUTPUT_DIR)

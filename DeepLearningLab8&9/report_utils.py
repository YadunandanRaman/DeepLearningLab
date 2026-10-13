"""Small helpers that write results as CSV, JSON and LaTeX tables (write_latex_table) for the lab report."""
import csv
import json
import os

import numpy as np


def _clean(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_clean)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("->", r"$\rightarrow$")):
        s = s.replace(a, b)
    return s


def write_latex_table(path, header, rows, caption, label, resize=False):
    """Write a booktabs-free LaTeX table (tabular with hlines, as in the lab manual)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    spec = "|l|" + "c|" * (len(header) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{tex_escape(caption)}}}", f"\\label{{{label}}}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{")
    lines += [f"\\begin{{tabular}}{{{spec}}}", r"\hline",
              " & ".join(f"\\textbf{{{tex_escape(h)}}}" for h in header) + r" \\", r"\hline"]
    for r in rows:
        lines += [" & ".join(tex_escape(c) for c in r) + r" \\", r"\hline"]
    lines.append(r"\end{tabular}" + ("}" if resize else ""))
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def pct(x, nd=2):
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


def millions(x, nd=2):
    return "n/a" if x is None else f"{x / 1e6:.{nd}f}M"


def ms(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"

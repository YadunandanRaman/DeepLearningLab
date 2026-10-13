"""
Registry of every training run in Experiments 8 and 9.

Each run is one row of the specification below. A run is described by
  positions   tuple of ResNet50 positions (1..4 for P1..P4) where the SAME attention stack is inserted
  components  ordered tuple of attention components applied at each position
  kernel      spatial attention kernel size
Runs with an "alias" are identical to another run (for example B3 is the same model as A3), so they
are not trained twice; the results of the aliased run are reused.

Group           Manual section                         Runs
type            14 Main Experiment 1                   A0 A1 A2 A3 A4 A5
position        15 Main Experiment 2                   B1 B2 B3 B4
multi           16 Main Experiment 3                   C1 C2 C3 C4
order           17 Main Experiment 4                   D1 D2
kernel          18 Main Experiment 5                   E3 E5 E7
mechanism       19 and 36 Additional mechanisms        M1 (ECA) M2 (CBAM) M3 (Coordinate)
"""
import config

# fmt: off
EXPERIMENTS = {
    # Experiment 1: attention type at P3
    "A0": dict(group="type", label="Baseline", positions=(), components=(), kernel=7),
    "A1": dict(group="type", label="Channel (SE)", positions=(3,), components=("channel",), kernel=7),
    "A2": dict(group="type", label="Spatial", positions=(3,), components=("spatial",), kernel=7),
    "A3": dict(group="type", label="Channel+Spatial", positions=(3,), components=("channel", "spatial"), kernel=7),
    "A4": dict(group="type", label="Self-Attention", positions=(3,), components=("self",), kernel=7),
    "A5": dict(group="type", label="Combined (C+S+Self)", positions=(3,), components=("channel", "spatial", "self"), kernel=7),
    # Experiment 2: position of Channel+Spatial
    "B1": dict(group="position", label="Channel+Spatial at P1", positions=(1,), components=("channel", "spatial"), kernel=7),
    "B2": dict(group="position", label="Channel+Spatial at P2", positions=(2,), components=("channel", "spatial"), kernel=7),
    "B3": dict(group="position", label="Channel+Spatial at P3", alias="A3"),
    "B4": dict(group="position", label="Channel+Spatial at P4", positions=(4,), components=("channel", "spatial"), kernel=7),
    # Experiment 3: multiple positions of Channel+Spatial
    "C1": dict(group="multi", label="P3", alias="A3"),
    "C2": dict(group="multi", label="P4", alias="B4"),
    "C3": dict(group="multi", label="P3+P4", positions=(3, 4), components=("channel", "spatial"), kernel=7),
    "C4": dict(group="multi", label="P2+P3+P4", positions=(2, 3, 4), components=("channel", "spatial"), kernel=7),
    # Experiment 4: order of Channel and Spatial at P3
    "D1": dict(group="order", label="Channel -> Spatial", alias="A3"),
    "D2": dict(group="order", label="Spatial -> Channel", positions=(3,), components=("spatial", "channel"), kernel=7),
    # Experiment 5: kernel size of spatial attention at P3 (spatial attention alone)
    "E3": dict(group="kernel", label="3x3", positions=(3,), components=("spatial",), kernel=3),
    "E5": dict(group="kernel", label="5x5", positions=(3,), components=("spatial",), kernel=5),
    "E7": dict(group="kernel", label="7x7", alias="A2"),
    # Additional mechanisms at P3 (Sections 19 and 36). SE is A1 and self-attention is A4.
    "M1": dict(group="mechanism", label="ECA", positions=(3,), components=("eca",), kernel=7),
    "M2": dict(group="mechanism", label="CBAM", positions=(3,), components=("cbam",), kernel=7),
    "M3": dict(group="mechanism", label="Coordinate Attention", positions=(3,), components=("coord",), kernel=7),
}
# fmt: on

GROUPS = {
    "type": ["A0", "A1", "A2", "A3", "A4", "A5"],
    "position": ["B1", "B2", "B3", "B4"],
    "multi": ["C1", "C2", "C3", "C4"],
    "order": ["D1", "D2"],
    "kernel": ["E3", "E5", "E7"],
    "mechanism": ["M1", "M2", "M3"],
}

# the runs behind the figures and tables (names used by the analysis scripts)
SELECTION_CANDIDATES = ["A1", "A2", "A3", "A4", "A5"]   # selected attention model = best validation accuracy
TABLE_19_1 = [("No Attention", "A0"), ("Channel / SE", "A1"), ("Spatial Attention", "A2"), ("ECA", "M1"),
              ("CBAM", "M2"), ("Self-Attention", "A4"), ("Additional: Coordinate Attention", "M3")]
TABLE_23 = [("Baseline", "A0", "--"), ("Channel Attention", "A1", "P3"), ("Spatial Attention", "A2", "P3"),
            ("Channel+Spatial", "B2", "P2"), ("Channel+Spatial", "A3", "P3"), ("Channel+Spatial", "B4", "P4"),
            ("Self-Attention", "A4", "P3"), ("Additional Attention-1 (ECA)", "M1", "P3"),
            ("Additional Attention-2 (CBAM)", "M2", "P3")]
BAR_GRAPH_RUNS = [("Baseline", "A0"), ("Channel (SE)", "A1"), ("Spatial", "A2"), ("Channel+Spatial", "A3"),
                  ("Self-Attention", "A4"), ("ECA", "M1"), ("CBAM", "M2"), ("Coordinate", "M3")]
CURVE_RUNS = ["A0", "A1", "A2", "A3", "A4"]              # Figures 24.1 and 24.2 (baseline and attention models)


def resolve(name):
    """Follow aliases and return (real_name, spec)."""
    spec = EXPERIMENTS[name]
    while "alias" in spec:
        name = spec["alias"]
        spec = EXPERIMENTS[name]
    return name, spec


def real_runs(names=None):
    """Unique runs that must really be trained, in registry order, honouring the ONLY filter."""
    names = names or list(EXPERIMENTS)
    if config.ONLY:
        names = [n for n in names if any(n.startswith(p) for p in config.ONLY)]
    seen, out = set(), []
    for n in names:
        r, _ = resolve(n)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def describe(name):
    r, spec = resolve(name)
    pos = "-" if not spec["positions"] else "+".join(f"P{p}" for p in spec["positions"])
    comp = "none" if not spec["components"] else " -> ".join(spec["components"])
    return f"{name} ({EXPERIMENTS[name]['label']}): positions={pos}, components={comp}, kernel={spec['kernel']}"

"""Figures from saved smoke traces only; never imply scientific evaluation."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(output: Path):
    summary = json.loads((output / "summary.json").read_text())
    trace = np.load(output / "traces.npz")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, key, assignments, title in (
        (axes[0], "noop_positions", summary["initial_assignments"], "No-op: matched branch"),
        (axes[1], "selected_positions", summary["selected_assignments"], "Selected: matched branch"),
    ):
        positions = trace[key]
        for i in range(3):
            line, = ax.plot(positions[:, i, 0], positions[:, i, 1], label=f"agent {i} → mission {assignments[i]}")
            ax.scatter(*positions[0, i], color=line.get_color(), marker="o")
            target = positions[0, 3 + assignments[i]]
            ax.plot([positions[0, i, 0], target[0]], [positions[0, i, 1], target[1]], linestyle=":", color=line.get_color(), alpha=0.5)
        ax.scatter(positions[0, 3:, 0], positions[0, 3:, 1], marker="*", c="black", s=80)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        ax.legend(fontsize=7)
    axes[2].plot(trace["noop_rewards"], label="no-op")
    axes[2].plot(trace["selected_rewards"], label="selected")
    axes[2].set(title="Realized native mean reward", xlabel="Step", ylabel="Reward")
    axes[2].legend()
    fig.suptitle("CORA engineering smoke test — no performance or safety claim")
    fig.savefig(output / "poc.png", dpi=160)
    plt.close(fig)


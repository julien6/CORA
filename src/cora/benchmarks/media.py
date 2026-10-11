"""Saved-data plots, native MPE screenshots/GIFs, and synchronized comparisons."""
import json
import html
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np


def render_trace(path: Path, output: Path, title: str, fps=10):
    import jaxmarl
    from jaxmarl.environments.mpe import MPEVisualizer
    with np.load(path) as trace:
        frames = len(trace["p_pos"])
        states = [SimpleNamespace(p_pos=trace["p_pos"][i], c=trace["c"][i], step=trace["step"][i]) for i in range(frames)]
        assignments = trace["assignments"].tolist()
        rewards = trace["rewards"].copy()
    output.mkdir(parents=True, exist_ok=True)
    visualizer = MPEVisualizer(jaxmarl.make("MPE_simple_spread_v3"), states)
    positions = np.array([s.p_pos for s in states])
    low = np.minimum(positions.min(axis=(0, 1)) - 0.3, -1)
    high = np.maximum(positions.max(axis=(0, 1)) + 0.3, 1)
    visualizer.ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
    labels = [visualizer.ax.text(*positions[0, i], f"a{i}→m{assignments[i]}", fontsize=8) for i in range(3)]
    for i in range(3):
        visualizer.ax.text(*positions[0, i + 3], f"m{i}", fontsize=8)
    native_update = visualizer.update
    def update(frame):
        native_update(frame)
        for i, label in enumerate(labels):
            label.set_position(positions[frame, i])
        visualizer.ax.set_title(f"{title}\nStep {frame}; native mean reward {rewards[frame]:.3f}", fontsize=10)
    for frame in sorted({0, frames // 2, frames - 1}):
        update(frame)
        visualizer.fig.savefig(output / f"frame_{frame:04d}.png", dpi=120)
    animation = FuncAnimation(visualizer.fig, update, frames=frames, interval=1000 / fps, blit=False)
    animation.save(output / "replay.gif", writer=PillowWriter(fps=fps), dpi=85)
    plt.close(visualizer.fig)
    organization_figure(assignments, output / "organization.png", title)


def organization_figure(assignments, path, title):
    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    for i in range(3):
        ax.text(0, i, f"Agent {i}\nexecutor", ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "lightblue"})
        ax.text(2, i, f"Mission {i}", ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "wheat"})
        ax.annotate("", xy=(1.7, assignments[i]), xytext=(0.35, i), arrowprops={"arrowstyle": "->"})
    ax.set(xlim=(-0.5, 2.5), ylim=(-0.5, 2.5), title=title)
    ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison(paths, names, output, fps=10):
    traces = []
    for path in paths:
        with np.load(path) as trace:
            traces.append({k: trace[k].copy() for k in ("p_pos", "rewards", "assignments")})
    if not traces:
        return
    frames = min(len(t["p_pos"]) for t in traces)
    if not all(np.array_equal(traces[0]["p_pos"][0], t["p_pos"][0]) for t in traces[1:]):
        raise ValueError("Comparison traces do not share their initial physical state")
    fig, axes = plt.subplots(1, len(traces), figsize=(4 * len(traces), 4), squeeze=False, layout="constrained")
    positions = np.concatenate([t["p_pos"].reshape(-1, 2) for t in traces])
    low, high = positions.min(axis=0) - 0.3, positions.max(axis=0) + 0.3
    def update(frame):
        for ax, trace, name in zip(axes[0], traces, names):
            ax.clear()
            p = trace["p_pos"]
            for i in range(3):
                ax.plot(p[:frame + 1, i, 0], p[:frame + 1, i, 1], color=f"C{i}")
                ax.scatter(*p[frame, i], color=f"C{i}", s=70)
                ax.text(*p[frame, i], f"a{i}→m{trace['assignments'][i]}", fontsize=8)
            ax.scatter(p[frame, 3:, 0], p[frame, 3:, 1], marker="*", color="black", s=90)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal",
                   title=f"{name}\nstep {frame}; reward {trace['rewards'][frame]:.2f}")
    output.mkdir(parents=True, exist_ok=True)
    for frame in sorted({0, frames // 2, frames - 1}):
        update(frame)
        fig.savefig(output / f"comparison_{frame:04d}.png", dpi=120)
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    animation.save(output / "comparison.gif", writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)


def result_figures(output: Path):
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines() if line.strip()]
    statistics = json.loads((output / "statistics.json").read_text())
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for benchmark in sorted({r["benchmark"] for r in rows}):
        entries = [r for r in statistics["summaries"] if r["benchmark"] == benchmark]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        metrics = (("simple_regret", "Simple regret"), ("heldout_mae", "Held-out value MAE"),
                   ("harmful_reference", "Harmful net value (reference)"), ("abstention", "No-op rate"),
                   ("constraint_violation", "Constraint violation rate"), ("wall_seconds", "Method compute seconds"))
        for ax, (metric, label) in zip(axes.flat, metrics):
            for method in sorted({r["method"] for r in entries}):
                group = sorted([r for r in entries if r["method"] == method and metric in r["metrics"]], key=lambda r: r["budget"])
                if not group:
                    continue
                x = [r["budget"] for r in group]
                y = [r["metrics"][metric]["iqm"] for r in group]
                line, = ax.plot(x, y, marker="o", markersize=3, label=method)
                if all(r["metrics"][metric]["iqm_ci"] is not None for r in group):
                    ci = np.array([r["metrics"][metric]["iqm_ci"] for r in group])
                    ax.fill_between(x, ci[:, 0], ci[:, 1], color=line.get_color(), alpha=0.12)
            ax.set(xlabel="Allocated intervention budget", ylabel=label)
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{benchmark} — observed results; descriptive IQM/seed bootstrap")
        fig.savefig(figures / f"{benchmark}.png", dpi=160)
        fig.savefig(figures / f"{benchmark}.pdf")
        plt.close(fig)
        latest = max(r["budget"] for r in entries)
        methods = sorted({r["method"] for r in rows if r["benchmark"] == benchmark})
        seeds = sorted({r["seed"] for r in rows if r["benchmark"] == benchmark})
        heatmap = np.full((len(methods), len(seeds)), np.nan)
        for r in rows:
            if r["benchmark"] == benchmark and r["budget"] == latest:
                heatmap[methods.index(r["method"]), seeds.index(r["seed"])] = r["simple_regret"]
        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
        image = ax.imshow(heatmap, aspect="auto", cmap="viridis")
        ax.set(xticks=range(len(seeds)), xticklabels=seeds, yticks=range(len(methods)), yticklabels=methods,
               xlabel="Seed", title=f"{benchmark}: per-seed regret, budget {latest}")
        fig.colorbar(image, ax=ax)
        fig.savefig(figures / f"{benchmark}_regret_heatmap.png", dpi=160)
        plt.close(fig)
    training = list((output / "learners").glob("*/*/training.json")) if (output / "learners").exists() else []
    if training:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for path in sorted(training):
            data = json.loads(path.read_text())
            loss = [m.get("total_loss", m.get("loss")) for m in data["metrics"]]
            ax.plot(range(1, len(loss) + 1), [float(np.asarray(v).mean()) for v in loss], marker="o", label=f"{data['learner']}, seed {data['seed']}")
        ax.set(xlabel="Training update", ylabel="Algorithm-specific loss", title="Execution diagnostics — losses are not comparable performance scores")
        ax.legend(fontsize=7)
        fig.savefig(figures / "training_diagnostics.png", dpi=160)
        plt.close(fig)
    prediction_figures(output, figures)


def prediction_figures(output, figures):
    """Representative fixed first-seed displays; no outcome-based replay picking."""
    for benchmark in sorted((output / "runs").iterdir()):
        seeds = sorted(benchmark.glob("seed_*"), key=lambda p: int(p.name.removeprefix("seed_")))
        if not seeds:
            continue
        directory = seeds[0]
        oracle = json.loads((directory / "evaluation_oracle.json").read_text())
        public = json.loads((directory / "public.json").read_text())
        truth = np.array(oracle["values"])
        valid = np.array(public["valid"], dtype=bool)
        methods = []
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text())
            if data.get("status") == "complete" and path.stem != "oracle":
                last = data["decisions"][-1]
                if last["predictions"] is not None:
                    methods.append((path.stem, last))
        if not methods:
            continue
        cols = 4
        rows = (len(methods) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), squeeze=False, layout="constrained")
        for ax, (method, data) in zip(axes.flat, methods):
            predicted = np.array(data["predictions"])
            low = min(truth[valid].min(), predicted[valid].min())
            high = max(truth[valid].max(), predicted[valid].max())
            ax.scatter(truth[valid], predicted[valid], s=12)
            ax.plot([low, high], [low, high], linestyle=":", color="black")
            ax.set(title=method, xlabel="Oracle/reference value", ylabel="Predicted value")
        for ax in list(axes.flat)[len(methods):]:
            ax.axis("off")
        fig.suptitle(f"{benchmark.name}, {directory.name}: predictions at final budget (includes training interventions)")
        fig.savefig(figures / f"{benchmark.name}_predictions.png", dpi=150)
        plt.close(fig)
        if not benchmark.name.startswith("synthetic") or not (directory / "cora.json").exists():
            continue
        predicted = np.array(json.loads((directory / "cora.json").read_text())["decisions"][-1]["predictions"])
        bits = np.array(public["bits"])
        m = bits.shape[1]
        index = {tuple(row): i for i, row in enumerate(bits)}
        def pairwise(values):
            matrix = np.zeros((m, m))
            zero = tuple([0] * m)
            for i in range(m):
                for j in range(i + 1, m):
                    a, b, pair = np.zeros(m, dtype=int), np.zeros(m, dtype=int), np.zeros(m, dtype=int)
                    a[i], b[j], pair[i], pair[j] = 1, 1, 1, 1
                    matrix[i, j] = matrix[j, i] = values[index[tuple(pair)]] - values[index[tuple(a)]] - values[index[tuple(b)]] + values[index[zero]]
            return matrix
        true_pairs, estimated_pairs = pairwise(truth), pairwise(predicted)
        limit = max(np.abs(true_pairs).max(), np.abs(estimated_pairs).max(), 1e-8)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
        for ax, matrix, title in zip(axes, (true_pairs, estimated_pairs), ("True", "CORA estimate")):
            im = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="coolwarm")
            ax.set(title=title, xlabel="Edit", ylabel="Edit", xticks=range(m), yticks=range(m))
        fig.colorbar(im, ax=axes)
        fig.suptitle(f"{benchmark.name}: pairwise Möbius interactions\nHigher-order terms omitted; forbidden combinations remain oracle-only")
        fig.savefig(figures / f"{benchmark.name}_interactions.png", dpi=150)
        plt.close(fig)


def gallery(output):
    """A local HTML artifact linking saved results, screenshots and animation."""
    manifest = json.loads((output / "manifest.json").read_text())
    media = json.loads((output / "media_manifest.json").read_text()) if (output / "media_manifest.json").exists() else []
    parts = ['<!doctype html><html><meta charset="utf-8"><title>CORA benchmark results</title>',
             '<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;background:#f7f8fa;color:#16202b}img{max-width:100%;border-radius:8px}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}a{color:#005bb5}small{color:#516070}</style>',
             '<h1>CORA benchmark execution</h1>',
             '<p>Observed computational results. Technical runs do not establish convergence, causal identification, or certified safety. Replays use the configured seed, without selecting favorable outcomes.</p>',
             f'<p>Purpose: {html.escape(manifest["purpose"])}. Seeds: {html.escape(str(manifest["config"]["seeds"]))}. Budgets: {html.escape(str(manifest["config"]["budgets"]))}.</p>',
             '<p><a href="results.csv">Raw results CSV</a> · <a href="statistics.json">Statistics</a> · <a href="manifest.json">Execution manifest</a> · <a href="config.yaml">Configuration</a></p>',
             '<section><h2>Baseline status</h2><ul>']
    for name, description in manifest["methods"].items():
        parts.append(f'<li><b>{html.escape(name)}</b>: {html.escape(description)}</li>')
    for name, reason in manifest["unresolved_baselines"].items():
        parts.append(f'<li><b>{html.escape(name)} — unavailable</b>: {html.escape(reason)}</li>')
    parts.append('</ul></section><h2>Quantitative figures</h2><div class="grid">')
    for path in sorted((output / "figures").glob("*.png")):
        target = path.relative_to(output).as_posix()
        parts.append(f'<section><h3>{html.escape(path.stem)}</h3><a href="{target}"><img loading="lazy" src="{target}"></a></section>')
    parts.append('</div><h2>Matched comparisons</h2><div class="grid">')
    for path in sorted((output / "media").glob("*/comparison/comparison.gif")) if (output / "media").exists() else []:
        target = path.relative_to(output).as_posix()
        parts.append(f'<section><h3>{html.escape(path.parents[1].name)}</h3><a href="{target}"><img loading="lazy" src="{target}"></a></section>')
    parts.append('</div><h2>Individual benchmark replays</h2><div class="grid">')
    for entry in media:
        parts.append(f'<section><h3>{html.escape(entry["benchmark"])} / {html.escape(entry["method"])}</h3><small>Seed {entry["seed"]}</small><img loading="lazy" src="{entry["gif"]}"><p><a href="{entry["trace"]}">Raw trace</a> · ')
        parts.extend(f'<a href="{path}">Screenshot {i+1}</a> ' for i, path in enumerate(entry["screenshots"]))
        parts.append('</p></section>')
    parts.append('</div></html>')
    (output / "index.html").write_text("\n".join(parts))

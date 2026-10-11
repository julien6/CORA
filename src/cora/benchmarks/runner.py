"""Run, resume, analyze, and render finite CORA benchmark suites."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jaxmarl
import numpy as np
from omegaconf import OmegaConf

from cora.benchmarks.learners import FrozenPolicy, SOURCES, load_policy, train
from cora.benchmarks.media import comparison, gallery, render_trace, result_figures
from cora.benchmarks.methods import METHODS, UNRESOLVED, run_method
from cora.benchmarks.problems import MPETrial, SyntheticTrial
from cora.benchmarks.simulation import save_trace
from cora.benchmarks.statistics import aggregate
from cora.env_wrapper import OrganizationWrapper
from cora.interventions import apply
from cora.ippo_adapter import ROOT, UPSTREAM_REVISION
from cora.organization import OrganizationSpec


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_config(config):
    seeds, budgets = config["seeds"], config["budgets"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be unique uint32 integers")
    if not budgets or budgets != sorted(set(budgets)) or any(type(b) is not int or b < 1 for b in budgets):
        raise ValueError("Budgets must be unique increasing positive integers")
    if not config["methods"] or len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("Methods must be a nonempty unique list")
    for name in config["methods"]:
        if name in UNRESOLVED:
            raise ValueError(f"{name}: {UNRESOLVED[name]}")
        if name not in METHODS:
            raise ValueError(f"Unknown method: {name}")
    if any(b not in ("synthetic", "simple_spread") for b in config["benchmarks"]):
        raise ValueError("Unknown benchmark")
    if config["media"]["enabled"] and config["media"]["seed"] not in seeds:
        raise ValueError("Media seed must be one of the specified experiment seeds")
    if not 1 <= int(config["mpe"]["horizon"]) <= 25:
        raise ValueError("MPE finite-horizon evaluation must fit its 25-step episode")
    if not 0 <= config["mpe"]["gamma"] <= 1:
        raise ValueError("Discount must be in [0,1]")
    if not np.isfinite(config["mpe"]["u_noise"]) or config["mpe"]["u_noise"] < 0:
        raise ValueError("Nonnegative finite action noise required")
    if any(p not in ("guided", "random", *SOURCES) for p in config["mpe"]["policies"]):
        raise ValueError("Unknown continuation learner")
    if any(s not in ("sparse", "dense", "misspecified", "policy_shift") for s in config["synthetic"]["scenarios"]):
        raise ValueError("Unknown synthetic scenario")
    if int(config["media"]["fps"]) < 1 or int(config["statistics"]["bootstrap_samples"]) < 1:
        raise ValueError("Positive rendering/statistics budgets required")


def learner_settings(name, config):
    settings = dict(config["training"])
    budget = int(settings.pop("total_environment_steps"))
    environments, updates = int(settings["environments"]), int(settings["updates"])
    if min(environments, updates) < 1:
        raise ValueError("Positive training dimensions required")
    # Q implementations execute a replay-shape initialization trajectory before
    # their updates. Include it in the shared total environment interaction budget.
    divisor = environments * (updates + int(name in ("iql", "vdn", "qmix")))
    if budget % divisor or budget // divisor < 2:
        raise ValueError(f"Training budget {budget} cannot be divided exactly for {name}; divisor={divisor}")
    settings["steps"] = budget // divisor
    return settings


def evaluate(trial, method, decisions, logs, seed, config):
    public = trial.public
    truth = trial.truth
    net = truth - public.costs
    oracle = int(np.flatnonzero(public.valid)[np.argmax(net[public.valid])])
    rows = []
    threshold = float(config["estimation"]["regret_threshold"])
    reached = None
    for decision in decisions:
        index = decision["selected_index"]
        budget = decision["budget"]
        sampled = {r["index"] for r in logs if r["step"] <= budget}
        heldout = np.array([i for i in range(len(truth)) if public.valid[i] and i not in sampled], dtype=int)
        predictions = np.asarray(decision["predictions"]) if decision["predictions"] is not None else None
        error = np.abs(predictions[heldout] - truth[heldout]) if predictions is not None and len(heldout) else None
        effects = np.abs((predictions[heldout] - predictions[public.noop]) - (truth[heldout] - truth[public.noop])) if predictions is not None and len(heldout) else None
        radius = np.asarray(decision["diagnostic_radius"]) if decision["diagnostic_radius"] is not None else None
        regret = max(0.0, float(net[oracle] - net[index])) if public.valid[index] else None
        if reached is None and regret is not None and regret <= threshold:
            reached = budget
        if trial.metadata["truth_kind"] == "monte_carlo":
            oracle_raw = np.asarray(trial.metadata["oracle_values"])
            difference = oracle_raw[:, index] - oracle_raw[:, public.noop] - public.costs[index] + public.costs[public.noop]
            harm_se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        else:
            harm_se = 0.0
        row = {"benchmark": public.name, "method": method, "seed": seed, "budget": budget,
               "queries": decision["queries"], "selected_index": index, "selected_intervention": public.bits[index].tolist(),
               "policy_version": public.policy_version, "simple_regret": regret,
               "heldout_mae": float(error.mean()) if error is not None else None,
               "heldout_effect_mae": float(effects.mean()) if effects is not None else None,
               "heldout_count": len(heldout), "selected_net_value": float(net[index]),
               "noop_net_value": float(net[public.noop]), "oracle_net_value": float(net[oracle]),
               "harmful_reference": int(index != public.noop and net[index] < net[public.noop]),
               "harm_reference_standard_error": harm_se,
               "harm_sign_resolved_at_approx_95pct": bool(abs(net[index] - net[public.noop]) > 1.96 * harm_se) if index != public.noop else True,
               "constraint_violation": int(not public.valid[index]), "abstention": int(index == public.noop),
               "truth_kind": trial.metadata["truth_kind"], "oracle_samples": trial.oracle_samples,
               "oracle_standard_error": float(trial.oracle_standard_error[index]),
               "diagnostic_coverage": float(np.mean(np.abs(predictions[public.valid] - truth[public.valid]) <= radius[public.valid])) if radius is not None else None,
               "certified": decision["certified"], "design_rank": decision["rank"],
               "experiment_cost": float(decision["experiment_cost"]), "wall_seconds": decision["wall_seconds"],
               "budget_to_regret_threshold": reached}
        rows.append(row)
    return rows


def trial_run(trial, seed, config, output, resume):
    public = trial.public
    directory = output / "runs" / public.name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    # Oracle is isolated in an explicitly evaluation-only file, not method input.
    write_json(directory / "evaluation_oracle.json", {"values": trial.truth.tolist(), "standard_errors": trial.oracle_standard_error.tolist(),
                                           "metadata": trial.metadata, "privileged": True})
    write_json(directory / "public.json", {"bits": public.bits.tolist(), "costs": public.costs.tolist(),
                                           "valid": public.valid.tolist(), "family": [list(t) for t in public.family],
                                           "policy_version": public.policy_version})
    for method in config["methods"]:
        if method == "cora_no_constraints" and public.name.startswith("simple_spread"):
            continue
        path = directory / f"{method}.json"
        if resume and path.exists():
            continue
        print(f"{public.name} seed={seed}: {method}", flush=True)
        try:
            if method == "oracle":
                net = trial.truth - public.costs
                index = int(np.flatnonzero(public.valid)[np.argmax(net[public.valid])])
                decisions = [{"budget": b, "selected_index": index, "predictions": trial.truth.tolist(), "diagnostic_radius": None,
                              "rank": 0, "certified": False, "queries": 0, "experiment_cost": 0, "wall_seconds": 0} for b in config["budgets"]]
                logs = []
            else:
                decisions, logs = run_method(method, public, trial.observe, config["budgets"], seed, config["estimation"])
            rows = evaluate(trial, method, decisions, logs, seed, config)
            write_json(path, {"status": "complete", "rows": rows, "decisions": decisions, "experiments": logs})
            if isinstance(trial, MPETrial) and seed == config["media"]["seed"] and config["media"]["enabled"]:
                index = decisions[-1]["selected_index"]
                trace = trial.trace(index, 900000)
                org = apply(trial.env.spec, trial.base.organization, trial.options[index])
                save_trace(directory / f"{method}_trace.npz", trace, org.assignments)
        except Exception:
            with (output / "failures.jsonl").open("a") as stream:
                stream.write(json.dumps({"benchmark": public.name, "seed": seed, "method": method, "traceback": traceback.format_exc()}) + "\n")
            raise


def analyze(output, config):
    rows = []
    for path in sorted((output / "runs").glob("*/*/*.json")):
        data = json.loads(path.read_text())
        if data.get("status") == "complete":
            rows.extend(data["rows"])
    if not rows:
        raise ValueError("No completed benchmark results found")
    (output / "results.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "statistics.json", aggregate(rows, config["statistics"]))
    result_figures(output)
    return rows


def render_media(output, config, force=False):
    if not config["media"]["enabled"]:
        return []
    manifest = []
    for directory in sorted((output / "runs").glob("simple_spread_*/seed_*")):
        paths = sorted(directory.glob("*_trace.npz"))
        if not paths:
            continue
        benchmark = directory.parent.name
        seed = int(directory.name.removeprefix("seed_"))
        for path in paths:
            method = path.stem.removesuffix("_trace")
            target = output / "media" / benchmark / directory.name / method
            if force or not (target / "replay.gif").exists() or len(list(target.glob("frame_*.png"))) != 3 or not (target / "organization.png").exists():
                print(f"Rendering {benchmark}: {method}", flush=True)
                render_trace(path, target, f"{benchmark}, seed {seed}, {method}; budget {max(config['budgets'])}", int(config["media"]["fps"]))
            manifest.append({"benchmark": benchmark, "seed": seed, "method": method,
                             "trace": str(path.relative_to(output)), "gif": str((target / "replay.gif").relative_to(output)),
                             "screenshots": [str(p.relative_to(output)) for p in sorted(target.glob("frame_*.png"))]})
        names = config["media"]["comparison_methods"]
        chosen = [(directory / f"{name}_trace.npz", name) for name in names if (directory / f"{name}_trace.npz").exists()]
        target = output / "media" / benchmark / "comparison"
        if chosen and (force or not (target / "comparison.gif").exists() or len(list(target.glob("comparison_*.png"))) != 3):
            comparison([p for p, n in chosen], [n for p, n in chosen], target, int(config["media"]["fps"]))
    write_json(output / "media_manifest.json", manifest)
    return manifest


def run(config, resume=False, render_only=False, analyze_only=False):
    validate_config(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    signature_config = {k: v for k, v in config.items() if k != "output"}
    signature = hashlib.sha256(json.dumps(signature_config, sort_keys=True).encode()).hexdigest()
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / "src/cora").rglob("*.py"))}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["config_sha256"] != signature:
            raise ValueError("Output contains a different configuration; choose a fresh output directory")
        if not (resume or render_only or analyze_only):
            raise ValueError("Output already exists; pass --resume or select a fresh output directory")
        if resume and existing.get("source_sha256") is not None and existing["source_sha256"] != source_hashes:
            raise ValueError("Source changed since execution; use a fresh destination for experiments or --analyze-only for saved results")
    elif render_only or analyze_only:
        raise ValueError("Rendering/analysis requires an existing execution manifest")
    manifest = {"schema_version": 1, "config_sha256": signature, "config": signature_config,
                "purpose": config["purpose"], "upstream_revision": UPSTREAM_REVISION,
                "hardware": {"platform": platform.platform(), "python": platform.python_version(), "devices": [str(d) for d in jax.devices()]},
                "dependencies": {name: version(name) for name in ("jax", "jaxlib", "jaxmarl", "flax", "flashbax", "optax", "numpy", "pillow")},
                "methods": {name: METHODS[name] for name in config["methods"]}, "unresolved_baselines": UNRESOLVED,
                "unsupported_combinations": {"cora_no_constraints": ["simple_spread"]}, "status": "running"}
    manifest["source_sha256"] = source_hashes
    if not (render_only or analyze_only):
        write_json(manifest_path, manifest)
        OmegaConf.save(OmegaConf.create(config), output / "config.yaml")
    start = time.perf_counter()
    try:
        if not (render_only or analyze_only):
            for seed in config["seeds"]:
                if "synthetic" in config["benchmarks"]:
                    for scenario in config["synthetic"]["scenarios"]:
                        trial_run(SyntheticTrial(config["synthetic"], seed, scenario), seed, config, output, resume)
                if "simple_spread" in config["benchmarks"]:
                    env = OrganizationWrapper(jaxmarl.make("MPE_simple_spread_v3", u_noise=np.full(3, config["mpe"]["u_noise"])), OrganizationSpec())
                    for name in config["mpe"]["policies"]:
                        if name in ("guided", "random"):
                            policy = FrozenPolicy(name, tuple(env.agents), version=f"{name}-fixed-v1")
                        else:
                            location = output / "learners" / name / f"seed_{seed}"
                            if resume and (location / "training.json").exists():
                                policy = load_policy(location, env)
                            else:
                                print(f"Training upstream {name}, seed={seed}", flush=True)
                                policy, _ = train(name, env, seed, learner_settings(name, config), location)
                        trial_run(MPETrial(env, policy, config["mpe"], seed), seed, config, output, resume)
                        # Compiled graphs are disposable; keep checkpointed actors.
                        del policy
                        jax.clear_caches()
        rows = analyze(output, config)
        media = [] if analyze_only else render_media(output, config, force=render_only)
        if not (render_only or analyze_only):
            manifest.update(status="complete", result_rows=len(rows), media_replays=len(media), wall_seconds=time.perf_counter() - start)
            write_json(manifest_path, manifest)
        gallery(output)
        return {"output": str(output), "result_rows": len(rows), "media_replays": len(media)}
    except Exception:
        if not (render_only or analyze_only):
            manifest.update(status="failed", traceback=traceback.format_exc(), wall_seconds=time.perf_counter() - start)
            write_json(manifest_path, manifest)
        raise


def main():
    import sys
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config',type=Path)
    requested,_=probe.parse_known_args()
    if len(sys.argv)==1 or (requested.config and 'domains' in OmegaConf.load(requested.config)):
        from cora.benchmarks.release import main as native_main
        return native_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bench_smoke.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--list", action="store_true", help="Show methods, learners, and unresolved paper baselines")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({"methods": METHODS, "learners": SOURCES, "unresolved": UNRESOLVED}, indent=2))
        return
    config = OmegaConf.to_container(OmegaConf.load(args.config))
    if args.output:
        config["output"] = str(args.output)
    print(json.dumps(run(config, args.resume, args.render_only, args.analyze_only), indent=2))


if __name__ == "__main__":
    main()

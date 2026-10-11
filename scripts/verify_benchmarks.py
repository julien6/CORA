"""Audit a completed suite's coverage, budgets, checkpoints and media artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jaxmarl
import numpy as np
from PIL import Image

from cora.benchmarks.learners import load_policy
from cora.env_wrapper import OrganizationWrapper
from cora.ippo_adapter import ROOT
from cora.organization import OrganizationSpec


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    config = manifest["config"]
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    expected = 0
    if "synthetic" in config["benchmarks"]:
        expected += len(config["synthetic"]["scenarios"]) * len(config["methods"])
    if "simple_spread" in config["benchmarks"]:
        count = len([m for m in config["methods"] if m != "cora_no_constraints"])
        expected += len(config["mpe"]["policies"]) * count
    expected *= len(config["seeds"]) * len(config["budgets"])
    assert len(rows) == expected
    assert len({(r["benchmark"], r["method"], r["seed"], r["budget"]) for r in rows}) == expected
    for path in sorted((output / "runs").glob("*/*/evaluation_oracle.json")):
        reference = json.loads(path.read_text())
        assert reference["privileged"] and np.isfinite(reference["values"]).all()
        comparator = path.parent / "oracle.json"
        if comparator.exists():
            logged = json.loads(comparator.read_text())
            np.testing.assert_array_equal(reference["values"], logged["decisions"][-1]["predictions"])
        if reference["metadata"]["truth_kind"] == "monte_carlo":
            np.testing.assert_allclose(np.mean(reference["metadata"]["oracle_values"], axis=0), reference["values"])
    for row in rows:
        assert not row["certified"]
        assert row["queries"] == (0 if row["method"] in ("no_op", "oracle") else row["budget"])
        if row["method"] != "cora_no_constraints":
            assert row["constraint_violation"] == 0
        if row["method"] == "oracle":
            assert row["simple_regret"] == 0
        for value in row.values():
            if isinstance(value, float):
                assert np.isfinite(value)
    checkpoints = []
    for path in sorted((output / "learners").glob("*/*/training.json")):
        data = json.loads(path.read_text())
        total = data["training_environment_steps"] + data["buffer_initialization_steps"]
        assert total == config["training"]["total_environment_steps"]
        assert data["optimizer_updates"] >= 1 and data["finite"]
        env = OrganizationWrapper(jaxmarl.make("MPE_simple_spread_v3", u_noise=np.full(3, config["mpe"]["u_noise"])), OrganizationSpec())
        policy = load_policy(path.parent, env)
        obs, _ = env.reset(jax.random.PRNGKey(data["seed"]))
        stream = jax.random.fold_in(jax.random.PRNGKey(data["seed"]), 900000)
        _, action_key, _ = jax.random.split(stream, 3)
        actions, hidden = policy.act(obs, policy.initial_hidden(), np.zeros(3, dtype=bool), action_key)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves((actions, hidden)))
        trace = output / "runs" / f"simple_spread_{data['learner']}" / f"seed_{data['seed']}" / "no_op_trace.npz"
        if trace.exists():
            with np.load(trace) as replay:
                np.testing.assert_array_equal([actions[a] for a in env.agents], replay["actions"][0])
        checkpoints.append({"learner": data["learner"], "seed": data["seed"], "total_environment_steps": total,
                            "optimizer_updates": data["optimizer_updates"], "checkpoint_loaded": True})
    gifs = list((output / "media").rglob("*.gif")) if (output / "media").exists() else []
    for path in gifs:
        with Image.open(path) as image:
            assert image.is_animated and image.n_frames == config["mpe"]["horizon"]
    screenshots = list((output / "media").rglob("frame_*.png")) if (output / "media").exists() else []
    for path in screenshots:
        with Image.open(path) as image:
            image.verify()
    result = {"result_rows": len(rows), "expected_result_rows": expected,
              "unique_result_keys": True, "finite_metrics": True, "budgets_match": True,
              "checkpoints": checkpoints, "animated_gifs": len(gifs), "individual_screenshots": len(screenshots),
              "implementation_snapshot_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in sorted((ROOT / "src/cora/benchmarks").glob("*.py"))},
              "purpose": "technical_verification; not scientific validation"}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(verify(parser.parse_args().output), indent=2))

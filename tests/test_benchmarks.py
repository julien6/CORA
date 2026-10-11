from copy import deepcopy
import json
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest

from cora.benchmarks.learners import FrozenPolicy, load_policy, train
from cora.benchmarks.methods import METHODS, acquisition, ridge_fit, run_method
from cora.benchmarks.problems import PublicProblem, SyntheticTrial, encode
from cora.benchmarks.runner import ROOT, learner_settings, run, validate_config
from cora.benchmarks.simulation import simulate, save_trace
from cora.benchmarks.media import render_trace, comparison
from cora.benchmarks.statistics import holm, iqm, summarize


@pytest.fixture
def bench_config(tmp_path):
    config = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/bench_smoke.yaml"))
    config["output"] = str(tmp_path / "suite")
    config["seeds"] = [7]
    config["budgets"] = [2, 4]
    config["benchmarks"] = ["synthetic"]
    config["synthetic"]["scenarios"] = ["sparse"]
    config["statistics"]["bootstrap_samples"] = 10
    config["media"]["enabled"] = False
    return config


def test_ground_truth_is_separate_and_constraints(bench_config):
    trial = SyntheticTrial(bench_config["synthetic"], 7)
    assert not hasattr(trial.public, "truth")
    assert not hasattr(trial.public, "coefficients")
    assert np.any(~trial.public.valid)
    assert trial.public.valid[trial.public.noop]
    repeat = SyntheticTrial(bench_config["synthetic"], 7)
    np.testing.assert_array_equal(trial.truth, repeat.truth)
    assert trial.observe(1, 0) == repeat.observe(1, 0)
    shift = SyntheticTrial(bench_config["synthetic"], 7, "policy_shift")
    np.testing.assert_array_equal(shift.reference_truth, trial.truth)
    assert not np.array_equal(shift.truth, trial.truth)


@pytest.mark.parametrize("method", list(METHODS.keys() - {"oracle", "cora_conditional"}))
def test_every_intervention_method_budget_and_reproducibility(method, bench_config):
    trial = SyntheticTrial(bench_config["synthetic"], 7)
    # Only callback and public structure are passed, never evaluation coefficients.
    decisions, logs = run_method(method, trial.public, trial.observe, [2, 4], 7, bench_config["estimation"])
    assert [d["budget"] for d in decisions] == [2, 4]
    assert len(logs) == (0 if method == "no_op" else 4)
    if method != "cora_no_constraints":
        assert all(trial.public.valid[d["selected_index"]] for d in decisions)
        assert all(log["admissible"] for log in logs)
    assert all(not d["certified"] for d in decisions)
    repeat, second_logs = run_method(method, trial.public, trial.observe, [2, 4], 7, bench_config["estimation"])
    assert logs == second_logs
    for first, second in zip(decisions, repeat):
        assert {k: v for k, v in first.items() if k != "wall_seconds"} == {k: v for k, v in second.items() if k != "wall_seconds"}


def test_decision_focused_acquisition_matches_direct_inverse():
    f = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
    fit = ridge_fit(f, [0], [0], 0.1, 100)
    scores = acquisition(f, fit, np.arange(3), np.zeros(3))
    diffs = (f[:, None] - f[None, :]).reshape(-1, 2)
    precision = f[[0]].T @ f[[0]] + 0.1 * np.eye(2)
    expected = []
    for row in f:
        inverse = np.linalg.inv(precision + np.outer(row, row))
        expected.append(np.einsum("ij,jk,ik->i", diffs, inverse, diffs).max())
    np.testing.assert_allclose(scores, expected)


def test_statistics_fractional_iqm_and_holm():
    assert iqm([0, 1, 2, 3]) == 1.5
    assert iqm([0, 1, 100]) == pytest.approx((0 / 4 + 1 + 100 / 4) / 1.5)
    np.testing.assert_allclose(holm([0.01, 0.03, 0.5]), [0.03, 0.06, 0.5])
    assert summarize([1], 10)["iqm_ci"] is None
    assert summarize([1, 2, 3], 10) == summarize([1, 2, 3], 10)


def test_matched_total_training_budget(bench_config):
    for name in ("ippo", "mappo", "iql", "vdn", "qmix"):
        settings = learner_settings(name, bench_config)
        total = settings["steps"] * settings["environments"] * (settings["updates"] + int(name in ("iql", "vdn", "qmix")))
        assert total == bench_config["training"]["total_environment_steps"]


def test_suite_results_resume_and_oracle(bench_config):
    result = run(bench_config)
    output = Path(bench_config["output"])
    assert result["result_rows"] == 2 * len(bench_config["methods"])
    assert "values" in json.loads((output / "runs/synthetic_sparse/seed_7/evaluation_oracle.json").read_text())
    assert "decisions" in json.loads((output / "runs/synthetic_sparse/seed_7/oracle.json").read_text())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert all(r["simple_regret"] == 0 for r in rows if r["method"] == "oracle")
    assert all(r["queries"] == 0 for r in rows if r["method"] in ("oracle", "no_op"))
    assert (output / "figures/synthetic_sparse.png").exists()
    assert (output / "figures/synthetic_sparse_predictions.png").exists()
    assert (output / "figures/synthetic_sparse_interactions.png").exists()
    assert "published_compositional" in (output / "index.html").read_text()
    before = (output / "runs/synthetic_sparse/seed_7/cora.json").read_bytes()
    run(bench_config, resume=True)
    assert before == (output / "runs/synthetic_sparse/seed_7/cora.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        run(bench_config)
    changed = deepcopy(bench_config)
    changed["budgets"] = [3, 4]
    with pytest.raises(ValueError, match="different configuration"):
        run(changed, resume=True)


def test_native_replay_and_screenshots(env, tmp_path):
    _, state = env.reset(jax.random.PRNGKey(7))
    policy = FrozenPolicy("guided", tuple(env.agents))
    trace = simulate(env, policy, state, jax.random.PRNGKey(8), 4, 0.99)
    path = tmp_path / "trace.npz"
    save_trace(path, trace, state.organization.assignments)
    render_trace(path, tmp_path / "media", "test")
    with Image.open(tmp_path / "media/replay.gif") as gif:
        assert gif.n_frames == 4
        assert gif.is_animated
    assert len(list((tmp_path / "media").glob("frame_*.png"))) == 3
    comparison([path, path], ["first", "second"], tmp_path / "comparison")
    with Image.open(tmp_path / "comparison/comparison.gif") as gif:
        assert gif.n_frames == 4


def test_reject_unidentified_external_methods(bench_config):
    bench_config["methods"] = ["published_compositional"]
    with pytest.raises(ValueError, match="not identified"):
        validate_config(bench_config)

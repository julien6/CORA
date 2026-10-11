import json
from pathlib import Path

import pytest

from cora.poc import run
from cora.rollout import load_dataset, save_dataset


def test_pipeline_repeatability_and_serialization(config, tmp_path):
    # No learner updates here: full acceptance invokes upstream update once below.
    one = run(config, run_ippo=False)
    rows = load_dataset(tmp_path / "dataset.json")
    assert len(rows) == 8 and one["design_rank"] == 8
    assert (tmp_path / "poc.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads((tmp_path / "summary.json").read_text()) == one
    config["output"] = str(tmp_path / "repeat")
    two = run(config, run_ippo=False)
    assert one == two
    assert rows == load_dataset(tmp_path / "repeat/dataset.json")
    config["seed"] += 1
    config["output"] = str(tmp_path / "second_key")
    other = run(config, run_ippo=False)
    assert other["continuation_returns"] != one["continuation_returns"]
    rows[0]["organization"] = [0, 0, 0]
    with pytest.raises(ValueError):
        save_dataset(tmp_path / "invalid.json", rows)


def test_full_pipeline_one_ippo_update(config):
    result = run(config)
    assert result["ippo"]["optimizer_updates"] == 1
    assert result["ippo"]["parameter_delta_l2"] > 0
    assert result["ippo"]["applied_gradient_l2"] > 0
    assert result["ippo"]["finite"]


def test_runtime_bounds(config):
    config["horizon"] = 100
    with pytest.raises(ValueError):
        run(config)

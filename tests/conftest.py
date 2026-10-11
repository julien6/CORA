import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jaxmarl
import pytest
from omegaconf import OmegaConf

from cora.env_wrapper import OrganizationWrapper
from cora.ippo_adapter import ROOT
from cora.organization import OrganizationSpec


@pytest.fixture(scope="session")
def env():
    return OrganizationWrapper(jaxmarl.make("MPE_simple_spread_v3"), OrganizationSpec())


@pytest.fixture
def config(tmp_path):
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/poc.yaml"))
    cfg["output"] = str(tmp_path)
    return cfg


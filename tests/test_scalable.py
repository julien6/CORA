import numpy as np
from cora.benchmarks.scalable import ScalableSynthetic,candidate_pool


def test_large_backend_without_enumeration():
    trial=ScalableSynthetic(edits=64,agents=64,order=4,candidates=80)
    assert trial.public.bits.shape == (80,64)
    assert len(trial.public.family)<=16*16
    assert np.isfinite(trial.observe(0,0))
    assert not hasattr(trial.public,'_theta')
    assert trial.diagnostics['global_optimization_gap'] is None
    assert not np.any(trial.public.bits[:,0]*trial.public.bits[:,1])
    bits,_=candidate_pool(64,size=80)
    assert bits.shape == (80,64)

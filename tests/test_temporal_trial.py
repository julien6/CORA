from cora.benchmarks.temporal_trial import ReservoirSCM


def test_locality_and_shared_coupling():
    assert (0,1) not in ReservoirSCM().graph().encoder().family
    assert (0,1) in ReservoirSCM(coupling=True).graph().encoder().family
    for coupling in (False,True):
        result = ReservoirSCM(coupling=coupling).run(budget=16)
        assert result['simultaneous_coverage']
        assert len(result['queries']) == 16
        truth = result['evaluation_only_values']
        chosen = result['selected']
        assert truth[chosen] - .1 * chosen.bit_count() >= truth[0]

import numpy as np
from cora.benchmarks.evaluation import recovery_time,performance_profile,structure_accuracy,simultaneous_coverage


def test_metrics_definitions():
    assert recovery_time([10,10,0,9,10],2,10)==1
    assert recovery_time([10,10,0,8,7],2,10) is None
    np.testing.assert_allclose(performance_profile([0,1,2],[0,1,2]),[1/3,2/3,1])
    assert structure_accuracy([(0,),(1,),(0,1)],[(0,),(1,)])['false_omissions']==1
    assert simultaneous_coverage([1,2],[1,2],[0,0])

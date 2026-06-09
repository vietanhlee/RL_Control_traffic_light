from metrics.calculator import MetricsCalculator
from metrics.collector import DirectionMetrics, GroupMetrics


def test_local_imbalance_formula():
    data = {
        1: DirectionMetrics(motorcycle=GroupMetrics(queue_length=2)),
        2: DirectionMetrics(motorcycle=GroupMetrics(queue_length=6)),
        3: DirectionMetrics(motorcycle=GroupMetrics(queue_length=4)),
    }
    result = MetricsCalculator.local_imbalance(data)
    # avg = 4, imbalance = |2-4| + |6-4| + |4-4| = 4
    assert result.avg_queue == 4
    assert result.local_imbalance == 4


def test_global_imbalance_sum():
    assert MetricsCalculator.global_imbalance({1: 3.0, 2: 2.5}) == 5.5

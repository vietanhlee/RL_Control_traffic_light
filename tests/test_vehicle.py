from simulation.road import RoadNetwork


def test_shortest_path_exists():
    network = RoadNetwork.from_defaults()
    path = network.shortest_path(0, 4)
    assert path[0] == 0
    assert path[-1] == 4
    assert len(path) >= 2

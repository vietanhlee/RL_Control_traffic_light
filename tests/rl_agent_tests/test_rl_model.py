from rl_agent.traffic_rl.agent import QMIXAgent
from rl_agent.traffic_rl.features import build_features


def test_feature_vector_has_stable_size():
    observation = {
        "time": 12.5,
        "local_imbalance": 3.0,
        "global_imbalance": 8.0,
        "light_states": {"1": "GREEN", "2": "RED"},
        "current_phase": 0,
        "incoming_nodes": [1, 2],
        "directions": {
            "1": {
                "queue_length": 4,
                "motorcycle_density": 1.0,
                "car_density": 2.0,
                "motorcycle_avg_speed": 8.0,
                "car_avg_speed": 10.0,
            },
            "2": {
                "queue_length": 2,
                "motorcycle_density": 0.5,
                "car_density": 1.5,
                "motorcycle_avg_speed": 6.0,
                "car_avg_speed": 9.0,
            },
        },
    }

    features = build_features(observation)
    assert len(features) == 12
    # queue_1 / 40.0 = 4 / 40.0 = 0.1
    assert features[0] == 0.1


def test_qmix_agent_selects_actions_and_persists(tmp_path):
    agent = QMIXAgent(
        n_agents=2,
        obs_dim=8,
        learning_rate=0.1,
        gamma=0.9,
        epsilon=0.0,
        batch_size=2,
        buffer_capacity=8,
        target_update_freq=1,
        seed=7,
    )

    observations = {
        1: [1.0, 0.2, 0.0, 0.5, 0.1, 0.0, 0.0, 1.0],
        2: [1.0, 0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
    }
    actions = agent.select_actions(observations, agent_ids=[1, 2], explore=False)

    assert set(actions) == {1, 2}
    assert all(action in (0, 1) for action in actions.values())

    model_path = tmp_path / "qmix_agent.pth"
    agent.save(model_path)

    loaded = QMIXAgent.load(
        model_path,
        default_n_agents=2,
        default_obs_dim=8,
    )

    assert loaded.n_agents == 2
    assert loaded.obs_dim == 8

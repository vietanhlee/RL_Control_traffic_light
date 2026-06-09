import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RL model"))

from traffic_rl.agent import LinearQAgent
from traffic_rl.features import build_features


def test_feature_vector_has_stable_size():
    observation = {
        "time": 12.5,
        "local_imbalance": 3.0,
        "global_imbalance": 8.0,
        "light_states": {"1": "GREEN", "2": "RED"},
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
    assert len(features) > 0
    assert features[0] == 1.0


def test_agent_update_changes_weights():
    agent = LinearQAgent(feature_size=8, learning_rate=0.1, gamma=0.9, epsilon=0.0)
    state = [1.0, 0.2, 0.0, 0.5, 0.1, 0.0, 0.0, 1.0]
    next_state = [1.0, 0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0]

    before = agent.q_values(state)
    agent.update(state, 1, reward=-2.0, next_features=next_state)
    after = agent.q_values(state)

    assert before != after

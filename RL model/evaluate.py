"""
evaluate.py – Chạy QMIX policy đã train để đánh giá hiệu suất.

Khác với training, evaluate chạy pure greedy (explore=False) và không
cập nhật weights. Dùng để kiểm tra policy đã học hoạt động tốt không
trên backend simulation đang chạy.
"""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from traffic_rl.agent import QMIXAgent
from traffic_rl.client import ApiError, TrafficApiClient
from traffic_rl.config import DEFAULT_BASE_URL, DEFAULT_DECISION_INTERVAL_SECONDS, DEFAULT_MODEL_PATH
from traffic_rl.environment import TrafficEnvironment
from traffic_rl.features import build_features


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng argument parser CLI cho evaluate script."""
    parser = argparse.ArgumentParser(
        description="Đánh giá QMIX policy đã train trên backend simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL của backend API")
    parser.add_argument("--decision-interval", type=float, default=DEFAULT_DECISION_INTERVAL_SECONDS)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Đường dẫn file .pth")
    parser.add_argument("--steps", type=int, default=500, help="Số step đánh giá")
    return parser


def main() -> int:
    """Chạy QMIX evaluation loop.

    Returns:
        0 nếu thành công, 1 nếu lỗi kết nối.
    """
    args = build_parser().parse_args()
    client = TrafficApiClient(base_url=args.base_url)
    env = TrafficEnvironment(
        client=client,
        decision_interval_seconds=args.decision_interval,
    )

    try:
        env.bootstrap()
        initial_obs = env.observe_all()
    except ApiError as exc:
        print(f"[ERROR] {exc}")
        return 1

    agent_ids = env.intersection_ids
    n_agents = len(agent_ids)
    obs_dim = len(build_features(next(iter(initial_obs.values()))))

    # Tải model (không update weights trong evaluate)
    agent = QMIXAgent.load(
        args.model_path,
        default_n_agents=n_agents,
        default_obs_dim=obs_dim,
    )

    print(f"[INFO] QMIX Evaluate | agents={n_agents} | obs_dim={obs_dim}")
    print(f"[INFO] Model: {args.model_path} | Steps: {args.steps}")
    print("-" * 60)

    observation = initial_obs
    queue_history: list[float] = []
    reward_history: list[float] = []

    pbar = tqdm(range(1, args.steps + 1), desc="Evaluate", unit="step", ncols=100, colour="green")

    for step in pbar:
        obs_array = np.array(
            [build_features(observation[aid]) for aid in agent_ids],
            dtype=np.float32,
        )

        # Chọn action greedy (explore=False)
        raw_actions = agent.select_actions(
            {aid: obs_array[i].tolist() for i, aid in enumerate(agent_ids)},
            agent_ids=agent_ids,
            explore=False,
        )
        actions = {
            aid: (0 if env.hold_required(aid) else raw_actions[aid])
            for aid in agent_ids
        }

        env.apply_actions(actions)
        observation = env.advance()

        step_queue = 0.0
        step_rewards: list[float] = []
        for aid in agent_ids:
            obs = observation[aid]
            step_rewards.append(env.reward_for(obs, actions.get(aid, 0)))
            step_queue += float(obs.get("queue_total", 0.0))

        avg_queue = step_queue / max(n_agents, 1)
        joint_reward = float(np.mean(step_rewards))
        queue_history.append(avg_queue)
        reward_history.append(joint_reward)

        if step % 20 == 0:
            pbar.set_postfix(
                {
                    "r̄": f"{sum(reward_history[-20:])/min(20,len(reward_history)):+.3f}",
                    "q̄": f"{sum(queue_history[-20:])/min(20,len(queue_history)):.1f}",
                },
                refresh=True,
            )

    pbar.close()
    print(f"\n[DONE] avg_joint_reward={sum(reward_history)/len(reward_history):.3f} | avg_queue={sum(queue_history)/len(queue_history):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

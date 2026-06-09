"""
train.py – Vòng lặp huấn luyện QMIX cho hệ thống 16 nút giao thông.

Quy trình mỗi step:
  1. observe_all()       → dict[agent_id → local_obs]
  2. select_actions()    → dict[agent_id → action]  (ε-greedy, decentralized)
  3. apply_actions()     → gửi hành động lên backend
  4. advance()           → chờ decision_interval, nhận next_obs
  5. reward_for()        → tính reward từng nút → mean → joint_reward
  6. buffer.push()       → lưu 1 joint transition
  7. agent.update()      → 1 bước QMIX gradient descent
  8. decay_epsilon()     → giảm exploration rate

Khác với DQN cũ:
  - Thay vì update N lần (1 per agent), QMIX chỉ update 1 lần/step với joint loss
  - Reward là mean của tất cả agents (cooperative)
  - tqdm postfix hiển thị: epsilon, loss, avg_reward, avg_queue
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from traffic_rl.agent import QMIXAgent
from traffic_rl.client import ApiError, TrafficApiClient
from traffic_rl.config import (
    DEFAULT_BASE_URL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER_CAPACITY,
    DEFAULT_DECISION_INTERVAL_SECONDS,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_GAMMA,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HISTORY_WINDOW,
    DEFAULT_LR,
    DEFAULT_MIN_EPSILON,
    DEFAULT_MIN_PHASE_HOLD_STEPS,
    DEFAULT_MIXING_HIDDEN_DIM,
    DEFAULT_MODEL_PATH,
    DEFAULT_N_AGENTS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_TARGET_UPDATE_FREQ,
)
from traffic_rl.environment import TrafficEnvironment
from traffic_rl.features import build_features


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng argument parser CLI cho QMIX training.

    Returns:
        ArgumentParser đã cấu hình đầy đủ các tham số QMIX.
    """
    parser = argparse.ArgumentParser(
        description="Train QMIX agent cho hệ thống điều khiển đèn giao thông 16 nút",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Backend
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL của backend API")
    parser.add_argument("--steps", type=int, default=5000, help="Tổng số training steps")
    parser.add_argument(
        "--decision-interval",
        type=float,
        default=DEFAULT_DECISION_INTERVAL_SECONDS,
        help="Giây chờ giữa các quyết định",
    )
    # Model
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Đường dẫn lưu/tải model (.pth)")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Lưu model mỗi N step")
    # QMIX hyperparams
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (Adam)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor γ")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Epsilon exploration ban đầu")
    parser.add_argument("--min-epsilon", type=float, default=DEFAULT_MIN_EPSILON, help="Epsilon tối thiểu")
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULT_EPSILON_DECAY, help="Hệ số suy giảm epsilon")
    parser.add_argument("--min-phase-hold", type=int, default=DEFAULT_MIN_PHASE_HOLD_STEPS, help="Số step giữ pha tối thiểu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Kích thước mini-batch")
    parser.add_argument("--buffer-capacity", type=int, default=DEFAULT_BUFFER_CAPACITY, help="Capacity của joint replay buffer")
    parser.add_argument("--target-update-freq", type=int, default=DEFAULT_TARGET_UPDATE_FREQ, help="Hard-update target nets mỗi N updates")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dim của Q-network")
    parser.add_argument("--mixing-hidden-dim", type=int, default=DEFAULT_MIXING_HIDDEN_DIM, help="Hidden dim của Mixing network")
    # Misc
    parser.add_argument("--reset-first", action="store_true", help="Reset backend trước khi training")
    parser.add_argument("--no-explore", action="store_true", help="Tắt ε-greedy (pure greedy)")
    return parser


def obs_dict_to_array(
    obs_dict: dict[int, dict[str, Any]],
    agent_ids: list[int],
) -> np.ndarray:
    """Chuyển dict observations sang numpy array (n_agents, obs_dim).

    Args:
        obs_dict  : Dict mapping agent_id → observation dict từ backend.
        agent_ids : Danh sách agent IDs theo thứ tự cố định.

    Returns:
        numpy array shape (n_agents, obs_dim).
    """
    return np.array(
        [build_features(obs_dict[aid]) for aid in agent_ids],
        dtype=np.float32,
    )


def main() -> int:
    """Điểm vào chính của QMIX training.

    Returns:
        0 nếu thành công, 1 nếu lỗi kết nối.
    """
    args = build_parser().parse_args()

    # ── Khởi tạo client và môi trường ─────────────────────────────────────
    client = TrafficApiClient(base_url=args.base_url)
    env = TrafficEnvironment(
        client=client,
        decision_interval_seconds=args.decision_interval,
        min_phase_hold_steps=args.min_phase_hold,
    )

    try:
        env.bootstrap()
        if args.reset_first:
            env.reset()
    except ApiError as exc:
        print(f"[ERROR] Không thể kết nối API: {exc}")
        return 1

    # ── Xác định kích thước features và agent IDs ─────────────────────────
    initial_obs = env.observe_all()
    agent_ids: list[int] = env.intersection_ids  # thứ tự cố định
    n_agents = len(agent_ids)
    obs_dim = len(build_features(next(iter(initial_obs.values()))))

    print(f"[INFO] Số nút giao (agents): {n_agents}")
    print(f"[INFO] Local obs dim: {obs_dim} | Input dim (+ ID): {obs_dim + n_agents}")
    print(f"[INFO] Global state dim: {n_agents * obs_dim}")
    print(f"[INFO] Lưu model vào: {args.model_path}")
    print(f"[INFO] Steps: {args.steps:,} | Batch: {args.batch_size} | Buffer: {args.buffer_capacity}")
    print("-" * 70)

    # ── Tải hoặc khởi tạo QMIX agent ─────────────────────────────────────
    agent = QMIXAgent.load(
        args.model_path,
        default_n_agents=n_agents,
        default_obs_dim=obs_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        hidden_dim=args.hidden_dim,
        mixing_hidden_dim=args.mixing_hidden_dim,
    )

    # ── Vòng lặp training ─────────────────────────────────────────────────
    observation = initial_obs
    reward_history: list[float] = []
    queue_history: list[float] = []
    loss_history: list[float] = []
    td_error_history: list[float] = []   # TD error để đo convergence
    change_count = 0                      # Đếm tổng số hành động CHANGE (action=1)

    pbar = tqdm(
        range(1, args.steps + 1),
        desc="QMIX Training",
        unit="step",
        ncols=130,
        colour="magenta",
        dynamic_ncols=False,
    )

    for step in pbar:

        # ── Bước 1: Chuyển obs dict → numpy array ─────────────────────────
        obs_array = obs_dict_to_array(observation, agent_ids)  # (N, obs_dim)
        global_state = agent._build_global_state(obs_array)     # (N*obs_dim,)

        # ── Bước 2: Chọn actions (ε-greedy, decentralized) ────────────────
        # Override action = 0 (KEEP) cho agents đang trong giai đoạn giữ pha
        raw_actions = agent.select_actions(
            {aid: obs_array[i].tolist() for i, aid in enumerate(agent_ids)},
            agent_ids=agent_ids,
            explore=not args.no_explore,
        )
        actions: dict[int, int] = {}
        for aid in agent_ids:
            if env.hold_required(aid):
                actions[aid] = 0
            else:
                actions[aid] = raw_actions[aid]

        # ── Bước 3: Áp dụng actions và advance ───────────────────────────
        env.apply_actions(actions)
        next_observation = env.advance()

        # ── Bước 4: Tính joint reward và queue stats ──────────────────────
        step_rewards: list[float] = []
        step_queue = 0.0
        next_obs_array = obs_dict_to_array(next_observation, agent_ids)
        next_global_state = agent._build_global_state(next_obs_array)

        for i, aid in enumerate(agent_ids):
            obs = next_observation[aid]
            reward = env.reward_for(obs, actions.get(aid, 0))
            step_rewards.append(reward)
            step_queue += sum(
                float(p.get("queue_length", 0.0))
                for p in obs.get("directions", {}).values()
                if isinstance(p, dict)
            )

        joint_reward = float(np.mean(step_rewards))   # mean (cooperative)
        avg_queue = step_queue / max(n_agents, 1)

        # ── Bước 5: Lưu joint transition vào buffer ───────────────────────
        actions_array = [actions.get(aid, 0) for aid in agent_ids]
        agent.buffer.push(
            obs_all=obs_array,
            actions_all=actions_array,
            global_state=global_state,
            joint_reward=joint_reward,
            next_obs_all=next_obs_array,
            next_global_state=next_global_state,
            done=False,
        )

        # ── Bước 6: Update QMIX (1 lần/step, joint loss) ─────────────────
        loss = agent.update()

        # Đếm số action CHANGE trong step này
        change_count += sum(1 for a in actions_array if a == 1)

        reward_history.append(joint_reward)
        queue_history.append(avg_queue)
        if loss > 0.0:
            loss_history.append(loss)
        agent.decay_epsilon()
        observation = next_observation

        # ── Bước 7: Cập nhật tqdm postfix mỗi 10 step ────────────────────
        if step % 10 == 0:
            w25 = min(len(reward_history), 25)
            avg_r  = sum(reward_history[-25:]) / w25
            avg_q  = sum(queue_history[-25:]) / w25
            avg_l  = sum(loss_history[-25:]) / max(len(loss_history[-25:]), 1)
            # Tỷ lệ action CHANGE tích lũy (%)
            chg_rate = change_count / (step * n_agents) * 100
            pbar.set_postfix(
                {
                    "ε":    f"{agent.epsilon:.3f}",
                    "loss": f"{avg_l:.4f}",
                    "r̄":   f"{avg_r:+.2f}",
                    "q̄":   f"{avg_q:.1f}",
                    "upd":  agent.update_count,
                    "chg%": f"{chg_rate:.0f}",
                    "buf":  len(agent.buffer),
                },
                refresh=True,
            )

        # ── Bước 7b: In log chi tiết mỗi 50 step ─────────────────────────
        if step % 50 == 0 and len(reward_history) >= 50:
            w50 = min(len(reward_history), 50)
            avg_r50  = sum(reward_history[-50:]) / w50
            best_r50 = max(reward_history[-50:])
            worst_r50= min(reward_history[-50:])
            avg_q50  = sum(queue_history[-50:]) / w50
            avg_l50  = sum(loss_history[-50:]) / max(len(loss_history[-50:]), 1)
            chg_rate = change_count / (step * n_agents) * 100
            tqdm.write(
                f"  step={step:05d} | "
                f"ε={agent.epsilon:.4f} | "
                f"loss={avg_l50:.5f} | "
                f"r̄={avg_r50:+.3f} (max={best_r50:+.2f}, min={worst_r50:+.2f}) | "
                f"q̄={avg_q50:.1f} | "
                f"upd={agent.update_count} | "
                f"chg%={chg_rate:.1f}% | "
                f"buf={len(agent.buffer)}/{agent.buffer.capacity}"
            )

        # ── Bước 8: Lưu model định kỳ ────────────────────────────────────
        if step % args.save_every == 0:
            agent.save(args.model_path)
            tqdm.write(f"  [SAVE] step={step:05d} | ε={agent.epsilon:.4f} | r̄(50)={sum(reward_history[-50:])/min(50,len(reward_history)):+.3f} | upd={agent.update_count}")

    pbar.close()

    # ── Lưu model lần cuối ────────────────────────────────────────────────
    agent.save(args.model_path)

    # ── Tóm tắt kết quả ──────────────────────────────────────────────────
    summary = {
        "algorithm": "QMIX",
        "steps": args.steps,
        "n_agents": n_agents,
        "obs_dim": obs_dim,
        "final_epsilon": round(agent.epsilon, 6),
        "avg_joint_reward": round(sum(reward_history) / max(len(reward_history), 1), 4),
        "avg_queue": round(sum(queue_history) / max(len(queue_history), 1), 2),
        "avg_loss": round(sum(loss_history) / max(len(loss_history), 1), 6),
        "model_path": str(args.model_path),
        "history_window": DEFAULT_HISTORY_WINDOW,
    }
    print("\n" + "=" * 70)
    print("  QMIX TRAINING COMPLETE")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

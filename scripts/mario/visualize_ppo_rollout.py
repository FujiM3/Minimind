#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
import sys
from collections import Counter, deque
from typing import Any, Dict, Optional, Tuple

import cv2
import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.mario.collect_dt_dataset_from_ppo import BatchedPolicyAdapter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def process_frame(frame):
    if frame is None:
        return np.zeros((1, 84, 84), dtype=np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))[None, :, :] / 255.0
    return resized.astype(np.float32)


def _extract_step(step_out: Tuple[Any, ...]):
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)
    if len(step_out) == 4:
        obs, reward, done, info = step_out
        done = bool(done)
        return obs, float(reward), done, False, dict(info)
    raise ValueError(f"Unsupported step() output length: {len(step_out)}")


class CustomRewardWrapper:
    def __init__(self, env, world: Optional[int], stage: Optional[int]):
        self.env = env
        self.world = world
        self.stage = stage
        self.curr_score = 0
        self.current_x = 40
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _ = out
        else:
            obs = out
        return process_frame(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = _extract_step(self.env.step(action))
        done = bool(terminated or truncated)
        processed = process_frame(obs)

        reward += (float(info.get("score", 0)) - self.curr_score) / 40.0
        self.curr_score = float(info.get("score", 0))

        if done:
            if bool(info.get("flag_get", False)):
                reward += 50.0
            else:
                reward -= 50.0

        if self.world == 7 and self.stage == 4:
            x = int(info.get("x_pos", 0))
            y = int(info.get("y_pos", 0))
            if (
                (506 <= x <= 832 and y > 127)
                or (832 < x <= 1064 and y < 80)
                or (1113 < x <= 1464 and y < 191)
                or (1579 < x <= 1943 and y < 191)
                or (1946 < x <= 1964 and y >= 191)
                or (1984 < x <= 2060 and (y >= 191 or y < 127))
                or (2114 < x < 2440 and y < 191)
                or x < self.current_x - 500
            ):
                reward -= 50.0
                done = True
                terminated = True
                truncated = False
        if self.world == 4 and self.stage == 4:
            x = int(info.get("x_pos", 0))
            y = int(info.get("y_pos", 0))
            if (x <= 1500 and y < 127) or (1588 <= x < 2380 and y >= 127):
                reward = -50.0
                done = True
                terminated = True
                truncated = False

        self.current_x = int(info.get("x_pos", self.current_x))
        return processed, float(reward / 10.0), done, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class CustomSkipFrameWrapper:
    def __init__(self, env, skip: int = 4):
        self.env = env
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], axis=0)
        return self.states[None, :, :, :].astype(np.float32)

    def step(self, action):
        total_reward = 0.0
        last_states = []
        done = False
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        for i in range(self.skip):
            state, reward, done, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if i >= self.skip // 2:
                last_states.append(state)
            if done:
                return self.states[None, :, :, :].astype(np.float32), total_reward, True, terminated, truncated, info

        max_state = np.max(np.concatenate(last_states, axis=0), axis=0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def parse_world_stage(env_id: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"SuperMarioBros-(\d+)-(\d+)-", env_id)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def get_actions(action_type: str):
    kind = action_type.lower()
    if kind == "right":
        return RIGHT_ONLY
    if kind == "complex":
        return COMPLEX_MOVEMENT
    return SIMPLE_MOVEMENT


def make_eval_env(env_id: str, render_mode: Optional[str], action_type: str, skip: int):
    kwargs: Dict[str, Any] = {
        "apply_api_compatibility": True,
        "disable_env_checker": True,
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    world, stage = parse_world_stage(env_id)
    actions = get_actions(action_type)
    env = gym_super_mario_bros.make(env_id, **kwargs)
    env = JoypadSpace(env, actions)
    env = CustomRewardWrapper(env, world=world, stage=stage)
    env = CustomSkipFrameWrapper(env, skip=skip)
    return env


def unpack_step(step_out: Tuple[Any, ...]):
    if len(step_out) == 6:
        obs, reward, done, terminated, truncated, info = step_out
        return obs, float(reward), bool(done), bool(terminated), bool(truncated), dict(info)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
        return obs, float(reward), done, bool(terminated), bool(truncated), dict(info)
    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, float(reward), bool(done), bool(done), False, dict(info)
    raise ValueError(f"Unsupported step() output length: {len(step_out)}")


def safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, dict(info) if isinstance(info, dict) else {}
    return out, {}


def obs_to_bchw(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected state shape [C,H,W], got {arr.shape}")
    return arr[None, ...]


def ensure_frame(frame: Any) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        out = frame
    else:
        return None
    if out.ndim != 3:
        return None
    if out.shape[2] == 3:
        return out
    if out.shape[2] == 4:
        return out[:, :, :3]
    return None


def create_video_writer(path: str, fps: int, frame_shape: Tuple[int, int]) -> cv2.VideoWriter:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    width, height = frame_shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at: {path}")
    return writer


def main():
    parser = argparse.ArgumentParser(description="Run multiple Mario rollouts with a trained PPO checkpoint.")
    parser.add_argument("--model_path", type=str, default="PPO_trained_models\\ppo_super_mario_bros_1_1")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-1-1-v3")
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0, help="Add random actions to probe robustness.")
    parser.add_argument("--action_type", type=str, choices=["right", "simple", "complex"], default="simple")
    parser.add_argument("--skip", type=int, default=4, help="Frame skip used during training in reference repo.")
    parser.add_argument("--max_actions", type=int, default=200, help="Stop rollout if same action repeats this many times.")
    parser.add_argument("--render_mode", type=str, choices=["human", "rgb_array", "none"], default="none")
    parser.add_argument("--save_video_path", type=str, default="")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--save_plot_path", type=str, default="")
    parser.add_argument("--print_every", type=int, default=60)
    parser.add_argument("--stuck_window", type=int, default=180)
    parser.add_argument("--stuck_delta_x", type=int, default=16)
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")
    if args.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be > 0")
    if args.skip <= 0:
        raise ValueError("--skip must be > 0")
    if args.max_actions <= 0:
        raise ValueError("--max_actions must be > 0")
    if args.video_fps <= 0:
        raise ValueError("--video_fps must be > 0")
    if not (0.0 <= args.epsilon <= 1.0):
        raise ValueError("--epsilon must be in [0, 1]")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = BatchedPolicyAdapter(args.model_path, device=device)
    action_set = get_actions(args.action_type)
    if policy.backend == "torch":
        model_act_dim = int(policy.torch_model.actor_linear.out_features)
        if model_act_dim != len(action_set):
            raise ValueError(
                f"Action-space mismatch: model outputs {model_act_dim} actions, "
                f"but action_type='{args.action_type}' has {len(action_set)} actions."
            )

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = make_eval_env(args.env_id, render_mode=render_mode, action_type=args.action_type, skip=args.skip)
    rollout_summaries = []
    all_action_hist = Counter()

    try:
        for rollout_idx in range(args.num_rollouts):
            set_seed(args.seed + rollout_idx)
            writer: Optional[cv2.VideoWriter] = None
            obs, info = safe_reset(env)

            x_positions = [int(info.get("x_pos", 0))]
            rewards = []
            rollout_actions = []
            flag_get = False
            done = False
            step = 0
            terminated = False
            truncated = False
            action_window = deque(maxlen=args.max_actions)
            early_stop_same_action = False

            while not done and step < args.max_steps:
                state_batch = obs_to_bchw(np.array(obs).copy())
                greedy_action = int(policy.deterministic_actions(state_batch)[0])
                if random.random() < args.epsilon:
                    action = int(env.action_space.sample())
                    used_random = True
                else:
                    action = greedy_action
                    used_random = False

                next_obs, reward, done, terminated, truncated, info = unpack_step(env.step(action))
                if "x_pos" not in info:
                    raise ValueError(f"step info missing required key 'x_pos' at step={step}.")

                step += 1
                obs = next_obs
                rewards.append(float(reward))
                rollout_actions.append(int(action))
                action_window.append(int(action))
                x_positions.append(int(info["x_pos"]))
                flag_get = bool(info.get("flag_get", False))
                if len(action_window) == args.max_actions and action_window.count(action_window[0]) == len(action_window):
                    early_stop_same_action = True
                    done = True

                if args.print_every > 0 and (step % args.print_every == 0 or done):
                    print(
                        f"[rollout={rollout_idx + 1:03d} step={step:4d}] x_pos={x_positions[-1]:4d} reward={reward:7.2f} "
                        f"action={action} random={used_random} flag_get={flag_get} early_stop={early_stop_same_action}"
                    )

                if args.save_video_path:
                    frame = ensure_frame(env.render())
                    if frame is not None:
                        if writer is None:
                            if args.num_rollouts == 1:
                                video_path = args.save_video_path
                            else:
                                base, ext = os.path.splitext(args.save_video_path)
                                video_path = f"{base}_r{rollout_idx + 1:03d}{ext or '.mp4'}"
                            writer = create_video_writer(
                                video_path,
                                fps=args.video_fps,
                                frame_shape=(frame.shape[1], frame.shape[0]),
                            )
                        draw = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.putText(draw, f"rollout:{rollout_idx + 1} step:{step}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(draw, f"x:{x_positions[-1]} a:{action} r:{reward:.2f}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        writer.write(draw)

            if writer is not None:
                writer.release()

            total_return = float(np.sum(rewards)) if rewards else 0.0
            max_x = int(np.max(x_positions)) if x_positions else 0
            action_hist = Counter(rollout_actions)
            all_action_hist.update(action_hist)
            stuck = False
            if len(x_positions) > args.stuck_window:
                recent = x_positions[-args.stuck_window :]
                stuck = (max(recent) - min(recent)) <= args.stuck_delta_x

            summary = {
                "rollout": rollout_idx + 1,
                "steps": step,
                "return": total_return,
                "max_x": max_x,
                "terminated": terminated,
                "truncated": truncated,
                "flag_get": flag_get,
                "early_stop_same_action": early_stop_same_action,
                "stuck": stuck,
                "action_hist": dict(sorted(action_hist.items())),
                "x_positions": x_positions,
                "rewards": rewards,
            }
            rollout_summaries.append(summary)

            print("\n=== Rollout Summary ===")
            print(f"rollout={summary['rollout']}, env_id={args.env_id}, model_path={args.model_path}")
            print(f"steps={summary['steps']}, return={summary['return']:.2f}, max_x={summary['max_x']}")
            print(f"terminated={summary['terminated']}, truncated={summary['truncated']}, flag_get={summary['flag_get']}")
            print(f"action_type={args.action_type}, skip={args.skip}, max_actions={args.max_actions}")
            print(f"early_stop_same_action={summary['early_stop_same_action']}, stuck={summary['stuck']}")
            print(f"action_hist={summary['action_hist']}")

            if args.save_plot_path:
                if args.num_rollouts == 1:
                    plot_path = args.save_plot_path
                else:
                    base, ext = os.path.splitext(args.save_plot_path)
                    plot_path = f"{base}_r{rollout_idx + 1:03d}{ext or '.png'}"
                out_dir = os.path.dirname(os.path.abspath(plot_path))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                plt.figure(figsize=(11, 4))
                plt.subplot(1, 2, 1)
                plt.plot(x_positions)
                plt.title("x_pos over time")
                plt.xlabel("step")
                plt.ylabel("x_pos")
                plt.grid(True, alpha=0.3)
                plt.subplot(1, 2, 2)
                plt.plot(np.cumsum(rewards) if rewards else [0.0])
                plt.title("cumulative return")
                plt.xlabel("step")
                plt.ylabel("return")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_path, dpi=140)
                plt.close()
                print(f"plot_saved={plot_path}")
    finally:
        env.close()

    clear_count = int(sum(1 for s in rollout_summaries if s["flag_get"]))
    mean_return = float(np.mean([s["return"] for s in rollout_summaries])) if rollout_summaries else 0.0
    mean_max_x = float(np.mean([s["max_x"] for s in rollout_summaries])) if rollout_summaries else 0.0
    print("\n=== Aggregate Summary ===")
    print(f"num_rollouts={args.num_rollouts}, clear_count={clear_count}, clear_rate={clear_count / args.num_rollouts:.3f}")
    print(f"mean_return={mean_return:.2f}, mean_max_x={mean_max_x:.2f}")
    print(f"action_hist_all={dict(sorted(all_action_hist.items()))}")


if __name__ == "__main__":
    main()

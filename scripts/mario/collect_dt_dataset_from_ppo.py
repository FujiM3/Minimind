#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2
import os
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3.common.vec_env import SubprocVecEnv
import gym


class CustomPPOPolicyNet(nn.Module):
    def __init__(self, in_channels: int = 4, act_dim: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.linear = nn.Linear(1152, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, act_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        value = self.critic_linear(x)
        logits = self.actor_linear(x)
        return logits, value


class BatchedPolicyAdapter:
    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device)
        self.backend = None
        self.sb3_model = None
        self.torch_model = None

        if model_path.lower().endswith(".zip"):
            try:
                from stable_baselines3 import PPO
            except Exception as e:
                raise ImportError(
                    "Model path looks like Stable-Baselines3 (.zip), but stable_baselines3 is not installed."
                ) from e
            self.sb3_model = PPO.load(model_path, device=device)
            self.backend = "sb3"
            return

        state_dict = torch.load(model_path, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported model file format for {model_path}: expected state_dict dict.")

        required_keys = {
            "conv1.weight", "conv1.bias",
            "conv2.weight", "conv2.bias",
            "conv3.weight", "conv3.bias",
            "conv4.weight", "conv4.bias",
            "linear.weight", "linear.bias",
            "critic_linear.weight", "critic_linear.bias",
            "actor_linear.weight", "actor_linear.bias",
        }
        if not required_keys.issubset(set(state_dict.keys())):
            missing = sorted(list(required_keys - set(state_dict.keys())))
            raise ValueError(
                f"Unsupported custom PPO checkpoint: missing keys {missing}. "
                "Expected conv/linear/actor/critic state_dict."
            )

        in_channels = int(state_dict["conv1.weight"].shape[1])
        act_dim = int(state_dict["actor_linear.weight"].shape[0])
        model = CustomPPOPolicyNet(in_channels=in_channels, act_dim=act_dim)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device).eval()
        self.torch_model = model
        self.backend = "torch"

    @staticmethod
    def _ensure_bchw(states: np.ndarray) -> np.ndarray:
        arr = np.asarray(states)
        if arr.ndim == 5 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 4:
            raise ValueError(f"Expected batched states with ndim=4, got shape {arr.shape}")
        if arr.shape[1] in (1, 2, 3, 4, 8):
            return arr
        if arr.shape[-1] in (1, 2, 3, 4, 8):
            return np.transpose(arr, (0, 3, 1, 2))
        raise ValueError(f"Cannot infer channel dimension from states shape {arr.shape}")

    def deterministic_actions(self, states: np.ndarray) -> np.ndarray:
        if self.backend == "sb3":
            out = []
            for s in states:
                a, _ = self.sb3_model.predict(s, deterministic=True)
                out.append(int(a))
            return np.asarray(out, dtype=np.int64)

        bchw = self._ensure_bchw(states)
        states_tensor = torch.from_numpy(bchw).to(self.device, non_blocking=True)
        with torch.no_grad():
            logits, _ = self.torch_model(states_tensor)
            actions_tensor = torch.argmax(logits, dim=-1)
        # CRITICAL: move actions off GPU immediately before vec_env.step().
        return actions_tensor.cpu().numpy().astype(np.int64, copy=False)


class SeedCompatibleWrapper(gym.Wrapper):
    """兼容 legacy reset() 不支持 seed/options 的环境包装。"""

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        kwargs.pop("options", None)
        if seed is not None:
            try:
                seed = int(seed)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            except Exception:
                pass
        try:
            out = self.env.reset(**kwargs)
        except TypeError:
            out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}


def process_frame(frame: Any) -> np.ndarray:
    if frame is None:
        return np.zeros((1, 84, 84), dtype=np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))[None, :, :] / 255.0
    return resized.astype(np.float32)


def extract_world_stage(env_id: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"SuperMarioBros-(\d+)-(\d+)-", env_id)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def get_action_set(action_type: str):
    k = action_type.lower()
    if k == "right":
        return RIGHT_ONLY
    if k == "complex":
        return COMPLEX_MOVEMENT
    return SIMPLE_MOVEMENT


def _extract_step(step_out: Tuple[Any, ...]):
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)
    if len(step_out) == 4:
        obs, reward, done, info = step_out
        done = bool(done)
        return obs, float(reward), done, False, dict(info)
    raise ValueError(f"Unsupported step() output length: {len(step_out)}")


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env=None, world: Optional[int] = None, stage: Optional[int] = None):
        super().__init__(env)
        self.curr_score = 0.0
        self.current_x = 40
        self.world = world
        self.stage = stage

    def reset(self, **kwargs):
        self.curr_score = 0.0
        self.current_x = 40
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return process_frame(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = _extract_step(self.env.step(action))
        state = process_frame(obs)
        reward += (float(info.get("score", 0)) - self.curr_score) / 40.0
        self.curr_score = float(info.get("score", 0))
        if bool(terminated or truncated):
            reward += 50.0 if bool(info.get("flag_get", False)) else -50.0

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
                terminated = True
                truncated = False
        if self.world == 4 and self.stage == 4:
            x = int(info.get("x_pos", 0))
            y = int(info.get("y_pos", 0))
            if (x <= 1500 and y < 127) or (1588 <= x < 2380 and y >= 127):
                reward = -50.0
                terminated = True
                truncated = False
        self.current_x = int(info.get("x_pos", self.current_x))
        return state, float(reward / 10.0), terminated, truncated, info


class CustomSkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            state, info = out
        else:
            state, info = out, {}
        self.states = np.concatenate([state for _ in range(self.skip)], axis=0)
        return self.states[None, :, :, :].astype(np.float32), info

    def step(self, action):
        total_reward = 0.0
        last_states = []
        done = False
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        for i in range(self.skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            if i >= self.skip // 2:
                last_states.append(state)
            if done:
                return self.states[None, :, :, :].astype(np.float32), total_reward, terminated, truncated, info
        max_state = np.max(np.concatenate(last_states, axis=0), axis=0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, terminated, truncated, info


def make_env(env_id: str, rank: int, seed: int, action_type: str, skip: int) -> Callable[[], Any]:
    def _init():
        env_seed = int(seed + rank)
        random.seed(env_seed)
        np.random.seed(env_seed)
        torch.manual_seed(env_seed)

        try:
            base_env = gym_super_mario_bros.make(
                env_id,
                apply_api_compatibility=True,
                disable_env_checker=True,
            )
        except TypeError:
            base_env = gym_super_mario_bros.make(env_id)
        if not hasattr(base_env, "render_mode"):
            base_env.render_mode = None
        world, stage = extract_world_stage(env_id)
        actions = get_action_set(action_type)
        env = JoypadSpace(base_env, actions)
        env = CustomRewardWrapper(env, world=world, stage=stage)
        env = CustomSkipFrameWrapper(env, skip=skip)
        env = SeedCompatibleWrapper(env)

        try:
            env.reset(seed=env_seed)
        except TypeError:
            env.reset()
        return env

    return _init


def compute_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1D array, got shape {rewards.shape}")
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + running
        rtg[i] = running
    return rtg


def normalize_state_batch(states: np.ndarray) -> np.ndarray:
    arr = np.asarray(states)
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :, :]
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def obs_for_storage(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def epsilon_bucket(eps: float) -> str:
    left = float(np.floor(eps * 10.0) / 10.0)
    right = min(1.0, left + 0.1)
    return f"[{left:.1f},{right:.1f})"


@dataclass
class EnvTracker:
    # Per-env async state tracker. Each env has its own lifecycle in VecEnv.
    epsilon: float = 0.0
    target_x: int = 0
    silent_steps: int = 0
    collect_steps: int = 0
    is_spawning: bool = True
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    terminateds: List[bool] = field(default_factory=list)
    truncateds: List[bool] = field(default_factory=list)
    flag_gets: List[bool] = field(default_factory=list)

    def clear_buffers(self):
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminateds.clear()
        self.truncateds.clear()
        self.flag_gets.clear()
        self.collect_steps = 0


def init_tracker(
    tracker: EnvTracker,
    epsilon_min: float,
    epsilon_max: float,
    target_x_min: int,
    target_x_max: int,
):
    if target_x_min > target_x_max:
        raise ValueError(f"target_x_min ({target_x_min}) must be <= target_x_max ({target_x_max}).")
    tracker.epsilon = random.uniform(epsilon_min, epsilon_max)
    tracker.target_x = random.randint(target_x_min, target_x_max)
    tracker.silent_steps = 0
    tracker.is_spawning = True
    tracker.clear_buffers()


def validate_episode_shapes(ep: Dict[str, np.ndarray]):
    keys = ["observations", "actions", "rewards", "returns_to_go", "timesteps", "terminateds", "truncateds", "flag_gets"]
    lengths = {k: len(ep[k]) for k in keys}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Inconsistent trajectory lengths detected: {lengths}")
    if lengths["actions"] == 0:
        raise RuntimeError("Collected zero-length episode.")


def build_episode_from_tracker(tracker: EnvTracker) -> Dict[str, np.ndarray]:
    rewards_np = np.asarray(tracker.rewards, dtype=np.float32)
    episode = {
        "observations": np.asarray(tracker.obs, dtype=np.uint8),
        "actions": np.asarray(tracker.actions, dtype=np.int64),
        "rewards": rewards_np,
        "returns_to_go": compute_returns_to_go(rewards_np),
        "timesteps": np.arange(len(tracker.actions), dtype=np.int32),
        "terminateds": np.asarray(tracker.terminateds, dtype=np.bool_),
        "truncateds": np.asarray(tracker.truncateds, dtype=np.bool_),
        "flag_gets": np.asarray(tracker.flag_gets, dtype=np.bool_),
    }
    validate_episode_shapes(episode)
    return episode


def atomic_pickle_dump(obj: Any, output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Vectorized DT dataset collection from PPO with SubprocVecEnv (legacy repro defaults)."
    )
    parser.add_argument("--model_path", type=str, default="PPO_trained_models\\ppo_super_mario_bros_1_1")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-1-1-v0")
    parser.add_argument("--output_path", type=str, default="dataset\\aligned_greedy\\ppo_1_1_greedy.pkl")
    parser.add_argument("--total_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--epsilon_min", type=float, default=0.0)
    parser.add_argument("--epsilon_max", type=float, default=0.0)
    parser.add_argument("--target_x_min", type=int, default=0)
    parser.add_argument("--target_x_max", type=int, default=0)
    parser.add_argument("--max_silent_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_return", type=float, default=-1000000.0)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--action_type", type=str, choices=["right", "simple", "complex"], default="simple")
    parser.add_argument("--skip", type=int, default=4)
    args = parser.parse_args()

    if args.total_episodes <= 0:
        raise ValueError("--total_episodes must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be > 0")
    if args.max_silent_steps <= 0:
        raise ValueError("--max_silent_steps must be > 0")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be > 0")
    if args.skip <= 0:
        raise ValueError("--skip must be > 0")
    if args.min_length < 1:
        raise ValueError("--min_length must be >= 1")
    if not (0.0 <= args.epsilon_min <= 1.0 and 0.0 <= args.epsilon_max <= 1.0):
        raise ValueError("epsilon_min/epsilon_max must be in [0, 1]")
    if args.epsilon_min > args.epsilon_max:
        raise ValueError("epsilon_min must be <= epsilon_max")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = BatchedPolicyAdapter(args.model_path, device=device)
    action_set = get_action_set(args.action_type)
    if policy.backend == "torch":
        model_act_dim = int(policy.torch_model.actor_linear.out_features)
        if model_act_dim != len(action_set):
            raise ValueError(
                f"Action-space mismatch: model outputs {model_act_dim} actions, "
                f"but action_type='{args.action_type}' has {len(action_set)} actions."
            )

    env_fns = [make_env(args.env_id, rank=i, seed=args.seed, action_type=args.action_type, skip=args.skip) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    try:
        if vec_env.action_space.__class__.__name__.lower() != "discrete":
            raise ValueError(f"Epsilon-greedy requires discrete action space, got: {vec_env.action_space}")
        if not hasattr(vec_env.action_space, "sample"):
            raise ValueError("Environment action_space does not support sample().")

        states = normalize_state_batch(vec_env.reset())
        if states.ndim != 4:
            raise ValueError(f"Expected vec_env.reset() to return 4D batch, got {states.shape}")

        trackers: List[EnvTracker] = [EnvTracker() for _ in range(args.num_envs)]
        for t in trackers:
            init_tracker(t, args.epsilon_min, args.epsilon_max, args.target_x_min, args.target_x_max)

        is_spawning = np.ones(args.num_envs, dtype=np.bool_)
        target_x = np.asarray([t.target_x for t in trackers], dtype=np.int32)

        episodes: List[Dict[str, np.ndarray]] = []
        bucket_stats = defaultdict(lambda: {"count": 0, "accepted": 0, "returns": [], "lengths": []})
        all_returns: List[float] = []

        pbar = tqdm(total=args.total_episodes, desc="Collecting episodes", dynamic_ncols=True)

        while len(episodes) < args.total_episodes:
            # Batched GPU inference (N,C,H,W)->N actions.
            greedy_actions = policy.deterministic_actions(states)
            if greedy_actions.shape[0] != args.num_envs:
                raise RuntimeError("Batched policy output size mismatch with num_envs.")

            # Per-env async action routing:
            # - spawning envs: always greedy (silent rollout, no logging)
            # - collecting envs: epsilon-greedy
            actions = np.empty(args.num_envs, dtype=np.int64)
            for i in range(args.num_envs):
                if is_spawning[i]:
                    actions[i] = int(greedy_actions[i])
                else:
                    if random.random() < trackers[i].epsilon:
                        actions[i] = int(vec_env.action_space.sample())
                    else:
                        actions[i] = int(greedy_actions[i])

            # Vec step; actions already detached numpy (no GPU tensors in long-lived buffers).
            next_states, rewards, dones, infos = vec_env.step(actions)
            next_states = normalize_state_batch(next_states)
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=np.bool_)

            for i in range(args.num_envs):
                info = infos[i] if isinstance(infos[i], dict) else {}
                if "x_pos" not in info:
                    raise ValueError(f"Missing 'x_pos' in info for env {i}.")

                done = bool(dones[i])
                # VecEnv auto-resets done envs. terminal_observation preserves last true state before reset.
                terminal_obs = info.get("terminal_observation", next_states[i])
                current_obs = states[i]
                next_obs_for_transition = terminal_obs if done else next_states[i]

                if is_spawning[i]:
                    trackers[i].silent_steps += 1
                    reached_target = int(info["x_pos"]) >= int(target_x[i])
                    timed_out_spawn = trackers[i].silent_steps >= args.max_silent_steps

                    # Spawn complete -> start collection from *new current state*.
                    if reached_target or timed_out_spawn:
                        is_spawning[i] = False
                        trackers[i].is_spawning = False
                        trackers[i].silent_steps = 0
                        trackers[i].collect_steps = 0
                        continue

                    # Spawn-phase death/truncation: VecEnv already reset; restart spawn task with new target/epsilon.
                    if done:
                        init_tracker(trackers[i], args.epsilon_min, args.epsilon_max, args.target_x_min, args.target_x_max)
                        is_spawning[i] = True
                        target_x[i] = trackers[i].target_x
                    continue

                # Collecting phase transition append.
                trackers[i].obs.append(obs_for_storage(np.array(current_obs).copy()))
                trackers[i].actions.append(int(actions[i]))
                trackers[i].rewards.append(float(rewards[i]))
                trackers[i].terminateds.append(bool(done))
                trackers[i].truncateds.append(bool(done and (trackers[i].collect_steps + 1 >= args.max_steps)))
                trackers[i].flag_gets.append(bool(info.get("flag_get", False)))
                trackers[i].collect_steps += 1

                # Harvest this env trajectory independently.
                should_harvest = done or (trackers[i].collect_steps >= args.max_steps)
                if should_harvest:
                    episode = build_episode_from_tracker(trackers[i])
                    ep_return = float(np.sum(episode["rewards"]))
                    ep_length = int(len(episode["actions"]))

                    bucket = epsilon_bucket(trackers[i].epsilon)
                    bucket_stats[bucket]["count"] += 1
                    bucket_stats[bucket]["returns"].append(ep_return)
                    bucket_stats[bucket]["lengths"].append(ep_length)

                    if ep_return >= args.min_return and ep_length >= args.min_length:
                        bucket_stats[bucket]["accepted"] += 1
                        episodes.append(episode)
                        pbar.update(1)
                        if len(episodes) >= args.total_episodes:
                            break

                    all_returns.append(ep_return)

                    # Re-arm env i for next episode: new epsilon + new target_x + spawn mode.
                    init_tracker(trackers[i], args.epsilon_min, args.epsilon_max, args.target_x_min, args.target_x_max)
                    is_spawning[i] = True
                    target_x[i] = trackers[i].target_x

            if len(episodes) >= args.total_episodes:
                break

            states = next_states
            avg_ret = float(np.mean(all_returns)) if all_returns else 0.0
            pbar.set_postfix(avg_return=f"{avg_ret:.2f}", kept=f"{len(episodes)}")

        pbar.close()

    finally:
        try:
            vec_env.close()
        except Exception:
            pass

    serializable_bucket_stats = {}
    total_attempted = 0
    for b in sorted(bucket_stats.keys()):
        c = bucket_stats[b]["count"]
        a = bucket_stats[b]["accepted"]
        rets = bucket_stats[b]["returns"]
        lens = bucket_stats[b]["lengths"]
        total_attempted += c
        serializable_bucket_stats[b] = {
            "count": c,
            "accepted": a,
            "accept_ratio": float(a / max(1, c)),
            "avg_return": float(np.mean(rets)) if rets else 0.0,
            "avg_length": float(np.mean(lens)) if lens else 0.0,
        }

    metadata = {
        "model_path": args.model_path,
        "env_id": args.env_id,
        "total_episodes_requested": args.total_episodes,
        "max_steps": args.max_steps,
        "epsilon_min": args.epsilon_min,
        "epsilon_max": args.epsilon_max,
        "target_x_min": args.target_x_min,
        "target_x_max": args.target_x_max,
        "max_silent_steps": args.max_silent_steps,
        "seed": args.seed,
        "min_return": args.min_return,
        "min_length": args.min_length,
        "num_envs": args.num_envs,
        "device": device,
        "policy_backend": policy.backend,
        "action_type": args.action_type,
        "skip": args.skip,
        "default_profile": "legacy_repro",
        "wrapper_order": [
            "JoypadSpace",
            "CustomReward(process_frame+reward_shaping)",
            f"CustomSkipFrame(skip={args.skip})",
        ],
        "episodes_kept": len(episodes),
        "episodes_attempted": total_attempted,
        "keep_ratio": float(len(episodes) / max(1, total_attempted)),
    }

    dataset = {
        "metadata": metadata,
        "episodes": episodes[: args.total_episodes],
        "epsilon_bucket_stats": serializable_bucket_stats,
    }
    atomic_pickle_dump(dataset, args.output_path)

    print("\n=== Epsilon Bucket Stats ===")
    for b in sorted(serializable_bucket_stats.keys()):
        s = serializable_bucket_stats[b]
        print(
            f"{b}: count={s['count']}, accepted={s['accepted']}, "
            f"accept_ratio={s['accept_ratio']:.3f}, avg_return={s['avg_return']:.2f}, "
            f"avg_length={s['avg_length']:.2f}"
        )
    print(f"\nSaved dataset to: {args.output_path}")
    print(f"Episodes kept: {len(dataset['episodes'])}")


if __name__ == "__main__":
    main()

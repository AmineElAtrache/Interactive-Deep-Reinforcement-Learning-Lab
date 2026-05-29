import os
from datetime import datetime
from typing import List, Optional, Tuple

import gym
import numpy as np

from config import EXPORTS_DIR


def _make_render_env(env_id: str):
    try:
        return gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        return gym.make(env_id)


def run_agent_episode(
    model,
    env_id: str,
    max_steps: int = 1000,
) -> Tuple[List[np.ndarray], float, int]:
    env = _make_render_env(env_id)
    frames: List[np.ndarray] = []
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        try:
            frame = env.render(mode="rgb_array")
        except TypeError:
            frame = env.render()
        if frame is not None:
            frames.append(np.array(frame))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    env.close()
    return frames, total_reward, steps


def save_frames_as_gif(frames: List[np.ndarray], filename: str) -> str:
    import imageio

    os.makedirs(EXPORTS_DIR, exist_ok=True)
    path = os.path.join(EXPORTS_DIR, filename if filename.endswith(".gif") else f"{filename}.gif")
    imageio.mimsave(path, frames, fps=30)
    return path


def save_frames_as_mp4(frames: List[np.ndarray], filename: str, fps: int = 30) -> str:
    import imageio

    os.makedirs(EXPORTS_DIR, exist_ok=True)
    path = os.path.join(EXPORTS_DIR, filename if filename.endswith(".mp4") else f"{filename}.mp4")
    imageio.mimsave(path, frames, fps=fps)
    return path


def export_episode(
    model,
    env_id: str,
    export_format: str = "gif",
    label: Optional[str] = None,
) -> Tuple[str, float, int, int]:
    frames, reward, steps = run_agent_episode(model, env_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = label or f"{env_id}_{timestamp}"

    if export_format.lower() == "mp4":
        path = save_frames_as_mp4(frames, base)
    else:
        path = save_frames_as_gif(frames, base)

    return path, reward, steps, len(frames)

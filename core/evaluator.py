from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.models import load_model, make_env


@dataclass
class EvaluationResult:
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def best_reward(self) -> float:
        return float(max(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def success_rate(self) -> float:
        if not self.episode_rewards:
            return 0.0
        threshold = 475 if max(self.episode_rewards) > 100 else 200
        successes = sum(1 for r in self.episode_rewards if r >= threshold)
        return successes / len(self.episode_rewards)


def evaluate_model(
    model,
    env_id: str,
    n_episodes: int = 10,
) -> EvaluationResult:
    env = make_env(env_id)
    result = EvaluationResult()
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
        result.episode_rewards.append(total_reward)
        result.episode_lengths.append(steps)

    env.close()
    return result


def evaluate_saved_model(
    algorithm: str,
    model_file: str,
    env_id: str,
    n_episodes: int = 10,
) -> EvaluationResult:
    env = make_env(env_id)
    model = load_model(algorithm, model_file, env=env)
    result = evaluate_model(model, env_id, n_episodes)
    env.close()
    return result

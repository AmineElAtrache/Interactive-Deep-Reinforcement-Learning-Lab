import os
from datetime import datetime
from typing import Optional

import gym
from stable_baselines3 import A2C, DQN, PPO

from config import LOGS_DIR, MODELS_DIR

ALGO_CLASSES = {"PPO": PPO, "A2C": A2C, "DQN": DQN}


def make_env(env_id: str):
    return gym.make(env_id)


def create_model(algorithm: str, env, learning_rate: float, gamma: float, run_name: str):
    algo_cls = ALGO_CLASSES[algorithm]
    common = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "verbose": 0,
        "tensorboard_log": LOGS_DIR,
    }
    if algorithm == "DQN":
        return algo_cls(
            **common,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_fraction=0.3,
        )
    return algo_cls(**common)


def save_model(model, algorithm: str, label: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = label or f"{algorithm.lower()}_{timestamp}"
    path = os.path.join(MODELS_DIR, algorithm, filename)
    model.save(path)
    return f"{filename}.zip"


def load_model(algorithm: str, model_file: str, env=None):
    if not model_file.endswith(".zip"):
        model_file = f"{model_file}.zip"
    path = os.path.join(MODELS_DIR, algorithm, model_file.replace(".zip", ""))
    return ALGO_CLASSES[algorithm].load(path, env=env)


def list_saved_models(algorithm: Optional[str] = None) -> dict:
    algorithms = [algorithm] if algorithm else ALGO_CLASSES.keys()
    result = {}
    for algo in algorithms:
        folder = os.path.join(MODELS_DIR, algo)
        if not os.path.isdir(folder):
            result[algo] = []
            continue
        result[algo] = sorted(
            f for f in os.listdir(folder) if f.endswith(".zip")
        )
    return result

import json
import os
from datetime import datetime
from typing import Optional

from stable_baselines3 import A2C, DQN, PPO

from config import LOGS_DIR, MODELS_DIR

ALGO_CLASSES = {"PPO": PPO, "A2C": A2C, "DQN": DQN}

OBS_SHAPE_TO_ENV = {
    (4,): "CartPole-v1",
    (8,): "LunarLander-v3",
}


class EnvironmentMismatchError(ValueError):
    """Raised when a saved model does not match the selected environment."""


def _model_path(algorithm: str, model_file: str) -> str:
    if not model_file.endswith(".zip"):
        model_file = f"{model_file}.zip"
    return os.path.join(MODELS_DIR, algorithm, model_file.replace(".zip", ""))


def _meta_path(model_path: str) -> str:
    return f"{model_path}.meta.json"


def infer_env_from_obs_shape(obs_shape: tuple) -> str:
    return OBS_SHAPE_TO_ENV.get(tuple(obs_shape), f"unknown (obs shape {obs_shape})")


def _env_from_filename(model_file: str) -> Optional[str]:
    lower = model_file.lower()
    if "cartpole" in lower:
        return "CartPole-v1"
    if "lunarlander" in lower or "lunar_lander" in lower:
        return "LunarLander-v3"
    return None


def get_model_metadata(algorithm: str, model_file: str) -> dict:
    path = _model_path(algorithm, model_file)
    meta_path = _meta_path(path)
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    guessed = _env_from_filename(model_file)
    if guessed:
        return {"env_id": guessed, "algorithm": algorithm}

    model = ALGO_CLASSES[algorithm].load(path, env=None)
    env_id = infer_env_from_obs_shape(tuple(model.observation_space.shape))
    return {
        "env_id": env_id,
        "algorithm": algorithm,
        "obs_shape": list(model.observation_space.shape),
    }


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


def save_model(
    model,
    algorithm: str,
    label: Optional[str] = None,
    env_id: Optional[str] = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = label or f"{algorithm.lower()}_{timestamp}"
    path = os.path.join(MODELS_DIR, algorithm, filename)
    model.save(path)

    meta = {
        "algorithm": algorithm,
        "env_id": env_id or infer_env_from_obs_shape(tuple(model.observation_space.shape)),
        "obs_shape": list(model.observation_space.shape),
        "saved_at": timestamp,
    }
    with open(_meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return f"{filename}.zip"


def load_model(algorithm: str, model_file: str, env=None):
    path = _model_path(algorithm, model_file)
    model = ALGO_CLASSES[algorithm].load(path, env=None)

    if env is not None:
        if model.observation_space != env.observation_space:
            trained_env = infer_env_from_obs_shape(tuple(model.observation_space.shape))
            meta = get_model_metadata(algorithm, model_file)
            if meta.get("env_id") and not meta["env_id"].startswith("unknown"):
                trained_env = meta["env_id"]
            selected_env = infer_env_from_obs_shape(tuple(env.observation_space.shape))
            raise EnvironmentMismatchError(
                f"This model was trained on **{trained_env}**, but you selected "
                f"**{selected_env}** in the sidebar. Switch the environment or pick "
                f"a model trained on {selected_env}."
            )
        model.set_env(env)

    return model


def list_saved_models(algorithm: Optional[str] = None) -> dict:
    algorithms = [algorithm] if algorithm else ALGO_CLASSES.keys()
    result = {}
    for algo in algorithms:
        folder = os.path.join(MODELS_DIR, algo)
        if not os.path.isdir(folder):
            result[algo] = []
            continue
        result[algo] = sorted(f for f in os.listdir(folder) if f.endswith(".zip"))
    return result

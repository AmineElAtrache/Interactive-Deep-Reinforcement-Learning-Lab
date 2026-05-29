import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

ENVIRONMENTS = {
    "CartPole-v1": {"id": "CartPole-v1", "success_threshold": 475},
    "LunarLander-v3": {"id": "LunarLander-v3", "success_threshold": 200},
}

ALGORITHMS = ["DQN", "PPO", "A2C"]

DEFAULT_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "total_timesteps": 50_000,
    "chunk_size": 2048,
}

CHART_COLORS = {
    "DQN": "#e74c3c",
    "PPO": "#f39c12",
    "A2C": "#3498db",
}


def ensure_dirs():
    for path in (LOGS_DIR, MODELS_DIR, EXPORTS_DIR):
        os.makedirs(path, exist_ok=True)
    for algo in ALGORITHMS:
        os.makedirs(os.path.join(MODELS_DIR, algo), exist_ok=True)

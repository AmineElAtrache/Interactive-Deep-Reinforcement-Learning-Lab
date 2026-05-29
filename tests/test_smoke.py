"""Fast smoke tests — no full RL training (keeps CI under a few minutes)."""

from pathlib import Path


def test_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app.py").is_file()
    assert (root / "pyproject.toml").is_file()
    assert (root / "core" / "trainer.py").is_file()


def test_config_constants():
    from config import ALGORITHMS, DEFAULT_HYPERPARAMS, ENVIRONMENTS

    assert "CartPole-v1" in ENVIRONMENTS
    assert set(ALGORITHMS) == {"DQN", "PPO", "A2C"}
    assert DEFAULT_HYPERPARAMS["total_timesteps"] > 0


def test_empty_metrics():
    from core.trainer import empty_metrics

    metrics = empty_metrics()
    assert metrics["status"] == "idle"
    assert metrics["episode_rewards"] == []
    assert metrics["current_timestep"] == 0


def test_cartpole_env_reset_and_step():
    from core.env_utils import make_env, reset_env, step_env

    env = make_env("CartPole-v1")
    try:
        obs = reset_env(env)
        assert obs is not None
        obs, reward, done, _info = step_env(env, env.action_space.sample())
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
    finally:
        env.close()


def test_create_model():
    from core.env_utils import make_env
    from core.models import create_model

    env = make_env("CartPole-v1")
    try:
        model = create_model("PPO", env, learning_rate=3e-4, gamma=0.99, run_name="ci_test")
        assert model is not None
    finally:
        env.close()


def test_app_syntax():
    import ast

    root = Path(__file__).resolve().parents[1]
    source = (root / "app.py").read_text(encoding="utf-8")
    ast.parse(source)

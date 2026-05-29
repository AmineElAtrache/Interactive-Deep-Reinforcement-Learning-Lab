import gymnasium as gym


def make_env(env_id: str, render_mode=None):
    if render_mode:
        return gym.make(env_id, render_mode=render_mode)
    return gym.make(env_id)


def reset_env(env):
    obs, _info = env.reset()
    return obs


def step_env(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, float(reward), terminated or truncated, info

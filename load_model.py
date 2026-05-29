import os
import sys

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO

modelPath = os.path.join(os.getcwd(), "models")

algorithms = ["A2C", "PPO", "DQN"]
modelType = None

if (
    len(sys.argv) == 3
    and sys.argv[1] in algorithms
    and os.path.exists(os.path.join(modelPath, sys.argv[1], sys.argv[2]))
):
    modelType = sys.argv[1]
    modelPath = os.path.join(modelPath, modelType, sys.argv[2])
else:
    raise Exception(
        "ERROR: missing arguments! Please specify the algorithm then the model file "
        "(e.g. PPO 20000.zip)"
    )

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

model = None
if modelType == "PPO":
    model = PPO.load(modelPath, env=env)
if modelType == "A2C":
    model = A2C.load(modelPath, env=env)
if modelType == "DQN":
    model = DQN.load(modelPath, env=env)

for _ in range(1, 5):
    obs, _info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
        env.render()

env.close()

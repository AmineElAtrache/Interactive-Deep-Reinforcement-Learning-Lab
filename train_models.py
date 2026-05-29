import os
import sys

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO

epochs = 25
if len(sys.argv) >= 2:
    if 1 <= int(sys.argv[1]) <= 100:
        epochs = int(sys.argv[1]) + 1
    else:
        raise Exception("ERROR: invalid # of epochs provided! (1 >= epochs <= 100")

logsPath = os.path.join(os.getcwd(), "logs")
ppoModelsPath = os.path.join(os.getcwd(), "models", "PPO")
a2cModelsPath = os.path.join(os.getcwd(), "models", "A2C")
dqnModelsPath = os.path.join(os.getcwd(), "models", "DQN")

for path in (logsPath, ppoModelsPath, a2cModelsPath, dqnModelsPath):
    os.makedirs(path, exist_ok=True)

env = gym.make("CartPole-v1")
env.reset()

ppoModel = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logsPath)
a2cModel = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logsPath)
dqnModel = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logsPath)

timesteps = 10000
for i in range(1, epochs):
    ppoModel.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
    ppoModel.save("%s/%s" % (ppoModelsPath, timesteps * i))

    a2cModel.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="A2C")
    a2cModel.save("%s/%s" % (a2cModelsPath, timesteps * i))

    dqnModel.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="DQN")
    dqnModel.save("%s/%s" % (dqnModelsPath, timesteps * i))

env.close()

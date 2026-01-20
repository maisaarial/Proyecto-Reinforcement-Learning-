from stable_baselines3 import PPO
from Robbery_env_complete import ThiefEnv_complete
from Robbery_env_cams_exitless import ThiefEnv_cams_exitless
from Robbery_env_camless_exitless import ThiefEnv_camexitless
from Robbery_env_exit import ThiefEnv_exit
import numpy as np
import time


env = ThiefEnv_complete(

    render_mode="human"
)

model1 = PPO.load("models/solo_c_thief_ppo_v2", env=env)
#model2 = PPO.load("models/solo_ce_perfect_move", env=env)
obs, _ = env.reset()
while True:

    action, _ = model1.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()


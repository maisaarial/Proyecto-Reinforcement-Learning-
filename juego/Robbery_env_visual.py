from stable_baselines3 import PPO
from Robbery_env_cont import ThiefEnv_cont
from Robbery_env_camless import ThiefEnv_camless
from Robbery_env_camless_exitless import ThiefEnv_camexitless
import numpy as np
import time


env = ThiefEnv_camexitless(
    render_mode="human"
)

model = PPO.load("models/solo_ce_thief_ppo_v4", env=env)
obs, _ = env.reset()

while True:
    # Deterministic = no exploration noise
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
        

from stable_baselines3 import PPO
from Robbery_multi_thief import Multiagent_Thief
import numpy as np

# Dummy patrol (or trained one)
class RandomPatrol:
    def predict(self, obs, deterministic):
        return np.random.uniform(-1, 1, size=2), None

env = Multiagent_Thief(
    patrol_model=RandomPatrol(),
    render_mode="human" 
)

model = PPO.load("models/thief_ppo_v0", env=env)
obs, _ = env.reset()

while True:
    # Deterministic = no exploration noise
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

from stable_baselines3 import PPO
from juego.Robbery_env_complete import ThiefEnv_cont
from Robbery_env_camless import ThiefEnv_camless
from Robbery_env_camless_exitless import ThiefEnv_camexitless
import gymnasium as gym
import random

NBR_ITER = 1
SEED= 23


thief_env = ThiefEnv_camexitless(max_steps=800)
thief_env.reset(seed=SEED)
thief_model = PPO(
    "MultiInputPolicy",
    thief_env,
    verbose=1,
    tensorboard_log="./Entrenamiento/continuo"
)

thief_model.learn(total_timesteps=300_000, tb_log_name=f"thief_v1")
thief_model.save(f"models/solo_ce_thief_ppo_v4")
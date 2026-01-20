from stable_baselines3 import PPO
from Robbery_env_complete import ThiefEnv_complete
from Robbery_env_cams_exitless import ThiefEnv_cams_exitless
from Robbery_env_camless_exitless import ThiefEnv_camexitless
from Robbery_env_exit import ThiefEnv_exit
import gymnasium as gym
import random

NBR_ITER = 1
SEED= 23


thief_env = ThiefEnv_cams_exitless(max_steps=800, nbr_cam=4)
thief_env.reset(seed=SEED)
thief_model = PPO.load("models/solo_ce_perfect_move_copy", env=thief_env)


thief_model.learn(total_timesteps=50_000, tb_log_name=f"thief_v1")
thief_model.save(f"models/solo_c_thief_ppo_v2")
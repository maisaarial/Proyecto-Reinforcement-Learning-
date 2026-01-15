from stable_baselines3 import PPO
from Robbery_multi_patrol import Multiagent_patrol
from Robbery_multi_thief import Multiagent_Thief
import gymnasium as gym
import random

NBR_ITER = 1
SEED= 23

# Random agent behaviour to start training
class RandomPolicy:
    def predict(self, obs, deterministic):
        a=random.random()*2 - 1
        b=random.random()*2 - 1
        action = (a,b)
        return action, None

#Alternating training one and the other : 
thief_env = Multiagent_Thief(patrol_model=RandomPolicy())
thief_env.reset(seed=SEED)
thief_model = PPO(
    "MultiInputPolicy",
    thief_env,
    verbose=1,
    tensorboard_log="./Entrenamiento/multi/thief/"
)

#thief_model = PPO.load("models/thief_ppo_v0", env=thief_env)
#thief_model.policy.set_training_mode(False)
#thief_model.policy.eval()


patrol_env = Multiagent_patrol(thief_model=thief_model, max_steps=300)
patrol_env.reset(seed=SEED)
patrol_model = PPO(
    "MultiInputPolicy",
    patrol_env,
    verbose=1,
    n_steps=512,
    batch_size=128,
    tensorboard_log="./Entrenamiento/multi/patrol/"
)

for i in range(NBR_ITER):
    thief_model.learn(total_timesteps=200_000, tb_log_name=f"thief_v{i}")
    thief_model.save(f"models/thief_ppo_v{i}")
    #patrol_model.learn(total_timesteps=50_000, tb_log_name=f"patrol_v{i}")
    #patrol_model.save(f"models/patrol_ppo_v{i}")
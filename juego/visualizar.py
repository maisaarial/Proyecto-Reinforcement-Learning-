#para mostrar el entorno visualmente
from stable_baselines3 import PPO
from Robbery_env_cont import ThiefEnv_cont
from Robbery_env_camless import ThiefEnv_camless

from Robbery_env_camless_exitless import ThiefEnv_camexitless
import numpy as np
import time
import pybullet as p

from Robbery_v1a import ThiefEnv_V1a
from Robbery_v1b import ThiefEnv_V1b
from Robbery_v1c import ThiefEnv_V1c

import pybullet as p

env = ThiefEnv_V1a(
    render_mode="human"
)

model = PPO.load(
    "cur_lear/20260115-211343_PPO_curriculum_lr1e-04_g0.99_ns2048_bs512_ep10_ent0.01/model_current.zip",
    env=env
)

obs, _ = env.reset()

def set_topdown_camera():
    # Ajusta estos valores a tu mapa
    cx, cy = 7.5, 7.5     # centro
    distance = 15         # altura / zoom
    yaw = 0               # orientación
    pitch = -89           # casi cenital
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=[cx, cy, 0.0]
    )

obs, _ = env.reset()
set_topdown_camera()

while True:
    # Deterministic = no exploration noise
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(1/30)

    if terminated or truncated:
        obs, _ = env.reset()
        set_topdown_camera()
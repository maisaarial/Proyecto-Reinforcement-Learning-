from stable_baselines3 import PPO
from Robbery_v1a import ThiefEnv_V1a
import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import pybullet as p

# ---------- Wrapper para igualar observation_space ----------
class GridAdapterFlattenFloat(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_space = env.observation_space
        new_spaces = dict(old_space.spaces)
        new_spaces["grid"] = spaces.Box(low=0.0, high=3.0, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        obs = dict(obs)
        obs["grid"] = np.array(obs["grid"], dtype=np.float32).reshape(-1)
        return obs

# ---------- Config ----------
MODEL_DIR = "cur_lear/20260115-103610_PPO_curriculum_lr1e-04_g0.99_ns2048_bs512_ep10_ent0.01"
MODEL_PATH = os.path.join(MODEL_DIR, "model_current.zip")

VIDEOS_DIR = "videos_modelos"
FPS = 30
SECONDS = 60
N_STEPS = FPS * SECONDS

# Nombre del video = nombre de la carpeta del modelo
model_name = os.path.basename(MODEL_DIR)
os.makedirs(VIDEOS_DIR, exist_ok=True)
video_path = os.path.join(VIDEOS_DIR, f"{model_name}.mp4")

# ---------- Env + Modelo ----------
env = ThiefEnv_V1a(render_mode="human")
env = GridAdapterFlattenFloat(env)
model = PPO.load(MODEL_PATH, env=env)

# ---------- Cámara top-down (captura independiente del GUI) ----------
def get_topdown_frame(width=640, height=640):
    # Ajusta centro y altura a tu mapa
    cx, cy = 7.5, 7.5
    cam_height = 25.0

    view = p.computeViewMatrix(
        cameraEyePosition=[cx, cy, cam_height],
        cameraTargetPosition=[cx, cy, 0.0],
        cameraUpVector=[0, 1, 0],
    )

    proj = p.computeProjectionMatrixFOV(
        fov=25,                 # más 2D (menos perspectiva)
        aspect=width / height,
        nearVal=0.1,
        farVal=200.0,
    )

    _, _, px, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,  # más estable (headless-friendly)
    )

    rgba = np.array(px, dtype=np.uint8).reshape(height, width, 4)
    rgb = rgba[:, :, :3]
    return rgb

# ---------- Writer MP4 ----------
# Nota: OpenCV escribe en BGR
frame0 = None

obs, _ = env.reset()

# Capturamos el primer frame para inicializar writer
rgb0 = get_topdown_frame(640, 640)
h, w = rgb0.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

# ---------- Loop ----------
for _ in range(N_STEPS):
    action, _ = model.predict(obs, deterministic=True)

    # FIX acción discreta
    if isinstance(action, np.ndarray):
        action = int(action.item())

    obs, reward, terminated, truncated, _ = env.step(action)

    rgb = get_topdown_frame(w, h)
    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # si quieres verlo “en directo” al mismo tiempo (opcional)
    time.sleep(1 / FPS)

    if terminated or truncated:
        obs, _ = env.reset()

writer.release()
env.close()

print(f"✅ Video guardado correctamente en: {video_path}")

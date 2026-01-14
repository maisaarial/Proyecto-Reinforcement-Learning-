# train_v1a.py
import os
import time
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# Importa tu entorno (asegúrate que el archivo se llama Robbery_v1a.py)
from Robbery_v1a import ThiefEnv_V1a


# ============================================================
# WRAPPERS (adaptan observaciones para SB3)
# ============================================================

class FlattenGridObs(gym.ObservationWrapper):
    """
    Convierte obs["grid"] de (3,3) a (9,)
    Esto suele ayudar a estabilizar el aprendizaje con MLP.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Dict)

        old = self.observation_space
        grid_space = old["grid"]
        assert isinstance(grid_space, gym.spaces.Box)

        # Creamos un nuevo Dict space, reemplazando "grid" por una versión flatten
        self.observation_space = gym.spaces.Dict({
            **{k: v for k, v in old.spaces.items() if k != "grid"},
            "grid": gym.spaces.Box(
                low=grid_space.low.min(),
                high=grid_space.high.max(),
                shape=(int(np.prod(grid_space.shape)),),
                dtype=grid_space.dtype
            )
        })

    def observation(self, obs):
        obs = dict(obs)
        obs["grid"] = obs["grid"].reshape(-1)
        return obs


class CastObsToFloat32(gym.ObservationWrapper):
    """
    Convierte arrays NumPy a float32 (SB3 suele ir mejor con float32).
    No toca espacios Discrete.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Dict)

        new_spaces = {}
        for k, sp in self.observation_space.spaces.items():
            if isinstance(sp, gym.spaces.Box):
                new_spaces[k] = gym.spaces.Box(
                    low=sp.low, high=sp.high, shape=sp.shape, dtype=np.float32
                )
            else:
                new_spaces[k] = sp

        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        obs = dict(obs)
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.dtype != np.float32:
                obs[k] = v.astype(np.float32)
        return obs


class RemoveKeysObs(gym.ObservationWrapper):
    """
    Elimina claves de la observación.

    Útil aquí porque tu env devuelve "exits" en obs,
    pero observation_space NO lo declara.
    Eso rompe SB3 (debe coincidir exactamente).

    Si más adelante decides usar "exits" en el policy,
    entonces mejor: añadir "exits" al observation_space del env.
    """
    def __init__(self, env, keys_to_remove=("exits",)):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Dict)

        self.keys_to_remove = set(keys_to_remove)

        # Construimos el nuevo observation_space sin esas claves
        old = self.observation_space
        self.observation_space = gym.spaces.Dict({
            k: v for k, v in old.spaces.items() if k not in self.keys_to_remove
        })

    def observation(self, obs):
        obs = dict(obs)
        for k in self.keys_to_remove:
            obs.pop(k, None)
        return obs


# ============================================================
# FACTORY: crea entornos para vectorización
# ============================================================

def make_env(rank: int, seed: int, render: bool = False):
    """
    Devuelve una función que crea un env.
    Esto es lo que necesita SubprocVecEnv/DummyVecEnv.
    """
    def _init():
        env = ThiefEnv_V1a(render_mode=None if render else None)

        # Monitor guarda sestadísticas de episodios (reward, length, etc.)
        env = Monitor(env)

        # Fix de observaciones + preparación para SB3
        env = RemoveKeysObs(env, keys_to_remove=("exits",))  # <--- clave para que no crashee
        env = FlattenGridObs(env)
        env = CastObsToFloat32(env)

        # Seed por entorno (recomendado en vec env)
        env.reset(seed=seed + rank)
        return env

    # Esto fija seeds de cosas internas (cuando se usa SubprocVecEnv)
    set_random_seed(seed)
    return _init


# ============================================================
# MAIN
# ============================================================

def main():
    # -------------------------
    # 1) Config general
    # -------------------------
    SEED = 42
    N_ENVS = 1
    TOTAL_TIMESTEPS = 100_000

    # -------------------------
    # 2) Hiperparámetros PPO (los usaremos en el nombre del run)
    # -------------------------
    LR = 3e-4
    GAMMA = 0.99
    N_STEPS = 1024
    BATCH_SIZE = 256
    N_EPOCHS = 10
    ENT_COEF = 0.01
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2

    # -------------------------
    # 3) Nombre del experimento: timestamp + hparams (sin env ni seed)
    # -------------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{timestamp}_PPO_v1a"
        f"_lr{LR:.0e}"
        f"_g{GAMMA}"
        f"_ns{N_STEPS}"
        f"_bs{BATCH_SIZE}"
        f"_ep{N_EPOCHS}"
        f"_ent{ENT_COEF}"
    )

    # Directorio base del run
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # -------------------------
    # 4) Crear entornos vectorizados
    # -------------------------
    # Entrenamiento (muchos envs en paralelo)
    vec_env = DummyVecEnv([make_env(0, seed=SEED, render=False)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(run_dir, "monitor.csv"))

    # Evaluación (1 env)
    eval_env = DummyVecEnv([make_env(0, seed=SEED, render=False)])
    eval_env = VecMonitor(eval_env)

    # -------------------------
    # 5) Callbacks: evaluación y checkpoints
    # -------------------------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=20_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="ppo_thief"
    )

    # -------------------------
    # 6) Definir el modelo (MultiInputPolicy porque obs es Dict)
    # -------------------------
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(run_dir, "tb"),
        seed=SEED,
        verbose=1
    )

    # -------------------------
    # 7) Entrenar
    # -------------------------
    '''
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=False,
        tb_log_name="PPO"  # aparece como subcarpeta dentro de tb/
    )
'''
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=False,
        tb_log_name="PPO"  # aparece como subcarpeta dentro de tb/
    )
    # -------------------------
    # 8) Guardar modelo final
    # -------------------------
    final_path = os.path.join(run_dir, "final_model.zip")
    model.save(final_path)
    print(f"\n Modelo final guardado en: {final_path}")

    # Cerrar entornos
    vec_env.close()
    eval_env.close()

    # -------------------------
    # 9)Reproducir el best_model con render
    # -------------------------
    best_path = os.path.join(run_dir, "best_model", "best_model.zip")
    if os.path.exists(best_path):
        print("\n🎮 Reproduciendo best_model con render...\n")

        play_env = make_env(0, seed=SEED + 999, render=True)()
        play_model = PPO.load(best_path)

        obs, _ = play_env.reset()
        for _ in range(500):
            action, _ = play_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = play_env.step(int(action))
            if terminated or truncated:
                obs, _ = play_env.reset()

        play_env.close()
    else:
        print("\n(No se ha encontrado best_model.zip; revisa el entrenamiento/evaluación.)")


if __name__ == "__main__":
    main()

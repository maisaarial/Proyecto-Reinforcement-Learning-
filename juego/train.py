import os
import cv2
import numpy as np
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from Robbery_env_cont import ThiefEnv_cont
from stable_baselines3.common.callbacks import BaseCallback
class TensorboardEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
        return True
    

def make_env(render=False):
    def _init():
        return ThiefEnv_cont(render_mode="human" if render else None)
    return _init


def record_video(model, vecnorm_path, video_path, max_steps=800):

    env = DummyVecEnv([make_env(render=True)])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()

    width, height = 1024, 1024
    fps = 15

    view = p.computeViewMatrix(
        cameraEyePosition=[7, 7, 20],
        cameraTargetPosition=[7, 7, 0],
        cameraUpVector=[0, 1, 0]
    )

    proj = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=1.0,
        nearVal=0.1,
        farVal=100
    )

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    done = False
    step = 0
    total_reward = 0.0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        total_reward += reward[0]
        step += 1

        _, _, px, _, _ = p.getCameraImage(
            width, height, view, proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        frame = np.reshape(px, (height, width, 4))[:, :, :3]
        writer.write(frame.astype(np.uint8))

    writer.release()
    env.close()

    print(f"Vídeo guardado: {video_path} | Reward: {total_reward:.2f}")



def main():

    TOTAL_TIMESTEPS = 300000

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)

    EXPERIMENTS = [
        {"name": "exp_lr3e4", "lr": 3e-4, "n_steps": 1024, "ent": 0.01},
        {"name": "exp_lr1e4", "lr": 1e-4, "n_steps": 1024, "ent": 0.01},
        {"name": "exp_ns2048", "lr": 1e-4, "n_steps": 2048, "ent": 0.01},
        {"name": "exp_ent001", "lr": 1e-4, "n_steps": 2048, "ent": 0.001},
    ]

    for exp in EXPERIMENTS:

        print(f"\n===== {exp['name']} =====")

        # Entornos
        train_env = DummyVecEnv([make_env(render=False)])
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=False, norm_obs_keys=["goal_vector", "ray_view"]
        )

        eval_env = DummyVecEnv([make_env(render=False)])
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False, training=False, norm_obs_keys=["goal_vector", "ray_view"]
        )

        tb_dir = f"tensorboard/{exp['name']}"

        eval_cb = EvalCallback(
            eval_env,
            eval_freq=5000,
            n_eval_episodes=20,
            deterministic=True,
            best_model_save_path=f"models/{exp['name']}_best",
            log_path=tb_dir,  
        )

        model = PPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=exp["lr"],
            n_steps=exp["n_steps"],
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=exp["ent"],
            verbose=1,
            tensorboard_log=tb_dir, 
            policy_kwargs=dict(net_arch=[256, 256])
        )
        eval_tb_cb = TensorboardEvalCallback(eval_env, eval_freq=50)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_cb, eval_tb_cb],
            tb_log_name="train"  
        )

        model.save(f"models/{exp['name']}")
        train_env.save(f"models/{exp['name']}_vecnorm.pkl")

        mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        print(f"Reward final: {mean:.2f} ± {std:.2f}")
        video_path = f"models/{exp['name']}_video.mp4"
        vecnorm_path = f"models/{exp['name']}_vecnorm.pkl"
        record_video(model, vecnorm_path, video_path, max_steps=800)

        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

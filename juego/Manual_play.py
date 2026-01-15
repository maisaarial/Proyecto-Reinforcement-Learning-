from Robbery_v1a import ThiefEnv_V1a
from Robbery_v1b import ThiefEnv_V1b
from Robbery_v1c import ThiefEnv_V1c
from Robbery_env_camless_exitless import ThiefEnv_camexitless
from Robbery_env_cams_exitless import ThiefEnv_cams_exitless
from Robbery_env_complete import ThiefEnv_complete
import numpy as np

def run_env(type):
    if type == "discrete" :
        for i in range(200):
        # Pedir acción al usuario
            try:
                action = int(input("Introduce acción (0: up, 1: down, 2: left, 3: right, 4: stay/take): "))
            except ValueError:
                print("Entrada no válida, usando acción 4 por defecto")
                action = 4

            # Asegurarse que la acción está dentro del rango permitido
            if action not in range(env.action_space.n):
                print(f"Acción fuera de rango, usando acción 4 por defecto")
                action = 4

            obs, reward, terminated, truncated, info = env.step(action)
            print("Observación:", obs)
            print("Recompensa:", reward)

            if terminated or truncated:
                print("Terminado")
                obs, info = env.reset()

    if type == "continuous":
        while True:
            user_input = input("Action (turn forward): ")

            if user_input.lower() == "q":
                print("Exiting.")
                break

            try:
                turn, forward = map(float, user_input.split())
            except ValueError:
                print("❌ Invalid input. Enter two numbers like: 0.2 -0.5")
                continue

            # Clip for safety
            action = np.clip(
                np.array([turn, forward], dtype=np.float32),
                env.action_space.low,
                env.action_space.high
            )

            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Reward: {reward:.3f}")
            print(f"Position: {env.goal_vector}")
            print(f"Has object: {env.has_object}, Alert: {env.alert_flag}")

            if terminated or truncated:
                print("Episode ended. Resetting...\n")
                obs, info = env.reset()

env = ThiefEnv_complete(render_mode="human")
obs, info = env.reset()
run_env("continuous")


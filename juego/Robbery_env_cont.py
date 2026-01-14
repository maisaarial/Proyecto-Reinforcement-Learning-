import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
from Pybullet_edificio import create_structure, create_floor, create_thief, create_object, set_exit_tiles
from Pybullet_sensors import *
"""
V2 : 
Espacio continuo
2 tipos acciones : rotacion [-1,1] equivalencia -70°,70° / velocidad [-1,1] 
recompensas : fijas (+ al encontrar objeto y salir  - al perder) + dinamica al acercarse del objetivo
zonas de camaras : circulares y rotativas 4 seleccionadas al azar entre 8
"""

class ThiefEnv_cont(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self._physics_client = None

        # Action: 2 continuous moves
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32)
        
        # Observation : discrete
        self.observation_space = spaces.Dict({
            "goal_vector":spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "ray_view": spaces.Box(low=0.0,high=1.0,shape=(22,),dtype=np.float32),
            "has_object": spaces.Discrete(2),
            "alert_flag": spaces.Discrete(6), # e.g. binary variable
        })

        self.grid = np.zeros((3,3), dtype=np.int8)
        self.goal = np.zeros((1,1), dtype=np.int8)
        self.exits = np.zeros((1,3), dtype=np.int8)
        self.has_object = 0
        self.alert_flag = 0

        #Other objects
        self.object_pos = None
        self.grid = np.zeros((15,15), dtype=np.int8) # Store world layout (walls, watched tiles, etc.)
        self.types = {}
        self.cameras = None

        
    
    def _load_world(self):
        self.grid.fill(0)  # 0 = empty

        #Agents
        self.thief_pos = np.array([7,0])
        self.material_thief = create_thief(self.thief_pos, pos_height=-3.5, Mass=1)
        self.sensor_thief = create_thief(self.thief_pos)

        # Create below structure for material thief
        create_floor(pos_height=-4.2)
        create_structure(self.grid, pos_height=-3)

        # Create above structure for sensor thief
        create_floor()
        self.types, self.grid = create_structure(self.grid, self.types)     # walls = 1
        self.cameras = create_cameras([self.sensor_thief], self.types)    # watched = 2
        self.object_body, self.grid, self.object_pos = create_object(self.grid)  # object = 3
        p.setCollisionFilterPair(self.object_body, self.sensor_thief, -1, -1, enableCollision=1)

        self.types[self.object_body] = 3
        self.types[-1] = 0
        self.types[1] = 0
        self.goal = np.array([self.object_pos[0], self.object_pos[1]], dtype=np.int32) # Knows position of object
        self.exits = np.array([[7, 0],[6, 0],[8, 0]], dtype=np.int32) # Knows position of exits

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #Connection to PyBullet
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
        self._physics_client = p.connect(p.GUI if self.render_mode=="human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=15, 
                cameraYaw=0, 
                cameraPitch=-89, 
                cameraTargetPosition=[7, 7, 0])

        # Load the world
        p.setGravity(0,0,-9.8)
        self._load_world()

        # Thief's position and initial obs
        obs = {"goal_vector":self.calculate_goal_vector(),
               "ray_view":self._get_observation(),
               "has_object": self.has_object,
               "alert_flag": self.alert_flag}
        return obs, {}
    
    def distance (self, object):
        dist = np.linalg.norm(self.thief_pos - object)/10
        return dist
    
    def calculate_reward(self):
        if self.has_object==0:
            dist = self.distance(self.goal)
        else :
            dist = min(self.distance(self.exits[0]),self.distance(self.exits[1]),self.distance(self.exits[2]))
        return -dist

    def calculate_goal_vector(self):
        if self.has_object == 0:
            vector = (self.goal - self.thief_pos)/14
        else :
            min_index = np.argmin([self.distance(self.exits[0]),self.distance(self.exits[1]),self.distance(self.exits[2])])
            vector = (self.exits[min_index] - self.thief_pos)/14
        return vector
    
    def step(self, action):
        #Update environment
        rotate_cameras(self.cameras, angle=0.05)

        # Continuous controls
        action = np.clip(action, self.action_space.low, self.action_space.high)
        turn = float(action[0])  # e.g. -1 = full left, 1 = full right
        forward = float(action[1])  # e.g. -1 = full backward, 1 = full forward

        move_thief(self.material_thief, self.sensor_thief, turn, forward)
        p.stepSimulation()

        
        pos, orn = p.getBasePositionAndOrientation(self.material_thief)
        pos_ghost=(pos[0],pos[1], 0.5)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        new_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(
            self.sensor_thief,
            pos_ghost,  # keep at same position
            new_orn)

        
        position, _ = p.getBasePositionAndOrientation(self.material_thief)
        x = position[0]
        y = position[1]
        self.thief_pos = np.array([x,y])
        reward = self.calculate_reward()

        if get_contact_cameras(self.sensor_thief, self.cameras) :
            reward -= 0.1
            self.alert_flag +=1
            self.alert_flag = min(self.alert_flag, 5)
        else :
            self.alert_flag = 0

        print(get_contact_object(self.sensor_thief, self.object_body))
        if get_contact_object(self.sensor_thief, self.object_body):
            self.has_object = 1
            p.removeBody(self.object_body)
            self.exit_ids = set_exit_tiles(height=0.1)
            for e in self.exit_ids :
                self.types[e] = 3
                p.setCollisionFilterPair(e, self.sensor_thief, -1, -1, enableCollision=1)
            reward +=10

        terminated = False
        truncated = False

        if self.alert_flag >4 : 
            terminated = True
            reward = -50

        if self.has_object == 1 and get_contact_exits(self.sensor_thief, self.exit_ids) :
            terminated= True
            reward = 50

        self.goal_vector = self.calculate_goal_vector()
        obs = {"goal_vector":self.calculate_goal_vector(),
               "ray_view":self._get_observation(),
               "has_object": self.has_object,
               "alert_flag": self.alert_flag}
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        debug=False
        if self.render_mode =="human" : 
            debug = True
        ray_results = raycast_view_cone(self.sensor_thief, fov=np.pi/2, num_rays=11, max_distance=5.0, height=0.05, debug=debug)
        l= []
        for r in ray_results : 
            id = r[0]
            type = self.types[id]/3
            distance_ratio = r[2]
            l.append(type)
            l.append(distance_ratio)
        return l
    
    def render(self):
        pass

    def close(self):
        if self._physics_client:
            p.disconnect(self._physics_client)


"""
env = ThiefEnv_cont(render_mode="human")
obs, info = env.reset()
print (env.grid)

for i in range(500):
    #action = env.action_space.sample()
    action = (0,1)
    print (action)
    obs, reward, terminated, truncated, info = env.step(action)
    print (reward)
    time.sleep(1)
    if terminated or truncated:
        print ("terminated")
        env.reset()

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
"""
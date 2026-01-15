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
Espacio continuo :
2 tipos acciones : rotacion [-1,1] equivalencia -60°,60° / velocidad [-1,1] 
recompensas : fijas (+ al encontrar objeto - al perder) + dinamica al acercarse del objetivo, o al dirigirse hasta el objeto
### Version con espacio de observaciones diferente y otra recompensa 

"""

class ThiefEnv_camexitless(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=1000, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self._physics_client = None
        self.max_steps = max_steps
        self.current_steps = 0

        # Action: 2 continuous moves
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32)
        
        # Observation : discrete
        self.observation_space = spaces.Dict({
            "goal_dist":spaces.Box(low=0, high=2.2, shape=(1,), dtype=np.float32),
            "agent_yaw": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "to_goal_yaw": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
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
        self.object_body, self.grid, self.object_pos = create_object(self.grid)  # object = 3
        p.setCollisionFilterPair(self.object_body, self.sensor_thief, -1, -1, enableCollision=1)

        self.blocks = []
        for key, values in self.types.items():
            if values == 1 : 
                p.setCollisionFilterPair(key, self.sensor_thief, -1, -1, enableCollision=1)
                self.blocks.append(key)


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
        self.current_steps = 0
        self.min_distance = 16
        pos, orn = p.getBasePositionAndOrientation(self.material_thief)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        self.yaw= yaw

        # Thief's position and initial obs
        self.goal_vector = self.calculate_goal_vector()
        goal_angle = np.arctan2(self.goal_vector[1], self.goal_vector[0])
        self.to_goal_yaw = goal_angle - self.yaw
        self.to_goal_yaw = ((self.to_goal_yaw + np.pi) % (2 * np.pi) - 3*np.pi/2)

        obs = {"goal_dist":np.array([self.distance(self.goal)], dtype=np.float32),
               "agent_yaw":np.array([self.yaw], dtype=np.float32),
               "to_goal_yaw":np.array([self.to_goal_yaw], dtype=np.float32),
               "ray_view":self._get_observation(),
               "has_object": self.has_object,
               "alert_flag": self.alert_flag}
        return obs, {}
    
    def distance (self, object):
        dist = np.linalg.norm(self.thief_pos - object)/10
        return dist
    
    def calculate_reward(self):
        alpha = 1
        if self.has_object==0:
            dist = self.distance(self.goal)
        else :
            dist = 0
        heading_reward = np.cos(self.to_goal_yaw)
        delta = self.prev_dist - dist
        if dist < 0.2 :
            alpha = 0
        return 10*delta + 2.0 * alpha * heading_reward

    def calculate_goal_vector(self):
        vector = (self.goal - self.thief_pos)
        return vector
    
    def step(self, action):
        # Continuous controls
        self.prev_dist = self.distance(self.goal)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        turn = float(action[0])  # e.g. -1 = full left, 1 = full right
        forward = float(action[1])  # e.g. -1 = full backward, 1 = full forward

        move_agent(self.material_thief, turn, forward)
        p.stepSimulation()
                
        pos, orn = p.getBasePositionAndOrientation(self.material_thief)
        pos_ghost=(pos[0],pos[1], 0.5)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        new_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        self.yaw = yaw
        p.resetBasePositionAndOrientation(
            self.sensor_thief,
            pos_ghost,  # keep at same position
            new_orn)
        
        x = pos[0]
        y = pos[1]
        self.thief_pos = np.array([x,y])
        reward = self.calculate_reward()   
        if get_contact_walls(self.sensor_thief, self.blocks):
            reward -=0.2   
        reward -= 0.005 * self.current_steps

        terminated = False
        truncated = False
        self.current_steps += 1

        if get_contact_object(self.sensor_thief, self.object_body):
            self.has_object = 1
            terminated = True
            reward +=50

        if self.current_steps>self.max_steps : 
            truncated = True
            reward = -50

        if self.alert_flag >4 : 
            terminated = True
            reward = -50

        self.goal_vector = self.calculate_goal_vector()
        goal_angle = np.arctan2(self.goal_vector[1], self.goal_vector[0])
        self.to_goal_yaw = goal_angle - self.yaw
        self.to_goal_yaw = ((self.to_goal_yaw + np.pi) % (2 * np.pi) - 3*np.pi/2)

        obs = {"goal_dist":np.array([self.distance(self.goal)], dtype=np.float32),
               "agent_yaw":np.array([self.yaw], dtype=np.float32),
               "to_goal_yaw":np.array([self.to_goal_yaw], dtype=np.float32),
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
        return np.array(l, dtype=np.float32)
    
    def render(self):
        pass

    def close(self):
        if self._physics_client:
            p.disconnect(self._physics_client)


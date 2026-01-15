import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from Pybullet_edificio import create_structure, create_floor, set_watched_tiles, create_thief, create_object, set_exit_tiles

"""
V1 : 
Espacio discreto
4 acciones (left, right, up, down)
recompensas : fijas (+ al encontrar objeto  - al perder o mover en una pared) + dinamica al acercarse del objetivo
zonas de camaras : cuadradas y seleccionada al azar entre 6
"""

class ThiefEnv_V1a(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self._physics_client = None

        # Action: 4 discrete moves
        self.action_space = spaces.Discrete(4)
        self.actions = {
            0: (0,1),
            1: (0,-1),
            2: (-1,0),
            3: (1,0)}
        
        # Observation : discrete
        self.observation_space = spaces.Dict({
            "goal_vector":spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "grid": spaces.Box(low=0, high=3, shape=(3, 3), dtype=np.int8), # 3x3 grid around thief (integers 0–3)
            "has_object": spaces.Discrete(2),
            "alert_flag": spaces.Discrete(4), # e.g. binary variable
        })
        self.grid = np.zeros((3,3), dtype=np.int8)
        self.goal = np.zeros((1,1), dtype=np.int8)
        self.has_object = 0
        self.alert_flag = 0

        #Other objects
        self.object_pos = None
        self.grid = np.zeros((15,15), dtype=np.int8) # Store world layout (walls, watched tiles, etc.)
        
    
    def _load_world(self):
        self.grid.fill(0)  # 0 = empty
        create_floor()
        self.grid = create_structure(self.grid)     # walls = 1
        print(f"before cameras: {self.grid}")
        self.grid = set_watched_tiles(self.grid)    # watched = 2
        print(f"after cameras: {self.grid}")
        self.object_body, self.grid, self.object_pos = create_object(self.grid)  # object = 3
        self.goal = np.array([self.object_pos[0], self.object_pos[1]], dtype=np.int32) # Knows position of object

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
                cameraPitch=-60, 
                cameraTargetPosition=[7, 7, 0])

        # Load the world
        p.setGravity(0,0,-9.8)
        self._load_world()

        # Thief's position and initial obs
        self.thief_pos = np.array([7,0])
        self.thief_body = create_thief(self.thief_pos)

        self.goal_vector = (self.goal - self.thief_pos)/14
        obs = {"goal_vector": self.goal_vector,
               "grid":self._get_observation(),
               "has_object": self.has_object,
               "alert_flag": self.alert_flag}
        
        return obs, {}

    def check_position(self):
        r = 0
        terminated = False
        if self.thief_pos[0] == self.object_pos[0] and self.thief_pos[1] == self.object_pos[1] and self.has_object == 0: 
            self.has_object = 1
            p.removeBody(self.object_body)
            self.object_body = None
            r +=50
            terminated = True
        return r, terminated
    
    def distance (self, object):
        dist = np.linalg.norm(self.thief_pos - object)/10
        return dist
    
    def calculate_reward(self):
        if self.has_object==0:
            dist = self.distance(self.goal)
        else :
            dist = min(self.distance(self.exits[0]),self.distance(self.exits[1]),self.distance(self.exits[2]))
        return -dist

    def step(self, action):
        terminated = False
        truncated = False

        if action < 4: #walking actions
            move = self.actions[action]
            target = self.thief_pos + move

            # Check collision with walls : if yes can't move
            if self.grid[target[0], target[1]] == 1 or target[0]<0  or target[1]<0:
                reward = self.calculate_reward()
                reward -= 0.5
            else:
                self.thief_pos = target
                p.resetBasePositionAndOrientation(self.thief_body,
                        [self.thief_pos[0], self.thief_pos[1], 0.5],
                        [0,0,0,1])
                reward = self.calculate_reward()
                r, terminated = self.check_position()
                reward += r
            
        # Check camera watched tile
        if self.grid[self.thief_pos[0], self.thief_pos[1]] == 2:
            reward -= 0.1  
            self.alert_flag +=1
            self.alert_flag =  min(self.alert_flag, 3)
        else : 
            self.alert_flag = 0

        if self.alert_flag >2 : 
            terminated = True
            reward = -50            

        self.goal_vector = (self.goal - self.thief_pos)/14
        obs = {"goal_vector": self.goal_vector,
               "grid":self._get_observation(),
               "has_object": self.has_object,
               "alert_flag": self.alert_flag}
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        x, y = self.thief_pos
        obs_grid = np.zeros((3,3), dtype=np.int8)

        for dx in range(-1,2):
            for dy in range(-1,2):
                xx = x+dx
                yy = y+dy
                if 0 <= xx < 15 and 0 <= yy < 15:
                    obs_grid[dx+1, dy+1] = self.grid[xx,yy]
                else:
                    obs_grid[dx+1, dy+1] = 1  # treat out of bounds as walls
        return obs_grid
    
    def render(self):
        pass

    def close(self):
        if self._physics_client:
            p.disconnect(self._physics_client)

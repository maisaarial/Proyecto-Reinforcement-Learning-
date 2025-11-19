from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
from gym_minigrid.minigrid import WorldObj, COLOR_NAMES

import random


### Creation of a particular type of floor to represent the field of a camera view
class camera_watched(WorldObj):
    def __init__(self, color='blue'):
        super().__init__('floor', color)
        self.is_camera_watched = True
        self.color = color

    def can_overlap(self):
        # Agent can step on it
        return True

    def render(self, img):
        # Simple visual: red square
        from gym_minigrid.rendering import fill_coords
        from gym_minigrid.minigrid import COLOR_TO_IDX, COLORS
        rgb = COLORS[self.color]
        fill_coords(img, lambda x, y: True, rgb)


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=15,
        agent_start_pos=(7, 14),
        agent_start_dir=3,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        ### Define custom Observation Space
        self.observation_space = gym.spaces.Dict({
        "image": self.observation_space['image'],  # keep the original image
        "mission": self.observation_space['mission'],  # keep mission text
        "in_camera": gym.spaces.Discrete(2),   # 0 = safe, 1 = camera
        "turns_in_camera": gym.spaces.Discrete(4), #0, 1 = been seen (-rew), 2 = still seen (-rew), 3 = end episode
        "phases": gym.spaces.Discrete(2), #0 = steal object, 1 = get to exit
        })

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(6, height-1, None)
        self.grid.set(7, height-1, None)
        self.grid.set(8, height-1, None)        

        camera_positions =[ 
            [(width-2, 1),(width-2, 2), (width-2, 3), (width-2, 4),
             (width-3, 1),(width-3, 2),(width-3, 3),(width-3, 4),
             (width-4, 1),(width-4, 2),(width-4, 3),(width-4, 4),
             (width-5, 1),(width-5, 2),(width-5, 3),(width-5, 4)],
            [(width-2, 8),(width-2, 9), (width-2, 10), (width-2, 11),
             (width-3, 8),(width-3, 9),(width-3, 10),(width-3, 11),
             (width-4, 8),(width-4, 9),(width-4, 10),(width-4, 11),
             (width-5, 8),(width-5, 9),(width-5, 10),(width-5, 11)],
            [(4, height-2),(4, height-3), (4, height-4), (4, height-5),
             (5, height-2),(5, height-3),(5, height-4),(5, height-5),
             (6, height-2),(6, height-3),(6, height-4),(6, height-5),
             (7, height-2),(7, height-3),(7, height-4),(7, height-5)],
            [(1, 4),(1, 5), (1, 6), (1, 7),
             (2, 4),(2, 5), (2, 6), (2, 7),
             (3, 4),(3, 5), (3, 6), (3, 7)],
            [(6, 6),(6, 7),(6, 8),
              (7, 6),(7, 7),(7, 8),
              (8, 6),(8, 7),(8, 8)],
            [(5, 2),(5, 3),(5, 4),
              (6, 2),(6, 3),(6, 4),
              (7, 2),(7, 3),(7, 4)]]

        cameras = random.sample(camera_positions, k=3)
        for camera in cameras:
            for pos in camera:
                self.grid.set(pos[0], pos[1], camera_watched())
        
        ### Get empty cells
        empty_cells = []

        for x in range(self.width):
            for y in range(self.height):
                if self.grid.get(x, y) is None:
                    empty_cells.append((x, y))

        # Place a goal in one of those cells
        self.goal_pos = random.choice(empty_cells)
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        #Place the pillars and their safe space
        pilares = [(2,5),(5,3),(9,3),(12,5),
                   (2,9),(5,11),(9,11),(12,9)]
        for pilar in pilares:
            self.grid.set(pilar[0], pilar[1], Wall())

            self.grid.set(pilar[0]-1, pilar[1], None)
            self.grid.set(pilar[0]+1, pilar[1], None)
            self.grid.set(pilar[0], pilar[1]-1, None)
            self.grid.set(pilar[0], pilar[1]+1, None)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get the object"
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # internal state (NOT in obs yet)
        self.in_camera = 0
        self.turns_in_camera = 0
        self.phases = 0

        # build extended obs
        obs = dict(obs)
        obs["in_camera"] = self.in_camera
        obs["turns_in_camera"] = self.turns_in_camera
        obs["phases"] = self.phases 

        return obs, info


    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        reward -= 0.01

        # check if current tile is camera
        obj = self.grid.get(*self.agent_pos)
        in_cam = int(hasattr(obj, "is_camera_watched") and obj.is_camera_watched)

        if in_cam:
            self.turns_in_camera += 1
            if self.turns_in_camera >= 3:
                reward -=1
                terminated = True
        else:
            self.turns_in_camera = 0

        self.in_camera = in_cam

        # extend obs
        obs = dict(obs)
        obs["in_camera"] = self.in_camera
        obs["turns_in_camera"] = self.turns_in_camera

        if self.phases == 0:
            if tuple(self.agent_pos) == tuple(self.goal_pos):
                #move goal to exit
                self.grid.set(self.goal_pos[0], self.goal_pos[1], None)
                self.put_obj(Goal(), 6, 14)
                self.put_obj(Goal(), 7, 14)
                self.put_obj(Goal(), 8, 14)
                #don't end episode
                terminated = False

                #update variable
                self.mission = "get out !"
                self.phases = 1
                obs["phases"] = self.phases



        return obs, reward, terminated, truncated, info




def main():
    env = SimpleEnv(render_mode="human")
    print(env.observation_space)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()
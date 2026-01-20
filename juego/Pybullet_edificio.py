import pybullet as p
import pybullet_data
import random
import numpy as np
import time

def create_thief(position, pos_height = 0.5, Mass = 0):
    # Create a collision shape (invisible, used for physics)
    thief_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    # Create a visual shape (this is what you see)
    thief_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.1, 0.1, 0.1, 1])
    # Create a rigid body with both collision and visual shapes
    thief_body = p.createMultiBody(baseMass=Mass,
                                baseCollisionShapeIndex=thief_collision,
                                baseVisualShapeIndex=thief_visual,
                                basePosition=[position[0], position[1], pos_height])
    p.changeDynamics(thief_body, -1, lateralFriction=1)
    return thief_body

def create_patrol(position, pos_height = 0.5, Mass = 0):
    # Create a collision shape (invisible, used for physics)
    patrol_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    # Create a visual shape (this is what you see)
    patrol_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0, 0, 1, 1])
    # Create a rigid body with both collision and visual shapes
    patrol_body = p.createMultiBody(baseMass=Mass,
                                baseCollisionShapeIndex=patrol_collision,
                                baseVisualShapeIndex=patrol_visual,
                                basePosition=[position[0], position[1], pos_height])
    p.changeDynamics(patrol_body, -1, lateralFriction=1)
    return patrol_body

def create_object(grid, pos_height=0.5):
    zeros = np.argwhere(grid == 0)
    position = random.choice(zeros)
    # Create a collision shape (invisible, used for physics)
    obj_collision = p.createCollisionShape(p.GEOM_CAPSULE, radius = 0.5)
    # Create a visual shape (this is what you see)
    obj_visual = p.createVisualShape(p.GEOM_CAPSULE, radius = 0.5, rgbaColor=[1, 1, 0, 1])
    # Create a rigid body with both collision and visual shapes
    grid[position[0],position[1]] = 3 #Guard position
    obj_body = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=obj_collision,
                                baseVisualShapeIndex=obj_visual,
                                basePosition=[position[0], position[1], pos_height])
    p.changeDynamics(obj_body, -1, lateralFriction=5)
    return obj_body, grid, position

def create_floor(pos_height=-0.2):
    # Create a collision shape (invisible, used for physics)
    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[7.5, 7.5, 0.2])
    # Create a visual shape (this is what you see)
    floor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[7.5, 7.5, 0.2], rgbaColor=[1, 1, 1, 0.8])
    # Create a rigid body with both collision and visual shapes that's fix
    floor_body = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=floor_collision,
                                baseVisualShapeIndex=floor_visual,
                                basePosition=[7, 7, pos_height])
    p.changeDynamics(floor_body, -1, lateralFriction=1)
    return floor_body

def create_wall(half_width, half_length, center_positions):
    # Create a collision shape (invisible, used for physics)
    wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_width,half_length, 1])
    # Create a visual shape (this is what you see)
    wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_width, half_length, 1], rgbaColor=[1, 1, 1, 1])
    # Create a rigid body with both collision and visual shapes
    wall_body = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=wall_collision,
                                baseVisualShapeIndex=wall_visual,
                                basePosition=center_positions)
    return wall_body

def create_pillar(center_positions):
    # Create a collision shape (invisible, used for physics)
    pillar_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5,0.5, 1])
    # Create a visual shape (this is what you see)
    pillar_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1], rgbaColor=[1, 1, 1, 1])
    # Create a rigid body with both collision and visual shapes
    pillar_body = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=pillar_collision,
                                baseVisualShapeIndex=pillar_visual,
                                basePosition=center_positions)
    return pillar_body

def create_structure(grid, types=None, pos_height=1):
    w1 = create_wall(0.5, 7.5, [0, 7, pos_height])
    w2 = create_wall(0.5, 7.5, [14, 7, pos_height])
    w3 = create_wall(6.5, 0.5, [7, 14, pos_height])
    w4 = create_wall(2.5, 0.5, [11, 0, pos_height])
    w5 = create_wall(2.5, 0.5, [3, 0, pos_height])
    w6 = create_wall(2.5, 0.5, [7, -1, pos_height])
    walls = [w1, w2, w3, w4, w5, w6]

    p1 = create_pillar([5,3,pos_height])
    p2 = create_pillar([9,3,pos_height])
    p3 = create_pillar([2,5,pos_height])
    p4 = create_pillar([12,5,pos_height])
    p5 = create_pillar([2,9,pos_height])
    p6 = create_pillar([12,9,pos_height])
    p7 = create_pillar([5,11,pos_height])
    p8 = create_pillar([9,11,pos_height])
    pillars = [p1, p2, p3, p4, p5, p6, p7, p8]

    #Account the Walls
    for y in range (0,15) : 
        grid[0,y] = 1
        grid[14,y] = 1
    for x in range (1,14):
        grid[x, 14] = 1
        if x != 6 and x!=7 and x!= 8 : 
            grid[x, 0] = 1
    #Account the pillars
    grid[5,3] = 1
    grid[9,3] = 1
    grid[2,5] = 1
    grid[12,5] = 1
    grid[2,9] = 1
    grid[12,9] = 1
    grid[5,11] = 1
    grid[9,11] = 1

    if types is not None :
        # Stock types of bodies for raycasting id
        for w in walls :
            types[w] = 1
        for p in pillars :
            types[p] = 1
        return types, grid
    
    return grid

def set_watched_tiles(grid):
    #Definition of fixed and squared camera zones.
    possible_cameras = [[(4,1),(4,2),(4,4),
                (5,1),
                (6,1),(6,2),(6,4),
                (7,1),(7,2),(7,3),(7,4)],
                [(10,4),(10,5),(10,6),
                 (11,3),(11,4),(11,6),
                 (12,3),
                 (13,3),(13,4),(13,6),],
                [(10,10),(10,12),(10,13),
                 (11,10),(11,11),(11,12),(11,13),
                 (12,11),(12,12),(12,13),
                 (13,10),(13,11),(13,12),(13,13)],
                [(6,6),(6,7),(6,8),
                 (7,6),(7,7),(7,8),
                 (8,6),(8,7),(8,8)],
                [(6,10),(6,12),
                 (7,10),(7,11),(7,12)],
                [(1,7),(1,8),(1,10),
                 (2,7),
                 (3,7),(3,8),(3,10)]]
    cameras = random.sample(possible_cameras, k=3)
    for camera in cameras:
        for (x,y) in camera:
            grid[x,y] = 2
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[0.5, 0.5, 0.05],   # 
                    rgbaColor=[1, 0, 0, 1]       
                ),
                basePosition=[x, y, -0.45]  
            )
    return grid

def set_exit_tiles(height=-0.45):
    exit_tiles = [(6,0),(7,0),(8,0)]
    ids = []
    for t in exit_tiles:
        exit_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        exit_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0, 1, 0, 1])
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=exit_collision,
            baseVisualShapeIndex=exit_visual,
            basePosition=[t[0], t[1], height]  
        )
        ids.append(body)
    return ids
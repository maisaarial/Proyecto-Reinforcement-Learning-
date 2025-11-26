import pybullet as p
import pybullet_data
import random

def create_thief():
    # Create a collision shape (invisible, used for physics)
    thief_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    # Create a visual shape (this is what you see)
    thief_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.1, 0.1, 0.1, 1])
    # Create a rigid body with both collision and visual shapes
    thief_body = p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=thief_collision,
                                baseVisualShapeIndex=thief_visual,
                                basePosition=[7, 0, 0.5])
    p.changeDynamics(thief_body, -1, lateralFriction=5)
    return thief_body

def create_object():
    # Create a collision shape (invisible, used for physics)
    obj_collision = p.createCollisionShape(p.GEOM_CAPSULE, radius = 0.5)
    # Create a visual shape (this is what you see)
    obj_visual = p.createVisualShape(p.GEOM_CAPSULE, radius = 0.5, rgbaColor=[1, 1, 0, 1])
    # Create a rigid body with both collision and visual shapes
    obj_body = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=obj_collision,
                                baseVisualShapeIndex=obj_visual,
                                basePosition=[7, 7, 0.5])
    p.changeDynamics(obj_body, -1, lateralFriction=5)
    return obj_body

def create_floor():
    # Create a collision shape (invisible, used for physics)
    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[7.5, 7.5, 0.2])
    # Create a visual shape (this is what you see)
    floor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[7.5, 7.5, 0.2], rgbaColor=[1, 1, 1, 0.8])
    # Create a rigid body with both collision and visual shapes that's fix
    floor_body = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=floor_collision,
                                baseVisualShapeIndex=floor_visual,
                                basePosition=[7, 7, -0.2])
    p.changeDynamics(floor_body, -1, lateralFriction=5)
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

def create_structure():
    wall_1 = create_wall(0.5, 7.5, [0, 7, 1])
    wall_2 = create_wall(0.5, 7.5, [14, 7, 1])
    wall_3 = create_wall(6.5, 0.5, [7, 14, 1])
    wall_4 = create_wall(2.5, 0.5, [11, 0, 1])
    wall_5 = create_wall(2.5, 0.5, [3, 0, 1])
    pillar_1 = create_pillar([5,3,1])
    pillar_2 = create_pillar([9,3,1])
    pillar_3 = create_pillar([2,5,1])
    pillar_4 = create_pillar([12,5,1])
    pillar_5 = create_pillar([2,9,1])
    pillar_6 = create_pillar([12,9,1])
    pillar_7 = create_pillar([5,11,1])
    pillar_8 = create_pillar([9,11,1])

def camera_watched():
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
        for tile in camera:
            floor_properties = {tile:"watched"}
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[0.5, 0.5, 0.05],   # 
                    rgbaColor=[1, 0, 0, 1]       
                ),
                basePosition=[tile[0], tile[1], -0.45]  
            )
    return floor_properties


# Connect to GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.resetDebugVisualizerCamera(
    cameraDistance=15, 
    cameraYaw=0, 
    cameraPitch=-60, 
    cameraTargetPosition=[7, 7, 0])

thief = create_thief()
floor = create_floor()
plane_id = p.loadURDF("plane.urdf",basePosition=[7.5, 7.5, -0.5],useFixedBase=True)
structure = create_structure()
create_object()
camera_watched()


# Turn on gravity (Earth-like)
p.setGravity(0, 0, -9.8)

# Run the simulation
while True:
    p.stepSimulation()

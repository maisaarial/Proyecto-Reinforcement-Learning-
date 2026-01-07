import pybullet as p
import pybullet_data
import random
import numpy as np
import time
from pyb_utils.ghost import GhostObject
from Pybullet_edificio import create_structure, create_floor, set_watched_tiles, create_thief, create_object, set_exit_tiles

def create_cameras(sensor_thief):
    #Defining Cameras
    triangle1_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle1.obj", meshScale=[1, 1, 1])
    triangle1_vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle1.obj", meshScale=[1, 1, 1], rgbaColor=[1, 0, 0, 1])
    triangle2_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle2.obj", meshScale=[1, 1, 1])
    triangle2_vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle2.obj", meshScale=[1, 1, 1], rgbaColor=[1, 0, 0, 1])
    triangle3_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle3.obj", meshScale=[1, 1, 1])
    triangle3_vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle3.obj", meshScale=[1, 1, 1], rgbaColor=[1, 0, 0, 1])
    triangle4_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle4.obj", meshScale=[1, 1, 1])
    triangle4_vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle4.obj", meshScale=[1, 1, 1], rgbaColor=[1, 0, 0, 1])

    cam_1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=triangle1_col, baseVisualShapeIndex=triangle1_vis,
    basePosition=[5, 3, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]))
    cam_2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=triangle1_col, baseVisualShapeIndex=triangle1_vis,
    basePosition=[9, 11, 0])
    cam_3 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=triangle2_col, baseVisualShapeIndex=triangle2_vis,
    basePosition=[2, 5, 0])
    cam_4 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=triangle2_col, baseVisualShapeIndex=triangle2_vis,
    basePosition=[12, 5, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]))
    cam_5 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=triangle3_col, baseVisualShapeIndex=triangle3_vis,
    basePosition=[9, 3, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]))
    cam_6 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=triangle3_col, baseVisualShapeIndex=triangle3_vis,
    basePosition=[5, 11, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]))
    cam_7 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=triangle4_col, baseVisualShapeIndex=triangle4_vis,
    basePosition=[2, 9, 0])
    cam_8 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=triangle4_col, baseVisualShapeIndex=triangle4_vis,
    basePosition=[12, 9, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]))
    
    p.setCollisionFilterPair(cam_1, sensor_thief, -1, -1, enableCollision=1)
    p.setCollisionFilterPair(cam_2, sensor_thief, -1, -1, enableCollision=1)
    p.setCollisionFilterPair(cam_3, sensor_thief, -1, -1, enableCollision=1)
    p.setCollisionFilterPair(cam_4, sensor_thief, -1, -1, enableCollision=1)
    p.setCollisionFilterPair(cam_5, sensor_thief, -1, -1, enableCollision=1)
    p.setCollisionFilterPair(cam_6, sensor_thief, -1, -1, enableCollision=1)
    p.setCollisionFilterPair(cam_7, sensor_thief, -1, -1, enableCollision=1)
    p.setCollisionFilterPair(cam_8, sensor_thief, -1, -1, enableCollision=1)

    cameras = {cam_1:"clockwise", cam_2:"clockwise", cam_3:"anti", cam_4:"clockwise", cam_5:"anti", cam_6:"anti", cam_7:"clockwise", cam_8:"anti"}
    return cameras

def rotate_cameras(cameras, angle):
    # Quaternion from Euler for rotation around Z
    for cam, rot in cameras.items() : 
        if rot == "clockwise":
            used_angle = -angle
        else : 
            used_angle = angle
        pos, orn = p.getBasePositionAndOrientation(cam)
        delta_quat = p.getQuaternionFromEuler([0, 0, used_angle])
        _, new_orn = p.multiplyTransforms(
            [0, 0, 0], orn,
            [0, 0, 0], delta_quat
        )
        p.resetBasePositionAndOrientation(
            cam,
            pos,  # keep at same position
            new_orn        # new orientation
        )  

def get_contact(sensor_thief, cameras):
    for cam in cameras.keys():
        contacts = p.getContactPoints(bodyA=cam, bodyB=sensor_thief)
        if contacts:
            print(f"Agent is touching the {cam}!")



# Connect to GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
"""
p.resetDebugVisualizerCamera(
    cameraDistance= 7, 
    cameraYaw=0, 
    cameraPitch=0, 
    cameraTargetPosition=[7, 7, 1])

p.resetDebugVisualizerCamera(
    cameraDistance=15, 
    cameraYaw=0, 
    cameraPitch=-60, 
    cameraTargetPosition=[7, 7, 0])
"""
p.resetDebugVisualizerCamera(
    cameraDistance=15, 
    cameraYaw=0, 
    cameraPitch=-89, 
    cameraTargetPosition=[7, 7, 0])

grid = np.zeros((15,15), dtype=np.int8)
thief_pos = np.array([7,0])
sensor_thief = create_thief(thief_pos)
floor = create_floor()
# plane_id = p.loadURDF("plane.urdf",basePosition=[7.5, 7.5, -0.5],useFixedBase=True)
structure = create_structure(grid)
# set_watched_tiles()


thief_pos = np.array([7,0])
material_thief = create_thief(thief_pos, pos_height=-3.5, Mass=1)
floor = create_floor(pos_height=-4.2)
# plane_id = p.loadURDF("plane.urdf",basePosition=[7.5, 7.5, -0.5],useFixedBase=True)
structure = create_structure(grid, pos_height=-3)

# Turn on gravity (Earth-like)
p.setGravity(0, 0, -9.8)


cameras = create_cameras(sensor_thief=sensor_thief)
angle = 0

while True:
    
    angle += 0.05  # rotation speed (rad per step)
    """
    # Quaternion from Euler for rotation around Z
    quat = p.getQuaternionFromEuler([0, 0, angle])
    
    p.resetBaseVelocity(body, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
    p.resetBasePositionAndOrientation(
        body,
        [9, 3, 0],  # keep at same position
        quat        # new orientation
    )
    contacts = p.getContactPoints(bodyA=body, bodyB=sensor_thief)
    if contacts:
        print("Agent is touching the triangle mesh!")
    """
    rotate_cameras(cameras=cameras, angle=0.05)
    p.resetBaseVelocity(material_thief, linearVelocity=[0, 50, 0])
    pos, orn = p.getBasePositionAndOrientation(material_thief)
    pos_ghost=(pos[0],pos[1], 0.5)
    p.resetBasePositionAndOrientation(
        sensor_thief,
        pos_ghost,  # keep at same position
        orn        # new orientation
    )
    get_contact(sensor_thief, cameras)

    p.stepSimulation()
    time.sleep(0.1)
import pybullet as p
import pybullet_data
import random
import numpy as np
import time
import math
from Pybullet_edificio import create_structure, create_floor, create_thief, create_object, set_exit_tiles

def create_cameras(sensor_bodies:list, types):
    #Defining Cameras
    triangle1_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle1.obj", meshScale=[1, 1, 1])
    triangle1_vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle1.obj", meshScale=[1, 1, 1], rgbaColor=[1, 0, 0, 1])
    triangle2_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle2.obj", meshScale=[1, 1, 1])
    triangle2_vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle2.obj", meshScale=[1, 1, 1], rgbaColor=[1, 0, 0, 1])
    triangle3_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle3.obj", meshScale=[1, 1, 1])
    triangle3_vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle3.obj", meshScale=[1, 1, 1], rgbaColor=[1, 0, 0, 1])
    triangle4_col = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle4.obj", meshScale=[1, 1, 1])
    triangle4_vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="/juego/mesh/triangle4.obj", meshScale=[1, 1, 1], rgbaColor=[1, 0, 0, 1])

    cameras = [
        {"collision":triangle1_col, "visual": triangle1_vis, "rotation":"clockwise", "position":[5, 3, 0], "orientation":p.getQuaternionFromEuler([0, 0, -np.pi/2])},
        {"collision":triangle1_col, "visual": triangle1_vis, "rotation":"clockwise", "position":[9, 11, 0], "orientation":p.getQuaternionFromEuler([0, 0, 0])},
        {"collision":triangle2_col, "visual": triangle2_vis, "rotation":"anti", "position":[2, 5, 0], "orientation":p.getQuaternionFromEuler([0, 0, 0])},
        {"collision":triangle2_col, "visual": triangle2_vis, "rotation":"clockwise", "position":[12, 5, 0], "orientation":p.getQuaternionFromEuler([0, 0, np.pi])},
        {"collision":triangle3_col, "visual": triangle3_vis, "rotation":"anti", "position":[9, 3, 0], "orientation":p.getQuaternionFromEuler([0, 0, -np.pi/2])},
        {"collision":triangle3_col, "visual": triangle3_vis, "rotation":"anti", "position":[5, 11, 0], "orientation":p.getQuaternionFromEuler([0, 0, -np.pi/2])},
        {"collision":triangle4_col, "visual": triangle4_vis, "rotation":"clockwise", "position":[2, 9, 0], "orientation":p.getQuaternionFromEuler([0, 0, 0])},
        {"collision":triangle4_col, "visual": triangle4_vis, "rotation":"anti", "position":[12, 9, 0], "orientation":p.getQuaternionFromEuler([0, 0, np.pi])},
        ]
    
    chosen_cameras = random.sample(cameras,k=4)
    built_cams={}
    for cam in chosen_cameras :
        new_cam = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cam["collision"], baseVisualShapeIndex=cam["visual"],
                                    basePosition=cam["position"], baseOrientation=cam["orientation"])
        built_cams[new_cam] = cam["rotation"]
        types[new_cam] = 2
        for body in sensor_bodies :
            p.setCollisionFilterPair(new_cam, body, -1, -1, enableCollision=1)
    return built_cams

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

def raycast_view_cone(sensor_thief,fov=np.pi / 2, num_rays=21, max_distance=6.0,height=0.25, debug=True):
    """
    Cast a batch of rays from sensor_thief in a viewing cone.
    Returns ray results from p.rayTestBatch.
    """
    pos, orn = p.getBasePositionAndOrientation(sensor_thief)
    # Ray origin at fixed height
    ray_start = np.array([pos[0], pos[1], height])

    # Forward direction in world frame
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    forward = rot_matrix @ np.array([0, 1, 0])   # X-forward
    right   = rot_matrix @ np.array([1, 0, 0])   # Y-right

    ray_from = []
    ray_to = []

    angles = np.linspace(-fov / 2, fov / 2, num_rays)
    for a in angles:
        direction = (
            np.cos(a) * forward +
            np.sin(a) * right
        )
        direction /= np.linalg.norm(direction)

        start = ray_start
        end = ray_start + direction * max_distance

        ray_from.append(start.tolist())
        ray_to.append(end.tolist())

        if debug:
            p.addUserDebugLine(start, end, [1, 0, 0], 1, 0.1)

    results = p.rayTestBatch(ray_from, ray_to)
    return results


def get_contact_cameras(sensor_thief, cameras):
    contact = False
    for cam in cameras.keys():
        contacts = p.getContactPoints(bodyA=cam, bodyB=sensor_thief)
        if contacts:
            contact=True
    return contact

def get_contact_object(sensor_thief, object):
    contact = False
    contacts = p.getContactPoints(bodyA=object, bodyB=sensor_thief)
    if contacts:
        print ("OBJECT")
        contact = True
    return contact

def get_contact_exits(sensor_thief, exits):
    contact = False
    for e in exits:
        contacts = p.getContactPoints(bodyA=e, bodyB=sensor_thief)
        if contacts:
            print("EXITS")
            contact=True
    return contact

def move_thief(material_thief, sensor_thief, rotation, velocity):
    """
    rotation: float in [-1, 1] → relative yaw offset [-75°, +75°]
    force:    float in [-1, 1] → forward/backward thrust
    """
    MAX_YAW_OFFSET_DEG = 60.0     # rotation range

    # Clamp inputs (important for RL stability)
    rotation = np.clip(rotation, -1.0, 1.0)
    velocity = np.clip(velocity, -1.0, 1.0)

    # Get current pose
    pos, orn = p.getBasePositionAndOrientation(material_thief)

    # Convert orientation to Euler
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)

    # Map rotation input to yaw offset
    yaw_offset = math.radians(rotation * MAX_YAW_OFFSET_DEG)
    new_yaw = yaw + yaw_offset

    # Build new orientation quaternion
    new_orn = p.getQuaternionFromEuler([roll, pitch, new_yaw])

    # Compute forward direction from new yaw
    forward_dir = np.array([
        math.cos(new_yaw),
        math.sin(new_yaw),
        0.0
    ])

    # Update thief orientation
    p.resetBasePositionAndOrientation(
        material_thief,
        pos,
        new_orn
    )

    factor = 500
    velocity *= factor 
    velocity = [0, velocity, 0]
    position, orientation = p.getBasePositionAndOrientation(material_thief)
    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
    new_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    p.resetBasePositionAndOrientation(
        material_thief,
        position,  # keep at same position
        new_orn        # new orientation
    )
    position, orientation = p.getBasePositionAndOrientation(material_thief)
    # Convert quaternion to rotation matrix
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    # Transform the force vector from local to world coordinates
    vel_world = np.dot(rotation_matrix, velocity)
    p.resetBaseVelocity(material_thief, linearVelocity=vel_world)

    



"""
# Connect to GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


p.resetDebugVisualizerCamera(
    cameraDistance=15, 
    cameraYaw=0, 
    cameraPitch=0, 
    cameraTargetPosition=[7, 7, -3])

p.resetDebugVisualizerCamera(
    cameraDistance= 7, 
    cameraYaw=0, 
    cameraPitch=0, 
    cameraTargetPosition=[7, 7, -3])


p.resetDebugVisualizerCamera(
    cameraDistance=15, 
    cameraYaw=0, 
    cameraPitch=-89, 
    cameraTargetPosition=[7, 7, 0])

grid = np.zeros((15,15), dtype=np.int8)
types = {}

thief_pos = np.array([7,0])
sensor_thief = create_thief(thief_pos)
floor = create_floor()
print(sensor_thief)
structure = create_structure(grid, types=types)
obj_body, grid, obj_position = create_object(grid=grid)
types[obj_body] = 3
types[-1] = 0
types[0] = 0

thief_pos = np.array([7,0])
material_thief = create_thief(thief_pos, pos_height=-3.5, Mass=1)




floor = create_floor(pos_height=-4.2)
structure = create_structure(grid, pos_height=-3)

# Turn on gravity (Earth-like)
p.setGravity(0, 0, -9.8)
cameras = create_cameras([sensor_thief], types=types)
print(types)
while True:

    rotate_cameras(cameras=cameras, angle=0.05)
    a=random.random()*2 - 1
    b=random.random()*2 - 1
    move_thief(material_thief, sensor_thief,0, 1 )

    ray_results = raycast_view_cone(sensor_thief,fov=2*np.pi,num_rays=20,max_distance=3.0,height=0.05,debug=True)
    l= []
    for r in ray_results : 
        id = r[0]
        l.append(types[id])
    print(set(l))

    get_contact_cameras(sensor_thief, cameras)
    p.stepSimulation()
    pos, orn = p.getBasePositionAndOrientation(material_thief)
    pos_ghost=(pos[0],pos[1], 0.5)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    new_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    p.resetBasePositionAndOrientation(
        sensor_thief,
        pos_ghost,  # keep at same position
        new_orn)
    time.sleep(0.01)
"""
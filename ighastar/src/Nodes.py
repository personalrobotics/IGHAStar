import numpy as np
import torch
from torch.utils.cpp_extension import load
from termcolor import colored
import os, sys
import time
import reeds_shepp

folder_name = os.getcwd()
for path in sys.path:
    folder_path = os.path.join(path, folder_name)
    if os.path.exists(folder_path):
        break
else:
    print("something wrong with pathing")

cpp_path = '{}/Dynamics/analytical_bicycle.cpp'.format(folder_path)

kernel = load(
    name="analytical_bicycle",
    sources=[cpp_path],
    verbose=True,
)
kinematic_rollout = kernel.rollout_kinematic
kinodynamic_rollout = kernel.rollout_kinodynamic
check_crop = kernel.check_crop

# =========== begin default values ===============
moves = [1, -1]
steering_angles = np.linspace(-25, 25, 3)/57.3  # convert to radians
kinematic_controls = []
for move in moves:
    for angle in steering_angles:
        kinematic_controls.append((angle, move))

kinematic_controls = np.array(kinematic_controls)
kinematic_controls = torch.Tensor(kinematic_controls).to(dtype=torch.float32)
kinematic_K = np.int32(len(kinematic_controls))

accels = [1, 0.4, 0, -0.4, -1]  # 1 for forward, -1 for backward
accel_angles = np.linspace(-25, 25, 3)/57.3  # convert to radians
kinodynamic_controls = []
for accel in accels:
    for angle in accel_angles:
        kinodynamic_controls.append((angle, accel))
kinodynamic_controls = np.array(kinodynamic_controls)
kinodynamic_controls = torch.Tensor(kinodynamic_controls).to(dtype=torch.float32)
kinodynamic_K = np.int32(len(kinodynamic_controls))

RI = np.float32(0.7)
max_vert = np.float32(4)
min_vel = np.float32(-2.0)
max_vel = np.float32(10.0)
vel_range = max_vel - min_vel
del_theta = 90
max_theta = np.float32(40/57.3)
# =========== end default values ===============

class node:
    def __init__(self, pose, resolution=1, cost=1e9, parent= None, node_info = None, isvalid=True):
        if node_info is None and parent is None:
            self.step_size = 1.0
        elif node_info is None and parent is not None:
            self.step_size = parent.step_size
        else:
            self.step_size = node_info['step_size']
        self.pose = pose # pose is a numpy array.
        self.hash = hash((int(pose[0]), int(pose[1])))
        self.pixel_pose = pose
        self.cost = cost
        self.f_value = 1e9
        self.parent = parent
        self.calc_index(resolution) # this is the "hash"
        self.active = 0
        self.rank = 0
        self.level = 0
        self.old_closed = 0
        self.expansions_needed = 0
        self.isvalid = isvalid

    def distance(self, goal):
        return np.linalg.norm(np.array(self.pose) - np.array(goal.pose))

    def reached_goal_region(self, goal, epsilon=1):
        return self.distance(goal) < epsilon

    def calc_index(self, resolution=1):
        self.index = hash((int(self.pose[0]/resolution), int(self.pose[1]/resolution)))
        return self.index
    
    def __lt__(self, other):
        return self.f_value < other.f_value

    def __eq__(self, other):
        # if self.pose[0] == other.pose[0] and self.pose[1] == other.pose[1]:
        if self.hash == other.hash:
            return True
        return False

    def Succ(self, bitmap, resolution=1, capture_tree=False):
        # get the neighbors of a node
        x, y = self.pose[0], self.pose[1]
        V_P = []

        for i in range(16):
            cost = self.cost + self.step_size
            new_node = node((x+self.step_size*(np.cos(i*np.pi/8) ), y+self.step_size*(np.sin(i*np.pi/8))), parent=self, resolution=resolution, cost = cost)
            valid = self.check_validity(new_node, bitmap, step_size=self.step_size)
            if valid or capture_tree:
                new_node.isvalid = valid
                V_P.append(new_node)
        return V_P

    def heuristic(self, goal):
        # calculate the heuristic value between the current node and the goal node
        return self.distance(goal)

    def calc_f(self, goal):
        return self.cost + self.heuristic(goal)

    def check_validity(self, v2, bitmap, step_size=1):
        x, y = v2.pose[0], v2.pose[1]
        if x < 0 or x >= bitmap.shape[0]:
            return False
        if y < 0 or y >= bitmap.shape[1]:
            return False
        if bitmap[int(y), int(x)] < 254:
            return False
        for i in range(int(step_size)):
            x = self.pose[0] + (i+1)*(v2.pose[0] - self.pose[0])/step_size
            y = self.pose[1] + (i+1)*(v2.pose[1] - self.pose[1])/step_size
            if bitmap[int(y), int(x)] < 254:
                return False
        return True


class kinematic_node:
    def __init__(self, pose, resolution=1, cost=1e9, parent=None, node_info=None, f_value=1e9, isvalid = True):
        if node_info is None and parent is None:
            self.step_size     = 5.0
            self.length        = 2.6
            self.width         = 1.5
            self.map_res       = np.float32(0.1)
            self.dt            = np.float32(0.1)
            self.timesteps     = np.int32(10)
            self.controls      = kinematic_controls
            self.K             = kinematic_K
            self.max_curvature = np.tan(steering_angles.max())/self.length
            self.del_theta     =  np.pi/4
            self.hash_theta    = True
        elif node_info is None and parent is not None:
            self.step_size     = parent.step_size
            self.length        = parent.length
            self.width         = parent.width
            self.map_res       = parent.map_res
            self.dt            = parent.dt
            self.timesteps     = parent.timesteps
            self.controls      = parent.controls
            self.K             = parent.K
            self.max_curvature = parent.max_curvature
            self.del_theta     = parent.del_theta
            self.hash_theta    = parent.hash_theta
        else:
            self.step_size     = node_info['step_size']
            self.length        = node_info['length']
            self.width         = node_info['width']
            self.map_res       = np.float32(node_info['map_res'])
            self.dt            = np.float32(node_info['dt'])
            self.timesteps     = np.int32(node_info['timesteps'])
            self.controls      = node_info['controls']
            self.K             = node_info['K']
            self.max_curvature = node_info['max_curvature']
            self.del_theta     = node_info['del_theta']
            if 'hash_theta' not in node_info: # this is to handle the bug that the node info is obtained from the experiment config not from the yaml and the old version didn't have this info.
                self.hash_theta = False
            else:
                self.hash_theta    = node_info['hash_theta']

        self.pose = pose  # (x, y, theta)
        self.hash = hash(self.pose)
        if self.hash_theta:
            self.hash = hash((int(self.pose[0]*20), int(self.pose[1]*20)))
        self.pixel_pose = (int(pose[0]/self.map_res), int(pose[1]/self.map_res))
        self.cost = cost
        self.f_value = f_value
        self.parent = parent
        self.active = 0
        self.rank = 0
        self.level = 0
        self.expansions_needed = 0
        self.isvalid = isvalid

        self.valid = torch.ones(self.K, dtype=torch.bool)
        self.fanout_cost = torch.zeros(self.K, dtype=torch.float32)
        self.max_curvature = np.tan(steering_angles.max())/self.length
        self.calc_index(resolution)

    def calc_index(self, resolution=1):
        if self.hash_theta and 0:
            # self.index = hash((int(self.pose[0]/4.0), int(self.pose[1]/4.0), int(self.pose[2]*57.3/(resolution * del_theta))))
            self.index = hash((int(self.pose[0]/resolution), int(self.pose[1]/resolution))) #, int(self.pose[2]*57.3/(resolution * del_theta)))) # using this only for visuals.
        else:
            self.index = hash((int(self.pose[0]/resolution), int(self.pose[1]/resolution), int(self.pose[2]*57.3/(resolution * del_theta))))
        return self.index

    def __lt__(self, other):
        return self.f_value < other.f_value

    def __eq__(self, other):
        # if self.pose[0] == other.pose[0] and self.pose[1] == other.pose[1] and self.pose[2] == other.pose[2]:
        if self.hash == other.hash:
            return True
        return False

    def Succ(self, cost_map, heightmap, resolution=1, capture_tree = False):
        states = torch.Tensor(self.pose)
        states = states.repeat(self.K, 1)
        V_P = []
        
        kinematic_rollout(states, self.controls, heightmap, cost_map, self.valid, self.fanout_cost, self.dt, self.K, self.timesteps, np.int32(3), np.int32(2),
              self.step_size, np.float32(1.0), heightmap.shape[0], self.map_res, self.length/2, self.width/2)
        cpu_states = states.cpu().numpy()

        for i in range(self.K):
            if self.valid[i] or capture_tree:
                cost = self.fanout_cost[i].item() + self.cost
                # if self.parent is not None and self.parent.parent is not None:
                #     if np.linalg.norm(np.array(cpu_states[i, :2]) - np.array(self.parent.parent.pose[:2])) < 0.1:
                #         continue
                new_node = kinematic_node(tuple(cpu_states[i, :]), parent=self, resolution=resolution, cost=cost, isvalid=self.valid[i])
                V_P.append(new_node)
        return V_P

    def check_validity(self, node, bitmap):
        state = np.array(node.pose)
        cy = np.cos(state[2])
        sy = np.sin(state[2])
        valid = check_crop(state[0], state[1], cy, sy,  bitmap[...,0], bitmap.shape[0], node.map_res, node.length/2, node.width/2)
        return valid

    def distance(self, goal):
        return self.heuristic(goal) # reeds_shepp.path_length(self.pose, goal.pose, 1/self.max_curvature)

    def reached_goal_region(self, goal, epsilon=1):
        if self.hash_theta:
            dist = np.linalg.norm(np.array(self.pose)[:2] - np.array(goal.pose)[:2])
        else:
            dist = reeds_shepp.path_length(self.pose, goal.pose, 1/self.max_curvature)
        return dist < epsilon

    def heuristic(self, goal):
        dist = np.linalg.norm(np.array(self.pose)[:2] - np.array(goal.pose)[:2])
        return dist

    def calc_f(self, goal):
        return self.cost + self.heuristic(goal)
    
class kinodynamic_node:
    def __init__(self, pose, resolution=1, cost=1e9, parent=None, node_info=None, f_value=1e9, isvalid=True):
        self.hash_theta = True # true by default
        if node_info is None and parent is None:
            self.step_size     = 5.0
            self.length        = 2.6
            self.width         = 1.5
            self.map_res       = np.float32(0.1)
            self.dt            = np.float32(0.1)
            self.timesteps     = np.int32(10)
            self.controls      = kinodynamic_controls
            self.K             = kinodynamic_K
            self.max_curvature = np.tan(steering_angles.max())/self.length
            self.del_theta     =  np.pi/4
            self.del_vel       =  1.0
            self.RI            = np.float32(0.7)
            self.max_vert      = np.float32(4)
            self.min_vel       = np.float32(-2.0)
            self.max_vel       = np.float32(10.0)
            self.max_theta     = np.float32(40/57.3)
            self.hash_theta    = True
        elif node_info is None and parent is not None:
            self.step_size     = parent.step_size
            self.length        = parent.length
            self.width         = parent.width
            self.map_res       = parent.map_res
            self.dt            = parent.dt
            self.timesteps     = parent.timesteps
            self.controls      = parent.controls
            self.K             = parent.K
            self.max_curvature = parent.max_curvature
            self.del_theta     = parent.del_theta
            self.del_vel       = parent.del_vel
            self.RI            = parent.RI
            self.max_vert      = parent.max_vert
            self.min_vel       = parent.min_vel
            self.max_vel       = parent.max_vel
            self.max_theta     = parent.max_theta
            self.hash_theta    = parent.hash_theta
        else:
            self.step_size     = node_info["step_size"]
            self.length        = node_info['length']
            self.width         = node_info['width']
            self.map_res       = np.float32(node_info['map_res'])
            self.dt            = np.float32(node_info['dt'])
            self.timesteps     = np.int32(node_info['timesteps'])
            self.controls      = node_info['controls']
            self.K             = np.int32(len(self.controls))
            self.max_curvature = node_info['max_curvature']
            self.del_theta     = node_info['del_theta']
            self.del_vel       = node_info['del_vel']
            self.RI            = np.float32(node_info['RI'])
            self.max_vert      = np.float32(node_info['max_vert'])
            self.min_vel       = np.float32(node_info['min_vel'])
            self.max_vel       = np.float32(node_info['max_vel'])
            self.max_theta     = np.float32(node_info['max_theta'])
            self.hash_theta    = node_info['hash_theta']

        self.pose = pose  # (x, y, theta, v, omega)
        self.hash = hash(self.pose)
        self.pixel_pose = (int(pose[0]/self.map_res), int(pose[1]/self.map_res))
        self.cost = cost
        self.f_value = f_value
        self.parent = parent
        self.active = 0
        self.rank = 0
        self.level = 0
        self.expansions_needed = 0
        self.isvalid = isvalid

        self.valid = torch.ones(self.K, dtype=torch.bool)
        self.fanout_cost = torch.zeros(self.K, dtype=torch.float32)
        self.max_curvature = np.tan(accel_angles.max())/self.length
        self.calc_index(resolution)

    def calc_index(self, resolution=1):
        if self.hash_theta:
            self.index = hash((int(self.pose[0]/4.0), int(self.pose[1]/4.0), int(self.pose[2]*57.3/(resolution * del_theta)), int(self.pose[3]/(self.del_vel) ) ) )
        else: # increase resolution in all directions
            self.index = hash((int(self.pose[0]/resolution), int(self.pose[1]/resolution), int(self.pose[2]*57.3/(resolution * del_theta)), int(self.pose[3]/(resolution*self.del_vel) ) ) )
        return self.index

    def __lt__(self, other):
        return self.f_value < other.f_value

    def __eq__(self, other):
        # if self.pose[0] == other.pose[0] and self.pose[1] == other.pose[1] and self.pose[2] == other.pose[2] and self.pose[3] == other.pose[3]:
        if self.hash == other.hash:
            return True
        return False

    def Succ(self, cost_map, heightmap, resolution=1, capture_tree=False):
        states = torch.Tensor(self.pose)
        states = states.repeat(self.K, 1)
        V_P = []
        kinodynamic_rollout(states, self.controls, heightmap, cost_map, self.valid, self.fanout_cost, self.dt, self.K, self.timesteps, np.int32(5), np.int32(2),
              self.step_size, np.float32(1.0), np.int32(heightmap.shape[0]), np.float32(self.map_res), self.length/2, self.width/2, self.max_vel, self.min_vel, self.RI, self.max_vert, self.max_theta)
        cpu_states = states.cpu().numpy()
        for i in range(self.K):
            if self.valid[i] or capture_tree:
                cost = self.fanout_cost[i].item() + self.cost
                new_node = kinodynamic_node(tuple(cpu_states[i, :]), parent=self, resolution=resolution, cost=cost, isvalid=self.valid[i])
                V_P.append(new_node)
        return V_P

    def check_validity(self, node, bitmap):
        state = np.array(node.pose)
        cy = np.cos(state[2])
        sy = np.sin(state[2])
        valid = check_crop(state[0], state[1], cy, sy,  bitmap[...,0], bitmap.shape[0], node.map_res, node.length/2, node.width/2)
        return valid

    def distance(self, goal):
        return np.linalg.norm(np.array(self.pose)[:2] - np.array(goal.pose)[:2])

    def reached_goal_region(self, goal, epsilon=1):
        dist = reeds_shepp.path_length(self.pose, goal.pose, 1/self.max_curvature)
        return dist < epsilon and abs(self.pose[3] - goal.pose[3]) < 2.0 # 2 m/s velocity difference

    def heuristic(self, goal):
        return self.distance(goal)/max_vel

    def calc_f(self, goal):
        return self.cost + self.heuristic(goal)

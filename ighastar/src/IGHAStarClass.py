import numpy as np
import sys
import pathlib
import heapq
import time
from termcolor import colored
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import time
import torch
import gc
from copy import deepcopy

class IGHAStarClass:
    def __init__(self, node, bitmap, start, goal, epsilon=14, resolution=1, node_info=None, level_limit=1e9, G_goal=1e9, division_factor=2, DEBUG=False, Oracle=None, hysteresis=0.5, subsampling_ratio=0.5, min_active=100, static_hysteresis=None):
        self.debug = DEBUG
        self.bshape = bitmap.shape
        self.bitmap = bitmap
        self.use_elevation = False
        if len(self.bitmap.shape) == 3:
            self.elevation_map = torch.clone(self.bitmap[..., 1])
            self.costmap = torch.clone(self.bitmap[..., 0])
            self.use_elevation = True
        self.start = node(start, cost = 0.0, resolution=resolution, node_info=node_info)
        self.goal = node(goal, resolution=resolution, node_info=node_info)
        self.epsilon = epsilon
        self.resolution = resolution

        self.level_limit = level_limit
        self.G_goal = G_goal
        self.Oracle = Oracle
        self.closed_set_hash = set()
        self.Q_v_hash = set()
        self.inactive_Q_v_hash = set()
        if not self.start.check_validity(self.start, self.bitmap):
            raise ValueError("Start position is out of bounds.")
        if not self.goal.check_validity(self.goal, self.bitmap):
            raise ValueError("Goal position is out of bounds.")
        # initialize open and closed set using heapq priority queue:
        self.Q_v = []
        self.closed_set = []
        self.start.f_value = self.start.calc_f(self.goal)
        self.start.active = 0
        heapq.heappush(self.Q_v, self.start)
        self.Q_v_hash.add(self.start.hash)
        # G is a dictionary that stores the cost to reach an index.
        g = {}
        g[self.start.index] = 0
        self.G = []
        self.G.append(g)
        # V is a dictionary that stores the vertex corresponding to indices
        v = {}
        v[self.start.index] = self.start
        self.V = []
        self.V.append(v)
        self.counter = 0
        self.last_counter = 0
        self.best_path = []
        self.level = 0
        self.next = 0
        self.division_factor = division_factor
        self.path_list = []
        self.path_cost_list = []
        self.heapify_counter = 0
        self.bottle_neck_ratio = []
        self.old_Qv = []
        self.tree_snapshot = []
        self.Q_v_snapshot = []
        self.freeze_ratio = []
        self.level_information = []
        self.seen_hash_table = []
        self.hysteresis = hysteresis
        self.low_level_dominance_count = 0
        self.len_last_Qv = 0
        self.subsampling_ratio = subsampling_ratio
        self.min_active = min_active
        self.low_res = False
        self.static_hysteresis = static_hysteresis

    def bubbleActive(self):
        while self.Q_v and self.Q_v[0].active != 0:
            v = heapq.heappop(self.Q_v)
            self.inactive_Q_v.append(v)
            self.inactive_Q_v_hash.add(v.hash)

    def Dense(self):
        self.bubbleActive() # this is a given
        run = self.Q_v and self.Q_v[0].active == 0 and self.Q_v[0].f_value < self.G_goal
        self.next = self.level + 1
        return run

    def DenseSparse(self):
        if self.Dense():
            if self.Q_v[0].level == self.level:
                return True
            else:
                self.next = self.Q_v[0].level
        return False

    def DenseSparseHysteresis(self):
        Dense_possible = self.Dense()
        self.low_res = False
        if Dense_possible:
            if self.Q_v[0].level == self.level:
                return Dense_possible
            else:
                self.low_level_dominance_count += 1
                if self.static_hysteresis is None:
                    if self.low_level_dominance_count > int(self.len_last_Qv*self.hysteresis):
                        self.next = self.Q_v[0].level
                        self.low_level_dominance_count = 0
                        self.low_res = True
                        return False
                else:
                    if self.low_level_dominance_count > self.static_hysteresis:
                        self.next = self.Q_v[0].level
                        self.low_level_dominance_count = 0
                        self.low_res = True
                        return False
                return Dense_possible
        return Dense_possible

    def ActivateMethod(self, Activate):
        Activate()
        # remove all inactive vertices from Q_v and put them in self.inactive_Q_v
        del self.inactive_Q_v
        self.inactive_Q_v = []
        self.inactive_Q_v_hash = set()
        self.Q_v_hash = set()
        new_q = []
        for i in range(len(self.Q_v)):
            self.Q_v[i].level = self.level # level reset
            if self.Q_v[i].active == 0:
                new_q.append(self.Q_v[i])
                self.Q_v_hash.add(self.Q_v[i].hash)
            else:
                self.inactive_Q_v.append(self.Q_v[i])
                self.inactive_Q_v_hash.add(self.Q_v[i].hash)
        del self.Q_v
        # gc.collect()
        self.Q_v = new_q
        heapq.heapify(self.Q_v)

    def Freeze(self, v):
        if v.index in self.G[self.level]:
            v_p = self.V[self.level][v.index]
            if v_p in self.Q_v and v_p.active == 0:
                self.Q_v[self.Q_v.index(v_p)].active = 1
                self.heapify = True
        self.G[self.level][v.index] = v.cost
        self.V[self.level][v.index] = v

    def GUpdate(self, v):
        for l in range(self.level):
            v.calc_index(self.resolution/(self.division_factor**l))
            if v.cost < self.get_G(l, v.index):
                v.level = l
                self.G[l][v.index] = v.cost
                self.V[l][v.index] = v
                break
        # continue operating as if nothing happened.
        v.calc_index(self.current_resolution)
        self.Freeze(v)

    def removeQ(self):
        new_q = []
        self.inactive_Q_v = []
        self.inactive_Q_v_hash = set()
        self.Q_v_hash = set()
        if len(self.Q_v):
            for i in range(len(self.Q_v)):
                if self.Q_v[i].f_value < self.G_goal:
                    new_q.append(self.Q_v[i])
                    self.Q_v_hash.add(self.Q_v[i].hash)
        del self.Q_v
        # gc.collect()
        self.Q_v = new_q

    def change_resolution_g_update(self):
        self.level = self.next
        if len(self.G) <= self.level:
            self.G.append({})
            self.V.append({})
        
        self.current_resolution = self.resolution/(self.division_factor**self.level)
        
        for i in range(len(self.Q_v)):
            self.Q_v[i].calc_index(self.current_resolution)
        # Organize nodes by grid index
        grid_cells = {}
        for node in self.Q_v:
            idx = node.index
            if idx not in grid_cells:
                grid_cells[idx] = []
            grid_cells[idx].append(node)
        new_q = []
        # Update G and V with sorted grid cells
        for idx, nodes in grid_cells.items():
            nodes.sort(key=lambda n: n.cost)
            for rank, node in enumerate(nodes):
                node.rank = rank
            # assign equal rank to equal cost nodes:
            # for i in range(1, len(nodes)):
            #     if nodes[i].cost == nodes[i-1].cost:
            #         nodes[i].rank = nodes[i-1].rank
            if nodes[0].cost < self.get_G( self.level, idx):
                self.G[self.level][idx] = nodes[0].cost
                self.V[self.level][idx] = nodes[0]
            for node in nodes:
                new_q.append(node)
        del self.Q_v
        # gc.collect()
        self.Q_v = new_q

    def reconstruct_path(self, start, goal):
        path = []
        v = goal
        path.append(v)
        while v.pose != start.pose:
            v = v.parent
            path.append(v)
        return path

    def get_G(self, level, index):
        if index in self.G[level]:
            return self.G[level][index]
        return 1e9

    def protectedRank(self):
        # only activate best rank node in a grid-cell in Q_v 
        # if its g is better than or equal to G existing (existing G may come from the closed list)

        for i in range(len(self.closed_set)):
            self.closed_set[i].calc_index(self.current_resolution)
            # update the lower level G and V with the best node from the higher level only if it is better.
            if self.closed_set[i].cost < self.get_G(self.level, self.closed_set[i].index):
                self.G[self.level][self.closed_set[i].index] = self.closed_set[i].cost
                self.V[self.level][self.closed_set[i].index] = self.closed_set[i]
        num_active = 0
        for i in range(len(self.Q_v)):
            # it is possible that G_existing was set using a vertex from the self.Q_v itself.
            if self.Q_v[i].rank == 0 and self.Q_v[i].cost <= self.get_G(self.level, self.Q_v[i].index):
                self.Q_v[i].active = 0
                num_active += 1
            else:
                self.Q_v[i].active = 1
        # self.G[self.level] = {}
        # self.V[self.level] = {}
        for i in range(len(self.Q_v)):
            if self.Q_v[i].active == 0:
                self.G[self.level][self.Q_v[i].index] = self.Q_v[i].cost
                self.V[self.level][self.Q_v[i].index] = self.Q_v[i]
        return num_active

    def protectedRankSub(self):
        num_active = self.protectedRank()
        if num_active > self.min_active and self.low_res == False:
            fraction = max(self.subsampling_ratio, self.min_active/num_active)
            # if not random, evenly distribute the active nodes.
            # skip nodes = 1/fraction
            # for i in range(len(self.Q_v)):
            #     if self.Q_v[i].active == 0:
            #         self.Q_v[i].active = np.random.choice([0, 1], p=[fraction, 1-fraction])
            count = 0
            skip = int(1/fraction)
            if fraction < 0.5:
                for i in range(len(self.Q_v)):
                    if self.Q_v[i].active == 0:
                        count += 1
                        if count % skip == 0:
                            self.Q_v[i].active = 0
                        else:
                            self.Q_v[i].active = 1
            else:
                for i in range(len(self.Q_v)):
                    if self.Q_v[i].active == 0:
                        count += 1
                        if count % skip == 0:
                            self.Q_v[i].active = 1
                        else:
                            self.Q_v[i].active = 0

    def reset(self):
        del self.Q_v
        del self.closed_set
        self.Q_v = []
        self.Q_v_hash = set()
        self.closed_set = []
        self.closed_set_hash = set()
        self.G[self.level]    = {}
        self.V[self.level]    = {}
        self.start.active = 0
        self.G[self.level][self.start.index] = 0
        self.V[self.level][self.start.index] = self.start
        heapq.heappush(self.Q_v, self.start)
        self.Q_v_hash.add(self.start.hash)


    def calculate_bottleneck_ratio(self):
        if self.old_Qv is not []:
            ratio = None
            for v in self.old_Qv:
                if v in self.best_path:
                    final_vertex = self.best_path[0]
                    ratio = v.expansions_needed/final_vertex.expansions_needed
                    self.bottle_neck_ratio.append(ratio)
            if ratio is None:
                for v in self.Q_v:
                    for u in self.best_path:
                        if abs(v.pose[0] - u.pose[0]) < 1e-2 and abs(v.pose[1] - u.pose[1]) < 1e-2:
                            final_vertex = self.best_path[0]
                            ratio = v.expansions_needed/final_vertex.expansions_needed
                            self.bottle_neck_ratio.append(ratio)

    def nodes_to_pose_array(self, nodes):
        poses = []
        for node in nodes:
            poses.append(np.array(list(node.pose)))
        return poses

    def vertex_list_to_tree(self, vertex_list):
        tree = []
        for v in vertex_list:
            path = []
            path.append(v.pose)
            while v.pose != self.start.pose:
                v = v.parent
                path.append(np.array(list(v.pose)))
            tree.extend(path)
        tree = np.array(tree)
        # keep only unique poses
        tree = list(np.unique(tree, axis=0))
        return tree

    def ratio_freeze_unfreeze(self):
        denominator = (len(self.Q_v) + len(self.inactive_Q_v))
        if denominator == 0:
            ratio = 0
        else:
            ratio = len(self.inactive_Q_v)/denominator
        self.freeze_ratio.append(ratio)

    def reset_blind(self):
        # self.expansion_counter = 0
        self.G_goal = 1e9
        self.reset()

    def run(self, condition=None, Activate=None, early_break=False, max_expansion=1e9, file=None, capture_tree=False):
        if(condition is None or Activate is None):
            raise ValueError("Condition and Activate methods must be provided")
        # reset expansion counter
        self.expansion_counter = 0
        self.expansion_list = []
        self.break_all_loops = False
        now = time.time()
        self.inconsistency = 0
        avg_Succ_time = []
        avg_heapify_time = []
        avg_g_update_time = []
        self.inconsistencies = []
        self.inactive_Q_v = []
        self.invalid_Q_v = []

        while self.expansion_counter < max_expansion and (self.Q_v or self.inactive_Q_v) and self.level <= self.level_limit and not self.break_all_loops:
            success = False
            self.current_resolution = self.resolution/(self.division_factor**self.level)
            self.inconsistency = 0
            self.vertex_list = []
            if self.static_hysteresis is not None:
                self.low_level_dominance_count = 0
            if file is not None:
                print("shifted resolution: ", self.current_resolution, file=file)
            while self.expansion_counter < max_expansion and (self.Q_v or self.inactive_Q_v) and condition() and not self.break_all_loops:
                u = heapq.heappop(self.Q_v)
                # if u.parent is not None and file is not None:
                #     print("pose: ", u.pose, "f value", u.f_value, "index:", u.index, "cost:", u.cost, "parent:", u.parent.pose, file=file)
                self.closed_set.append(u)
                self.closed_set_hash.add(u.hash)
                self.vertex_list.append(u) # this is for recording purposes only.
                if u.reached_goal_region(self.goal, epsilon=self.epsilon):
                    success = True
                    self.G_goal = u.cost # should this be f_value?
                    self.best_path = self.reconstruct_path(self.start, u)
                    self.path_cost_list.append(self.G_goal)
                    self.path_list.append(self.best_path)
                    self.expansion_list.append(self.expansion_counter) # the diff will automatically tell you how many expansions were done at this level.
                    self.calculate_bottleneck_ratio()
                    self.old_Qv = deepcopy(self.Q_v)
                    self.old_Qv.extend(deepcopy(self.inactive_Q_v))
                    self.vertex_list.extend(self.old_Qv)
                    if early_break:
                        self.break_all_loops = True
                    if self.debug:
                        print(colored("found goal at {} expansions, cost: {}".format(self.expansion_counter, self.G_goal), "green")) # bag this data somehow
                        # if file is not None:
                        #     print("best path: ", file=file)
                        #     for i in range(len(self.best_path)):
                        #         print(self.best_path[i].pose, self.best_path[i].cost, self.best_path[i].f_value, file=file)
                        #     print("=====", file=file)
                    break
                self.heapify = False
                self.expansion_counter += 1
                start_time = time.perf_counter()
                if self.use_elevation:
                    succ = u.Succ(self.costmap, self.elevation_map, resolution=self.current_resolution, capture_tree=capture_tree)
                else:
                    succ = u.Succ(self.bitmap, resolution=self.current_resolution, capture_tree=capture_tree)
                end_time = time.perf_counter()
                avg_Succ_time.append((end_time - start_time))
                for v in succ:
                    v.f_value = v.calc_f(self.goal)
                    v.expansions_needed = self.expansion_counter
                    # if u.parent is not None:
                    #     condition1 = v.pose != u.parent.pose
                    # else:
                    #     condition1 = True
                    # condition1 = v not in self.closed_set and v not in self.Q_v and v not in self.inactive_Q_v
                    condition1 = v.hash not in self.closed_set_hash and v.hash not in self.Q_v_hash and v.hash not in self.inactive_Q_v_hash and v.isvalid
                    if condition1:
                        if v.cost < self.get_G(self.level, v.index):
                            self.GUpdate(v)
                            v.active = 0
                            heapq.heappush(self.Q_v, v)
                            self.Q_v_hash.add(v.hash)
                        else:
                            v.active = 1
                            self.inactive_Q_v.append(v)
                            self.inactive_Q_v_hash.add(v.hash)
                    else:
                        if v.isvalid == False:
                            self.invalid_Q_v.append(v)

            self.inconsistencies.append(self.inconsistency)
            self.level_information.append(self.level)
            if not success:
                self.path_cost_list.append(self.G_goal)
                self.path_list.append(self.best_path)
                self.expansion_list.append(self.expansion_counter) # the diff will automatically tell you how many expansions were done at this level.
            # combine self.Q_v and self.inactive_Q_v:
            self.Q_v.extend(self.inactive_Q_v)
            self.len_last_Qv = len(self.Q_v) # this was introduced for DenseSparseHysteresis
            # for i in range(len(self.Q_v)):
            #     print(colored("hit", "blue"), self.Q_v[i].pose, self.Q_v[i].cost, self.Q_v[i].f_value, self.Q_v[i].active)
            if capture_tree:
                # for v in self.Q_v:
                #     if v.active == 0:
                #         self.vertex_list.append(v)
                self.tree_snapshot.append(deepcopy(self.closed_set))
                output_q_v = deepcopy(self.Q_v)
                output_q_v.extend(deepcopy(self.invalid_Q_v))
                self.Q_v_snapshot.append(output_q_v)
                self.invalid_Q_v = []
            # if len(self.Q_v) < 20:
            #     for v in self.Q_v:
            #         print(v.pose, v.cost, v.f_value, v.hash, v.active)
            self.removeQ()

            # self.closed_update()
            if len(self.Q_v) == 0 and Activate != self.reset_blind:
                break
            if self.debug:
                print("number of expansions at this level:", self.expansion_counter - self.last_counter)
                self.last_counter = self.expansion_counter
                print("length of Q_vertex", len(self.Q_v))
                print("going from level {} to level {}".format(self.level, self.next))
                print("number of heapifies: ", self.heapify_counter)
                print("number of inconsistencies: ", self.inconsistency)
                print("average time for Succ: ", np.mean(np.array(avg_Succ_time)))
                if len(avg_heapify_time):
                    print("average time for heapify: ", np.mean(np.array(avg_heapify_time)))
                else:
                    print("no heapify was done")
            start_time = time.perf_counter()
            self.change_resolution_g_update()
            end_time = time.perf_counter()
            print("time for g_update: ", end_time - start_time)
            self.ActivateMethod(Activate)
            self.ratio_freeze_unfreeze()
        if Activate == self.reset_blind:
            diff_exp = np.diff(np.array(self.expansion_list))
            exp_list = np.array(deepcopy(self.expansion_list))
            exp_list[1:] = diff_exp
            self.expansion_list = list(exp_list)
        return
        # return self.path_list, self.path_cost_list, self.expansion_list, self.closed_set, self.Q_v, self.expansion_counter, self.heapify_counter, self.inconsistencies

    def run_minimal(self, condition=None, Activate=None, early_break=False, max_expansion=1e9, file=None, capture_tree=False):
        if(condition is None or Activate is None):
            raise ValueError("Condition and Activate methods must be provided")
        # reset expansion counter
        self.expansion_counter = 0
        self.expansion_list = []
        self.break_all_loops = False
        now = time.time()
        self.inconsistency = 0
        avg_Succ_time = []
        avg_heapify_time = []
        avg_g_update_time = []
        self.inconsistencies = []
        self.inactive_Q_v = []
        self.invalid_Q_v = []

        while self.expansion_counter < max_expansion and (self.Q_v or self.inactive_Q_v) and self.level <= self.level_limit and not self.break_all_loops:
            success = False
            self.current_resolution = self.resolution/(self.division_factor**self.level)
            inner_loop_start = time.perf_counter()
            check_time = time.perf_counter()
            inactive_insertions = 0
            while self.expansion_counter < max_expansion and (self.Q_v or self.inactive_Q_v) and condition() and not self.break_all_loops:
                u = heapq.heappop(self.Q_v)
                self.closed_set.append(u)
                self.closed_set_hash.add(u.hash)
                # print("pop node:", "{:.4f},".format(u.pose[0]), "{:.4f},".format(u.pose[1]), "{}".format(u.index))
                if u.reached_goal_region(self.goal, epsilon=self.epsilon):
                    success = True
                    self.G_goal = u.cost # should this be f_value?
                    self.best_path = self.reconstruct_path(self.start, u)
                    self.path_cost_list.append(self.G_goal)
                    self.path_list.append(self.best_path)
                    self.expansion_list.append(self.expansion_counter) # the diff will automatically tell you how many expansions were done at this level.
                    if early_break:
                        self.break_all_loops = True
                    if self.debug:
                        print(colored("found goal at {} expansions, cost: {}".format(self.expansion_counter, self.G_goal), "green")) # bag this data somehow
                    break

                self.expansion_counter += 1
                start_time = time.perf_counter()
                if self.use_elevation:
                    succ = u.Succ(self.costmap, self.elevation_map, resolution=self.current_resolution, capture_tree=capture_tree)
                else:
                    succ = u.Succ(self.bitmap, resolution=self.current_resolution, capture_tree=capture_tree)
                end_time = time.perf_counter()
                avg_Succ_time.append((end_time - start_time))
                start_time = time.perf_counter()
                for v in succ:
                    # print(v.pose)
                    v.f_value = v.calc_f(self.goal)
                    v.expansions_needed = self.expansion_counter
                    # print("active node:", "{:.4f},".format(v.pose[0]), "{:.4f},".format(v.pose[1]), "{}".format(v.index))
                    condition1 = v.hash not in self.closed_set_hash and v.hash not in self.Q_v_hash and v.hash not in self.inactive_Q_v_hash
                    if condition1:
                        if v.cost < self.get_G(self.level, v.index):
                            self.GUpdate(v)
                            v.active = 0
                            heapq.heappush(self.Q_v, v)
                            self.Q_v_hash.add(v.hash)
                        else:
                            v.active = 1
                            self.inactive_Q_v.append(v)
                            self.inactive_Q_v_hash.add(v.hash)
                            inactive_insertions += 1
                            # print("froze node:", "{:.4f},".format(v.pose[0]), "{:.4f},".format(v.pose[1]), "{}".format(v.index))

                            # print the pose with 4 decimal points
                            # print("Inactive node:", "{:.4f},".format(v.pose[0]), "{:.4f}".format(v.pose[1]))
                end_time = time.perf_counter()
                avg_g_update_time.append(end_time - start_time)
            end_check_time = time.perf_counter()
            print("time for check: ", 1e3*(end_check_time - check_time))
            if not success:
                self.path_cost_list.append(self.G_goal)
                self.path_list.append(self.best_path)
                self.expansion_list.append(self.expansion_counter) # the diff will automatically tell you how many expansions were done at this level.
            # combine self.Q_v and self.inactive_Q_v:
            self.Q_v.extend(self.inactive_Q_v)
            self.len_last_Qv = len(self.Q_v) # this was introduced for DenseSparseHysteresis
            self.removeQ()
            print(colored(f"Size of Q_v {len(self.Q_v)}, {inactive_insertions}", "red"))

            if len(self.Q_v) == 0 and Activate != self.reset_blind:
                break
            # if self.debug:
            #     print("number of expansions at this level:", self.expansion_counter - self.last_counter)
            #     self.last_counter = self.expansion_counter
            #     print("length of Q_vertex", len(self.Q_v))
            #     print("going from level {} to level {}".format(self.level, self.next))
            #     print("number of heapifies: ", self.heapify_counter)
            #     print("average time for Succ: ", np.mean(np.array(avg_Succ_time)))
            #     if len(avg_heapify_time):
            #         print("average time for heapify: ", np.mean(np.array(avg_heapify_time)))
            #     else:
            #         print("no heapify was done")
            print("average succ time: ", np.mean(np.array(avg_Succ_time)))
            start_time = time.perf_counter()
            self.change_resolution_g_update()
            end_time = time.perf_counter()
            resolution_update_time = end_time - start_time
            # print("time for resolution change g_update: ", end_time - start_time)
            start_time = time.perf_counter()
            self.ActivateMethod(Activate)
            end_time = time.perf_counter()
            activation_time = end_time - start_time
            # print("time for activation: ", end_time - start_time)
            inner_loop_end = time.perf_counter()
            inner_loop_time = inner_loop_end - inner_loop_start
            print(colored(f"inner loop time: {(inner_loop_end - inner_loop_start)}", "yellow"))
            if len(self.expansion_list) > 1:
                min_time = np.mean(np.array(avg_Succ_time)) * (self.expansion_list[-1] - self.expansion_list[-2])
                g_update_time = np.mean(np.array(avg_g_update_time)) * (self.expansion_list[-1] - self.expansion_list[-2])
            else:
                min_time = np.mean(np.array(avg_Succ_time)) * self.expansion_list[-1]
                g_update_time = np.mean(np.array(avg_g_update_time)) * self.expansion_list[-1]
            print(colored(f"successor time: {min_time/inner_loop_time}, g_update: {g_update_time/inner_loop_time} resolution_update: {resolution_update_time/inner_loop_time}, activation: {activation_time/inner_loop_time}", "yellow"))
            print("Size of Q_v just before reboot: ", len(self.Q_v))
        if Activate == self.reset_blind:
            diff_exp = np.diff(np.array(self.expansion_list))
            exp_list = np.array(deepcopy(self.expansion_list))
            exp_list[1:] = diff_exp
            self.expansion_list = list(exp_list)
        return
        # return self.path_list, self.path_cost_list, self.expansion_list, self.closed_set, self.Q_v, self.expansion_counter, self.heapify_counter, self.inconsistencies
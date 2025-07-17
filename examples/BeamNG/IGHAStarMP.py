import torch.multiprocessing as mp
import numpy as np
import time
import traceback
from queue import Empty

class IGHAStarMP:
    """
    Multiprocessing wrapper for the IGHA* planner. Runs the planner in a separate process and communicates via queues.
    """

    def __init__(self, configs):
        mp.set_start_method("spawn", force=True)  # Safe for CUDA
        self.query_queue = mp.Queue(5)
        self.result_queue = mp.Queue(5)
        self.process = mp.Process(
            target=self._planner_process,
            args=(configs, self.query_queue, self.result_queue),
        )
        self.process.start()
        self.path = None
        self.success = False
        self.completed = True
        self.expansion_counter = 0

    def _planner_process(self, configs, query_queue, result_queue):
        """
        Planner process: loads the CUDA/C++ kernel and runs the IGHA* planner in response to queries.
        """
        import torch
        import numpy as np
        from ighastar.scripts.common_utils import create_planner

        planner = create_planner(configs["Planner_config"])
        print("[IGHAStarMP] Planner loaded.")

        map_res = configs["experiment_info_default"]["node_info"]["map_res"]
        offset = None
        avg_dt = 0.1
        stream = torch.cuda.Stream()
        while True:
            task = query_queue.get()
            if task is None:
                time.sleep(0.1)
                continue
            (
                map_center,
                start_state,
                goal_,
                costmap,
                heightmap,
                hysteresis,
                expansion_limit,
                stop,
            ) = task
            try:
                map_size = costmap.shape
                bitmap = torch.ones((map_size[0], map_size[1], 2), dtype=torch.float32)
                bitmap[..., 0] = torch.from_numpy(costmap)
                bitmap[..., 1] = torch.from_numpy(heightmap)
                if offset is None:
                    offset = map_res * np.array(bitmap.shape[:2]) * 0.5
                start = torch.zeros(4, dtype=torch.float32)
                goal = torch.zeros(4, dtype=torch.float32)
                start[:2] = torch.from_numpy(start_state[:2] + offset - map_center)
                goal[:2] = torch.from_numpy(goal_[:2] + offset - map_center)
                start[2] = start_state[5]
                start[3] = float(np.linalg.norm(start_state[6:8]))
                if start[3] > 1.0 and start_state[6] > 0.5:
                    V = start_state[6:8]
                    theta = start_state[5]
                    dx = V[0] * np.cos(theta) - V[1] * np.sin(theta)
                    dy = V[0] * np.sin(theta) + V[1] * np.cos(theta)
                    start[2] = np.arctan2(dy, dx)
                dx = goal[0] - start[0]
                dy = goal[1] - start[1]
                goal[2] = torch.atan2(dy, dx)
                goal[3] = 0.0 if stop else 10.0
                now = time.perf_counter()
                with torch.cuda.stream(stream):
                    success = planner.search(
                        start, goal, bitmap, expansion_limit, hysteresis, True
                    )  # ignore goal validity
                    (
                        avg_successor_time,
                        avg_goal_check_time,
                        avg_overhead_time,
                        avg_g_update_time,
                        switches,
                        max_level_profile,
                        Q_v_size,
                        expansion_counter,
                        expansion_list,
                    ) = planner.get_profiler_info()

                end = time.perf_counter()
                avg_dt = 0.9 * avg_dt + 0.1 * (end - now)
                print(
                    f"[IGHAStarMP] Avg_dt: {avg_dt:.3f}s, Expansions: {expansion_counter}, Success: {success}"
                )
                if success:
                    path = planner.get_best_path().numpy()
                    path = np.flip(path, axis=0)
                    path[..., :2] -= offset
                    path[..., :2] += map_center
                    result_queue.put((True, True, path, expansion_counter))
                else:
                    result_queue.put((False, True, None, expansion_counter))
            except Exception as e:
                print(f"[IGHAStarMP] Planner error: {e}")
                traceback.print_exc()
                result_queue.put((False, True, None, 3))

    def set_query(
        self,
        map_center,
        start_state,
        goal_,
        costmap,
        heightmap,
        hysteresis,
        expansion_limit,
        stop=False,
        disable=False,
    ):
        """
        Submit a new planning query. Returns immediately; results are available via update().
        """
        if disable:
            return
        if np.linalg.norm(start_state[:2] - goal_[:2]) < 5.0:
            print("[IGHAStarMP] Start and goal are too close, skipping query.")
            return
        if not self.completed:
            return
        self.query_queue.put(
            (
                map_center,
                start_state,
                goal_,
                costmap,
                heightmap,
                hysteresis,
                expansion_limit,
                stop,
            )
        )
        self.completed = False
        self.success = False

    def reset(self):
        """Reset planner state."""
        self.path = None
        self.success = False
        self.completed = True
        self.expansion_counter = 0

    def update(self):
        """
        Call this periodically in your main loop to check for planner results.
        Returns (success, path, expansion_counter).
        """
        try:
            (
                self.success,
                self.completed,
                path,
                expansions,
            ) = self.result_queue.get_nowait()
            self.expansion_counter = expansions
            if self.success:
                self.path = path
                return self.success, self.path, self.expansion_counter
            else:
                return False, self.path, self.expansion_counter
        except Empty:
            return False, self.path, self.expansion_counter

    def shutdown(self):
        """Shut down the planner process."""
        self.query_queue.put(None)
        self.process.terminate()
        self.process.join()

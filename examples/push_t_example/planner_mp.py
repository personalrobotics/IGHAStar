"""Async IGHA* for Push-T. Same queue pattern as examples/BeamNG/IGHAStarMP.py."""
import time
import traceback
from queue import Empty
from typing import Optional, Tuple

import numpy as np
import torch.multiprocessing as mp


class PushTPlannerMP:
    def __init__(self, configs):
        mp.set_start_method("spawn", force=True)
        info = configs["experiment_info_default"]
        self.hysteresis = info["hysteresis"]
        self.expansion_limit = info["max_expansions"]
        self.query_queue = mp.Queue(5)
        self.result_queue = mp.Queue(5)
        self.process = mp.Process(
            target=self._planner_process,
            args=(configs, self.query_queue, self.result_queue, self.expansion_limit, self.hysteresis),
        )
        self.process.start()
        self.path = None
        self.controls = None
        self.success = False
        self.completed = True
        self.expansion_counter = 0

    @staticmethod
    def _planner_process(configs, query_queue, result_queue, expansion_limit, hysteresis):
        import numpy as np
        import torch
        from ighastar.scripts.common_utils import create_planner

        planner = create_planner(configs, bidirectional=False)
        print("[PushTPlanner] loaded")
        world = torch.zeros(1)
        while True:
            task = query_queue.get()
            if task is None:
                break
            state, goal, t_world = task
            try:
                start = torch.tensor(state, dtype=torch.float32)
                goal_t = torch.tensor(goal, dtype=torch.float32)
                now = time.perf_counter()
                success = planner.search(start, goal_t, world, expansion_limit, hysteresis, True)
                expansions = planner.get_profiler_info()[7]
                print(
                    f"[PushTPlanner] dt={time.perf_counter() - now:.2f}s "
                    f"expansions={expansions} success={success}"
                )
                if success:
                    path_t, ctrl_t = planner.get_best_path_with_controls()
                    path = np.flip(path_t.numpy(), axis=0)
                    controls = np.flip(ctrl_t.numpy(), axis=0)
                    path[:, -1] += float(t_world)
                    result_queue.put((True, True, path, controls, expansions))
                else:
                    result_queue.put((False, True, None, None, expansions))
            except Exception as e:
                print(f"[PushTPlanner] error: {e}")
                traceback.print_exc()
                result_queue.put((False, True, None, None, 0))

    def set_query(self, state, goal, t_world) -> None:
        if not self.completed:
            return
        self.query_queue.put(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(goal, dtype=np.float32),
                float(t_world),
            )
        )
        self.completed = False
        self.success = False

    def reset(self) -> None:
        self.path = None
        self.controls = None
        self.success = False
        self.completed = True
        self.expansion_counter = 0

    def update(self) -> Tuple[bool, Optional[np.ndarray], int]:
        try:
            self.success, self.completed, path, controls, expansions = self.result_queue.get_nowait()
            self.expansion_counter = expansions
            if self.success:
                self.path = path
                self.controls = controls
            return self.success, self.path, self.expansion_counter
        except Empty:
            return False, self.path, self.expansion_counter

    def shutdown(self) -> None:
        self.query_queue.put(None)
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

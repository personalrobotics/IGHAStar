"""Batched MJWarp rollouts for abc_sim scenes.

One ``WarpBatchRollout`` owns a few MJWarp worlds-batches ("capacities") of the
*same* compiled MuJoCo model. A rollout loads B arbitrary full simulator states
(qpos, qvel) into the first B worlds, ramps the position-actuator targets from
``ctrl_from`` to ``ctrl_to`` over ``n_ctrl`` control periods (each period is
``substeps`` physics steps, replayed from a captured CUDA graph), and records
the state after every control period. Everything stays on the GPU; the caller
gets torch CUDA tensors.

The physics is exactly abc_sim's batched-eval physics: the MJWarp runtime comes
from ``abc_sim.rendering.replay.renderer.WarpReplayRuntime`` (which works around
MJWarp's zero-size-array quirks) and the step is ``mujoco_warp.step``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch


@dataclass
class RolloutResult:
    qpos: torch.Tensor  # [B, T, nq]
    qvel: torch.Tensor  # [B, T, nv]
    site_xpos: torch.Tensor  # [B, T, n_sites, 3]
    overflow: bool  # contact / constraint buffer overflowed at some step


class WarpBatchRollout:
    def __init__(
        self,
        mjm: mujoco.MjModel,
        mjd: mujoco.MjData,
        *,
        capacities: list[int],
        substeps: int,
        site_ids: list[int],
        nconmax: int = 128,
        njmax: int = 1200,
    ) -> None:
        import warp as wp
        from abc_sim.rendering.replay.renderer import WarpReplayRuntime

        self._wp = wp
        self.mjm = mjm
        self.substeps = int(substeps)
        self.site_ids = torch.as_tensor(site_ids, dtype=torch.long)
        self.capacities = sorted(int(c) for c in capacities)
        self.runtimes = {}
        for cap in self.capacities:
            rt = WarpReplayRuntime(mjm, mjd, nworld=cap, nconmax=nconmax, njmax=njmax)
            rt.sync_from_mujoco()
            self.runtimes[cap] = rt
        d0 = self.runtimes[self.capacities[0]].d_warp
        self.device = wp.to_torch(d0.qpos).device
        self.site_ids = self.site_ids.to(self.device)
        self.calls = 0
        self.world_steps = 0  # control periods simulated (incl. padding)

    def _views(self, cap: int):
        wp = self._wp
        d = self.runtimes[cap].d_warp
        views = {
            "qpos": wp.to_torch(d.qpos),
            "qvel": wp.to_torch(d.qvel),
            "ctrl": wp.to_torch(d.ctrl),
            "site_xpos": wp.to_torch(d.site_xpos),
            "nacon": wp.to_torch(d.nacon),
            "nefc": wp.to_torch(d.nefc),
        }
        for name in ("qacc_warmstart", "qacc", "act", "time"):
            arr = getattr(d, name, None)
            if arr is not None and arr.size:
                views[name] = wp.to_torch(arr)
        return d, views

    @torch.no_grad()
    def rollout(
        self,
        qpos0: torch.Tensor,
        qvel0: torch.Tensor,
        ctrl_from: torch.Tensor,
        ctrl_to: torch.Tensor,
        n_ctrl: int,
    ) -> RolloutResult:
        """All inputs are [B, ...] tensors in MuJoCo actuator space."""
        B = qpos0.shape[0]
        out_q, out_v, out_s = [], [], []
        overflow = False
        cap_max = self.capacities[-1]
        for start in range(0, B, cap_max):
            stop = min(B, start + cap_max)
            r = self._rollout_chunk(
                qpos0[start:stop], qvel0[start:stop], ctrl_from[start:stop],
                ctrl_to[start:stop], n_ctrl,
            )
            out_q.append(r.qpos)
            out_v.append(r.qvel)
            out_s.append(r.site_xpos)
            overflow |= r.overflow
        return RolloutResult(
            torch.cat(out_q), torch.cat(out_v), torch.cat(out_s), overflow
        )

    def _rollout_chunk(self, qpos0, qvel0, ctrl_from, ctrl_to, n_ctrl) -> RolloutResult:
        B = qpos0.shape[0]
        cap = next(c for c in self.capacities if c >= B)
        rt = self.runtimes[cap]
        d, v = self._views(cap)
        dev = self.device

        def pad(x):
            x = x.to(dev, torch.float32)
            if B == cap:
                return x
            reps = torch.zeros((cap - B,) + x.shape[1:], device=dev, dtype=x.dtype)
            reps[:] = x[:1]
            return torch.cat([x, reps], 0)

        q0, v0, c0, c1 = pad(qpos0), pad(qvel0), pad(ctrl_from), pad(ctrl_to)
        v["qpos"].copy_(q0)
        v["qvel"].copy_(v0)
        # Fresh solver warm start + clock per edge, so a replay of the same edge
        # from the same state is reproducible.
        for name in ("qacc_warmstart", "qacc", "act", "time"):
            if name in v:
                v[name].zero_()

        T = int(n_ctrl)
        nq, nv = q0.shape[1], v0.shape[1]
        ns = int(self.site_ids.numel())
        rec_q = torch.empty((cap, T, nq), device=dev)
        rec_v = torch.empty((cap, T, nv), device=dev)
        rec_s = torch.empty((cap, T, ns, 3), device=dev)
        max_con = torch.zeros((), device=dev, dtype=torch.int64)
        max_efc = torch.zeros((), device=dev, dtype=torch.int64)
        for t in range(T):
            a = float(t + 1) / float(T)
            v["ctrl"].copy_(c0 + a * (c1 - c0))
            rt.step(nstep=self.substeps)
            rec_q[:, t] = v["qpos"]
            rec_v[:, t] = v["qvel"]
            rec_s[:, t] = v["site_xpos"][:, self.site_ids]
            max_con = torch.maximum(max_con, v["nacon"].max().to(torch.int64))
            max_efc = torch.maximum(max_efc, v["nefc"].max().to(torch.int64))
        overflow = bool(max_con.item() >= d.naconmax) or bool(max_efc.item() >= d.njmax)
        self.calls += 1
        self.world_steps += cap * T
        return RolloutResult(rec_q[:B], rec_v[:B], rec_s[:B], overflow)

    def close(self) -> None:
        for rt in self.runtimes.values():
            rt.close()
        self.runtimes = {}

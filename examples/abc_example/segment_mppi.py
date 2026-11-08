"""UNREVIEWED: written by Claude, not yet read or reviewed by a human. Treat as experimental.

Segment-wise MPPI refinement of a coarse trajectory (system agnostic).

Idea: a trajectory planned with a *coarse* model (large physics step) is turned into one
that works under the *fine* model by re-optimising each control segment: the fine model
is rolled out from its own current state, the coarse plan's state at the end of the
segment is the target (a "knot"), and the coarse plan's action warm-starts the search.
Segments are chained, so the fine rollout follows the plan closed-loop and the refined
action sequence can then be replayed open-loop.

The component interfaces follow ``sidtalia/mppi`` so its pieces can be mixed in:

    Dynamics.forward(states[M,K,T,NX], controls[M,K,T,nu]) -> states[M,K,T,NX]
    Costs.forward(states[M,K,T,NX], controls[M,K,T,nu])    -> cost[K]
    Sampling.sample(states, U[T,nu])        -> controls[M,K,T,nu], perturbation_cost[K]
    Sampling.update_control(cost[K], U, x)  -> controls, U

Here ``states[..., t, :]`` is the state *after* control t (so the state at the end of
the segment is the last one), M is the number of bins (1 here), K the rollouts, T the
horizon in control periods. Controls are absolute (the action itself), not deltas.
Nothing below knows about robots or MuJoCo; ``abc_refine.py`` holds the adapter.
"""

from __future__ import annotations

import torch


class GaussianSampling(torch.nn.Module):
    """Gaussian perturbations of absolute controls around the current mean U.

    Rollout 0 is always the unperturbed mean (the warm start can never be lost).
    ``sigma`` is per control channel (0 freezes a channel); ``lo``/``hi`` clip.
    ``temperature`` > 0 is used as given, <= 0 makes it adaptive:
    ``adaptive_scale * (mean cost - min cost)``, so the weights stay well conditioned
    whatever the units of the cost.
    """

    def __init__(self, sigma, lo, hi, K: int, T: int, temperature: float = 0.0,
                 adaptive_scale: float = 0.3, seed: int = 0, device="cpu", dtype=torch.float64):
        super().__init__()
        self.d, self.dtype, self.K, self.T = device, dtype, K, T
        self.sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
        self.lo = torch.as_tensor(lo, dtype=dtype, device=device)
        self.hi = torch.as_tensor(hi, dtype=dtype, device=device)
        self.nu = int(self.sigma.numel())
        self.temperature, self.adaptive_scale = float(temperature), float(adaptive_scale)
        self.scale = 1.0  # multiplies sigma; annealed by the driver across iterations
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.noise = torch.zeros(K, T, self.nu, dtype=dtype, device=device)

    def reset(self):
        self.scale = 1.0

    def sample(self, states, U):
        n = torch.randn(self.K, self.T, self.nu, generator=self.gen, dtype=self.dtype, device=self.d)
        n = n * self.sigma * self.scale
        n[0] = 0.0
        controls = torch.minimum(torch.maximum(U + n, self.lo), self.hi)
        self.noise = controls - U  # what was actually applied, after clipping
        return controls.unsqueeze(0), torch.zeros(self.K, dtype=self.dtype, device=self.d)

    def update_control(self, cost_total, U, state=None):
        beta = cost_total.min()
        temp = self.temperature if self.temperature > 0 else self.adaptive_scale * (cost_total.mean() - beta)
        w = torch.exp(-(cost_total - beta) / torch.clamp(temp, min=1e-9))
        w = w / w.sum()
        U = torch.minimum(torch.maximum(U + (w.view(-1, 1, 1) * self.noise).sum(0), self.lo), self.hi)
        self.last_weights = w
        return U.unsqueeze(0), U


class SegmentMPPI(torch.nn.Module):
    """The MPPI shell of ``sidtalia/mppi`` for one segment: sample, roll out, score, update."""

    def __init__(self, Dynamics, Costs, Sampling, K: int, T: int, device="cpu", dtype=torch.float64):
        super().__init__()
        self.Dynamics, self.Costs, self.Sampling = Dynamics, Costs, Sampling
        self.K, self.T, self.d, self.dtype = K, T, device, dtype

    def optimize(self, x, U):
        """One MPPI iteration from state ``x`` [NX] with mean controls ``U`` [T,nu]."""
        states = x.view(1, 1, 1, -1).repeat(1, self.K, self.T, 1)
        controls, pert = self.Sampling.sample(states, U)
        states = self.Dynamics.forward(states, controls)
        cost = torch.nan_to_num(self.Costs.forward(states, controls) + pert, nan=1e9, posinf=1e9)
        self.last_cost = cost
        _, U = self.Sampling.update_control(cost, U, x)
        return U

    def forward(self, x, U0, num_iters: int = 4, anneal: float = 0.7):
        U = U0.clone()
        self.Sampling.reset()
        for _ in range(num_iters):
            U = self.optimize(x, U)
            self.Sampling.scale *= anneal
        return U


@torch.no_grad()
def refine_trajectory(mppi: SegmentMPPI, x0, ref_states, ref_controls, horizon: int = 1,
                      num_iters: int = 4, anneal: float = 0.7, log=None):
    """Refine a coarse trajectory under the fine model, one segment at a time.

    x0           [NX]       fine-model start state
    ref_states   [N+1,NX]   coarse state at each knot (ref_states[0] ~ x0)
    ref_controls [N,nu]     coarse controls, ref_controls[t] takes knot t to knot t+1
    Returns (refined controls [N,nu], fine states [N+1,NX], per-segment info list).
    For segment t the horizon is ``min(horizon, N-t)`` periods: the mean starts at the
    coarse controls t..t+h-1, the targets are knots t+1..t+h, only control t is kept.
    """
    N = ref_controls.shape[0]
    x = x0.clone()
    xs, us, info = [x.clone()], [], []
    dyn, cost = mppi.Dynamics, mppi.Costs
    for t in range(N):
        h = min(horizon, N - t)
        mppi.T = h
        mppi.Sampling.T = h
        mppi.Sampling.noise = torch.zeros(mppi.K, h, mppi.Sampling.nu, dtype=mppi.dtype, device=mppi.d)
        cost.set_targets(ref_states[t + 1:t + 1 + h])
        U0 = ref_controls[t:t + h]
        c_before = float(cost.forward(dyn.forward(x.view(1, 1, 1, -1).repeat(1, 1, h, 1), U0.view(1, 1, h, -1)),
                                      U0.view(1, 1, h, -1))[0])
        U = mppi(x, U0, num_iters=num_iters, anneal=anneal)
        u = U[0]
        x = dyn.step(x, u)
        us.append(u.clone())
        xs.append(x.clone())
        info.append({"t": t, "cost_warm": c_before, "cost_best_sample": float(mppi.last_cost.min()),
                     "du_max": float((u - ref_controls[t]).abs().max())})
        if log is not None:
            log(t, info[-1])
    return torch.stack(us), torch.stack(xs), info

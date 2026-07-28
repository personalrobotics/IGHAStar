#!/usr/bin/env python3
"""Self-contained smoke test for IGHA* latent-space planning with a PCA grid
subspace -- the IGHA*-side plumbing for the SimDist world-model plan.

This validates the integration WITHOUT any SimDist / JAX / flax dependency, by
substituting:
  - the world-model encoder with a placeholder that yields a 64-D latent z,
  - the learned latent dynamics g(z, a) with a placeholder (a low-rank
    integrator embedded in the 64-D latent), and
  - the real dataset PCA with PCA fit on sampled placeholder latents.

It demonstrates the locked-in design:
  - dynamics runs over the FULL 64-D latent (no reduction),
  - IGHA* grids/dedups only the top-k (whitened) PCA dims via HASH_DIMS=k,
  - goal test / heuristic are L2 in that k-D subspace,
and confirms IGHA* needs no C++/core changes (config + Python callbacks only).

Swap the three placeholders for SimDist's encode_latent, the trained g, and PCA
fit on the real dataset to get the actual Go2 latent planner.

Run:  python generic_latent_planning_smoke.py
"""
import os

import numpy as np
import torch

from ighastar.scripts.common_utils import create_planner

torch.set_default_dtype(torch.float32)

LATENT_DIM = 64  # world-model latent dim (N_DIMS)
ACT_DIM = 12  # Go2 joint-target action dim (N_CONT)
DT = 0.5
ACTIVE = 2  # placeholder dynamics moves only the first ACTIVE latent dims
VAR_THRESHOLD = 0.99  # pick k = dims capturing this much variance


# --------------------------------------------------------------------------
# Placeholder world model (stands in for SimDist encode_latent + trained g)
# --------------------------------------------------------------------------
def encode_latent_placeholder(seed_vec: np.ndarray) -> np.ndarray:
    """A real encoder maps obs -> z64. Here we just place a point in latent."""
    z = np.zeros(LATENT_DIM, dtype=np.float32)
    z[: len(seed_vec)] = seed_vec
    return z


def raw_dynamics(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Placeholder latent dynamics g(z64, a) over the FULL latent. Here only the
    first ACTIVE latent dims are driven (1:1) by the first ACTIVE action dims --
    a low-dim manifold embedded in 64-D, which is what PCA should discover."""
    delta = torch.zeros_like(z)
    delta[:, :ACTIVE] = a[:, :ACTIVE] * DT
    return z + delta


# --------------------------------------------------------------------------
# Whitened PCA (fit on sampled latents) -> rotation used for the grid subspace
# --------------------------------------------------------------------------
def fit_whitened_pca(Z: np.ndarray, var_threshold: float, eps: float = 1e-6):
    mean = Z.mean(axis=0)
    Zc = Z - mean
    cov = (Zc.T @ Zc) / max(1, Zc.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]  # descending
    eigvals = np.clip(eigvals[order], 0.0, None)
    components = eigvecs[:, order]  # columns = components, desc variance
    scale = np.sqrt(eigvals + eps)
    ratio = eigvals / max(eigvals.sum(), 1e-12)
    k = int(np.searchsorted(np.cumsum(ratio), var_threshold) + 1)
    k = max(1, min(k, Z.shape[1]))
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "scale": scale.astype(np.float32),
        "k": k,
        "explained": ratio.astype(np.float32),
    }


def sample_reachable_latents(z0: np.ndarray, n: int = 4000) -> np.ndarray:
    """Roll out random actions from z0 to sample the reachable latent set."""
    rng = np.random.default_rng(0)
    z = torch.from_numpy(np.tile(z0, (n, 1)))
    out = [z.clone()]
    for _ in range(8):
        a = torch.from_numpy(
            (rng.standard_normal((n, ACT_DIM)) * 1.5).astype(np.float32)
        )
        z = raw_dynamics(z, a)
        out.append(z.clone())
    return torch.cat(out, dim=0).numpy()


def main() -> None:
    # Placeholder "encodings" of the current and goal states.
    z0_raw = encode_latent_placeholder(np.array([0.0, 0.0], dtype=np.float32))
    zg_raw = encode_latent_placeholder(np.array([2.0, 2.0], dtype=np.float32))

    # Fit whitened PCA on sampled reachable latents and choose k.
    pca = fit_whitened_pca(sample_reachable_latents(z0_raw), VAR_THRESHOLD)
    k = pca["k"]
    print(f"PCA: chose k={k} grid dims; top-6 explained var = "
          f"{np.round(pca['explained'][:6], 4)}")

    mean = torch.from_numpy(pca["mean"])
    comp = torch.from_numpy(pca["components"])  # [64,64]
    scale = torch.from_numpy(pca["scale"])  # [64]

    def to_pca(z_raw: torch.Tensor) -> torch.Tensor:
        return ((z_raw - mean) @ comp) / scale

    def from_pca(z_pca: torch.Tensor) -> torch.Tensor:
        return (z_pca * scale) @ comp.T + mean

    z0_pca = to_pca(torch.from_numpy(z0_raw).unsqueeze(0))[0]
    zg_pca = to_pca(torch.from_numpy(zg_raw).unsqueeze(0))[0]

    # 8-direction control primitives in the first two action dims.
    angles = torch.linspace(0, 2 * np.pi, 9)[:-1]
    base = torch.zeros(8, ACT_DIM)
    base[:, 0] = torch.cos(angles)
    base[:, 1] = torch.sin(angles)
    GOAL_EPS = 0.5

    def sample_controls():
        return base

    def dynamics(z_pca, controls):
        # operate in PCA basis: rotate out -> apply g over full 64-D -> rotate in
        return to_pca(raw_dynamics(from_pca(z_pca), controls))

    def cost(z_pca, controls, z_pca_next):
        return torch.norm((z_pca_next - z_pca)[:, :k], dim=1)

    def validity(z_pca):
        return torch.isfinite(z_pca).all(dim=1).float()

    def heuristic(z_pca):
        return torch.norm((z_pca - zg_pca.unsqueeze(0))[:, :k], dim=1)

    def goal_test(z_pca):
        return bool(torch.norm((z_pca - zg_pca)[:k]).item() < GOAL_EPS)

    # First k dims get a real grid resolution (whitened units); the rest are
    # large so they never participate in the hash/dominance grid.
    resolution = [0.25] * k + [1e9] * (LATENT_DIM - k)
    tolerance = [0.1] * k + [1e9] * (LATENT_DIM - k)

    config = {
        "experiment_info_default": {
            "state_dim": LATENT_DIM,
            "control_dim": ACT_DIM,
            "hash_dims": k,  # <-- grid only the top-k PCA dims
            "num_controls": base.shape[0],
            "resolution": resolution,
            "tolerance": tolerance,
            "bounds_lower": [-1e9] * LATENT_DIM,
            "bounds_upper": [1e9] * LATENT_DIM,
            "max_level": 4,
            "division_factor": 2.0,
            "max_expansions": 20000,
            "hysteresis": 2000,
            "preemptive_expansion": {"enabled": False, "min_preemptive": 8,
                                     "max_preemptive": 32},
            "node_info": {"node_type": "generic", "timesteps": 1},
        },
        "sample_controls_fn": sample_controls,
        "dynamics_fn": dynamics,
        "cost_fn": cost,
        "validity_fn": validity,
        "heuristic_fn": heuristic,
        "goal_test_fn": goal_test,
    }

    planner = create_planner(config, bidirectional=False)
    start = z0_pca.clone().float()
    goal = zg_pca.clone().float()  # ignored by goal_test, but must be n_dims
    world = torch.zeros(1, dtype=torch.float32)

    ok = planner.search(start, goal, world, 20000, 2000, True)
    info = planner.get_profiler_info()
    print(f"success={ok} expansions={info[7]} preempt={planner.get_preemptive_expansions()}")
    if ok:
        path = planner.get_best_path().numpy()  # rows in PCA basis (64-D) + g
        print(f"latent plan: {path.shape[0]} states, cost {path[0, -1]:.3f}")
        # Map the planned latents back to raw space and show the active dims.
        zpath = torch.from_numpy(path[::-1, :LATENT_DIM].copy())  # start->goal
        raw = from_pca(zpath).numpy()
        print(f"start active dims: {np.round(raw[0, :ACTIVE], 3)} -> "
              f"goal active dims: {np.round(raw[-1, :ACTIVE], 3)} "
              f"(target {zg_raw[:ACTIVE]})")
    else:
        print("No latent plan found.")


if __name__ == "__main__":
    main()

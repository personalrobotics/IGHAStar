"""Push-T control sampling. Pixel-space delta MPPI; clones sidtalia/MPPI if needed."""
import importlib.util
import subprocess
from pathlib import Path

import torch

_MPPI_URL = "https://github.com/sidtalia/MPPI.git"
# Sibling of the ighastar repo: .../src/MPPI next to .../src/ighastar.
_REPO_PARENT = Path(__file__).resolve().parents[3]
_MPPI_ROOT = _REPO_PARENT / "MPPI"


def _ensure_mppi() -> Path:
    # Prefer an existing checkout (local or cloned). Layout differs across MPPI versions.
    for root in (_MPPI_ROOT / "mppi", _MPPI_ROOT):
        if (root / "MPPI.py").is_file() or (root / "mppi" / "MPPI.py").is_file():
            return root if (root / "MPPI.py").is_file() else root / "mppi"
    print(f"[push_t] cloning {_MPPI_URL} into {_MPPI_ROOT}")
    subprocess.check_call(["git", "clone", "--depth", "1", _MPPI_URL, str(_MPPI_ROOT)])
    if (_MPPI_ROOT / "mppi" / "MPPI.py").is_file():
        return _MPPI_ROOT / "mppi"
    if (_MPPI_ROOT / "MPPI.py").is_file():
        return _MPPI_ROOT
    raise FileNotFoundError(f"cloned MPPI at {_MPPI_ROOT} but MPPI.py was not found")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find(root: Path, *candidates: str) -> Path:
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"none of {candidates} under {root}")


_root = _ensure_mppi()
_DeltaSampling = _load(
    "delta_sampling",
    _find(_root, "scripts/Sampling/Delta_Sampling.py", "Sampling/Delta_Sampling.py"),
).Delta_Sampling
MPPI = _load("mppi_mod", _find(_root, "MPPI.py")).MPPI


class Delta_Sampling(_DeltaSampling):
    def __init__(self, sampling_config, *args, **kwargs):
        # Integration gain. Commands are pixel targets, so a delta of 1 px stays 1 px.
        # Limits are filled by set_command_box each step.
        # Upstream MPPI may want dt or scaled_dt; we only need the integrate hook below.
        cfg = {
            **sampling_config,
            "dt": 1.0,
            "scaled_dt": 1.0,
            "limits": {k: 0.0 for k in ("min_0", "max_0", "min_1", "max_1")},
        }
        super().__init__(cfg, *args, **kwargs)
        self.dt = 1.0

    def set_command_box(self, center, radius, lo, hi) -> None:
        c = torch.as_tensor(center, dtype=self.dtype, device=self.d)
        self.min_val = torch.clamp(c - radius, lo, hi)
        self.max_val = torch.clamp(c + radius, lo, hi)

    def _integrate(self, origin, delta):
        # Pixel targets. Upstream Delta_Sampling clamps to car units; do not call it.
        return torch.clamp(origin + self.dt * torch.cumsum(delta, dim=-2), self.min_val, self.max_val)

    def sample(self, state, U):
        self.noise = (
            torch.matmul(
                torch.randn((self.K, self.T, self.nu), device=self.d, dtype=self.dtype),
                self.CTRL_NOISE,
            )
            + self.CTRL_NOISE_MU
        )
        perturbed = U + self.noise
        controls = self._integrate(state[..., -self.nu :], perturbed.unsqueeze(0))
        perturbed[:, 1:, :] = torch.diff(controls - state[..., -self.nu :], dim=-2).squeeze(0) / self.dt
        self.noise = perturbed - U
        action_cost = self.temperature * torch.matmul(self.noise, self.CTRL_NOISE_inv)
        return controls, torch.sum(U * action_cost, dim=(1, 2))

    def update_control(self, cost_total, U, state):
        beta = torch.min(cost_total)
        self.cost_total = cost_total.clone()
        weights = torch.exp((-1 / self.temperature) * (cost_total - beta))
        omega = weights / torch.sum(weights)
        U = U + (omega.view(-1, 1, 1) * self.noise).sum(dim=0)
        return self._integrate(state[-self.nu :], U), U

"""Chipmunk Push-T rollouts via one gym-pusht env.

ponytail: serial O(K * T * n_substeps) pymunk steps. Upgrade path is a batched stepper.
"""
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch

# Match pusht_physics.h. The planner does not read these from YAML.
DT = 0.01
KP = 100.0
KV = 20.0


def hold_action(core, action, n_sub, vmax):
    # Same PD and agent speed cap as pusht_physics::control_step. gym's env.step has no cap.
    act = np.asarray(action, dtype=np.float64)
    a = core.agent
    for _ in range(int(n_sub)):
        acc = KP * (act - a.position) + KV * (-a.velocity)
        a.velocity = a.velocity + acc * DT
        speed = float(a.velocity.length)
        if vmax > 0.0 and speed > vmax:
            a.velocity = a.velocity * (vmax / speed)
        core.space.step(DT)


def _core(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def _read(core) -> np.ndarray:
    a = core.agent
    b = core.block
    return np.array(
        [
            a.position.x,
            a.position.y,
            b.position.x,
            b.position.y,
            b.angle,
            a.velocity.x,
            a.velocity.y,
            b.velocity.x,
            b.velocity.y,
            b.angular_velocity,
        ],
        dtype=np.float32,
    )


class PushTDynamics:
    def __init__(self, dynamics_config, mppi_config, dtype=torch.float32, device=torch.device("cpu")):
        self.dtype = dtype
        self.d = device
        self.n_sub = int(dynamics_config["n_substeps"])
        self.lo = float(dynamics_config["workspace_lo"])
        self.hi = float(dynamics_config["workspace_hi"])
        self.vmax = float(dynamics_config["max_agent_vel"])
        self.K = int(mppi_config["ROLLOUTS"])
        self.T = int(mppi_config["TIMESTEPS"])
        self.env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
        self.env.reset(seed=0)
        self.core = _core(self.env)
        self.states = None

    def _clip_agent(self):
        a = self.core.agent
        a.position = (
            float(np.clip(a.position.x, self.lo, self.hi)),
            float(np.clip(a.position.y, self.lo, self.hi)),
        )
        speed = float(a.velocity.length)
        if self.vmax > 0.0 and speed > self.vmax:
            a.velocity = a.velocity * (self.vmax / speed)

    def _write(self, s):
        b = self.core.block
        a = self.core.agent
        b.angle = float(s[4])
        b.position = (float(s[2]), float(s[3]))
        a.position = (float(s[0]), float(s[1]))
        a.velocity = (float(s[5]), float(s[6]))
        b.velocity = (float(s[7]), float(s[8]))
        b.angular_velocity = float(s[9])

    def _hold(self, action):
        act = np.asarray(action, dtype=np.float64)
        a = self.core.agent
        for _ in range(self.n_sub):
            acc = KP * (act - a.position) + KV * (-a.velocity)
            a.velocity = a.velocity + acc * DT
            self._clip_agent()
            self.core.space.step(DT)
            self._clip_agent()

    def forward(self, states, controls):
        # states/controls: [M, K, T, *]. M is 1.
        s0 = states.detach().to("cpu").numpy()
        u = controls.detach().to("cpu").numpy()
        out = np.empty_like(s0)
        for k in range(self.K):
            pose = s0[0, k, 0, :10].copy()
            for t in range(self.T):
                self._write(pose)
                self._hold(u[0, k, t])
                pose = _read(self.core)
                out[0, k, t, :10] = pose
                out[0, k, t, 10:] = u[0, k, t]
        self.states = torch.as_tensor(out, dtype=self.dtype, device=self.d)
        return self.states

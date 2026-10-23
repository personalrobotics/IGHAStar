"""Track a Push-T reference: pusher xy, block xy, block angle."""
import torch


class PushTCost(torch.nn.Module):
    def __init__(self, cost_config, dtype=torch.float32, device=torch.device("cpu")):
        super().__init__()
        self.d = device
        self.w_pusher = torch.tensor(cost_config["w_pusher"], dtype=dtype, device=device)
        self.w_block = torch.tensor(cost_config["w_block"], dtype=dtype, device=device)
        self.w_theta = torch.tensor(cost_config["w_theta"], dtype=dtype, device=device)
        self.path = None

    def set_path(self, path: torch.Tensor) -> None:
        # [T, 10] pose, same T as the MPPI horizon.
        self.path = path.to(device=self.d)

    def forward(self, state, controls):
        ref = self.path
        dth = state[..., 4] - ref[:, 4]
        dth = torch.atan2(torch.sin(dth), torch.cos(dth))
        err = (
            self.w_pusher * ((state[..., 0:2] - ref[:, 0:2]) ** 2).sum(-1)
            + self.w_block * ((state[..., 2:4] - ref[:, 2:4]) ** 2).sum(-1)
            + self.w_theta * dth ** 2
        )
        return err.mean(dim=0).sum(dim=-1)

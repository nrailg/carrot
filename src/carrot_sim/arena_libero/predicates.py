import torch


def bowl_on_plate(
    relative_position: torch.Tensor,
    support_force: torch.Tensor,
    linear_velocity: torch.Tensor,
    finger_positions: torch.Tensor,
) -> torch.Tensor:
    return (
        (relative_position[:, :2].norm(dim=-1) < 0.06)
        & (relative_position[:, 2] > 0.0)
        & (relative_position[:, 2] < 0.12)
        & (support_force[:, 2] > 0.1)
        & (linear_velocity.norm(dim=-1) < 0.05)
        & (finger_positions.min(dim=-1).values > 0.025)
    )

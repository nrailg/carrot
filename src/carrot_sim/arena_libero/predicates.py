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


def released_and_stable(
    object_position: torch.Tensor,
    tcp_position: torch.Tensor,
    linear_velocity: torch.Tensor,
) -> torch.Tensor:
    return ((object_position - tcp_position).norm(dim=-1) > 0.10) & (
        linear_velocity.norm(dim=-1) < 0.05
    )


def supported_on(
    relative_position: torch.Tensor,
    half_size: torch.Tensor,
    support_force: torch.Tensor,
) -> torch.Tensor:
    return (
        (relative_position[:, :2].abs() < half_size[:2]).all(dim=-1)
        & (relative_position[:, 2] > 0)
        & (relative_position[:, 2] < half_size[2])
        & (support_force[:, 2] > 0.1)
    )


def inside_drawer_and_closed(
    local_position: torch.Tensor,
    object_half_extent: torch.Tensor,
    interior_half_size: torch.Tensor,
    joint_position: torch.Tensor,
    closed_position: float,
    closed_tolerance: float,
) -> torch.Tensor:
    return ((local_position.abs() + object_half_extent) < interior_half_size).all(dim=-1) & (
        (joint_position - closed_position).abs() <= closed_tolerance
    )


def stove_is_on(joint_position: torch.Tensor) -> torch.Tensor:
    angle = joint_position.abs()
    return (angle >= 0.35) & (angle <= 2 * torch.pi - 0.35)

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
    release_distance: float = 0.10,
) -> torch.Tensor:
    return ((object_position - tcp_position).norm(dim=-1) > release_distance) & (
        linear_velocity.norm(dim=-1) < 0.05
    )


def placed_beside(
    relative_position: torch.Tensor,
    target_half_size: tuple[float, float, float] | torch.Tensor,
    support_half_size: tuple[float, float, float] | torch.Tensor,
    side: str,
    edge_distance: tuple[float, float],
    side_tolerance: float,
) -> torch.Tensor:
    axis = 0 if side in ("left", "right") else 1
    sign = 1 if side in ("left", "back") else -1
    target_half = torch.as_tensor(target_half_size, device=relative_position.device)
    support_half = torch.as_tensor(support_half_size, device=relative_position.device)
    separation = target_half + support_half
    edge = relative_position[:, axis] * sign - separation[..., axis]
    return (
        (relative_position[:, 1 - axis].abs() < separation[..., 1 - axis] * side_tolerance)
        & (edge > edge_distance[0])
        & (edge < edge_distance[1])
        & (relative_position[:, 2].abs() < separation[..., 2] + 0.02)
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

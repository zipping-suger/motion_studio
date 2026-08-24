"""Sync-free port of spider.math's quaternion difference.

spider's `quat_to_vel` boolean-indexes, which costs a host-device sync
per call — twice per reward call in the rollout hot loop. These compute
the identical values with `torch.where`, so torch.compile traces through.

Solve venv only (torch).
"""

from __future__ import annotations

import torch


def mul_quat(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Quaternion product, (..., 4) wxyz."""
    uw, ux, uy, uz = u.unbind(-1)
    vw, vx, vy, vz = v.unbind(-1)
    return torch.stack(
        [
            uw * vw - ux * vx - uy * vy - uz * vz,
            uw * vx + ux * vw + uy * vz - uz * vy,
            uw * vy - ux * vz + uy * vw + uz * vx,
            uw * vz + ux * vy - uy * vx + uz * vw,
        ],
        dim=-1,
    )


def quat_to_vel(quat: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz quaternion -> (..., 3) angular velocity."""
    axis = quat[..., 1:4]
    sin_a_2 = torch.norm(axis, dim=-1)
    speed = 2.0 * torch.atan2(sin_a_2, quat[..., 0])
    # axis-angle beyond pi rotates the opposite way
    speed = torch.where(speed > torch.pi, speed - 2.0 * torch.pi, speed)
    zero = sin_a_2 == 0.0
    scale = torch.where(zero, torch.zeros_like(speed),
                        speed / torch.where(zero, torch.ones_like(sin_a_2),
                                            sin_a_2))
    return axis * scale.unsqueeze(-1)


def quat_sub(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
    """Angular difference qa ⊖ qb -> (..., 3), wxyz."""
    qneg = qb * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qb.device)
    return quat_to_vel(mul_quat(qneg, qa))

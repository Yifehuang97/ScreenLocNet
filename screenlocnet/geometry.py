"""Geometric parameterisations of a screen.

The screen is a rectangle in 3D, so its four corners have only 8 degrees of freedom,
not 12. Predicting the constrained 8-DOF form rather than four free corners is what
makes the largest single difference in the paper (11.22 cm vs 16.81 cm Corner-Dist).

`two_corners_sc` is the parameterisation used for every reported number. The others
are kept because the supplementary compares against them.
"""
import math

import torch

# number of raw outputs the head must produce for each parameterisation
DOF = {
    "four_corners": 12,
    "three_corners": 9,
    "two_corners_sc": 8,
    "transformation_quaternion": 8,
    "transformation_angle_axis": 8,
}


def decode(pred_type: str, pred: torch.Tensor) -> torch.Tensor:
    """Map the head's raw output to four corners, returned as (B, 12) in tl,tr,bl,br order."""
    if pred_type == "four_corners":
        return pred

    if pred_type == "three_corners":
        tl, tr, bl = pred[:, :3], pred[:, 3:6], pred[:, 6:9]
        centre = (tr + bl) / 2
        br = 2 * centre - tl
        return torch.cat([tl, tr, bl, br], dim=1)

    if pred_type == "two_corners_sc":
        # Predict the two diagonal corners, then place the remaining diagonal by a
        # direction on the sphere. The rectangle and coplanarity constraints then fix
        # the other two corners exactly.
        tl, br = pred[:, :3], pred[:, 3:6]
        centre = (tl + br) / 2
        theta = torch.sigmoid(pred[:, 6]) * (2 * math.pi)
        phi = torch.sigmoid(pred[:, 7]) * math.pi
        r = torch.norm(tl - centre, dim=1)
        bl = centre + torch.stack(
            [
                r * torch.sin(phi) * torch.cos(theta),
                r * torch.sin(phi) * torch.sin(theta),
                r * torch.cos(phi),
            ],
            dim=1,
        )
        tr = 2 * centre - bl
        return torch.cat([tl, tr, bl, br], dim=1)

    raise ValueError(
        f"unsupported pred_type {pred_type!r}; the transformation_* variants from the "
        f"supplementary are not needed to reproduce the paper and are omitted here"
    )

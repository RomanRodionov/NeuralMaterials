import torch
from .torch_coords import *

class RusinkiewiczTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            wi = wi / (torch.norm(wi, dim=-1, keepdim=True) + 1e-6)
            wo = wo / (torch.norm(wo, dim=-1, keepdim=True) + 1e-6)

            # half vector
            h = wi + wo
            h = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-6)

            n = torch.tensor([0.0, 0.0, 1.0], device=wi.device, dtype=wi.dtype)

            # angle between h and normal
            theta_h = torch.acos(torch.clamp((h * n).sum(-1), -1.0, 1.0))

            # define half-vector frame
            up = torch.tensor([0.0, 1.0, 0.0], device=wi.device, dtype=wi.dtype)
            right = torch.nn.functional.normalize(torch.linalg.cross(up.expand_as(h), h), dim=-1)
            new_up = torch.linalg.cross(h, right)

            # transform wi into half-vector frame
            wi_hx = (wi * right).sum(-1)
            wi_hy = (wi * new_up).sum(-1)
            wi_hz = (wi * h).sum(-1)

            # arccos of z component
            theta_d = torch.acos(torch.clamp(wi_hz, -1.0, 1.0))

            # phi_d = atan2(y, x)
            phi_d = torch.atan2(wi_hy, wi_hx)

            theta_h = 2.0 * (theta_h / torch.pi) - 1.0
            theta_d = 2.0 * (theta_d / torch.pi) - 1.0
            phi_d   = (phi_d   / torch.pi)

            return torch.stack([theta_h , theta_d, phi_d], dim=-1)

class Rusinkiewicz6DTransform(torch.nn.Module):
    """Returns (...,6) = [h_x,h_y,h_z, d_x,d_y,d_z]."""
    def __init__(self):
        super().__init__()

    def forward(self, wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            half = wi + wo
            half = half / half.norm(dim=-1, keepdim=True)

            sph_coords = xyz2sph(half)
            theta_h, phi_h = sph_coords[..., 1], sph_coords[..., 2]

            b_i_normal = torch.tensor([0.0, 1.0, 0.0], dtype=wi.dtype, device=wi.device)
            normal = torch.tensor([0.0, 0.0, 1.0], dtype=wi.dtype, device=wi.device)

            wi = rotate_vector(wi, normal.expand_as(wi), -phi_h)
            diff = rotate_vector(wi, b_i_normal.expand_as(wi), -theta_h)

            out = torch.cat([half, diff], dim=-1)
            return out
import torch

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

def _norm(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)

class Rusinkiewicz6DTransform(torch.nn.Module):
    """Returns (...,6) = [h_x,h_y,h_z, d_x,d_y,d_z]."""
    def __init__(self):
        super().__init__()

    def forward(self, wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        # wi, wo: (...,3)
        with torch.no_grad():
            wi = _norm(wi)
            wo = _norm(wo)

            h = wi + wo
            h_norm = h.norm(dim=-1, keepdim=True)
            # fallback when wi ~ -wo
            h = torch.where(h_norm < 1e-6, wi, h)
            h = _norm(h)

            # stable up vector: use (0,1,0) except when nearly parallel, then use (1,0,0)
            leading = [1] * (h.dim() - 1)  # e.g. for (N,3) -> [1]
            up0 = torch.tensor([0.0, 1.0, 0.0], device=h.device, dtype=h.dtype).view(*leading, 3).expand_as(h)
            up1 = torch.tensor([1.0, 0.0, 0.0], device=h.device, dtype=h.dtype).view(*leading, 3).expand_as(h)

            dot_up0 = (h * up0).sum(dim=-1).abs()   # (...,)
            use_up1 = dot_up0 > 0.999               # (...)
            mask = use_up1.unsqueeze(-1)            # (...,1)
            up = torch.where(mask, up1, up0)        # (...,3)

            # Orthonormal basis around h
            x_axis = _norm(torch.linalg.cross(up, h))      # (...,3)
            y_axis = torch.linalg.cross(h, x_axis)         # (...,3)

            # Express wo in this frame
            d_x = (wo * x_axis).sum(dim=-1, keepdim=True)
            d_y = (wo * y_axis).sum(dim=-1, keepdim=True)
            d_z = (wo * h).sum(dim=-1, keepdim=True)

            # Output shape (...,6)
            out = torch.cat([h, torch.cat([d_x, d_y, d_z], dim=-1)], dim=-1)
            return out
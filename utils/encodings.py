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

            enc = torch.stack([
                torch.sin(theta_h), torch.cos(theta_h),
                torch.sin(theta_d), torch.cos(theta_d),
                torch.sin(phi_d),   torch.cos(phi_d),
            ], dim=-1)

            #return torch.stack([theta_h , theta_d, phi_d], dim=-1)
            return enc

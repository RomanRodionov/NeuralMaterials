import torch

def rotate_vector(v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor):
    axis_unit = axis / axis.norm(dim=-1, keepdim=True)
    
    cos_theta = torch.cos(angle).unsqueeze(-1)
    sin_theta = torch.sin(angle).unsqueeze(-1)
    
    term1 = v * cos_theta
    term2 = torch.cross(axis_unit, v, dim=-1) * sin_theta
    term3 = axis_unit * (torch.sum(axis_unit * v, dim=-1, keepdim=True) * (1 - cos_theta))
    
    return term1 + term2 + term3

def xyz2sph(v: torch.Tensor):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r_xy2 = x**2 + y**2
    r = torch.sqrt(r_xy2 + z**2)
    theta = torch.atan2(torch.sqrt(r_xy2), z)
    phi = torch.atan2(y, x)
    return torch.stack((r, theta, phi), dim=-1)

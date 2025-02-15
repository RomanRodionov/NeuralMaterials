import numpy as np
import torch

def phong(view, light):
    c = 50
    d_color = torch.tensor([0.1, 0.6, 0.8]) * 0.2
    s_color = torch.tensor([0.9, 0.9, 0.9]) * 0.8
    n = torch.tensor([0, 0, 1], dtype=torch.float32)
    r = 2 * torch.dot(n, light) * n - light
    diffuse = torch.dot(n, light)
    s = torch.dot(r, view)
    specular = 0
    if s > 0:
        specular = s**c
    return specular * s_color + diffuse * d_color

import numpy as np
from numpy.random import randn

import numpy as np


def sample_rusinkiewicz(lobe_degree=1):
    # Sample half-angle (theta_h) and difference (theta_d) angles
    theta_h = np.arccos(np.random.rand()**(1/lobe_degree))
    phi_h = 2 * np.pi * np.random.rand()
    theta_d = np.arccos(2 * np.random.rand() - 1)
    phi_d = 2 * np.pi * np.random.rand()

    # Half-vector h
    sh, ch = np.sin(theta_h), np.cos(theta_h)
    sph, cph = np.sin(phi_h), np.cos(phi_h)
    h = np.array([sh * cph, sh * sph, ch])

    # Construct local tangent/bitangent frame for h
    t = np.cross(np.array([0, 0, 1]), h)
    t_norm = np.linalg.norm(t)
    t = t / t_norm if t_norm > 1e-7 else np.array([1, 0, 0])
    b = np.cross(h, t)

    # Difference vector in local frame
    sd, cd = np.sin(theta_d), np.cos(theta_d)
    spd, cpd = np.sin(phi_d), np.cos(phi_d)
    d_local = np.array([sd * cpd, sd * spd, cd])

    # Incident vector in world coordinates
    wi = d_local[0] * t + d_local[1] * b + d_local[2] * h
    wi /= np.linalg.norm(wi)

    # Outgoing vector by reflection
    dot_hi = np.dot(wi, h)
    wo = 2 * dot_hi * h - wi
    wo /= np.linalg.norm(wo)

    # Stack wi and wo into a single (2,3) array
    rays = np.stack([wi, wo], axis=0)
    return rays

def sample_sphere(num, dim=3):
    sample = randn(num, dim)
    sample = sample / np.linalg.norm(sample, axis=-1, keepdims=True)
    return sample

def sample_hemisphere(num, dim=3):
    sample = randn(num, dim)
    sample = sample / np.linalg.norm(sample, axis=-1, keepdims=True)
    sample[:, -1] = np.abs(sample[:, -1])
    return sample

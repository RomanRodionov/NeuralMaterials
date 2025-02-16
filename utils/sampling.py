import numpy as np
from numpy.random import randn

def sample_sphere(num, dim=3):
    sample = randn(num, dim)
    sample = sample / np.linalg.norm(sample, axis=-1, keepdims=True)
    return sample

def sample_hemisphere(num, dim=3):
    sample = randn(num, dim)
    sample = sample / np.linalg.norm(sample, axis=-1, keepdims=True)
    sample[:, -1] = np.abs(sample[:, -1])
    return sample

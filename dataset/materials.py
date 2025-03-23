import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from brdf_models import phong
from graphics_utils import film_refl, principled_bsdf
from utils.sampling import *
from external.compact_spectra import *

class TextureDataset(Dataset):
    #texture_types = ["basecolor", "diffuse", "displacement", "height", "metallic", "normal", "opacity", "roughness", "specular"]
    texture_types = ["basecolor", "metallic", "roughness", "specular"]

    def __init__(self, path, resolution=(1024, 1024), n_samples=1000):
        self.n_samples = n_samples
        self.path = path
        self.resolution = resolution
        self.textures = {}
        for tex in self.texture_types:
            tex_path = os.path.join(self.path, f"{tex}.png")
            self.textures[tex] = self.load_texture(tex_path)
        self.channels = np.concatenate([tex for tex in self.textures.values()], axis=-1)
    
    def load_texture(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load texture: {path}")
        if not self.resolution is None:
            img = cv2.resize(img, self.resolution)
        img = img.astype(np.float32) / 255
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1) 
        return img
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        u, v = np.random.rand(), np.random.rand()
        return self.sample(u, v)
    
    def sample(self, u, v):
        h, w = self.resolution

        i = u * (h - 1)
        j = v * (w - 1)

        i0, j0 = int(np.floor(i)), int(np.floor(j))
        i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)

        di, dj = i - i0, j - j0

        def interpolate(tex_array):
            return (
                (1 - di) * (1 - dj) * tex_array[i0, j0] +
                (1 - di) * dj * tex_array[i0, j1] +
                di * (1 - dj) * tex_array[i1, j0] +
                di * dj * tex_array[i1, j1]
            )

        vec_params = torch.tensor(interpolate(self.channels), dtype=torch.float32)
        dict_params = {tex: torch.tensor(interpolate(self.textures[tex]), dtype=torch.float32) for tex in self.texture_types}

        return vec_params, dict_params
    
class PrincipledDataset(Dataset):
    def __init__(self, textures, n_samples=1000):
        self.n_samples = n_samples
        self.textures = textures

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        vec_params, dict_params = self.textures[idx]        
        
        samples = sample_hemisphere(2)
        normal = np.array((0, 0, 1))

        gt_bsdf = principled_bsdf(samples[0], samples[1], normal, dict_params["basecolor"], dict_params["metallic"], dict_params["roughness"], dict_params["specular"])
        
        samples    = torch.tensor(samples, dtype=torch.float32)
        gt_bsdf    = torch.tensor(gt_bsdf, dtype=torch.float32)

        return samples[0], samples[1], vec_params, gt_bsdf
    
    def sample(self, u, v):
        vec_params, dict_params = self.textures.sample(u, v)       
        
        samples = sample_hemisphere(2)
        normal = np.array((0, 0, 1))

        gt_bsdf = principled_bsdf(samples[0], samples[1], normal, dict_params["basecolor"], dict_params["metallic"], dict_params["roughness"], dict_params["specular"])
        
        samples    = torch.tensor(samples, dtype=torch.float32)
        gt_bsdf    = torch.tensor(gt_bsdf, dtype=torch.float32)

        return samples[0], samples[1], vec_params, gt_bsdf
    
class PhongDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        rnd = torch.randn(2, 3, dtype=torch.float32)
        sample = rnd / torch.linalg.norm(rnd, dim=-1, keepdim=True)
        return sample[0], sample[1], phong(sample[0], sample[1])

class IridescenceDataset(Dataset):
    def __init__(self, min_wavelength=380, max_wavelength=780, film_thickness=300, n_samples=1000):
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.range = max_wavelength - min_wavelength
        self.film_thickness = film_thickness
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        samples = sample_hemisphere(2)
        wavelength = np.random.rand() * self.range + self.min_wavelength
        value = film_refl(samples[0], samples[1], self.film_thickness, wavelength)

        samples    = torch.tensor(samples, dtype=torch.float32)
        wavelength = torch.tensor(wavelength, dtype=torch.float32)
        value      = torch.tensor(value, dtype=torch.float32)

        return samples[0], samples[1], wavelength, value

class MomentsDataset(Dataset):
    def __init__(self, min_wavelength=380, max_wavelength=780, film_thickness=300, n_moments=6, n_points=100, n_samples=1000):
        self.film_thickness = film_thickness
        self.n_points  = n_points
        self.n_moments = n_moments
        self.n_samples = n_samples
        
        self.mirror_signal = True
        self.use_warp      = False
        self.use_lagrange  = True
    
        self.wavelengths = np.linspace(min_wavelength, max_wavelength, n_points)
        self.phases = WavelengthToPhase(self.wavelengths, self.wavelengths.min(), self.wavelengths.max(), self.mirror_signal, self.use_warp)
    
    def __len__(self):
        return self.n_samples
    
    def moments_to_spectrum(self, bounded_moments):
        if(self.use_lagrange):
            bounded_mese = EvaluateBoundedMESELagrange(self.phases, bounded_moments)
        else:
            bounded_mese = EvaluateBoundedMESEDirect(self.phases, bounded_moments)
        return bounded_mese
    
    def spectrum_to_moments(self, spectrum):
        return ComputeTrigonometricMoments(self.phases, spectrum, self.n_moments, self.mirror_signal)
    
    def __getitem__(self, idx):
        samples = sample_hemisphere(2)

        reflectance = np.zeros(self.n_points)
        for i, wavelength in enumerate(self.wavelengths):
            reflectance[i] = film_refl(samples[0], samples[1], self.film_thickness, wavelength)

        bounded_moments = ComputeTrigonometricMoments(self.phases, reflectance, self.n_moments, self.mirror_signal)

        samples         = torch.tensor(samples, dtype=torch.float32)
        bounded_moments = torch.tensor(bounded_moments, dtype=torch.float32)

        return samples[0], samples[1], bounded_moments

if __name__ == "__main__":
    # just example
    path = "../resources/materials/test/Metal/tc_metal_029"
    dataset = TextureDataset(path)
    for i in range(5):
        sample = dataset[i]
        print(sample.shape, sample)
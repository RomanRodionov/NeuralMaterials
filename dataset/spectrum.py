import torch
import numpy as np
from scipy.stats import norm
from torch.utils.data import Dataset

class PolynomialDataset(Dataset):
    def __init__(self, n_samples=1000, degree=3, num_points=50, noise=True):
        self.n_samples = n_samples
        self.degree = degree
        self.num_points = num_points
        self.noise = noise
        self.x = np.linspace(-1, 1, num_points)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        coeffs = np.random.uniform(-1, 1, self.degree + 1)
        y = np.polyval(coeffs, self.x)

        if self.noise:
            y += np.sin(self.x * np.random.uniform(1, 10)) * np.random.uniform(0.1, 0.5)
            y += np.random.normal(0, 0.05, size=self.num_points)

        # random scaling
        y = (y - y.min()) / (y.max() - y.min())
        scale = np.random.uniform(0.25, 1)
        y = y * scale + np.random.uniform(0, 1 - scale)

        return torch.tensor(y, dtype=torch.float32)
    
# Based on:
# https://github.com/calvin-brown/Spectral-encoder/blob/main/generate_spectra.py

def calculate_std(fwhm):
    return fwhm / 2.355

class SpectrumDataset(Dataset):
    def __init__(self, n_samples=1000, num_points=50, noise=True):
        self.n_samples = n_samples
        self.num_points = num_points
        self.noise = noise
        self.x = np.linspace(-1, 1, num_points)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        min_wavelength = 380
        max_wavelength = 780
        wavelengths = np.linspace(min_wavelength, max_wavelength, self.num_points)
        max_n_peaks = 5
        min_power = 0.1
        max_power = 2
        power_range = max_power - min_power
        min_fwhm = 1
        max_fwhm = 100
        fwhm_range = max_fwhm - min_fwhm
        noise_power = 0.005

        curr_spectrum = np.zeros(self.num_points)
        n_peaks = np.random.randint(1, max_n_peaks + 1)

        for peak in range(n_peaks):
            center = np.random.choice(wavelengths)
            power = np.random.rand() * power_range + min_power
            fwhm = np.random.rand() * fwhm_range + min_fwhm
            curr_peak = norm.pdf(wavelengths, center, calculate_std(fwhm))
            curr_peak = curr_peak * (power/np.max(curr_peak))
            curr_spectrum += curr_peak

        if self.noise:
            curr_spectrum += np.random.normal(0, noise_power, self.num_points)

        return torch.tensor(curr_spectrum, dtype=torch.float32)

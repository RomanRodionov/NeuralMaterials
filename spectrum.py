import numpy as np

def read_spectrum(filename, wavelength_range):
    data = np.loadtxt(filename)
    wavelengths, intensities = data[:, 0], data[:, 1]

    new_intensities = np.interp(wavelength_range, wavelengths, intensities, left=intensities[0], right=intensities[-1])
    min_intensity, max_intensity = new_intensities.min(), new_intensities.max()
    new_intensities = (new_intensities - min_intensity) / (max_intensity - min_intensity)

    return new_intensities
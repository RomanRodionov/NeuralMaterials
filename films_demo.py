import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from graphics_utils import film_refl
from dataset.materials import IridescenceDataset
from brdf_decoder import SpectralDecoder, initialize_weights

DECODER_RAW_PATH = "saved_models/spectral_decoder.bin"
WL_MIN = 360
WL_MAX = 830

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = 10000000
    batch_size = 1024
    wl_samples = 128

    dataset = IridescenceDataset(max_wavelength=WL_MAX, min_wavelength=WL_MIN, n_samples=samples, film_thickness=300, wl_samples=wl_samples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    decoder = SpectralDecoder(hidden_dim=16, output_dim=1, wavelength_max=WL_MAX, wavelength_min=WL_MIN).to(device)
    initialize_weights(decoder, "normal")
    optimizer = optim.Adam(decoder.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=samples // batch_size, eta_min=1e-5)

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, wavelengths, labels = data
        w_i = w_i.flatten(end_dim=-2).to(device)
        w_o = w_o.flatten(end_dim=-2).to(device)
        wavelengths = wavelengths.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        value = decoder(w_i, w_o, wavelengths.flatten()).reshape((wavelengths.shape[0], wl_samples))

        loss1 = F.mse_loss(value, labels)

        sam = F.cosine_similarity(value, labels).mean()
        loss2 = (1 - sam)

        loss = loss1 + 0.02 * loss2
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch % 25 == 0:
            progress_bar.set_description(f"Batch {batch}, MSE loss: {loss1.item():.6f}, SAM: {sam.item():.4f}")

    decoder.save_raw(DECODER_RAW_PATH)
    print(decoder.decoder)
    
    return decoder.to("cpu")

if __name__ == "__main__":
    decoder = train_decoder().cpu()

    num_points = 100
    wavelength_range = np.linspace(360, 830, num_points)
    thickness = 300
    num_angles = 50

    w_i_vectors = [
        (np.cos(theta), 0.0, np.sin(theta)) for theta in np.linspace(0, np.pi / 2, num_angles)
    ]
    w_o = (0.5, 0.0, 0.5)  # doesn't mean yet

    eta_i = np.array([1.0, 0.0], dtype=np.float32)
    eta_f = np.array([2.0, 0.0], dtype=np.float32)
    eta_t = np.array([1.5, 0.0], dtype=np.float32)

    gt_map = np.zeros((num_angles, num_points))
    predicted_map = np.zeros((num_angles, num_points))

    for i, w_i in enumerate(w_i_vectors):
        gt_map[i, :] = film_refl(w_i, w_o, eta_i, eta_f, eta_t, thickness, wavelength_range)
        for j, wavelength in enumerate(wavelength_range):
            with torch.no_grad():
                predicted_map[i, j] = decoder(
                    torch.tensor(w_i, dtype=torch.float32),
                    torch.tensor(w_o, dtype=torch.float32),
                    torch.tensor(wavelength, dtype=torch.float32)
                ).item()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axs[0].imshow(gt_map, aspect='auto', cmap='jet', extent=[360, 830, 0, 90])
    axs[0].set_title("Ground Truth Reflection")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Incident Angle (degrees)")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(predicted_map, aspect='auto', cmap='jet', extent=[360, 830, 0, 90])
    axs[1].set_title("Predicted Reflection")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Incident Angle (degrees)")
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.show()
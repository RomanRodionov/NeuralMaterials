import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from graphics_utils import film_refl
from dataset.materials import IridescenceDataset
from brdf_decoder import SpectralDecoder, initialize_weights

DECODER_RAW_PATH = "saved_models/spectral_decoder.bin"

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = 25000000
    batch_size = 2048

    dataset = IridescenceDataset(n_samples=samples, film_thickness=300)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)

    decoder = SpectralDecoder(hidden_dim=16, output_dim=1).to(device)
    initialize_weights(decoder, "normal")
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, wavelength, labels = data
        w_i = w_i.to(device)
        w_o = w_o.to(device)
        wavelength = wavelength.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        value = decoder(w_i, w_o, wavelength).squeeze()
        loss = loss_fn(value, labels)
        loss.backward()
        optimizer.step()
        
        if batch % 25 == 0:
            progress_bar.set_description(f"Batch {batch}, MSE: {loss.item():.6f}")

    decoder.save_raw(DECODER_RAW_PATH)
    
    return decoder.to("cpu")

if __name__ == "__main__":
    decoder = train_decoder().cpu()

    num_points = 100
    wavelength_range = np.linspace(380, 780, num_points)
    thickness = 300
    num_angles = 50

    w_i_vectors = [
        (np.cos(theta), 0.0, np.sin(theta)) for theta in np.linspace(0, np.pi / 2, num_angles)
    ]
    w_o = (0.5, 0.0, 0.5)  # doesn't mean yet

    gt_map = np.zeros((num_angles, num_points))
    predicted_map = np.zeros((num_angles, num_points))

    for i, w_i in enumerate(w_i_vectors):
        for j, wavelength in enumerate(wavelength_range):
            gt_map[i, j] = film_refl(w_i, w_o, thickness, wavelength)
            
            with torch.no_grad():
                predicted_map[i, j] = decoder(
                    torch.tensor(w_i, dtype=torch.float32),
                    torch.tensor(w_o, dtype=torch.float32),
                    torch.tensor(wavelength, dtype=torch.float32)
                ).item()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axs[0].imshow(gt_map, aspect='auto', cmap='jet', extent=[380, 780, 0, 90])
    axs[0].set_title("Ground Truth Reflection")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Incident Angle (degrees)")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(predicted_map, aspect='auto', cmap='jet', extent=[380, 780, 0, 90])
    axs[1].set_title("Predicted Reflection")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Incident Angle (degrees)")
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.show()
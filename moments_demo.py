import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from graphics_utils import film_refl
from dataset.materials import MomentsDataset
from brdf_decoder import MomentsDecoder, initialize_weights


def train_decoder(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 128
    n_moments = dataset.n_moments
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    decoder = MomentsDecoder(hidden_dim=16, output_dim=n_moments + 1).to(device)
    initialize_weights(decoder, "normal")
    optimizer = optim.Adam(decoder.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    loss_fn = nn.MSELoss()

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, labels = data
        w_i = w_i.to(device)
        w_o = w_o.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        value = decoder(w_i, w_o).squeeze()
        loss = loss_fn(value, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch % 10 == 0:
            progress_bar.set_description(f"Batch {batch}, MSE: {loss.item():.8f}")
    
    return decoder.to("cpu")

if __name__ == "__main__":
    film_thickness=300
    eta_i = np.array([1.0, 0.0], dtype=np.float32)
    eta_f = np.array([2.0, 0.0], dtype=np.float32)
    eta_t = np.array([1.5, 0.0], dtype=np.float32)
    wl_samples=300
    n_moments=15
    n_samples=50000

    dataset = MomentsDataset(film_thickness=film_thickness,
                             eta_i=eta_i,
                             eta_f=eta_f,
                             eta_t=eta_t,
                             wl_samples=wl_samples,
                             n_moments=n_moments,
                             n_samples=n_samples)
    decoder = train_decoder(dataset).cpu()

    wavelengths = np.linspace(380, 780, wl_samples)
    num_angles = 50

    w_i_vectors = [
        (np.cos(theta), 0.0, np.sin(theta)) for theta in np.linspace(0, np.pi / 2, num_angles)
    ]
    w_o = (0.5, 0.0, 0.5)  # doesn't mean yet

    gt_map        = np.zeros((num_angles, wl_samples))
    mese_map      = np.zeros((num_angles, wl_samples))
    predicted_map = np.zeros((num_angles, wl_samples))

    for i, w_i in enumerate(w_i_vectors):
        with torch.no_grad():
            predicted = decoder(
                torch.tensor(w_i, dtype=torch.float32),
                torch.tensor(w_o, dtype=torch.float32)
            )
            predicted_map[i] = dataset.moments_to_spectrum(predicted.numpy())


        gt_map[i] = film_refl(w_i, w_o, eta_i, eta_f, eta_t, film_thickness, wavelengths)

        mese_map[i] = dataset.moments_to_spectrum(dataset.spectrum_to_moments(gt_map[i]))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axs[0].imshow(gt_map, aspect='auto', cmap='jet', extent=[380, 780, 0, 90])
    axs[0].set_title("Ground Truth Reflection")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Incident Angle (degrees)")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(mese_map, aspect='auto', cmap='jet', extent=[380, 780, 0, 90])
    axs[1].set_title("Unpacked MESE Reflection (n_moments=15)")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Incident Angle (degrees)")
    fig.colorbar(im2, ax=axs[1])

    im3 = axs[2].imshow(predicted_map, aspect='auto', cmap='jet', extent=[380, 780, 0, 90])
    axs[2].set_title("Predicted Reflection")
    axs[2].set_xlabel("Wavelength (nm)")
    axs[2].set_ylabel("Incident Angle (degrees)")
    fig.colorbar(im3, ax=axs[2])

    plt.tight_layout()
    plt.show()
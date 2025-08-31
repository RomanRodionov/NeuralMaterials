import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from graphics_utils import film_brdf, real_fourier_moments
from dataset.materials import RoughFilmsDataset
from brdf_decoder import NBRDFDecoder, initialize_weights

DECODER_RAW_PATH = "saved_models/rgb_decoder.bin"

def mean_absolute_logarithmic_error(y_true, y_pred):
    return torch.mean(torch.abs(torch.log(1 + y_true) - torch.log(1 + y_pred)))

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = 10000000
    batch_size = 2048

    dataset = RoughFilmsDataset(n_samples=samples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    decoder = NBRDFDecoder(hidden_dim=21).to(device)
    initialize_weights(decoder, "uniform")
    optimizer = optim.Adam(decoder.parameters(), lr=5e-4,
                           betas=(0.9, 0.999),
                         eps=1e-15,  # eps=None raises error
                         weight_decay=0.0,
                         amsgrad=False)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=samples // batch_size, eta_min=5e-4)

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, labels = data
        w_i = w_i.flatten(end_dim=-2).to(device)
        w_o = w_o.flatten(end_dim=-2).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        value = decoder(w_i, w_o)

        value = value*w_i[..., -1].unsqueeze(-1)
        labels = labels*w_i[..., -1].unsqueeze(-1)
        #f_value = torch.log(1 + f_value*w_i[..., -1].unsqueeze(-1))
        #f_labels = torch.log(1 + f_labels*w_i[..., -1].unsqueeze(-1))
        #print(value)

        loss1 = mean_absolute_logarithmic_error(value, labels)

        loss = loss1
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        if batch % 25 == 0:
            progress_bar.set_description(f"Batch {batch}, MSE loss: {loss1.item():.6f}")

    decoder.save_raw(DECODER_RAW_PATH)
    print(decoder.decoder)
    
    return decoder.to("cpu")

if __name__ == "__main__":
    decoder = train_decoder().cpu()

    num_points = 3
    thickness = 400
    alpha=0.25
    num_angles = 50
    wavelengths = np.array([650, 550, 425])

    w_i_vectors = np.array([
        (np.cos(theta), 0.0, np.sin(theta)) for theta in np.linspace(0, np.pi / 2, num_angles)
    ])
    w_o = np.array((0.0, 0.0, 1.0))  # doesn't mean yet

    eta_i = np.array([1.0, 0.0], dtype=np.float32)
    eta_f = np.array([2.0, 0.0], dtype=np.float32)
    eta_t = np.array([1.5, 0.0], dtype=np.float32)
    normal = np.array([0, 0, 1], dtype=np.float32)

    gt_map        = np.zeros((num_angles, num_points))
    mese_map      = np.zeros((num_angles, num_points))
    predicted_map = np.zeros((num_angles, num_points))

    for i, w_i in enumerate(w_i_vectors):
        w_o = torch.tensor(w_i)
        w_o[0] = -w_o[0]
        #print(w_i, w_o)
        with torch.no_grad():
            predicted = decoder(
                torch.tensor(w_i, dtype=torch.float32),
                torch.tensor(w_o, dtype=torch.float32)
            )
            predicted_map[i] = predicted.numpy()


        gt_map[i] = film_brdf(w_i, w_o, normal, eta_i, eta_f, eta_t, thickness, alpha, wavelengths)


    norm = colors.Normalize(vmin=0.0, vmax=25.0)

    # 2. Choose a blue-green colormap, e.g. "viridis" or "Blues"
    cmap = "Spectral_r"

    plt.rcParams.update({
        'font.size': 14,           # общий размер текста
        'axes.titlesize': 14,      # размер заголовков осей
        'axes.labelsize': 14,      # размер подписей осей
        'xtick.labelsize': 14,     # размер подписей делений по x
        'ytick.labelsize': 14,     # размер подписей делений по y
        'legend.fontsize': 14,     # размер текста в легенде
        'figure.titlesize': 14     # размер заголовка всей фигуры
    })

    # 3. Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')

    # 4. Plot each with shared norm
    im1 = axs[0].imshow(gt_map, aspect='auto', cmap=cmap, norm=norm,
                        extent=[380, 780, 0, 90])
    axs[0].set_title("Оригинальный спектр")
    axs[0].set_xlabel("Длина волны (нм)")
    axs[0].set_ylabel("Угол наблюдения (°)")

    im3 = axs[1].imshow(predicted_map, aspect='auto', cmap=cmap, norm=norm,
                        extent=[380, 780, 0, 90])
    axs[1].set_title("Предсказанный спектр")
    axs[1].set_xlabel("Длина волны (нм)")
    axs[1].set_ylabel("Угол наблюдения (°)")

    # 5. Add a single colorbar for all subplots
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Отражательная способность (0—1)")

    plt.show()
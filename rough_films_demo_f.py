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
from dataset.materials import FourierRoughFilmsDataset
from brdf_decoder import FSpectralDecoder, initialize_weights

DECODER_RAW_PATH = "saved_models/fourier_decoder.bin"
WL_MIN = 360
WL_MAX = 830
MOMENTS = 10
WL_SAMPLES = 300

def weighted_fourier_loss(pred, target, weights=None, reduction='mean'):
    diff = pred - target
    loss = diff.pow(2)
    if weights is not None:
        loss = loss * weights.view(1, -1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def real_fourier_series(phases, moments):
    moments = moments.copy()
    return np.sum(np.cos(phases[:, None] * np.arange(moments.shape[0])[None]) * moments[None], axis=1)

def real_fourier_series_torch(phases, moments):
    _, K = moments.shape

    k = torch.arange(K, dtype=phases.dtype, device=phases.device)
    angles = phases.unsqueeze(1) * k.unsqueeze(0)
    cos_mat = torch.cos(angles)
    res = moments.unsqueeze(1) * cos_mat.unsqueeze(0)
    return res.sum(dim=-1)

def to_phases(wl, wl_min, wl_max):
    return np.pi * (wl - wl_min) / (wl_max - wl_min) - np.pi

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = 2500000
    batch_size = 5000
    wl_samples = 100

    loss_weights = torch.tensor([(1.0 + 1.0 / (k+1)) for k in range(MOMENTS)]).to(device)
    loss_weights /= torch.sum(loss_weights)

    dataset = FourierRoughFilmsDataset(max_wavelength=WL_MAX, min_wavelength=WL_MIN, n_samples=samples, wl_samples=wl_samples, moments=MOMENTS)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    phases = torch.tensor(dataset.phases, dtype=torch.float32).to(device)

    decoder = FSpectralDecoder(hidden_dim=32, output_dim=MOMENTS, wavelength_max=WL_MAX, wavelength_min=WL_MIN).to(device)
    initialize_weights(decoder, "normal")
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=samples // batch_size, eta_min=5e-4)

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, labels, f_labels = data
        w_i = w_i.flatten(end_dim=-2).to(device)
        w_o = w_o.flatten(end_dim=-2).to(device)
        labels = labels.to(device)
        f_labels = f_labels.to(device)
        optimizer.zero_grad()
        f_value = decoder(w_i, w_o)

        #print(f"Labels: {f_labels[0]}, \nValues: {f_value[0]}")

        value = real_fourier_series_torch(phases, f_value)

        sam = F.cosine_similarity(value, labels).mean()

        #value = torch.log(1 + value*w_i[..., -1].unsqueeze(-1))
        #labels = torch.log(1 + labels*w_i[..., -1].unsqueeze(-1))
        #f_value = torch.log(1 + f_value*w_i[..., -1].unsqueeze(-1))
        #f_labels = torch.log(1 + f_labels*w_i[..., -1].unsqueeze(-1))
        #print(value)

        loss1 = F.mse_loss(value, labels)
        loss2 = F.mse_loss(f_value, f_labels)
        #loss2 = weighted_fourier_loss(f_value, f_labels, weights=loss_weights)

        loss3 = (1 - sam)

        loss = loss1 + 0.1 * loss2 + 0.02 * loss3
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch % 25 == 0:
            progress_bar.set_description(f"Batch {batch}, MSE loss: {loss1.item():.6f}, {loss2.item():.6f}, {loss3.item():.6f}, SAM: {sam.item():.4f}")

    decoder.save_raw(DECODER_RAW_PATH)
    print(decoder.decoder)
    
    return decoder.to("cpu")

if __name__ == "__main__":
    decoder = train_decoder().cpu()

    num_points = 100
    wavelength_range = np.linspace(WL_MIN, WL_MAX, num_points, endpoint=False)
    thickness = 400
    alpha=0.2
    num_angles = 50

    wavelengths = np.linspace(WL_MIN, WL_MAX, num_points, endpoint=False) + 0.5 * (wavelength_range / num_points) 
    phases = to_phases(wavelengths, WL_MIN, WL_MAX)

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
            predicted_map[i] = real_fourier_series(phases, predicted.numpy())


        gt_map[i] = film_brdf(w_i, w_o, normal, eta_i, eta_f, eta_t, thickness, alpha, wavelengths)

        mese_map[i] = real_fourier_series(phases, real_fourier_moments(phases, gt_map[i], MOMENTS))


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
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')

    # 4. Plot each with shared norm
    im1 = axs[0].imshow(gt_map, aspect='auto', cmap=cmap, norm=norm,
                        extent=[380, 780, 0, 90])
    axs[0].set_title("Оригинальный спектр")
    axs[0].set_xlabel("Длина волны (нм)")
    axs[0].set_ylabel("Угол наблюдения (°)")

    im2 = axs[1].imshow(mese_map, aspect='auto', cmap=cmap, norm=norm,
                        extent=[380, 780, 0, 90])
    axs[1].set_title(f"Реконструкция ряда Фурье\n(первые {MOMENTS} коэффициентов)")
    axs[1].set_xlabel("Длина волны (нм)")
    axs[1].set_ylabel("Угол наблюдения (°)")

    im3 = axs[2].imshow(predicted_map, aspect='auto', cmap=cmap, norm=norm,
                        extent=[380, 780, 0, 90])
    axs[2].set_title("Предсказанный спектр")
    axs[2].set_xlabel("Длина волны (нм)")
    axs[2].set_ylabel("Угол наблюдения (°)")

    # 5. Add a single colorbar for all subplots
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Отражательная способность (0—1)")

    plt.show()
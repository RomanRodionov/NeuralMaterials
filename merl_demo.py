import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from external.merl.dataset import MerlDataset, brdf_to_rgb
from brdf_decoder import NBRDFDecoder, initialize_weights
from utils.encodings import Rusinkiewicz6DTransform

DECODER_RAW_PATH = "saved_models/rgb_decoder.bin"

def mean_absolute_logarithmic_error(y_true, y_pred):
    return torch.mean(torch.abs(torch.log(1 + y_true) - torch.log(1 + y_pred)))

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #samples = 5000000
    batch_size = 2048
    epochs = 100

    dataset = MerlDataset(merlPath="./external/merl/BRDFDatabase/brdfs/orange-paint.binary", batchsize=batch_size)

    decoder = NBRDFDecoder(hidden_dim=21, encoder=False).to(device)
    initialize_weights(decoder, "uniform")
    optimizer = optim.Adam(decoder.parameters(), lr=5e-4,
                           betas=(0.9, 0.999),
                           eps=1e-15,
                           weight_decay=0.0,
                           amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    for epoch in range(epochs):
        dataset.shuffle()
        num_batches = int(dataset.train_samples.shape[0] / batch_size)
        progress_bar = tqdm(range(num_batches))

        for batch in progress_bar:

            optimizer.zero_grad()

            mlp_input, groundTruth = dataset.get_trainbatch(batch * dataset.bs)
            output = decoder(mlp_input).to(device)

            rgb_pred = brdf_to_rgb(mlp_input, output)
            rgb_true = brdf_to_rgb(mlp_input, groundTruth)

            loss = mean_absolute_logarithmic_error(y_true=rgb_true, y_pred=rgb_pred)
            loss.backward()
            optimizer.step()

            if batch % 25 == 0:
                progress_bar.set_description(f"Batch {batch}, MSE loss: {loss.item():.6f}")
        scheduler.step()

    decoder.save_raw(DECODER_RAW_PATH)
    print(decoder.decoder)
    
    return decoder.to("cpu")

if __name__ == "__main__":
    decoder = train_decoder().cpu()
    decoder.encoder = Rusinkiewicz6DTransform()
    #decoder.scale = torch.tensor([1./1500, 1.15/1500, 1.66/1500])

    num_points = 3
    num_angles = 50

    w_i_vectors1 = np.array([
        (np.cos(theta), 0.0, np.sin(theta)) for theta in np.linspace(0, np.pi / 2, num_angles)
    ])

    w_i_vectors2 = np.array([
        (0.0, -np.cos(theta), np.sin(theta)) for theta in np.linspace(0, np.pi / 2, num_angles)
    ])

    normal = np.array([0, 0, 1], dtype=np.float32)

    predicted_map1 = np.zeros((num_angles, num_points))
    predicted_map2 = np.zeros((num_angles, num_points))

    for i, w_i in enumerate(w_i_vectors1):
        w_o = w_i.copy()
        w_o[0] = -w_o[0]
        #print(w_i, w_o)
        with torch.no_grad():
            predicted = decoder(
                torch.tensor(w_i, dtype=torch.float32),
                torch.tensor(w_o, dtype=torch.float32)
            )
            predicted_map1[i] = predicted.numpy()

    for i, w_i in enumerate(w_i_vectors2):
        w_o = w_i.copy()
        w_o[1] = -w_o[1]
        #print(w_i, w_o)
        with torch.no_grad():
            predicted = decoder(
                torch.tensor(w_i, dtype=torch.float32),
                torch.tensor(w_o, dtype=torch.float32)
            )
            predicted_map2[i] = predicted.numpy()


    norm = colors.Normalize(vmin=0.0, vmax=1000.0)

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
    fig, axs = plt.subplots(1, 2, figsize=(5, 5), layout='constrained')

    # 4. Plot each with shared norm
    im1 = axs[0].imshow(predicted_map1, aspect='auto', cmap=cmap, norm=norm,
                        extent=[380, 780, 0, 90])
    axs[0].set_title("Предсказанный спектр")
    axs[0].set_xlabel("Длина волны (нм)")
    axs[0].set_ylabel("Угол наблюдения (°)")

    im2 = axs[1].imshow(predicted_map2, aspect='auto', cmap=cmap, norm=norm,
                        extent=[380, 780, 0, 90])
    axs[1].set_title("Предсказанный спектр")
    axs[1].set_xlabel("Длина волны (нм)")
    axs[1].set_ylabel("Угол наблюдения (°)")

    # 5. Add a single colorbar for all subplots
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Отражательная способность (0—1)")

    plt.show()
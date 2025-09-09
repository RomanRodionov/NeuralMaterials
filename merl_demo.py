import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from external.merl.dataset import MerlDataset, brdf_to_rgb, brdf_to_rgb_, fastmerl, brdf_values
from brdf_decoder import NBRDFDecoder, KAN_BRDF, initialize_weights
from utils.encodings import Rusinkiewicz6DTransform

DECODER_RAW_PATH = "saved_models/rgb_decoder.bin"
merlPath = "./external/merl/BRDFDatabase/brdfs/red-metallic-paint.binary"

def mean_absolute_logarithmic_error(y_true, y_pred):
    return torch.mean(torch.abs(torch.log(1 + y_true) - torch.log(1 + y_pred)))

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #samples = 5000000
    batch_size = 2048
    epochs = 60

    dataset = MerlDataset(merlPath=merlPath, batchsize=batch_size, nsamples=800000)

    #decoder = NBRDFDecoder(hidden_dim=21, encoder=False).to(device)
    decoder = KAN_BRDF(dim=[6, 6, 6, 3], k=3, nCps=10, encoder=False).to(device)
    #initialize_weights(decoder, "uniform")
    optimizer = optim.AdamW(decoder.parameters(), lr=1e-3,
                           betas=(0.9, 0.999),
                           eps=1e-15,
                           weight_decay=0.0,
                           amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    for epoch in range(epochs):
        dataset.shuffle()

        train_loss = [] 
        val_loss = []

        train_batches = int(dataset.train_samples.shape[0] / batch_size)
        train_progress = tqdm(range(train_batches), leave=False)
        for batch in train_progress:

            optimizer.zero_grad()

            train_input, train_gt = dataset.get_trainbatch(batch * dataset.bs)

            output = decoder(train_input)

            rgb_pred = brdf_to_rgb(train_input, output)
            rgb_true = brdf_to_rgb(train_input, train_gt)

            loss = mean_absolute_logarithmic_error(y_true=rgb_true, y_pred=rgb_pred)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if batch % 25 == 0:
                train_progress.set_description(f"Epoch {epoch}, Batch {batch}, Train loss: {loss.item():.6f}")
        
        print(f"Epoch {epoch}, Train loss: {sum(train_loss)/len(train_loss):.6f}")

        scheduler.step()
        batch_time = []

        with torch.no_grad():
            val_batches = int(dataset.test_samples.shape[0] / batch_size)
            val_progress = tqdm(range(val_batches), leave=False)
            for batch in val_progress:

                val_input, val_gt = dataset.get_testbatch(batch * dataset.bs)

                torch.cuda.synchronize()
                start_time = time.time()

                output = decoder(val_input)

                torch.cuda.synchronize()
                batch_time.append(time.time() - start_time)

                rgb_pred = brdf_to_rgb(val_input, output)
                rgb_true = brdf_to_rgb(val_input, val_gt)

                loss = mean_absolute_logarithmic_error(y_true=rgb_true, y_pred=rgb_pred)

                val_loss.append(loss.item())

                if batch % 25 == 0:
                    val_progress.set_description(f"Epoch {epoch}, Batch {batch}, Test loss: {loss.item():.6f}")
        
            print(f"Epoch {epoch}, Test loss: {sum(val_loss)/len(val_loss):.6f}, Batch time: {sum(batch_time)/len(batch_time):.6f}")

    #decoder.save_raw(DECODER_RAW_PATH)
    print(decoder.decoder)
    
    return decoder.to("cpu")

if __name__ == "__main__":
    decoder = train_decoder().cpu()
    decoder.encoder = Rusinkiewicz6DTransform()
    #decoder.scale = torch.tensor([1./1500, 1.15/1500, 1.66/1500])

    num_points = 3
    num_angles = 50

    w_i_vectors = np.array([
        (np.cos(theta), 0.0, np.sin(theta)) for theta in np.linspace(0, np.pi / 2, num_angles)
    ])[1:]

    normal = np.array([0, 0, 1], dtype=np.float32)

    predicted_map1 = np.zeros((num_angles, num_points))
    predicted_map2 = np.zeros((num_angles, num_points))

    for i, w_i in enumerate(w_i_vectors):
        w_o = w_i.copy()
        w_o[0] = -w_o[0]
        with torch.no_grad():
            predicted = decoder(
                torch.tensor(w_i[np.newaxis, :], dtype=torch.float32),
                torch.tensor(w_o[np.newaxis, :], dtype=torch.float32)
            )
            predicted_map1[i] = np.log(predicted.numpy())

    BRDF = fastmerl.Merl(merlPath)
    for i, w_i in enumerate(w_i_vectors):
        w_o = w_i.copy()
        w_o[0] = -w_o[0]
        rvectors = decoder.encoder(torch.tensor(w_i), torch.tensor(w_o)).unsqueeze(-1)
        brdf_vals = brdf_values(rvectors, brdf=BRDF)
        with torch.no_grad():
            predicted_map2[i] = np.log(brdf_vals)

    norm = colors.Normalize(vmin=0.0, vmax=np.max(np.array([predicted_map1, predicted_map2])))

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
    axs[0].set_title("NBRDF")
    axs[0].set_xlabel("Длина волны (нм)")
    axs[0].set_ylabel("Угол наблюдения (°)")

    im2 = axs[1].imshow(predicted_map2, aspect='auto', cmap=cmap, norm=norm,
                        extent=[380, 780, 0, 90])
    axs[1].set_title("GT")
    axs[1].set_xlabel("Длина волны (нм)")
    axs[1].set_ylabel("Угол наблюдения (°)")

    # 5. Add a single colorbar for all subplots
    cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Отражательная способность (0—1)")

    plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from external.merl.dataset import MerlDataset, brdf_to_rgb, brdf_to_rgb_, fastmerl, brdf_values
from brdf_decoder import NBRDFDecoder, KAN_BRDF, initialize_weights
from utils.encodings import Rusinkiewicz6DTransform
from utils.torch_utils import total_parameters

DECODER_RAW_PATH = "saved_models/rgb_decoder.bin"
merlPath = "./external/merl/BRDFDatabase/brdfs/red-metallic-paint.binary"
PROFILE_INFERENCE = True


TRAIN_BS = 512
VAL_BS = 50000

def mean_absolute_logarithmic_error(y_true, y_pred):
    return torch.mean(torch.abs(torch.log(1 + y_true) - torch.log(1 + y_pred)))

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #samples = 5000000
    epochs = 100

    dataset = MerlDataset(merlPath=merlPath, batchsize=TRAIN_BS, nsamples=800000, train_size=0.75, test_batchsize=VAL_BS, angles=True)

    #decoder = NBRDFDecoder(hidden_dim=64, encoder=False).to(device)
    decoder = KAN_BRDF(dim=[3, 2, 3], k=1, nCps=8, encoder=False).to(device)
    initialize_weights(decoder, "xavier_uniform")
    optimizer = optim.AdamW(decoder.parameters(), lr=1e-3,
                           betas=(0.9, 0.999),
                           eps=1e-15,
                           weight_decay=0.0,
                           amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    train_batches = int(dataset.train_samples.shape[0] / dataset.train_bs)
    val_batches = int(dataset.test_samples.shape[0] / dataset.test_bs)

    params_count = total_parameters(decoder)
    print(f"Model has {params_count} parameters")
    
    progress_bar = tqdm(range(epochs), leave=True)
    for epoch in progress_bar:
        dataset.shuffle()

        train_loss = [] 
        val_loss = []
        val_time = []

        for batch in range(train_batches):

            optimizer.zero_grad()

            train_input, train_gt = dataset.get_trainbatch(batch * dataset.train_bs)

            output = decoder(train_input)

            rgb_pred = brdf_to_rgb_(train_input, output)
            rgb_true = brdf_to_rgb_(train_input, train_gt)

            loss = mean_absolute_logarithmic_error(y_true=rgb_true, y_pred=rgb_pred)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        scheduler.step()

        with torch.no_grad():
            for batch in range(val_batches):

                val_input, val_gt = dataset.get_testbatch(batch * dataset.test_bs)

                if PROFILE_INFERENCE:
                    torch.cuda.synchronize()
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        profile_memory=False
                    ) as prof:
                        with record_function("forward_pass"):
                            output = decoder(val_input)

                    evt = next(e for e in prof.key_averages() if e.key == "forward_pass")
                    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    #print(prof.key_averages().self_cpu_time_total)
                    #print(evt.device_time_total)
                    val_time.append(evt.device_time_total)
                else:
                    output = decoder(val_input)

                rgb_pred = brdf_to_rgb_(val_input, output)
                rgb_true = brdf_to_rgb_(val_input, val_gt)

                loss = mean_absolute_logarithmic_error(y_true=rgb_true, y_pred=rgb_pred)

                val_loss.append(loss.item())

            mean_train_loss = sum(train_loss)/len(train_loss)

            if len(val_loss) == 0:
                mean_val_loss = 0
            else:
                mean_val_loss = sum(val_loss)/len(val_loss)

            if len(val_time) == 0:
                mean_val_time = 0
            else:
                mean_val_time = sum(val_time) / len(val_time)

            progress_bar.set_description(f"Epoch {epoch}, Train loss: {mean_train_loss:.6f}, Val loss: {mean_val_loss:.6f}, CUDA time: {mean_val_time / 1000.:.6f}ms")

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
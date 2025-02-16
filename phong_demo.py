import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from brdf_models import phong
from dataset.materials import PhongDataset
from brdf_decoder import SimpleDecoder
from skimage.metrics import peak_signal_noise_ratio as base_psnr
import cv2

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = 500000
    batch_size = 128

    dataset = PhongDataset(n_samples=samples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    decoder = SimpleDecoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, labels = data
        w_i = w_i.to(device)
        w_o = w_o.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        value = decoder(w_i, w_o)
        loss = loss_fn(value, labels)
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            progress_bar.set_description(f"Batch {batch}, MSE: {loss.item():.6f}")
    
    return decoder.to("cpu")

def neural_phong_demo(decoder, view_dir=(0, 0, 1), size=128):
    view_dir = torch.tensor(view_dir, dtype=torch.float32)
    view_dir = view_dir / view_dir.norm()
    bg = np.array((0, 0, 0))

    x = np.linspace(-1, 1, num=size)
    y = np.linspace(-1, 1, num=size)

    image = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            if x[i]**2 + y[j] ** 2 > 1:
                image[i, j] = bg
            else:
                z = np.sqrt(1 - x[i]**2 - y[j] ** 2)
                light_dir = torch.tensor([x[i], y[j], z], dtype=torch.float32)
                with torch.no_grad():
                    image[i, j] = decoder(view_dir, light_dir).numpy()
    return image

def phong_demo(view_dir=(0, 0, 1), size=128):
    view_dir = torch.tensor(view_dir, dtype=torch.float32)
    view_dir = view_dir / view_dir.norm()
    bg = np.array((0, 0, 0))

    x = np.linspace(-1, 1, num=size)
    y = np.linspace(-1, 1, num=size)

    image = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            if x[i]**2 + y[j] ** 2 > 1:
                image[i, j] = bg
            else:
                z = np.sqrt(1 - x[i]**2 - y[j] ** 2)
                light_dir = torch.tensor([x[i], y[j], z], dtype=torch.float32)
                image[i, j] = phong(view_dir, light_dir).numpy()
    return image

if __name__ == "__main__":
    decoder = train_decoder()
    views = [(0, 0, 1), (0.5, 0, 0.5), (0, 0.5, 0.5)]

    for i, view in enumerate(views):
        gt = phong_demo(view)
        cv2.imwrite(f'./tests/phong/gt_{i + 1}.png', gt * 255.0)
        img = neural_phong_demo(decoder, view)
        cv2.imwrite(f'./tests/phong/neural_{i + 1}.png', img * 255.0)
        print("PSNR: ", base_psnr(gt, img))

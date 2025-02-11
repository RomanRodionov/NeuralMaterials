import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from spectrum import *
from autoencoder import *
from dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_points = 128
latent_dim = 32
batches = 2500000
batch_size = 512

#dataset = PolynomialDataset(n_samples=batches, degree=16, num_points=num_points)
dataset = SpectrumDataset(n_samples=batches, num_points=num_points)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)

autoencoder = CNNAutoencoder(input_dim=num_points, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=5e-3)
loss_fn = nn.MSELoss()

progress_bar = tqdm(data_loader)
for batch, data in enumerate(progress_bar):
    data = data.to(device)
    optimizer.zero_grad()
    reconstructed = autoencoder(data)
    loss = loss_fn(reconstructed, data)
    loss.backward()
    optimizer.step()
    
    if batch % 100 == 0:
        progress_bar.set_description(f"Batch {batch}, MSE: {loss.item():.6f}")
        #plt.plot(data[0].cpu().numpy().flatten(), label='Ground Truth')
        #plt.legend()
        #plt.show()

wavelength_range = np.arange(380, 780, 400 / num_points)
sample = torch.tensor(read_spectrum("data/light/cie.stdillum.D6500.spd", wavelength_range), dtype=torch.float).unsqueeze(0).to(device)
predicted = autoencoder(sample).detach().cpu().numpy().flatten()

plt.plot(sample.cpu().numpy().flatten(), label='Ground Truth')
plt.plot(predicted, label='Reconstructed', linestyle='dashed')
plt.legend()
plt.show()
print(f"D6500 MSE: {loss_fn(sample, autoencoder(sample))}")

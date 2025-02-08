import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=32):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class PolynomialDataset(Dataset):
    def __init__(self, n_samples=1000, degree=3, num_points=50):
        self.n_samples = n_samples
        self.degree = degree
        self.num_points = num_points
        self.x = np.linspace(-1, 1, num_points)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        coeffs = np.random.uniform(-1, 1, self.degree + 1)
        y = np.polyval(coeffs, self.x)
        return torch.tensor(y, dtype=torch.float32)

num_points = 100
latent_dim = 32
batches = 10000
batch_size = 32

dataset = PolynomialDataset(n_samples=batches * batch_size, degree=32, num_points=num_points)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

autoencoder = Autoencoder(input_dim=num_points, latent_dim=latent_dim)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for batch, data in enumerate(tqdm(data_loader)):
    optimizer.zero_grad()
    reconstructed = autoencoder(data)
    loss = loss_fn(reconstructed, data)
    loss.backward()
    optimizer.step()
    
    if batch % 100 == 0:
        tqdm.write(f"Batch {batch}, MSE: {loss.item():.6f}")

sample = dataset[0].unsqueeze(0)
predicted = autoencoder(sample).detach().numpy().flatten()

plt.plot(sample.numpy().flatten(), label='Ground Truth')
plt.plot(predicted, label='Reconstructed', linestyle='dashed')
plt.legend()
plt.show()

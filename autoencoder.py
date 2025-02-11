import torch
import torch.nn as nn

class Sin(nn.Module):
  def forward(self, x):
    return torch.sin(x)

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(MLPAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, latent_dim),
            Sin()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            Sin(),
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
    
class CNNAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(CNNAutoencoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((input_dim // 8) * 16, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (input_dim // 8) * 16),
            nn.ReLU(),
            nn.Unflatten(1, (16, input_dim // 8)),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def encode(self, x):
        x = x.unsqueeze(1)
        return self.encoder(x).squeeze(1)
    
    def decode(self, x):
        x = x.unsqueeze(1)
        return self.decoder(x).squeeze(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)

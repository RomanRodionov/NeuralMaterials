import torch
import torch.nn as nn

class TextureDecoder(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=64):
        super(TextureDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 6, hidden_dim), # latent vector + w_i + w_o
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x, w_i, w_o):
        z = torch.cat([x, w_i, w_o], dim=-1)
        z = self.decoder(z)
        return z

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super(SimpleDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
            nn.GELU(),
            nn.Linear(6, 3)
        )

    def forward(self, w_i, w_o):
        z = torch.cat([w_i, w_o], dim=-1)
        z = self.decoder(z)
        return z

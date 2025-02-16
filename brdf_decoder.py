import torch
import torch.nn as nn
import torch.nn.init as init

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
    def __init__(self, hidden_dim=64, output_dim=3):
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
            nn.Linear(6, output_dim)
        )

    def forward(self, w_i, w_o):
        z = torch.cat([w_i, w_o], dim=-1)
        z = self.decoder(z)
        return z
    
class SpectralDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=1):
        super(SpectralDecoder, self).__init__()

        self.vectors_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.GELU()
        )

        self.wavelength_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
            nn.GELU(),
            nn.Linear(6, output_dim)
        )

    def forward(self, w_i, w_o, wavelength):
        v = self.vectors_encoder(torch.cat([w_i, w_o], dim=-1))
        w = self.wavelength_encoder(wavelength.unsqueeze(-1))
        z = torch.cat([v, w], dim=-1)
        z = self.decoder(z)
        return z
    
class MomentsDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=6):
        super(MomentsDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, w_i, w_o):
        z = torch.cat([w_i, w_o], dim=-1)
        z = self.decoder(z)
        return z



def initialize_weights(model, init_type="xavier_uniform"):
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if init_type == "xavier_uniform":
                init.xavier_uniform_(layer.weight)
            elif init_type == "xavier_normal":
                init.xavier_normal_(layer.weight)
            elif init_type == "kaiming_uniform":
                init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif init_type == "orthogonal":
                init.orthogonal_(layer.weight)
            elif init_type == "normal":
                init.normal_(layer.weight, mean=0.0, std=0.02)
            elif init_type == "uniform":
                init.uniform_(layer.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            
            if layer.bias is not None:
                init.zeros_(layer.bias)
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
#https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_paper.pdf

class LatentTexture(nn.Module):
    def __init__(self, resolution=(128, 128), latent_dim=8):
        super(LatentTexture, self).__init__()
        
        self.resolution = resolution
        self.latent_dim = latent_dim
        
        self.latent_texture = nn.Parameter(torch.randn(1, latent_dim, *resolution))

    def set(self, latent_texture):
        assert latent_texture.shape == (*self.resolution, self.latent_dim)
        self.latent_texture.data = latent_texture.permute(2, 0, 1).unsqueeze(0)

    def get_texture(self):
        return self.latent_texture.data.squeeze(0).permute(1, 2, 0)

    def forward(self, uv):
        N = uv.shape[0]
        uv = uv * 2 - 1
        uv = uv.view(1, 1, N, 2)

        sampled_latents = F.grid_sample(self.latent_texture, uv, align_corners=True)
        
        return sampled_latents.permute(0, 2, 3, 1).view(N, self.latent_dim)

class TextureEncoder(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=8):
        super(TextureEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

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
            nn.Linear(6, 3)
        )

    def forward(self, x, w_i, w_o):
        z = torch.cat([x, w_i, w_o], dim=-1)
        z = self.decoder(z)
        return z

    def save_raw(self, path):
        with open(path, "wb") as f:
            f.write("hydrann1".encode("utf-8"))
            layers = [x for x in self.decoder.children() if isinstance(x, nn.Linear)]
            f.write(len(layers).to_bytes(4, "little"))
            for layer in layers:
                weight = np.ascontiguousarray(layer.weight.cpu().detach().numpy(), dtype=np.float32)
                bias = np.ascontiguousarray(layer.bias.cpu().detach().numpy(), dtype=np.float32)

                f.write(weight.shape[0].to_bytes(4, "little"))
                f.write(weight.shape[1].to_bytes(4, "little"))
                f.write(weight.tobytes())
                f.write(bias.tobytes())

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=3):
        super(SimpleDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),           
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),           
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )

    def forward(self, w_i, w_o):
        z = torch.cat([w_i, w_o], dim=-1)
        z = self.decoder(z)
        return z
    
    def save_raw(self, path):
        with open(path, "wb") as f:
            f.write("hydrann1".encode("utf-8"))
            layers = [x for x in self.decoder.children() if isinstance(x, nn.Linear)]
            f.write(len(layers).to_bytes(4, "little"))
            for layer in layers:
                weight = np.ascontiguousarray(layer.weight.cpu().detach().numpy(), dtype=np.float32)
                bias = np.ascontiguousarray(layer.bias.cpu().detach().numpy(), dtype=np.float32)

                f.write(weight.shape[0].to_bytes(4, "little"))
                f.write(weight.shape[1].to_bytes(4, "little"))
                f.write(weight.tobytes())
                f.write(bias.tobytes())
    
class SpectralDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=1):
        super(SpectralDecoder, self).__init__()

        #self.vectors_encoder = nn.Sequential(
        #    nn.Linear(6, hidden_dim // 2),
        #    nn.ReLU()
        #)

        #self.wavelength_encoder = nn.Sequential(
        #    nn.Linear(1, hidden_dim // 2),
        #    nn.ReLU()
        #)
        
        self.decoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
            nn.ReLU(),
            nn.Linear(6, output_dim)
        )

    def forward(self, w_i, w_o, wavelength):
        #v = self.vectors_encoder(torch.cat([w_i, w_o], dim=-1))
        #w = self.wavelength_encoder(wavelength.unsqueeze(-1))
        w = (wavelength - 380) / 400
        z = torch.cat([w_i, w_o, w.unsqueeze(-1)], dim=-1)
        z = self.decoder(z)
        return z
        
    def save_raw(self, path):
        with open(path, "wb") as f:
            f.write("hydrann1".encode("utf-8"))
            layers = [x for x in self.decoder.children() if isinstance(x, nn.Linear)]
            f.write(len(layers).to_bytes(4, "little"))
            for layer in layers:
                weight = np.ascontiguousarray(layer.weight.cpu().detach().numpy(), dtype=np.float32)
                bias = np.ascontiguousarray(layer.bias.cpu().detach().numpy(), dtype=np.float32)

                f.write(weight.shape[0].to_bytes(4, "little"))
                f.write(weight.shape[1].to_bytes(4, "little"))
                f.write(weight.tobytes())
                f.write(bias.tobytes())
    
class MomentsDecoder(nn.Module):
    def __init__(self, hidden_dim=16, output_dim=6):
        super(MomentsDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
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
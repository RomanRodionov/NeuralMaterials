import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math
from utils.encodings import RusinkiewiczTransform, Rusinkiewicz6DTransform
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

class Sin(nn.Module):
    def __init__(self, A_init=1.0, w0_init=30.0, phi0_init=0.0, trainable=True):
        super().__init__()
        A = torch.tensor(A_init, dtype=torch.float32)
        w0 = torch.tensor(w0_init, dtype=torch.float32)
        phi0 = torch.tensor(phi0_init, dtype=torch.float32)
        if trainable:
            self.A   = nn.Parameter(A)
            self.w0  = nn.Parameter(w0)
            self.phi0 = nn.Parameter(phi0)
        else:
            self.register_buffer('A',   A)
            self.register_buffer('w0',  w0)
            self.register_buffer('phi0', phi0)

    def forward(self, x):
        return self.A * torch.sin(self.w0 * x + self.phi0)


class SpectralDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=1, wavelength_min=360, wavelength_max=830):
        super(SpectralDecoder, self).__init__()

        #self.vectors_encoder = nn.Sequential(
        #    nn.Linear(6, hidden_dim // 2),
        #    nn.ReLU()
        #)

        #self.wavelength_encoder = nn.Sequential(
        #    nn.Linear(1, hidden_dim // 2),
        #    nn.ReLU()
        #)

        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.range = wavelength_max - wavelength_min

        self.encoder = RusinkiewiczTransform()
        
        self.decoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, w_i, w_o, wavelength):
        w = (wavelength - self.wavelength_min) / self.range
        z = torch.cat([self.encoder(w_i, w_o) , w.unsqueeze(-1)], dim=-1)
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

class FSpectralDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=1, wavelength_min=360, wavelength_max=830):
        super(FSpectralDecoder, self).__init__()

        #self.vectors_encoder = nn.Sequential(
        #    nn.Linear(6, hidden_dim // 2),
        #    nn.ReLU()
        #)

        #self.wavelength_encoder = nn.Sequential(
        #    nn.Linear(1, hidden_dim // 2),
        #    nn.ReLU()
        #)

        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.range = wavelength_max - wavelength_min

        self.encoder = Rusinkiewicz6DTransform()

        self.decoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, w_i, w_o):
        z = self.encoder(w_i, w_o)
        z = self.decoder(z)# / w_i[..., -1].unsqueeze(-1)
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

class NBRDFDecoder(nn.Module):
    def __init__(self, hidden_dim=21, output_dim=3, encoder=True, scale=None):
        super(NBRDFDecoder, self).__init__()

        if encoder:
            self.encoder = Rusinkiewicz6DTransform()
        else:
            self.encoder = None

        self.scale = scale

        self.decoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, w_i, w_o=None):
        if self.encoder:
            z = self.encoder(w_i, w_o)
        else:
            if w_o is not None:
                z = torch.cat([w_i, w_o], dim=-1)
            else:
                z = w_i
        z = self.decoder(z)
        res = F.relu(torch.exp(z) - 1)
        if self.scale is not None:
            res = res / self.scale
        return res
        
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
                init.uniform_(layer.weight, a=-0.05, b=0.05)
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            
            if layer.bias is not None:
                init.zeros_(layer.bias)
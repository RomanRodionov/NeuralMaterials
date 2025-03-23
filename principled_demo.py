import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from dataset.materials import TextureDataset, PrincipledDataset
from brdf_decoder import TextureEncoder, TextureDecoder, LatentTexture
from skimage.metrics import peak_signal_noise_ratio as base_psnr

ENCODER_PATH = "saved_models/texture_encoder.pth"
DECODER_PATH = "saved_models/texture_decoder.pth"
LATENT_TEXTURE_PATH = "saved_models/latent_texture.npy"

def train_principled(tex_path="resources/materials/test/Metal/tc_metal_029", resolution = (128, 128), save_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = 15000000
    batch_size = 128

    writer = SummaryWriter("runs/principled_training")

    tex_dataset = TextureDataset(tex_path, resolution=resolution)
    bsdf_dataset = PrincipledDataset(textures=tex_dataset, n_samples=samples)
    data_loader = DataLoader(bsdf_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    encoder = TextureEncoder(hidden_dim=64, latent_dim=8).to(device)
    decoder = TextureDecoder(hidden_dim=64, latent_dim=8).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, params, gt_bsdf = data
        w_i = w_i.to(device)
        w_o = w_o.to(device)
        params = params.to(device)
        gt_bsdf = gt_bsdf.to(device)

        optimizer.zero_grad()

        latent = encoder(params)
        bsdf = decoder(latent, w_i, w_o)

        loss = loss_fn(bsdf, gt_bsdf)
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("Loss/train", loss.item(), batch)
        
        if batch % 100 == 0:
            progress_bar.set_description(f"Batch {batch}, MSE: {loss.item():.6f}")

    if save_model:
        torch.save(encoder.state_dict(), ENCODER_PATH)
        torch.save(decoder.state_dict(), DECODER_PATH)
        print(f"Models are saved to: {ENCODER_PATH}, {DECODER_PATH}")

    writer.close()
    return decoder.to("cpu")

def generate_latent_texture(tex_path="resources/materials/test/Metal/tc_metal_029", resolution=(128, 128), latent_dim=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TextureDataset(tex_path, resolution=resolution)
    h, w = resolution

    encoder = torch.load(ENCODER_PATH).to(device)

    latent_texture_model = LatentTexture(resolution=resolution, latent_dim=latent_dim).to(device)
    latent_texture_model.eval()

    latent_texture = torch.zeros((h, w, latent_dim), device=device)

    for i in range(h):
        for j in range(w):
            vec_params, _ = dataset.sample(i / h, j / w)
            latent_vector = encoder(vec_params.to(device))
            latent_texture[i, j] = latent_vector.squeeze(0)

    latent_texture_model.set(latent_texture)

    torch.save(latent_texture.cpu(), LATENT_TEXTURE_PATH)
    print(f"Latent texture is saved to {LATENT_TEXTURE_PATH}")

    return latent_texture

if __name__ == "__main__":
    train_principled()

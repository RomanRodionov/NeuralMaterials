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
import cv2

ENCODER_PATH = "saved_models/texture_encoder.pth"
DECODER_PATH = "saved_models/texture_decoder.pth"
DECODER_RAW_PATH = "saved_models/texture_decored.bin"
LATENT_TEXTURE_PATH = "saved_models/latent_texture.npy"
FINETUNED_LATENT_PATH = "saved_models/finetuned_latent_texture.npy"
RESOLUTION = (256, 256)
LATENT_DIM = 3

def train_encoder(tex_path="resources/materials/test/Metal/tc_metal_029", resolution = RESOLUTION, save_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = 3000
    batch_size = 128

    writer = SummaryWriter("runs/training_encoder")

    tex_dataset = TextureDataset(tex_path, resolution=resolution)
    bsdf_dataset = PrincipledDataset(textures=tex_dataset, n_samples=samples)
    data_loader = DataLoader(bsdf_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    encoder = TextureEncoder(hidden_dim=64, latent_dim=LATENT_DIM).to(device)
    decoder = TextureDecoder(hidden_dim=64, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam([
                            {'params': encoder.parameters()},
                            {'params': decoder.parameters()}
                           ], lr=1e-3)
    loss_fn = nn.MSELoss()

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, params, gt_bsdf, uv = data
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

    decoder.save_raw(DECODER_RAW_PATH)

    return decoder.to("cpu")

def generate_latent_texture(tex_path="resources/materials/test/Metal/tc_metal_029", resolution=RESOLUTION, latent_dim=LATENT_DIM):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TextureDataset(tex_path, resolution=resolution)
    h, w = resolution

    encoder = TextureEncoder(hidden_dim=64, latent_dim=LATENT_DIM)
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    encoder = encoder.to(device)

    latent_texture_model = LatentTexture(resolution=resolution, latent_dim=latent_dim).to(device)
    latent_texture_model.eval()

    latent_texture = torch.zeros((h, w, latent_dim), device=device)

    for i in range(h):
        for j in range(w):
            vec_params, _ = dataset.sample(i / h, j / w)
            latent_vector = encoder(vec_params.to(device))
            latent_texture[i, j] = latent_vector.squeeze(0)

    latent_texture_model.set(latent_texture.permute(2, 0, 1))

    torch.save(latent_texture_model.state_dict(), LATENT_TEXTURE_PATH)
    print(f"Latent texture is saved to {LATENT_TEXTURE_PATH}")

    return latent_texture_model

def train_decoder(tex_path="resources/materials/test/Metal/tc_metal_029", resolution=RESOLUTION, latent_dim=LATENT_DIM, save_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = 10000000
    batch_size = 128

    writer = SummaryWriter("runs/training_decoder")

    tex_dataset = TextureDataset(tex_path, resolution=resolution)
    bsdf_dataset = PrincipledDataset(textures=tex_dataset, n_samples=samples)
    data_loader = DataLoader(bsdf_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    latent_texture = LatentTexture(resolution=resolution, latent_dim=latent_dim).to(device)
    latent_texture.load_state_dict(torch.load(LATENT_TEXTURE_PATH, weights_only=False))
 
    decoder = TextureDecoder(hidden_dim=64, latent_dim=LATENT_DIM).to(device)
    decoder.load_state_dict(torch.load(DECODER_PATH))

    optimizer = optim.Adam([
                            {'params': latent_texture.parameters(), 'lr': 1e-4},
                            {'params': decoder.parameters()}
                           ], lr=1e-3)
    loss_fn = nn.MSELoss()

    progress_bar = tqdm(data_loader)
    for batch, data in enumerate(progress_bar):
        w_i, w_o, params, gt_bsdf, uv = data
        w_i = w_i.to(device)
        w_o = w_o.to(device)
        params = params.to(device)
        gt_bsdf = gt_bsdf.to(device)
        uv = uv.to(device)

        optimizer.zero_grad()

        latent = latent_texture(uv)
        bsdf = decoder(latent, w_i, w_o)

        loss = loss_fn(bsdf, gt_bsdf)
        loss.backward()

        #print(torch.max(latent_texture.latent_texture.grad), torch.max(decoder.decoder[0].weight.grad))

        optimizer.step()
        
        writer.add_scalar("Loss/train", loss.item(), batch)
        
        if batch % 100 == 0:
            progress_bar.set_description(f"Batch {batch}, MSE: {loss.item():.6f}")

    if save_model:
        torch.save(latent_texture.state_dict(), FINETUNED_LATENT_PATH)
        torch.save(decoder.state_dict(), DECODER_PATH)
        print(f"Models are saved to: {FINETUNED_LATENT_PATH}, {DECODER_PATH}")

    writer.close()
    return latent_texture.to("cpu"), decoder.to("cpu")

def visualize_latent_texture(latent_texture, save_path = './tests/principled/latent.png'):
    latent_texture = latent_texture.latent_texture.squeeze(0).permute(1, 2, 0)
    latent_gray = latent_texture.mean(dim=-1)
    latent_gray = (latent_gray - latent_gray.min()) / (latent_gray.max() - latent_gray.min())
    latent_gray = (latent_gray * 255).detach().cpu().numpy().astype(np.uint8)
    
    cv2.imwrite(save_path, latent_gray)
    print(f"Saved latent texture visualization to {save_path}")

if __name__ == "__main__":
    train_encoder()
    latent_texture = generate_latent_texture()
    visualize_latent_texture(latent_texture, './tests/principled/latent.png')
    #finetuned_latent, _ = train_decoder()
    #visualize_latent_texture(finetuned_latent, './tests/principled/finetuned_latent.png')
    


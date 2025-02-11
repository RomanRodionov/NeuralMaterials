import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class MaterialDataset(Dataset):
    texture_types = ["basecolor", "diffuse", "displacement", "height", "metallic", "normal", "opacity", "roughness", "specular"]

    def __init__(self, path, resolution=(1024, 1024), n_samples=1000):
        self.n_samples = n_samples
        self.path = path
        self.resolution = resolution
        self.textures = {}
        for tex in self.texture_types:
            tex_path = os.path.join(self.path, f"{tex}.png")
            self.textures[tex] = self.load_texture(tex_path)
        self.channels = np.concatenate([tex for tex in self.textures.values()], axis=-1)
    
    def load_texture(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load texture: {path}")
        if not self.resolution is None:
            img = cv2.resize(img, self.resolution)
        img = img.astype(np.float32) / 255
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1) 
        return img
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        h, w, _ = self.textures[self.texture_types[0]].shape
        u, v = np.random.randint(0, h), np.random.randint(0, w)
        sample = self.channels[u, v]
        return torch.tensor(sample, dtype=torch.float32)

if __name__ == "__main__":
    # just example
    path = "../resources/materials/test/Metal/tc_metal_029"
    dataset = MaterialDataset(path)
    for i in range(5):
        sample = dataset[i]
        print(sample.shape, sample)

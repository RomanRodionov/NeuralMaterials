import torch
import numpy as np
from brdf_models import phong
import cv2

def phong_demo(view_dir=(0, 0, 1), size=128):
    view_dir = torch.tensor(view_dir, dtype=float)
    view_dir = view_dir / view_dir.norm()
    bg = np.array((0, 0, 0))

    x = np.linspace(-1, 1, num=size)
    y = np.linspace(-1, 1, num=size)

    image = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            if x[i]**2 + y[j] ** 2 > 1:
                image[i, j] = bg
            else:
                z = np.sqrt(1 - x[i]**2 - y[j] ** 2)
                light_dir = torch.tensor([x[i], y[j], z], dtype=float)
                image[i, j] = phong(view_dir, light_dir).numpy()
    return image

if __name__ == "__main__":
    cv2.imwrite('material_1.png', phong_demo((0, 0, 1)) * 255.0)
    cv2.imwrite('material_2.png', phong_demo((0.5, 0, 0.5)) * 255.0)
    cv2.imwrite('material_3.png', phong_demo((0, 0.5, 0.5)) * 255.0)
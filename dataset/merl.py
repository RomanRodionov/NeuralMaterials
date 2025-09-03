'''
Original paper:

"A Data-Driven Reflectance Model",
Wojciech Matusik, Hanspeter Pfister, Matt Brand and Leonard McMillan,
ACM Transactions on Graphics 22, 3(2003), 759-769.
BibTeX:
@article {Matusik:2003, 
	author = "Wojciech Matusik and Hanspeter Pfister and Matt Brand and Leonard McMillan",
	title = "A Data-Driven Reflectance Model",
	journal = "ACM Transactions on Graphics",
	year = "2003",
	month = jul,
	volume = "22",
	number = "3",
	pages = "759-769"
}
'''

import torch
from torch.utils.data import Dataset
from utils.sampling import *
from utils.torch_coords import *
from external.merl.fastmerl import Merl

class MerlDataset(Dataset):
    def __init__(self, filename, n_samples=1000):
        self.n_samples = n_samples
        self.data_object = Merl(filename)
        #self.data_object.convert_to_fullmerl()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        theta_h, phi_h, theta_d, phi_d = np.random.uniform([0, 0, 0, 0], [1., 2 * np.pi, np.pi / 2., 2 * np.pi])

        theta_h = (theta_h) * np.pi / 2

        trunc_phi_d = np.where(phi_d > np.pi, phi_d - np.pi, phi_d)
        values = self.data_object.eval_interp(theta_h, theta_d, trunc_phi_d)
        values = np.where(values < 0, 0, values)

        #print(f"theta_h: {theta_h}, theta_d: {theta_d}, phi_d: {phi_d}, trunc_phi_d: {trunc_phi_d}, BRDF: {values}\n")

        half = torch.tensor(sph2xyz(theta_h, phi_h), dtype=torch.float32)
        diff = torch.tensor(sph2xyz(theta_d, phi_d), dtype=torch.float32)
        values  = torch.tensor(values, dtype=torch.float32)

        #wi, _ = hd_to_io(half, diff)

        return half, diff, values
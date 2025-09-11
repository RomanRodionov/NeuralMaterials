from . import common
from . import coords
from . import fastmerl
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

torch.set_default_dtype(torch.float32)
Xvars = ['hx', 'hy', 'hz', 'dx', 'dy', 'dz']
Xvars_ = ['theta_h', 'theta_d', 'phi_d']
Yvars = ['brdf_r', 'brdf_g', 'brdf_b']
device = 'cuda'

def brdf_to_rgb(rvectors, brdf):
    hx = torch.reshape(rvectors[:, 0], (-1, 1))
    hy = torch.reshape(rvectors[:, 1], (-1, 1))
    hz = torch.reshape(rvectors[:, 2], (-1, 1))
    dx = torch.reshape(rvectors[:, 3], (-1, 1))
    dy = torch.reshape(rvectors[:, 4], (-1, 1))
    dz = torch.reshape(rvectors[:, 5], (-1, 1))

    theta_h = torch.atan2(torch.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = torch.atan2(torch.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = torch.atan2(dy, dx)
    wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
          torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clamp(wiz, 0, 1)
    return rgb

class MerlDataset(Dataset):
    def __init__(self, merlPath, batchsize, nsamples=800000, angles=False, train_size=0.8, test_batchsize=None):
        super(MerlDataset, self).__init__()

        self.train_bs = batchsize
        if test_batchsize is not None:
            self.test_bs = test_batchsize
        else:
            self.test_bs = batchsize
        self.BRDF = fastmerl.Merl(merlPath)

        if angles:
            xvars=Xvars_
        else:
            xvars=Xvars

        self.reflectance_train = generate_nn_datasets(self.BRDF, nsamples=nsamples, pct=train_size, angles=angles)
        self.reflectance_test = generate_nn_datasets(self.BRDF, nsamples=nsamples, pct=(1. - train_size), angles=angles)

        self.train_samples = torch.tensor(self.reflectance_train[xvars].values, dtype=torch.float32, device=device)
        self.train_gt = torch.tensor(self.reflectance_train[Yvars].values, dtype=torch.float32, device=device)

        self.test_samples = torch.tensor(self.reflectance_test[xvars].values, dtype=torch.float32, device=device)
        self.test_gt = torch.tensor(self.reflectance_test[Yvars].values, dtype=torch.float32, device=device)

    def __len__(self):
        return self.train_samples.shape[0]

    def get_trainbatch(self, idx):
        return self.train_samples[idx:idx + self.train_bs, :], self.train_gt[idx:idx + self.train_bs, :]

    def get_testbatch(self, idx):
        return self.test_samples[idx:idx + self.test_bs, :], self.test_gt[idx:idx + self.test_bs, :]

    def shuffle(self):
        r = torch.randperm(self.train_samples.shape[0])
        self.train_samples = self.train_samples[r, :]
        self.train_gt = self.train_gt[r, :]

    def __getitem__(self, idx):
        pass


def brdf_to_rgb(rvectors, brdf):
    hx = torch.reshape(rvectors[:, 0], (-1, 1))
    hy = torch.reshape(rvectors[:, 1], (-1, 1))
    hz = torch.reshape(rvectors[:, 2], (-1, 1))
    dx = torch.reshape(rvectors[:, 3], (-1, 1))
    dy = torch.reshape(rvectors[:, 4], (-1, 1))
    dz = torch.reshape(rvectors[:, 5], (-1, 1))

    theta_h = torch.atan2(torch.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = torch.atan2(torch.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = torch.atan2(dy, dx)
    wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
          torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clamp(wiz, 0, 1)
    return rgb

def brdf_to_rgb_(rvectors, brdf):
    theta_h = torch.reshape(rvectors[:, 0], (-1, 1))
    theta_d = torch.reshape(rvectors[:, 1], (-1, 1))
    phi_d = torch.reshape(rvectors[:, 2], (-1, 1))

    wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
          torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clamp(wiz, 0, 1)
    return rgb

def brdf_values(rvectors, brdf=None, model=None):
    if brdf is not None:
        rangles = coords.rvectors_to_rangles(*rvectors)
        brdf_arr = brdf.eval_interp(*rangles).T
    elif model is not None:
        # brdf_arr = model.predict(rvectors.T)        # nnModule has no .predict
        raise RuntimeError("Should not have entered that branch at all from the original code")
    else:
        raise NotImplementedError("Something went really wrong.")
    brdf_arr *= common.mask_from_array(rvectors.T).reshape(-1, 1)
    return brdf_arr


def generate_nn_datasets(brdf, nsamples=800000, pct=0.8, angles=False):
    rangles = np.random.uniform([0, 0, 0], [np.pi / 2., np.pi / 2., 2 * np.pi], [int(nsamples * pct), 3]).T
    rangles[2] = common.normalize_phid(rangles[2])

    rvectors = coords.rangles_to_rvectors(*rangles)
    brdf_vals = brdf_values(rvectors, brdf=brdf)

    if angles:
        df = pd.DataFrame(np.concatenate([rangles.T, brdf_vals], axis=1), columns=[*Xvars_, *Yvars])
    else:
        df = pd.DataFrame(np.concatenate([rvectors.T, brdf_vals], axis=1), columns=[*Xvars, *Yvars])
    df = df[(df.T != 0).any()]
    df = df.drop(df[df['brdf_r'] < 0].index)
    return df

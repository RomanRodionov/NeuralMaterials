import torch
import torch.nn as nn

def total_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())
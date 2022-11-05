import tempfile
import torch
import numpy as np

from thop import profile
from tqdm.auto import tqdm


def memory_size(model):
    with tempfile.TemporaryFile() as f:
        torch.save(model.state_dict(), f)
        size = f.tell() / 2**20
    return size


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


def macs(model, batch):
    return profile(model, (batch,), verbose=False)[0]

def get_mean_macs(model, loader, log_melspec, device):
    model.eval()
    total_macs = []

    for i, (batch, _) in tqdm(enumerate(loader), total=len(loader)):
        batch = batch.to(device)
        batch = log_melspec(batch)
        total_macs.append(macs(model, batch))
    
    return np.mean(total_macs)
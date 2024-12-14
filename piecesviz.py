import numpy as np
import math
from matplotlib.colors import Normalize, hsv_to_rgb
import torch

def linear_pieces(model, data, receptive_fields=False):
    numneuro = model.numneuro
    np.random.seed(1)
    hues = np.linspace(0, 1, numneuro, endpoint=False)  # Evenly spaced hues
    hues = np.random.permutation(hues)  # Randomize order
    saturation = 0.9  # High saturation for vibrant colors
    value = 0.9  # High brightness
    neuron_colors = hsv_to_rgb(np.array([[h, saturation, value] for h in hues]))  # Convert to RGB
    _, hidden = model(data, return_hidden=True)
    Mact = (hidden>0).float()
    dims = math.ceil(math.sqrt(data.shape[0]))
    linpieces = np.dot(Mact.numpy(), neuron_colors)
    norm = Normalize()
    linpieces = norm(linpieces)
    linpieces = linpieces.reshape(dims, dims, 3).transpose(1, 0, 2)

    if receptive_fields:
        Rfall = []
        for k in range(model.numneuro):
            Rf = torch.zeros_like(Mact)
            Rf[:, k] = Mact[:, k]
            Rf = np.dot(Rf.numpy(), neuron_colors)
            norm = Normalize()
            Rf = norm(Rf)
            Rf = Rf.reshape(dims, dims, 3).transpose(1, 0, 2)
            Rfall.append(Rf)
    
    if not receptive_fields:
        return linpieces
    else:
        return linpieces, Rfall
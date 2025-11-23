import os
import numpy as np
import matplotlib.pyplot as plt
import torch

folder_counters = {}


def save_heatmaps(features, folder='heatmaps', prefix='frame', apply_sigmoid=False, take_max=True):
    return
    global folder_counters
    os.makedirs(folder, exist_ok=True)

    # Torch → NumPy
    data = features.detach().cpu()

    # optional Sigmoid (z. B. bei cls_head-Output)
    if apply_sigmoid:
        data = torch.sigmoid(data)

    # für Klassenscores: max über Kanäle (Anker/Klassen)
    if take_max:
        data = data.amax(dim=1, keepdim=True)  # [B,1,H,W]
    else:
        data = data.mean(dim=1, keepdim=True)  # z. B. bei Backbone-Features

    data = data.squeeze(1).numpy()  # [B,H,W]

    if folder not in folder_counters:
        folder_counters[folder] = 0

    for fmap in data:
        # Normierung pro Bild
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
        filename = os.path.join(folder, f"{prefix}_{folder_counters[folder]:06d}.png")
        plt.imsave(filename, fmap, cmap='hot')
        folder_counters[folder] += 1


# OLD

#import os

#import numpy as np
#from matplotlib import pyplot as plt

#folder_counters = {}

#def save_heatmaps(spatial_features_2d, folder='heatmaps', prefix='frame'):
#    global folder_counters
#    os.makedirs(folder, exist_ok=True)
#    features_2d = spatial_features_2d.detach().cpu().numpy()
#    features_2d = np.mean(features_2d, axis=1)
#    if folder not in folder_counters:
#        folder_counters[folder] = 0
#    for feature_map in features_2d:
#        filename = os.path.join(folder, f"{prefix}_{folder_counters[folder]:06d}.png")
#        plt.imsave(filename, feature_map, cmap='hot')
#        folder_counters[folder] += 1
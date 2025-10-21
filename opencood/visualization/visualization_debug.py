import os

import numpy as np
from matplotlib import pyplot as plt

folder_counters = {}

def save_heatmaps(spatial_features_2d, folder='heatmaps', prefix='frame'):
    global folder_counters
    os.makedirs(folder, exist_ok=True)
    features_2d = spatial_features_2d.detach().cpu().numpy()
    features_2d = np.mean(features_2d, axis=1)
    if folder not in folder_counters:
        folder_counters[folder] = 0
    for feature_map in features_2d:
        filename = os.path.join(folder, f"{prefix}_{folder_counters[folder]:06d}.png")
        plt.imsave(filename, feature_map, cmap='hot')
        folder_counters[folder] += 1

import numpy as np
import matplotlib.pyplot as plt

def plot_light_curve(torch_dataset, index_in_dataset, figsize=(6, 3)):
    lc_data, label, lc_id = torch_dataset.__getitem__(index_in_dataset)
    
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    for band, band_name in enumerate('ugrizY'):
        mask = lc_data[band, 3, :] == 1
        mjd, flux, flux_err = lc_data[band, :3, mask]
        ax.errorbar(mjd, flux, flux_err, fmt='.', label=band_name)
    ax.legend()
    ax.set_ylabel('Flux')
    ax.set_xlabel('Modified Julian Data')
    ax.set_title(f'PLAsTiCC ID: {lc_id} Label: {label}')
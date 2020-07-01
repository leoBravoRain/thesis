from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import sys

def make_lc_tensor(df_np, max_lc_length):
    lc_numpy = np.zeros(shape=(6, 4, max_lc_length))
    for band in range(6):
        #lc_data = df_lc[["mjd", "flux", "flux_err"]][df_lc.passband==band].values.astype('float32')
        mask = df_np[:, 1] == band
        lc_data = df_np[mask, :][:, [0, 2, 3]]
        lc_numpy[band, :3, :lc_data.shape[0]] = lc_data.T
        lc_numpy[band, 3, :lc_data.shape[0]] = 1
    return torch.from_numpy(lc_numpy.astype('float32'))


def find_longest_lightcurve(path):
    p = Path(path)
    data_paths = sorted(p.glob('plasticc_*_lightcurves*.csv'))
    
    max_length_single_band = 0
    for data_path in data_paths:
        df_data = pd.read_csv(data_path).set_index("object_id")
        max_length_single_band_test = df_data.groupby(["object_id", "passband"]).count().max().max()
        if max_length_single_band_test > max_length_single_band:
            max_length_single_band = max_length_single_band_test
    return max_length_single_band


def populate_light_curve_folder(path, overwrite_light_curves=False):
    #max_length_single_band = max_longest_find_longest_lightcurve(path)
    print(f"Looking for plasticc data at {path}")
    p = Path(path)    
    data_paths = sorted(p.glob('plasticc_*_lightcurves*.csv'))
    print(f"Found {len(data_paths)} csv files at given path")
    
    if len(data_paths) > 0:
        # Create light_curves folder if it does not exist
        (p / 'light_curves').mkdir(parents=True, exist_ok=True)
    
    if not overwrite_light_curves:
        print("Existing light curves will not be overwritten")
    else:
        print("All light curves will be overwritten")
    for data_path in data_paths:
        print(f"Creating light curves from {data_path.name}")
        df_data = pd.read_csv(data_path).set_index("object_id")
        lc_ids = df_data.index.unique()
        
        for lc_id in tqdm(lc_ids):
            tensor_file = p / 'light_curves' / f'{lc_id}.pt'
            if overwrite_light_curves or not tensor_file.exists():
                with open(tensor_file, 'wb') as f:
                    torch.save(make_lc_tensor(df_data.loc[lc_id].values, 72), f, 
                               _use_new_zipfile_serialization=True, pickle_protocol=4)
        

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please give the path to the plasticc CSVs"
    path = sys.argv[1] #"/home/shared/astro/PLAsTiCC/"
    populate_light_curve_folder(path)
    
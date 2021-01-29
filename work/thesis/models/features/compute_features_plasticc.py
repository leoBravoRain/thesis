import os
os.environ["MKL_NUM_THREADS"]="1"
# print(os.environ["MKL_NUM_THREADS"])
import sys
from os.path import join, exists
import pandas as pd
from joblib import Parallel, delayed, dump
import pickle
from itertools import zip_longest 
import turbofats
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


plasticc_path = "/home/shared/astro/PLAsTiCC/"

def compute_fats_features(batch_names):
    """
    Receives a list of file names and a path

    Returns a dataframe with the features
    """
    # TODO: STUDY BIASED FEATURES
    feature_list = ['CAR_sigma','CAR_mean', 'Meanvariance', 'Mean',                          'PercentAmplitude', 'Skew', 'AndersonDarling', 'Std', 'Rcs', 'StetsonK',
                         'MedianAbsDev', 'Q31', 'Amplitude', 'PeriodLS_v2', 'Harmonics',
                'Autocor_length', 'SlottedA_length', 'StetsonK_AC',  'Con', 'Beyond1Std',
                'SmallKurtosis', 'MaxSlope','MedianBRP', 'PairSlopeTrend', 
                'LinearTrend', 'Eta_e', 'Period_fit_v2', 'PeriodPowerRate',
                'Psi_CS_v2', 'Psi_eta_v2', 'StructureFunction_index_21', 'Pvar', 'StructureFunction_index_31',
                'ExcessVar', 'IAR_phi']
    features = []
    for name in batch_names:
        # Check that name is valid
        if name is None:
            continue
        # Read light curve
        lc_data = parse_light_curve_data(name)
        if lc_data is None:
            continue
        features_lc = []
        for fid in range(6):
            # Check that lc has more than 10 points
            #if lc_data.shape[0] < 7:
            #    print("Light curve %s has less than 7 points, skipping" %(name))
            #    continue
            # Compute features
            df_lc_fid = lc_data.loc[lc_data.fid == fid][["mjd", "mag", "err"]]
            feature_space = turbofats.FeatureSpace(feature_list=feature_list,
                                                      data_column_names=["mag", "mjd", "err"])
            features_fid = feature_space.calculate_features(df_lc_fid)
            features_fid = features_fid.rename(lambda x: x+"_"+str(fid), axis='columns')
            features_lc.append(features_fid)
        with open(join(plasticc_path, "features/fats"+str(int(name))+".pkl"), "wb") as f:
            dump(pd.concat(features_lc, axis=1), f, protocol=3)


        #features.append(pd.concat(features_lc, axis=1))
    #eturn pd.concat(features, axis=0)

def split_list_in_chunks(iterable, chunk_size, fillvalue=None):
    """
    Receives an iterable object (i.e. list) and a chunk size

    Returns an iterable object with the same elements on of the original but arranged in chunks
    """
    args = [iter(iterable)] * chunk_size
    return zip_longest(*args, fillvalue=fillvalue)

def parse_light_curve_data(light_curve_id):
    path_to_light_curve = join(plasticc_path, "light_curves", str(int(light_curve_id))+".pt")
    if not exists(path_to_light_curve):
        if raise_error:
            raise FileNotFoundError("File not found at: %s" %(path_to_light_curve))
        return None
    with open(path_to_light_curve, "rb") as f:
        lc_torch = torch.load(f)
    data = []
    for band in range(6):
        mask = lc_torch[band, -1, :] == 1
        tmp = lc_torch[band, :3, mask].T
        data.append(np.concatenate((tmp.numpy(), np.ones(shape=(tmp.shape[0], 1))*band),axis=1))
    df_lc = pd.DataFrame(data=np.concatenate(data, axis=0), columns=['mjd', 'mag', 'err', 'fid'])
    df_lc.index = [int(light_curve_id)]*len(df_lc.index.unique())
    return df_lc


if __name__ == "__main__":


    with open("/home/phuijse/plasticc_rf/dataset_ids_before_balancing.pkl", "rb") as f:
        ids = pickle.load(f)
    for subset in ["train", "validation", "test"]:
        print(subset)
        Parallel(n_jobs=10)(delayed(compute_fats_features)(batch_names) for batch_names in split_list_in_chunks(ids[subset], 100))
        #result = [compute_fats_features(ids[subset][:2])]
        #with open(join(plasticc_path, "features/features_fats_"+subset+".pkl"), "wb") as f:
        #    dump(result, f, protocol=3)

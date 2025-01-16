from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List

DTYPE = torch.float32

def read_data(years: list[str]):
    dfs = []
    for year in years:
        df = pd.read_parquet(f"~/fast_read/full/training_data_0610_{year}.parquet")
        trasparent_windows = [[0.7703, 0.7704], [1.6056, 1.6059], [2.0596, 2.0598]]
        band_names = ["o2", "weak_co2", "strong_co2"]
        for band_name in band_names:
            df[("features", band_name)] = 0.0
        for windows, band_name in zip(trasparent_windows, band_names):
            wl = df[f"wavelengths_{band_name}"].values
            radiances = df[f"radiances_{band_name}"].values
            index = (wl > windows[0]) & (wl < windows[1])
            radiances[~index] = 0.0
            n = np.sum(index, axis=1)
            ratio = np.sum(radiances, axis=1) / n
            df[("features", band_name)] = ratio
            df[f"radiances_{band_name}"] = df[f"radiances_{band_name}"].values / ratio[:, None]

        df[("labels", "xco2_diff")] = df[("labels", "xco2")] - df[("features", "xco2_apriori")]
        df[("features", "relative_azimuth")] = df[("features", "solar_azimuth")] - df[("features", "azimuth")]
        df[("features", "surface_pressure_fph")] = df[("labels", "surface_pressure_fph")]
        df[("features", "sounding_index")] = df.index.values % 10
        dfs.append(df)
    return pd.concat(dfs)

def build_scaler(df, features_names: List[str], band_names: List[str], 
                 array_names: List[str], label_names: List[str], 
                 error_scale_names: List[str]):
    label_scaler = StandardScaler()
    label_scaler = label_scaler.fit(df["labels"][label_names].values.reshape(-1, 1))

    feature_scaler = StandardScaler()
    feature_scaler = feature_scaler.fit(df['features'][features_names])

    spectrum_scalers = {}
    for array_name in array_names:
        for band_name in band_names:
            name = f"{array_name}_{band_name}"
            scaler = StandardScaler()
            scaler.fit(df[name])
            spectrum_scalers[name] = scaler

    error_scaler = MinMaxScaler()
    error_scaler = error_scaler.fit(df["labels"][error_scale_names].values.reshape(-1, 1))
    return label_scaler, feature_scaler, spectrum_scalers, error_scaler

def create_dataset(df, label_scaler, feature_scaler, spectrum_scalers, error_scaler,
                   features_names, band_names, array_names, label_names, 
                   error_scale_names):
    data_features = df['features'][features_names].astype(float)
    data_features = feature_scaler.transform(data_features)
    data_labels = df['labels'][label_names]
    data_labels = label_scaler.transform(data_labels.values.reshape(-1, 1))

    data_spectrums = {}
    for array_name in array_names:
        for band_name in band_names:
            name = f"{array_name}_{band_name}"
            data_spectrums[name] = spectrum_scalers[name].transform(df[name])

    bad_sample_lists = {}
    for band_name in band_names:
        bad_sample_lists[band_name] = df[f"bad_sample_list_{band_name}"].values
    
    data_error_scale = df['labels'][error_scale_names]
    data_error_scale = error_scaler.transform(data_error_scale.values.reshape(-1, 1))
    
    dataset = SpectraDataset(data_features, data_spectrums, bad_sample_lists, data_labels, data_error_scale, band_names, array_names)
    return dataset

def random_subset(dataset, subset_size):
    indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, indices)
    return subset


class SpectraDataset(Dataset):
    def __init__(self, data_features: np.ndarray,
                 data_spectrums: Dict[str, np.ndarray],
                 bad_sample_lists: Dict[str, np.ndarray],
                 data_labels: np.ndarray,
                 data_error_scale: np.ndarray, band_names: List[str], 
                 array_names: List[str]):
        self.data_features = data_features
        self.data_spectrums = data_spectrums
        self.bad_sample_lists = bad_sample_lists
        self.data_labels = data_labels
        self.data_error_scale = data_error_scale
        self.band_names = band_names
        self.array_names = array_names


    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        features = self.data_features[idx, :]
        labels = self.data_labels[idx, :]
        error_weights = (1 - self.data_error_scale[idx, :]) ** 2
        
        spectra = []
        for band_name in self.band_names:
            raw_array = np.stack([self.data_spectrums[f"{array_name}_{band_name}"][idx] for array_name in self.array_names], axis=1)
            bsl = self.bad_sample_lists[band_name][idx]
            raw_array[bsl!=0] = -10
            spectra.append(torch.tensor(raw_array, dtype=DTYPE))
            
        return {"features": torch.tensor(features, dtype=DTYPE),
                "spectra": torch.cat(spectra, dim=0),
                "labels": torch.tensor(labels, dtype=DTYPE),
                "error_weights": torch.tensor(error_weights, dtype=DTYPE)}
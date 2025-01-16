import json
import joblib
import pandas as pd
from .model import MyModel
from .process_data import create_dataset
from .train import predict
import numpy as np
import torch

def index2time(index):
    data = (index.values//100).astype(str)
    formatted_data = [
        f"{date[:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:{date[10:12]}:{date[12:]}"
        for date in data
    ]
    # 转换为 numpy 的 datetime64 类型
    timestamps = np.array(formatted_data, dtype="datetime64")
    return timestamps

class Post:
    def __init__(self, model_name, device="cpu", batch_size = 40):
        self.model_name = model_name
        self.device = device
        self.base_dir = f"model_status/{model_name}"
        self.load_columns_names()
        self.load_scaler()
        self.create_model()
        self.load_violin()
        self.dataset_additional_parameters = [
            self.label_scaler, self.feature_scaler, 
            self.spectrum_scalers, self.error_scaler,
            self.features_names, self.band_names,
            self.array_names, self.label_names,
            self.error_scale_names
        ]
        self.batch_size = batch_size
        self.stad_info = [self.label_scaler.mean_, self.label_scaler.scale_]
    
    def load_columns_names(self):
        with open(f"{self.base_dir}/column_names.json", "r") as file:
            save_dict = json.load(file)
        self.features_names = save_dict["features_names"]
        self.band_names = save_dict["band_names"]
        self.array_names = save_dict["array_names"]
        self.label_names = save_dict["label_names"]
        self.error_scale_names = save_dict["error_scale_names"]
    
    def load_scaler(self):
        self.label_scaler = joblib.load(f"{self.base_dir}/label_scaler.pkl")
        self.feature_scaler = joblib.load(f"{self.base_dir}/feature_scaler.pkl")
        self.spectrum_scalers = joblib.load(f"{self.base_dir}/spectrum_scalers.pkl")
        self.error_scaler = joblib.load(f"{self.base_dir}/error_scaler.pkl")
    
    def load_config(self):
        with open(f"{self.base_dir}/config.json", "r") as file:
            self.config = json.load(file)
    
    def create_model(self):
        self.load_config()
        self.model = MyModel(**self.config)
    
    def load_model_parameter(self, epoch=0):
        self.model.load_state_dict(torch.load(f"model_status/{self.model_name}/weight/{epoch:03d}.pth"))
        self.model = self.model.to(self.device)
        
    def load_plume(self, plume_name):
        df = pd.read_parquet(f"post_data_full/{plume_name}/data.parquet")
        df[("features", "sounding_index")] = df.index.values % 10
        df[("features", "relative_azimuth")] = df[("features", "solar_azimuth")] - df[("features", "azimuth")]
        with open(f"post_data_full/{plume_name}/additional_info.json", "r") as f:
            additional_info = json.load(f)
        ds = create_dataset(df, *self.dataset_additional_parameters)
        return df, ds, additional_info

    def prepare_plume(self, plume_name):
        df, ds, additional_info = self.load_plume(plume_name)
        latitudes = df[("features", "latitude")]
        longitudes = df[("features", "longitude")]
        # pressures = df[("labels", "surface_pressure_fph")]
        lbl, mlp = predict(ds, self.model, self.device, self.batch_size, 
                           self.stad_info, self.feature_scaler)
        
        return latitudes, longitudes, lbl, mlp, additional_info

    def load_violin(self):
        self.violin_df = pd.read_parquet("post_data_full/violin_17_24.parquet")
        self.violin_df[("features", "sounding_index")] = self.violin_df.index.values % 10
        self.violin_df[("features", "relative_azimuth")] = self.violin_df[("features", "solar_azimuth")] - self.violin_df[("features", "azimuth")]
    
    def prepare_violin(self, begin_year=2017, end_year=2019):
        df = self.violin_df[(self.violin_df.index // 1e12 >= begin_year) & \
            (self.violin_df.index // 1e12 <= end_year)]
        oco_time = index2time(df.index)
        ds = create_dataset(df, *self.dataset_additional_parameters)
        lbl, mlp = predict(ds, self.model, self.device, self.batch_size, 
                           self.stad_info, self.feature_scaler)
        return oco_time, lbl, mlp
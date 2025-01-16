import json
import joblib
import os
import shutil

class Save:
    def __init__(self, model_name):
        self.base_dir = f"model_status/{model_name}"
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir)
        os.makedirs(f"{self.base_dir}/img")
        os.makedirs(f"{self.base_dir}/weight")
        os.makedirs(f"{self.base_dir}/post")
        
    def save_columns_names(self, features_names, band_names, array_names, 
                           label_names, error_scale_names):
        save_dict = {
            "features_names": features_names, 
            "band_names": band_names, 
            "array_names": array_names, 
            "label_names": label_names, 
            "error_scale_names": error_scale_names
        }
        with open(f"{self.base_dir}/column_names.json", "w") as file:
            json.dump(save_dict, file)
    
    def save_scaler(self, label_scaler, feature_scaler, spectrum_scalers, error_scaler):
        joblib.dump(label_scaler, f"{self.base_dir}/label_scaler.pkl")
        joblib.dump(feature_scaler, f"{self.base_dir}/feature_scaler.pkl")
        joblib.dump(spectrum_scalers, f"{self.base_dir}/spectrum_scalers.pkl")
        joblib.dump(error_scaler, f"{self.base_dir}/error_scaler.pkl")
    
    def save_config(self, config):
        with open(f"{self.base_dir}/config.json", "w") as file:
            json.dump(config, file)
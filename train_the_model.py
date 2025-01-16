from SpT.model import *
from SpT.process_data import *
from SpT.train import *
from SpT.save import Save

import numpy as np
import pandas as pd
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1237)  # 你可以选择任何喜欢的数字作为种子

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")
    
    
file_name = __file__.split("/")[-1]
file_name = file_name.split(".")[0]
TRAIN_NAME = f'runs_xco2/{file_name}'

df_train = read_data(["2017", "2018", "2019"])
df_future = read_data(["2020"])

features_names = ["relative_azimuth", "solar_zenith", "zenith", "polarization_angle", 
            "solar_distance", "relative_velocity", "sounding_index",
            "surface_pressure_apriori_fph", "xco2_apriori"]
band_names = ["o2", "weak_co2", "strong_co2"]
array_names = ["radiances", "snrs"]
label_names = ["xco2_diff"]
error_scale_names = ["xco2_uncert"]

column_names = [features_names, band_names, array_names, label_names, error_scale_names]
label_scaler, feature_scaler, spectrum_scalers, error_scaler = build_scaler(df_train, *column_names)
scalers = [label_scaler, feature_scaler, spectrum_scalers, error_scaler]

save = Save(file_name)
save.save_columns_names(features_names, band_names, array_names, label_names, error_scale_names)
save.save_scaler(*scalers)


dataset = create_dataset(df_train, *scalers, *column_names)
idxs = np.random.permutation(np.arange(len(dataset)))
train_dataset = Subset(dataset, idxs[:int(len(idxs)*0.8)])
valid_dataset = Subset(dataset, idxs[int(len(idxs)*0.8):int(len(idxs)*0.9)])
test_dataset = Subset(dataset, idxs[int(len(idxs)*0.9):])

future_dataset = create_dataset(df_future, *scalers, *column_names)
idxs = np.random.permutation(np.arange(len(future_dataset)))
future_dataset = Subset(future_dataset, idxs)

datasets = [train_dataset, valid_dataset, test_dataset, future_dataset]

for dataset in datasets:
    print(len(dataset))

d_model = 256

# channels = [len(array_names), 16, d_model]
# kernels = [21, 15]

feature_dim = datasets[0][0]["features"].shape[0]

config = {
    'patch_size': 64,
    'stride': 64,
    'feature_dim': feature_dim,
    'd_model': d_model,
    'nhead': 2,
    'num_transformer_layers': 8,
    'dim_feedforward': 1024,
    'final_mlp_layers': [128, 1]
}

save.save_config(config)

model = MyModel(**config)
model.to(device)
with torch.no_grad():
    model.eval()
    for data in DataLoader(train_dataset, batch_size=10):
        features_names = data["features"]
        radiances = data["spectra"]
        labels = data["labels"]
        
        features_names = features_names.to(device)
        radiances = radiances.to(device)
        print(features_names.shape, radiances.shape)
        print(model.forward(features_names, radiances))
        print(labels)
        break


label_info = [label_scaler.mean_, label_scaler.scale_]
print(label_info)

# Training parameter
learning_rate = 1e-4
decay_rate = 1e-2
decay_steps = 20
batch_size = 128
epochs = 141

# Early stop parameter
patience = 40.0  # 没有改进时将等待的epochs数量
min_delta = 1e-6  # 认为性能改进是显著的最小变化
best_loss = float('inf')  # 最好的损失值
val_loss = 0.0

model = MyModel(**config).to(device)

# model.load_state_dict(torch.load(f"model_status/11_train_04_positional_encoding/weight/060.pth"))

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)
# criterion = nn.MSELoss()
# criterion = weighted_mse_loss
criterion = nn.HuberLoss(delta=0.5)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=decay_steps,  # 这里的 T_max 是一个周期内的步骤数，之后会重置
    eta_min=0  # 设置学习率衰减的最小值
)
early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
train_in_class = TrainInClass(TRAIN_NAME, file_name, model, criterion, optimizer,
                              scheduler, epochs, batch_size, early_stopper, device,
                              datasets, label_info, feature_scaler)

train_in_class.train()
fig = train_in_class.make_plot(10000)
fig_name =  f"model_status/{file_name}/img/final.png"
fig.savefig(fig_name)


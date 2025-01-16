from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import List
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from .process_data import *
from .support_func import r2


SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

def my_plot(x, y, fig, ax):
    LBL = x[:10000] * 1e6
    MLP = y[:10000] * 1e6

    xy = np.vstack([MLP, LBL])
    z = gaussian_kde(xy)(xy)
    z=(z-np.min(z))/(np.max(z)-np.min(z))
    idx = z.argsort()
    MMLP, LLBL, z = MLP[idx], LBL[idx], z[idx]
    
    # min_value = 0.99*np.min(LBL)
    # max_value = 1.01*np.max(LBL)
    min_value = 380
    max_value = 430
    xyline = np.linspace(min_value,max_value,61)
    

    # fig,ax = plt.subplots(figsize=(4,3))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax.set_ylabel('Predicted [ppm]')
    ax.set_xlabel('Ground truth [ppm]')
    ax.axis([min_value,max_value,min_value,max_value])
    ax.set_aspect(1.0)
    text1 = 'N:'+str(len(LBL)) 
    R2_test = r2(LBL,MLP)
    text2 = 'R$^2$:'+str("%.3f" % R2_test)
    text3 = 'ME: '+str("%.3f" % np.mean(LBL-MLP))+' ppm'
    text4 = 'RMSE: '+str("%.3f" % np.sqrt(mean_squared_error(LBL, MLP)))+' ppm'
    ax.plot(xyline, xyline, 'r-',label='Ideal$\pm$1%', linewidth=2)
    ax.fill_between(xyline, xyline*1.01, xyline*0.99,
        alpha=0.5, edgecolor='0.4', facecolor='0.4',
        linewidth=1, linestyle='--', antialiased=True)
    cf = ax.scatter(LLBL, MMLP, c=z,s=3, alpha=1.0)#facecolors='none',
    fig.colorbar(cf, label='Number density')

    ax.set_xlim([380, 430])
    ax.set_ylim([380, 430])

    z0 = 0.97
    dz = 0.07
    ax.text(0.02,z0,text1,ha='left',va='top',transform=ax.transAxes, fontsize=6)
    ax.text(0.02,z0-dz,text2,ha='left',va='top',transform=ax.transAxes, fontsize=6)
    ax.text(0.02,z0-dz*2,text3,ha='left',va='top',transform=ax.transAxes, fontsize=6)
    ax.text(0.02,z0-dz*3,text4,ha='left',va='top',transform=ax.transAxes, fontsize=6)
    return fig, ax

def predict(data, model, device, batch_size, stad_info, feature_scaler):
    model.eval()
    with torch.no_grad():
        x_all, y_all = [], []
        label_mean = stad_info[0]
        label_std = stad_info[1]
        
        for data in DataLoader(data, batch_size=batch_size):
            features = data["features"].to(device)
            radiances = data["spectra"].to(device)
            labels = data["labels"]
            output = model(features, radiances)
            
            xco2_priori = feature_scaler.inverse_transform(features.cpu().numpy())[:, -1]
            x_all.append(labels.numpy().flatten() * float(label_std) + float(label_mean) + xco2_priori)
            y_all.append(output.cpu().numpy().flatten() * float(label_std) + float(label_mean) + xco2_priori)
    return np.concatenate(x_all), np.concatenate(y_all)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        return False

class TrainInClass:
    def __init__(self, name: str, file_name: str, model: torch.nn.Module, 
                 criterion: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler, 
                 epochs: int, batch_size: int, 
                 early_stopper: EarlyStopper, device: torch.device, 
                 datas: List[torch.utils.data.Dataset], stad_info: List, feature_scaler):
        self.writer = SummaryWriter(name)
        self.file_name = file_name
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopper = early_stopper
        self.device = device
        self.datasets = datas
        self.stad_info = stad_info
        self.feature_scaler = feature_scaler

    def make_plot(self, plot_max_n = 1000):
        """_summary_

        Parameters
        ----------
        model : _type_
            _description_
        datas : _type_
            datas = [train_data, test_data, valid_data, future_data]
        plot_max_n : int, optional
            _description_, by default 500

        Returns
        -------
        _type_
            _description_
        """
        fig, axs = plt.subplots(1,4, figsize=[14, 2.7])

        
        titles = ["Train", "Test", "Valid", "Future"]
        for i, dataset in enumerate(self.datasets):
            ax = axs[i]
            n = len(dataset)
            n = min(plot_max_n, n)
            subset = random_subset(dataset, n)
            x, y = predict(subset, self.model, self.device, self.batch_size, 
                           self.stad_info, self.feature_scaler)
            fig, _ = my_plot(x, y, fig, ax)
            ax.set_title(titles[i])
        fig.tight_layout()
        return fig

    def train(self):
        valid_loss = float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            train_dataset, valid_dataset, _, _ = self.datasets
            
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, 
                num_workers=5)
            
            progress_bar = tqdm(train_dataloader, total = len(train_dataloader), 
                                desc=f'Epoch {epoch+1} Progress', leave=True)
            for i, data in enumerate(progress_bar):
                features = data["features"].to(self.device)
                radiances = data["spectra"].to(self.device)
                labels = data["labels"].to(self.device)
                error_weights = data["error_weights"].to(self.device)
                self.optimizer.zero_grad()
                
                output = self.model(features, radiances)
                loss = self.criterion(output, labels)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                progress_bar.set_postfix(loss=f"{train_loss/(i+1):.3e}", 
                                        last_val_loss = f"{valid_loss:.3e}", 
                                        stop_counter = f"{self.early_stopper.counter}",
                                        min_val_loss = f"{self.early_stopper.min_validation_loss:.3e}",
                                        learning_rate = f"{self.optimizer.param_groups[0]['lr']:.3e}",
                                        refresh=True)
            self.writer.add_scalars('Loss', {"train": train_loss/(i+1)}, epoch)
            
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                i = 0
                valid_dataloader = DataLoader(
                    valid_dataset, batch_size=self.batch_size, shuffle=True, 
                    num_workers=5)
                for data in valid_dataloader:
                    features = data["features"].to(self.device)
                    radiances = data["spectra"].to(self.device)
                    labels = data["labels"].to(self.device)
                    error_weights = data["error_weights"].to(self.device)

                    output = self.model(features, radiances)
                    loss = self.criterion(output, labels)
                    valid_loss += loss.item() * len(features)
                valid_loss /= len(valid_dataset)
            
            self.writer.add_scalars('Loss', {"valid": valid_loss}, epoch)
            
            if epoch % 5 == 0:
                if epoch % 40 == 20:
                    fig = self.make_plot(10000)
                else:
                    fig = self.make_plot()
                fig_name = f"model_status/{self.file_name}/img/{epoch:03d}.png"
                fig.savefig(fig_name)
                self.writer.add_figure('Scatter', fig, epoch)
        
            if self.early_stopper.early_stop(valid_loss):
                print(f"Early stopping at epoch {epoch}")
                break
            torch.save(self.model.state_dict(), f"model_status/{self.file_name}/weight/{epoch:03d}.pth")
            self.scheduler.step()

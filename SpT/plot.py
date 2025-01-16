import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from .train import predict

sns.set_context("paper", font_scale = 0.75)
sns.set_palette(sns.color_palette("bright", 8))

sns.set_style('white')

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

rc = {
    'xtick.major.size': 2,
    'xtick.major.width': 0.5,
    'ytick.major.size': 2,
    'ytick.major.width': 0.5,
    'xtick.bottom': True,
    'ytick.left':True,
    'font.size': MEDIUM_SIZE,
    'axes.titlesize': MEDIUM_SIZE,
    'axes.labelsize': MEDIUM_SIZE,
    'xtick.labelsize': SMALL_SIZE,
    'ytick.labelsize': SMALL_SIZE,
    'legend.fontsize': SMALL_SIZE,
    'figure.titlesize': BIGGER_SIZE,
    'savefig.dpi': 300,
    'figure.dpi': 300,
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif", "Nimbus Roman No9 L", "Times"]
}

plt.rcParams.update(rc)

lbl_name = "OCO-2"
mlp_name = "SpT"

def plot_plume(latitudes, longitudes, lbl, mlp, additional_info, vmin, vmax,
               resolution = "i"):
    factory_position = additional_info["factory_position"]
    lat_mid = additional_info["lat_mid"]
    lon_mid = additional_info["lon_mid"]
    delta_lat = additional_info["delta_lat"] / 2
    delta_lon = additional_info["delta_lon"] / 2
    titles = [lbl_name, mlp_name]
    
    # vmin = np.min(lbl) * 1e6
    # vmax = np.max(lbl) * 1e6
    
    figs = []
    for xco2s, title in zip([lbl, mlp], titles):
        fig, ax = plt.subplots(figsize=[3, 2.3])
        map = Basemap(projection='merc', llcrnrlat=lat_mid - delta_lat, urcrnrlat=lat_mid + delta_lat,
                        llcrnrlon=lon_mid - delta_lon, urcrnrlon=lon_mid + delta_lon, resolution=resolution, ax=ax)

        map.drawcoastlines()
        map.fillcontinents(color='lightgray', lake_color='aqua')
        map.drawmapboundary(fill_color='aqua')

        map.drawparallels(np.arange(int(lat_mid - delta_lat), int(lat_mid + delta_lat) + 1, 0.5), labels=[1,0,0,0], linewidth=0.3)
        map.drawmeridians(np.arange(int(lon_mid - delta_lon), int(lon_mid + delta_lon) + 1, 0.5), labels=[0,0,0,1], linewidth=0.3)

        x, y = map(longitudes, latitudes)

        cf = map.scatter(x, y, marker='o', edgecolor='k', 
                        c=xco2s*1e6, vmin=vmin, vmax=vmax, linewidth=0.2, cmap = "jet", s=5)
        
        x, y = map([factory_position[0]], [factory_position[1]])
        map.scatter(x, y, marker = "^", s=15)
        cb = plt.colorbar(cf)
        cb.set_label("XCO$_2$ [ppm]")
        
        ax.set_title(title)
        plt.tight_layout()
        figs.append(fig)
    return figs

def plot_plume_scatter(longitudes, lbl, mlp, vmin=405, vmax=425):
    fig, ax = plt.subplots(figsize=[3, 2.5])
    ax.scatter(longitudes, mlp*1e6, marker="x", s=8, linewidths=0.5, label=mlp_name, color="b")
    ax.scatter(longitudes, lbl*1e6, marker="x", s=8, linewidths=0.5, label=lbl_name, color="r")
    ax.set_xlabel(r"Longitude [$^\circ$E]")
    ax.set_ylabel("XCO$_2$ [ppm]")
    ax.legend()
    ax.set_ylim([vmin, vmax])
    return fig, ax

def plot_viloin(oco_time, lbl, mlp):
    data = {
        'oco_time': oco_time,
        mlp_name: mlp*1e6,
        lbl_name: lbl*1e6
    }

    df_plot = pd.DataFrame(data)
    df_plot['oco_time'] = pd.to_datetime(df_plot['oco_time'])
    df_plot['month'] = df_plot['oco_time'].dt.to_period('M')

    # 重塑数据以便于绘图
    df_melted = pd.melt(df_plot, id_vars=['oco_time', 'month'], 
                        value_vars=[mlp_name, lbl_name], var_name='type', 
                        value_name='value')
    df_melted = df_melted.sort_values("oco_time")

    # 生成violin plot
    fig, ax = plt.subplots(figsize=(6, 2))
    sns.violinplot(x='month', y='value', hue='type', data=df_melted, split=True, width=1.1,
                linewidth=0.5, inner='quart', ax = ax, linecolor='white', density_norm='width')
    # plt.title('Distribution of mlp and lbl by Month')
    ax.set_xlabel('Time')
    ax.set_ylabel('XCO$_2$ [ppm]')
    ax.legend(ncol=2)
    for i, tick in enumerate(ax.get_xticklabels()):
        if i % 6 != 0:
            tick.set_visible(False)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if i % 6 == 0:
            tick.tick1line.set_markersize(4)  # 设置tick长度
            tick.tick2line.set_markersize(4)  # 设置tick长度
    plt.tight_layout()
    return fig, ax


import time
import logging
import torch
import pandas as pd
from utils.utils import combine_dataset
from utils.function_utils import build_input_features, log
from stage1_trainer.fm_s1_trainer import evolution_search
from config.configs import General_Config, FM_Config
from utils.utils import load_pd_data, make_data, set_seed
from utils.utils import USE_FEATURES
import numpy as np
from model.FM_Model import FM_Model
from utils.function_utils import random_selected_interaction_type
import os
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import numpy as np


class CustomHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        lw = orig_handle.get_linewidth() * 4  
        line = plt.Line2D([xdescent, width - xdescent], [height/2. - ydescent, height/2. - ydescent],
                          linestyle=orig_handle.get_linestyle(),
                          linewidth=lw, color=orig_handle.get_color(), alpha=0.3)
        artists.insert(0, line)
        
        return artists

# Draw Temp Figure Functon
def draw_temp(save_path, depth, obs, sim, pred, dates, ice_flag, model_type):
    plt.figure(figsize=(64, 36))
    ax = plt.gca()
    ax.set_prop_cycle('color', plt.cm.viridis(np.linspace(0, 1, depth.shape[0])))
    # filter layer which has obs data:
    for i in range(depth.shape[0]):
        if i == 0:
            if not np.isnan(obs[i,:]).all():
                ax.plot(dates, obs[i, :], 'o', label=f'Obs Depth {depth[i]:.2f}m', color= '#eb5834')  # Observed data
            ax.plot(dates, sim[i, :], '--', label=f'Sim Depth {depth[i]:.2f}m', dashes=(5, 8)) # Simulated data
            ax.plot(dates, pred[i, :], '-', label=f'{model_type} Pred Depth {depth[i]:.2f}m', color= '#eb3434') # Predicted data
        else:
            if not np.isnan(obs[i,:]).all():
                ax.plot(dates, obs[i, :], 'o', label=f'Obs Depth {depth[i]:.2f}m')  # Observed data
            ax.plot(dates, sim[i, :], '--', label=f'Sim Depth {depth[i]:.2f}m', dashes=(5, 8)) # Simulated data
            ax.plot(dates, pred[i, :], '-', label=f'{model_type} Pred Depth {depth[i]:.2f}m') # Predicted data

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 20,
           }
    ax.set_xlabel('Date', fontdict=font)
    ax.set_ylabel('Values', fontdict=font)
    ax.set_title(f'Observation, Simulation, and Prediction', fontdict=font)
    
    ax.tick_params(axis='x', labelsize=20, labelcolor='blue')
    ax.tick_params(axis='y', labelsize=20, labelcolor='blue')

    plt.legend()
    plt.savefig(save_path)
    plt.close()
    return

def draw_temp_new(save_path, depth, obs, sim, pred, dates, ice_flag, model_type):
    plt.figure(figsize=(87.5, 35))
    FONT_SIZE_LEG = 70
    FONT_LABEL = 120
    LINEWIDTH = 8
    MAKER_S = 2000

    if model_type == 'lstm':
        model_type = 'LSTM'
    elif model_type == 'fm-pg':
        model_type = 'PGFM'
    plt.style.use('seaborn-whitegrid')
    ax = plt.gca()
    ax.grid(True, linewidth=0.5, linestyle='-', color='#dadada')
    # color_list = ['#2ca02c','#03fcdb','#03fcdb','#6703fc','#2ca02c','#fc0398','#fc034e', '#d62728']
    # color_list = ['#41bf80', '#2ca02c', '#9ee8b6', '#abc8d3', '#eb8334', '#d98f99','#d62728', '#8a231c']
    color_list = ['#1a6499', '#3b9ade', '#92c7ed', '#abc8d3', '#eb8334', '#d98f99','#d62728', '#8a231c']
    
    # filter layer which has obs data:
    range_list = list(range(0, depth.shape[0], 5)) + [depth.shape[0] - 1]
    
    # ax.set_prop_cycle('color', plt.cm.viridis(np.linspace(0, 1, len(range_list))))
    for index, i in enumerate(range_list):
        if i == 0:
            if not np.isnan(obs[i,:]).all():
                # ax.plot(dates, obs[i, :], 'o', label=f'Obs Depth {depth[i]:.2f}m', color= '#eb5834')  # Observed data
                for size, alpha in zip(range(3000, 8000, 2100), np.linspace(0.05, 0.15, 4)):
                    ax.scatter(dates, obs[i, :], marker='o', s=size, color=color_list[index], alpha=alpha, edgecolors='none')
                ax.scatter(dates, obs[i, :], label=f'Obs (-{depth[i]:.1f}$m$)', marker='o', s=MAKER_S, color=color_list[index], alpha=0.9)
            ax.plot(dates, sim[i, :], '--', linewidth=LINEWIDTH, label=f'Phy-based (-{depth[i]:.1f}$m$)', dashes=(3, 4), color=color_list[index]) # Simulated data
            ax.plot(dates, pred[i, :], '-', linewidth=LINEWIDTH, label=f'{model_type} (-{depth[i]:.1f}$m$)', color=color_list[index]) # Predicted data
        else:
            if not np.isnan(obs[i,:]).all():
                # ax.plot(dates, obs[i, :], 'o', label=f'Obs Depth {depth[i]:.2f}m', )  # Observed data
                for size, alpha in zip(range(3000, 8000, 2100), np.linspace(0.05, 0.15, 4)):
                    ax.scatter(dates, obs[i, :], marker='o', s=size, color=color_list[index], alpha=alpha, edgecolors='none')
                ax.scatter(dates, obs[i, :], label=f'Obs (-{depth[i]:.1f} $m$)', marker='o', s=MAKER_S, alpha=0.9, color=color_list[index])
            ax.plot(dates, sim[i, :], '--', linewidth=LINEWIDTH, label=f'Phy-based (-{depth[i]:.1f}$m$)', dashes=(3, 4), color=color_list[index]) # Simulated data
            ax.plot(dates, pred[i, :], '-', linewidth=LINEWIDTH, label=f'{model_type} (-{depth[i]:.1f}$m$)', color=color_list[index]) # Predicted data


    ax.set_ylabel('Water temperature ($°C$)', fontsize = FONT_LABEL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    # 设置刻度标签的字体
    ax.tick_params(axis='x', labelsize=FONT_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_LABEL)

    plt.legend(fontsize=FONT_SIZE_LEG, loc='upper left', frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return
# Draw Do Figure Function
def draw_do(save_path, obs, pred, sim, date, is_stratified, pic_model_name): 
    """
    Draws a DO chart and saves the image.

    Parameters:
    save_path (str): The path to save the image.
    obs (numpy.array): Observed data.
    pred (numpy.array): Predicted data.
    sim (numpy.array): Simulated data.
    date: Date
    is_stratified (numpy.array): Whether the lake is stratified.

    Returns:
    None: This function does not return a value. It saves the generated image to the specified path.
    """
    # 设置图表大小和分辨率
    plt.figure(figsize=(12, 4))

    # 绘制epi层数据

    # Updated color palette for clear distinction
    c_pred = '#1f77b4'  # Blue for NGCE predictions
    c_label = '#ff7f0e'  # Orange for Simulated DO concentrations
    c_obs_epi = '#2ca02c'  # Green for Obs DO conc. (epi)
    c_obs_hyp = '#d62728'  # Red for Obs DO conc. (hyp)
    c_fill_pred = '#8ea7d4'  # Lighter blue for prediction fill
    c_fill_label = '#ffd59e'  # Lighter orange for label fill
    c_obs_mixed = '#bebbbb' # Gray
    

    plt.style.use('seaborn-whitegrid')  # Use a clean and professional style
    FONT_SIZE1 = 18  # Slightly reduced for balance
    FONT_SIZE2 = 15
    FONT_SIZE_LEG = 15.5
    LINEWIDTH = 0.9 # More elegant line width
    LINEWIDTH2 = 0.9
    LINEWIDTH_LASER = 6.5
    # 绘制预测数据 
    pred[1, :] = np.where(is_stratified == 0, pred[0, :], pred[1, :])
    pred_epi = np.where(is_stratified != 0, pred[0,:], pred[0,:])
    pred_hypo = np.where(is_stratified != 0, pred[1,:], np.nan)
    pred_mixed = np.where(is_stratified == 0, pred[0,:], np.nan)
    print("pred_mixed shape:", pred_mixed.shape)
    if pic_model_name == 'April' or pic_model_name == 'Pril':
        plt.plot(date, pred_epi, label= f"$\t{{{pic_model_name}}}$ (epi)", color=c_pred, linewidth=LINEWIDTH)  
        plt.plot(date, pred_hypo, label= f"$\t{{{pic_model_name}}}$ (hyp)", color=c_pred, linewidth=LINEWIDTH, linestyle='--') 
        # laser line
        line_mixed_pred, = plt.plot(date, pred_mixed, label= f"$\t{{{pic_model_name}}}$ (total)", color=c_pred, linewidth=LINEWIDTH2) 
        plt.plot(date, pred_mixed, color=c_pred, linewidth=LINEWIDTH_LASER, alpha=0.3) 
    else:
        line_mixed_pred, = plt.plot(date, pred_mixed, label=f"{pic_model_name} (total)", color=c_pred, linewidth=LINEWIDTH2)
        plt.plot(date, pred_mixed, color=c_pred, linewidth=LINEWIDTH_LASER, alpha=0.3)
        
        plt.plot(date, pred_epi, label=f"{pic_model_name} (epi)", color=c_pred, linewidth=LINEWIDTH)  
        plt.plot(date, pred_hypo, label=f"{pic_model_name} (hyp)", color=c_pred, linewidth=LINEWIDTH, linestyle='--') 
        # laser line 



    # Fill between Pred APRIL DO concentrations
    plt.fill_between(date, pred_epi, pred_hypo, color=c_fill_pred, alpha=0.4)


    # 绘制模拟数据 - epi
    sim_data_epi = np.where(is_stratified != 0, sim[0, :], np.nan)
    sim_data_hypo = np.where(is_stratified != 0, sim[1, :], np.nan)
    sim_data_mixed = np.where(is_stratified == 0, sim[0, :], np.nan)


    plt.plot(date, sim_data_epi, label='Process (epi)', color=c_label, alpha=0.8, linewidth=LINEWIDTH)
    # 绘制模拟数据 - hypo
    plt.plot(date, sim_data_hypo, label='Process (hyp)', linestyle='--', color=c_label, alpha=0.8, linewidth=LINEWIDTH)
    # 绘制模拟数据 - mixed
    line_mixed_sim, = plt.plot(date, sim_data_mixed, label='Process (total)', color=c_label, alpha=0.8, linewidth=LINEWIDTH2)
    plt.plot(date, sim_data_mixed, color=c_label, alpha=0.3, linewidth=LINEWIDTH_LASER)

    # Fill between Simulated DO concentrations
    plt.fill_between(date, sim_data_epi, sim_data_hypo, color=c_fill_label, alpha=0.3)


    # 绘制观测数据 - epi
    obs_data_epi = np.where(is_stratified != 0, obs[0, :], np.nan)
    obs_data_hypo = np.where(is_stratified != 0, obs[1, :], np.nan)
    obs_data_mixed = np.where(is_stratified == 0, obs[0, :], np.nan)

    for size, alpha in zip(range(300, 1200, 300), np.linspace(0.05, 0.15, 4)):
        plt.scatter(date, obs_data_epi, marker='o', s=size, color=c_obs_epi, alpha=alpha, edgecolors='none')
    plt.scatter(date, obs_data_epi, marker='o', s=100, color=c_obs_epi, label='Obs DO conc. (epi)', alpha=0.8)

    # 绘制观测数据 - hypo
    for size, alpha in zip(range(300, 1200, 300), np.linspace(0.05, 0.15, 4)): 
        plt.scatter(date, obs_data_hypo, marker='o', s=size, color=c_obs_hyp, alpha=alpha, edgecolors='none')
    plt.scatter(date, obs_data_hypo, marker='o', s=100, color=c_obs_hyp, label='Obs DO conc. (hyp)', alpha=0.8)

        # 绘制观测数据 - mixed
    # if len(obs_indices_mixed) > 0: 
    for size, alpha in zip(range(300, 1200, 300), np.linspace(0.05, 0.15, 4)):
        plt.scatter(date, obs_data_mixed, marker='o', s=size, color=c_obs_mixed, alpha=alpha, edgecolors='none')
    plt.scatter(date, obs_data_mixed, marker='o', s=100, color=c_obs_mixed, label='Obs DO conc. (total)', alpha=0.8)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # 设置图表标题、坐标轴标签和图例
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=0, ha="right")
    plt.ylabel("DO concentration ($g \, m^{-3}$)", fontsize=FONT_SIZE1)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE2)
    plt.legend(fontsize=FONT_SIZE_LEG, loc='upper right', frameon=True, edgecolor='black')
    plt.legend(handler_map={line_mixed_pred: CustomHandler(), line_mixed_sim: CustomHandler()}, fontsize=FONT_SIZE_LEG, loc='upper right', frameon=True, edgecolor='black')
    # mixed 
    # dates_num = mdates.date2num(Date_epi)  
    # date_150 = dates_num[0]  # 
    # date_290 = dates_num[350-160]  #
    # plt.axvspan(dates_num[0], date_150, color='lightyellow', alpha=0.5)
    # plt.axvspan(date_290, dates_num[-1], color='lightyellow', alpha=0.5)

    # 展示图表
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # plt.show

def draw_do_new(save_path, obs, pred, sim, date, is_stratified, pic_model_name): 
    """
    Draws a DO chart and saves the image.

    Parameters:
    save_path (str): The path to save the image.
    obs (numpy.array): Observed data.
    pred (numpy.array): Predicted data.
    sim (numpy.array): Simulated data.
    date: Date
    is_stratified (numpy.array): Whether the lake is stratified.

    Returns:
    None: This function does not return a value. It saves the generated image to the specified path.
    """
    # 设置图表大小和分辨率
    # plt.figure(figsize=(10, 3))

    plt.figure(figsize=(87.5, 26))
    # 绘制epi层数据
    color_list = ['#1a6499', '#3b9ade', '#92c7ed', '#abc8d3', '#eb8334', '#d98f99','#d62728', '#8a231c']
    
    c_pred_epi = '#1a6499'  # Blue for NGCE predictions
    c_pred_hypo = '#d62728'
    c_sim_epi = '#3b9ade'  # Orange for Simulated DO concentrations
    c_sim_hypo = '#d98f99'  # Orange for Simulated DO concentrations

    c_obs_epi = '#1a6499'  # Green for Obs DO conc. (epi)
    c_obs_hyp = '#d62728'  # Red for Obs DO conc. (hyp)
    c_fill_pred = '#8ea7d4'  # Lighter blue for prediction fill
    c_fill_label = '#ffd59e'  # Lighter orange for label fill
    c_obs_mixed = '#bebbbb' # Gray
    

    

    plt.style.use('seaborn-whitegrid')  # Use a clean and professional style
    FONT_SIZE1 = 120 # Slightly reduced for balance
    FONT_SIZE2 = 120
    FONT_SIZE_LEG = 120
    LINEWIDTH = 15 # More elegant line width
    LINEWIDTH2 = 15
    LINEWIDTH_LASER = 75
    MAKER_S = 4000
    if pic_model_name == 'lstm':
        pic_model_name = 'LSTM'
    elif pic_model_name == 'fm-pg':
        pic_model_name = 'PGFM'
    # 绘制观测数据 - epi
    obs_data_epi = np.where(is_stratified != 0, obs[0, :], np.nan)
    obs_data_hypo = np.where(is_stratified != 0, obs[1, :], np.nan)
    obs_data_mixed = np.where(is_stratified == 0, obs[0, :], np.nan)

    # 绘制观测数据 - mixed
    # if len(obs_indices_mixed) > 0: 
    # for size, alpha in zip(range(300, 1200, 300), np.linspace(0.05, 0.15, 4)):
    #     plt.scatter(date, obs_data_mixed, marker='o', s=size, color=c_obs_mixed, alpha=alpha, edgecolors='none')
    # plt.scatter(date, obs_data_mixed, marker='o', s=100, color=c_obs_mixed, label='Obs (total)', alpha=0.8)

    # 绘制模拟数据 - epi
    sim_data_epi = np.where(is_stratified != 0, sim[0, :], sim[0, :])
    sim_data_hypo = np.where(is_stratified != 0, sim[1, :], sim[0, :])
    sim_data_mixed = np.where(is_stratified == 0, sim[0, :], np.nan)

    # Fill between Simulated DO concentrations
    # plt.fill_between(date, sim_data_epi, sim_data_hypo, color=c_fill_label, alpha=0.3)

    # 绘制预测数据 
    pred[1, :] = np.where(is_stratified == 0, pred[0, :], pred[1, :])
    pred_epi = np.where(is_stratified != 0, pred[0,:], pred[0,:])
    pred_hypo = np.where(is_stratified != 0, pred[1,:], pred[0,:])
    pred_mixed = np.where(is_stratified == 0, pred[0,:], np.nan)

    # Epi
    for size, alpha in zip(range(6000, 24000, 4000), np.linspace(0.05, 0.15, 4)):
        plt.scatter(date, obs_data_epi, marker='o', s=size, color=c_obs_epi, alpha=alpha, edgecolors='none')
    plt.scatter(date, obs_data_epi, marker='o', s=MAKER_S, color=c_obs_epi, label='Obs (epi)', alpha=0.8)

    plt.plot(date, sim_data_epi, label='Phy-based (epi)', linestyle='--', color=c_sim_epi, alpha=0.8, linewidth=LINEWIDTH)

    plt.plot(date, pred_epi, label=f"{pic_model_name} (epi)", color=c_pred_epi, linewidth=LINEWIDTH)  


    # Hypo
    for size, alpha in zip(range(6000, 240000, 4000), np.linspace(0.05, 0.15, 4)): 
        plt.scatter(date, obs_data_hypo, marker='o', s=size, color=c_obs_hyp, alpha=alpha, edgecolors='none')
    plt.scatter(date, obs_data_hypo, marker='o', s=MAKER_S, color=c_obs_hyp, label='Obs (hyp)', alpha=0.8)

    plt.plot(date, sim_data_hypo, label='Phy-based (hyp)', linestyle='--', color=c_sim_hypo, alpha=0.8, linewidth=LINEWIDTH)

    plt.plot(date, pred_hypo, label=f"{pic_model_name} (hyp)", color=c_pred_hypo, linewidth=LINEWIDTH, linestyle='-') 
    

    # Mixed
    line_mixed_sim, = plt.plot(date, sim_data_mixed, linestyle='--', label='Phy-based (total)', color=c_sim_epi, alpha=0.8, linewidth=LINEWIDTH2)
    plt.plot(date, sim_data_mixed, color=c_sim_epi, alpha=0.3, linewidth=LINEWIDTH_LASER)

    line_mixed_pred, = plt.plot(date, pred_mixed, label=f"{pic_model_name} (total)", color=c_pred_epi, linewidth=LINEWIDTH2)
    plt.plot(date, pred_mixed, color=c_pred_epi, linewidth=LINEWIDTH_LASER, alpha=0.3)
    
    # laser line 
    # Fill between Pred APRIL DO concentrations
    # plt.fill_between(date, pred_epi, pred_hypo, color=c_fill_pred, alpha=0.4)

    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # 设置图表标题、坐标轴标签和图例
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=0, ha="right")
    plt.ylabel("DO concentration ($g \, m^{-3}$)", fontsize=FONT_SIZE1)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE2)
    plt.legend(fontsize=FONT_SIZE_LEG, loc='upper right', frameon=True, edgecolor='black')
    plt.legend(handler_map={line_mixed_pred: CustomHandler(), line_mixed_sim: CustomHandler()}, fontsize=FONT_SIZE_LEG, loc='upper right', frameon=True, edgecolor='black')
    # mixed 
    # dates_num = mdates.date2num(Date_epi)  
    # date_150 = dates_num[0] 
    # date_290 = dates_num[350-160]  
    # plt.axvspan(dates_num[0], date_150, color='lightyellow', alpha=0.5)
    # plt.axvspan(date_290, dates_num[-1], color='lightyellow', alpha=0.5)


    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # plt.show

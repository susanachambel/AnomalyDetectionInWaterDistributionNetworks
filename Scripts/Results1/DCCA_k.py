# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan

@about: Heat Maps
    
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *
from event_archive_2 import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from itertools import combinations, product

def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])

def get_combo_name_2(sensor1, sensor2):
    return str(sensor1) + "-" + str(sensor2)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", title="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.pcolormesh(data, **kwargs, vmin=-1, vmax=1)
    
    """
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, pad=0.02)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.outline.set_visible(False)
    """
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    ax.invert_yaxis()
    
    """
    plt.text(-0.04, 0.60, 'Pressure\nSensors', color='k', rotation=90, transform=ax.transAxes, va="center", ha="center")
    plt.text(-0.04, 0.10, 'Volumetric Flowrate\nSensors', color='k', rotation=90, transform=ax.transAxes, va="center", ha="center")
    plt.text(0.405, -0.04, 'Pressure\nSensors', color='k', transform=ax.transAxes, va="center", ha="center")
    plt.text(0.905, -0.04, 'Volumetric Flowrate\nSensors', color='k', transform=ax.transAxes, va="center", ha="center")
    """
    
    """
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", #45
             rotation_mode="anchor")
    """
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    """
    ax.set_xticks(np.arange(data.shape[1]), minor=True)
    ax.set_yticks(np.arange(data.shape[0]), minor=True)
    """
    ax.axis('off')
    
    ax.set_title(title)
    
    ax.hlines([21.0], *ax.get_xlim(), color='white', linewidth=1.5)
    ax.vlines([21.0], *ax.get_ylim(), color='white', linewidth=1.5)
    
    ax.grid(which="minor", color="w", linestyle='-', linewidth=4)
    #ax.tick_params(which="minor", bottom=False, left=False)
    cbar = None
    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
        
    threshold1 = im.norm(threshold+0.5)
    threshold2 = im.norm(threshold-0.5)
        
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int((im.norm(data[i, j]) > threshold1) | (im.norm(data[i, j]) < threshold2))])
            #kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def get_correlation_map_simulated(sensors, columns, df_row): 
    rows_result = []
    for sensor in sensors:
        row = list(product([sensor],sensors))
        row_result = []
        for pair in row:
            sensor1 = pair[0]
            sensor2 = pair[1]
            corr = 0
            if (sensor1==sensor2):
                corr = 1
            elif (get_combo_name(pair) in columns):
                corr = df_row[get_combo_name(pair)]
            else:
                corr = df_row[get_combo_name_2(sensor2,sensor1)]
            row_result.append(round(corr,2))
        rows_result.append(row_result)
    return np.array(rows_result)

def get_sensors_n(data_type):
    if data_type == 'p':
        sensors = list(range(1, 21, 1))
        n = 3
    elif data_type == 'q':    
        sensors = ['1', '2', '6', '9', '10']
        n = 1
    else:
        sensors = list(range(1, 27, 1))
        n = 3
        
    return sensors, n

def get_df(path_init, width, correlation_type, dcca_k):
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_' + correlation_type.lower() +'_' + str(width) + '_' + str(dcca_k) + '.csv'
    df = pd.read_csv(path, index_col=0)
    columns = df.columns
    return df, columns

def leakage_sizes_hm(path_init):

    width = 40
    correlation_type = 'dcca'
    data_types = ['all']
    dcca_k = list(range(2,11,1))
    
    for data_type in data_types:
    
        sensors, n = get_sensors_n(data_type)
        df, columns = get_df(path_init, data_type, width, correlation_type, dcca_k)
        
        event_id_init = 1393
        event_id_final = event_id_init+9
        events_id = list(range(event_id_init-1, event_id_final+1, 2))
        
        coefficients = ['0.05', '0.1', '0.5', '1.0', '1.5', '2.0']
        
        fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(10.4,6)) #23 13 , 
        
        i = 0
        for ax in axs.flat:
        
            event_id = events_id[i]
        
            ax.set_aspect('equal')
            
            df_row = df.iloc[event_id,:]
            print(df_row)
            correlations = get_correlation_map_simulated(sensors, columns, df_row)
            
            im, cbar = heatmap(correlations, sensors, sensors, ax=ax,
                               cmap="RdBu", cbarlabel=correlation_type.upper(), title='Coef = ' + coefficients[i])
            #annotate_heatmap(im, valfmt="{x:.2f}", threshold=0)
            
            i += 1
        
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), pad=0.03)
        cbar.ax.set_ylabel(correlation_type.upper(), rotation=-90, va="bottom")
        cbar.outline.set_visible(False)
        
        #fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results1\\Leakage Sizes\\' + data_type + '_' + correlation_type + '_' + str(width) + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig=fig)

def dcca_k(path_init):

    width = 40
    correlation_type = 'dcca'
    data_type = 'all'
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    
    row_names = ['1-4', '22-25', '1-25']
    dic = {}
    for row_name in row_names:
        dic[row_name] = {'1':[],'10':[]}
    
    for dcca_k in range(2,41,2):
    
        df, columns = get_df(path_init, width, correlation_type, dcca_k)
        
        df_row_1 = df.iloc[1,:]
        df_row_10 = df.iloc[10,:]
        
        for row_name in row_names:
            dic[row_name]['1'].append(df_row_1[row_name])
            dic[row_name]['10'].append(df_row_10[row_name])
        
    titles = row_names
    
    row_names = []
    for dcca_k in range(2,41,2):
        row_names.append(dcca_k-1)
    
    
    ha_1 = [['left','','','','','','center','','','','','','center','','','','','','center','','','','','','center'],
            ['left','','','','','','center','','','','','','center','','','','','','center','','','','','','center']]
    ha_2 = [['left','','','','','','center','','','','','','center','','','','','','center','','','','','','center'],
            ['left','','','','','','center','','','','','','center','','','','','','center','','','','','','center']]
    ha_3 = [['left','','','','','','center','','','','','','center','','','','','','center','','','','','','center'],
            ['left','','','','','','center','','','','','','center','','','','','','center','','','','','','center']]
    ha_all = [ha_1,ha_2,ha_3] 
    
    xytext_1 = [[12,0,0,0,0,0,-18,0,0,0,0,0,-18,0,0,0,0,0,-18,0,0,0,0,0,-18],
                [-18,0,0,0,0,0,-18,0,0,0,0,0,-18,0,0,0,0,0,-18,0,0,0,0,0,-18]]
    xytext_2 = [[12,0,0,0,0,0,12,0,0,0,0,0,-18,0,0,0,0,0,-18,0,0,0,0,0,-18],
                [-18,0,0,0,0,0,-18,0,0,0,0,0,-18,0,0,0,0,0,-18,0,0,0,0,0,12]]
    xytext_3 = [[-18,0,0,0,0,0,12,0,0,0,0,0,12,0,0,0,0,0,12,0,0,0,0,0,12],
                [12,0,0,0,0,0,12,0,0,0,0,0,12,0,0,0,0,0,12,0,0,0,0,0,12]]
    xytext_all = [xytext_1,xytext_2,xytext_3]
    
        
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10.5,5), sharey=True, sharex=True)
    i = 0
    for ax in axs.flat:
            
        if i == 0:
            ax.set_ylabel('DCCA')
        if i == 1:
            ax.set_xlabel('n')
        
        title = titles[i]
        ha_i = ha_all[i]
        xytext_i = xytext_all[i] 
            
        x = row_names 
        y1 = dic[title]['1']
        y2 = dic[title]['10']
                
        ax.plot(x, y1, color='tab:blue', marker='s', markersize=3, label='w/o leakage')
        ax.plot(x, y2, color='tab:orange', marker='o', markersize=3, label='w/ leakage (coef=2.0)')
        
            
        title_split = title.split('-')
        
        ax.grid(True, axis='y', alpha=0.3, which='both')
        
        j=0
        for x,y1,y2,ha1,ha2,xytext1,xytext2 in zip(x,y1,y2,ha_i[0],ha_i[1],xytext_i[0],xytext_i[1]):

            if j%6 == 0:
                label = "{:.2f}".format(y1)
                ax.annotate(label, # this is the text
                            (x,y1), # this is the point to label
                            textcoords="offset points", # how to position the text
                            xytext=(0,xytext1), # distance from text to points (x,y)
                            ha=ha1, bbox=dict(facecolor="w",edgecolor='tab:blue',alpha=0.4,boxstyle="round")
                            ) # horizontal alignment can be left, right or center
                label = "{:.2f}".format(y2)
                ax.annotate(label, # this is the text
                            (x,y2), # this is the point to label
                            textcoords="offset points", # how to position the text
                            xytext=(0,xytext2), # distance from text to points (x,y)
                            ha=ha2, bbox=dict(facecolor="w",edgecolor='tab:orange',alpha=0.4,boxstyle="round")
                            ) # horizontal alignment can be left, right or center
            j+=1
        
        
        title = 'Sensors ' + title_split[0] + ' & ' + title_split[1]
        
        if i==0:
            ax.legend(loc='lower left')
            title += '\n(two pressure sensors)'
        elif i==1:
            title += '\n(two flowrate sensors)'
        else:
            title += '\n(one sensor of each type)'
            
        ax.set_title(title)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
        i += 1
        
    plt.ylim(-1,1)
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results1\\DCCA K\\dcca_k_lc.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig=fig)
        
    

config = Configuration()
path_init = config.path

dcca_k(path_init)
#leakage_sizes_hm(path_init)


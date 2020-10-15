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
            cbar_kw={}, cbarlabel="", **kwargs):
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
    im = ax.imshow(data, **kwargs, clim=(-1,1))

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.outline.set_visible(False)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", #45
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.set_xticks(np.arange(data.shape[1]), minor=True)
    ax.set_yticks(np.arange(data.shape[0]), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

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
        
    #threshold1 = im.norm(threshold+0.5)
    #threshold2 = im.norm(threshold-0.5)
        
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
            #kw.update(color=textcolors[int((im.norm(data[i, j]) > threshold1) | (im.norm(data[i, j]) < threshold2))])
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
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
    else:    
        sensors = ['1', '2', '6', '9', '10']
        n = 1
    return sensors, n

def get_df(path_init, data_type, width, correlation_type):
    path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_2\\dataset_'+ data_type + '_' + correlation_type.lower() +'_' + str(width) + '.csv'
    df = pd.read_csv(path, index_col=0)
    columns = df.columns
    return df, columns


def leakage_sizes(path_init):

    width = 40
    correlation_type = 'dcca'
    data_types = ['p', 'q']
    
    for data_type in data_types:
    
        sensors, n = get_sensors_n(data_type)
        df, columns = get_df(path_init, data_type, width, correlation_type)
        
        event_id_init = 1393
        event_id_final = event_id_init+9
        events_id = []
        events_id.extend(list(range(event_id_init-1, event_id_final+1, 2)))
        
        for event_id in events_id: #37-46, 1393-1402
            
            df_row = df.iloc[event_id,:]
            print(df_row)
            correlations = get_correlation_map_simulated(sensors, columns, df_row)
            
            fig, ax = plt.subplots(figsize=(6.4*n,4.8*n)) #23 13
            im, cbar = heatmap(correlations, sensors, sensors, ax=ax,
                               cmap="Blues", cbarlabel="")
            annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.25)
            fig.tight_layout()
            plt.savefig(path_init + '\\Images\\Results1\\Leakage Sizes\\' + data_type + '_' + correlation_type + '_' + str(width) + '_' + str(event_id) + '.png', format='png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig=fig)
    

def window_sizes(path_init):
    
    correlation_type = 'dcca'
    data_types = ['q', 'p']
    events_id = [1393, 1402]
    
    for data_type in data_types:
    
        for event_id in events_id:
        
            sensors, n = get_sensors_n(data_type)
            
            for width in range(15, 41, 5):
                df, columns = get_df(path_init, data_type, width, correlation_type)
                df_row = df.iloc[event_id,:]
                print(df_row)
                correlations = get_correlation_map_simulated(sensors, columns, df_row)
                
                fig, ax = plt.subplots(figsize=(6.4*n,4.8*n)) #23 13
                im, cbar = heatmap(correlations, sensors, sensors, ax=ax,
                                   cmap="Blues", cbarlabel="") #RdBu
                annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.25)
                fig.tight_layout()
                plt.savefig(path_init + '\\Images\\Results1\\Window Sizes\\' + str(event_id) +'\\' + data_type + '_' + correlation_type + '_' + str(width) + '_' + str(event_id) + '.png', format='png', dpi=300, bbox_inches='tight')
                plt.show()
                plt.close(fig=fig)
            


def correlation_methods(path_init):
    
    correlation_types = ['dcca','pearson']
    width = 40
    events_id = [1393, 1402]
    data_types = ['q', 'p']
    
    for correlation_type in correlation_types:
    
        for data_type in data_types:
            
            sensors, n = get_sensors_n(data_type)
            df, columns = get_df(path_init, data_type, width, correlation_type)
        
            for event_id in events_id:
    
                df_row = df.iloc[event_id,:]
                print(df_row)
                correlations = get_correlation_map_simulated(sensors, columns, df_row)
                        
                fig, ax = plt.subplots(figsize=(6.4*n,4.8*n)) #23 13
                im, cbar = heatmap(correlations, sensors, sensors, ax=ax,
                                   cmap="Blues", cbarlabel="") #RdBu
                annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.25)
                fig.tight_layout()
                plt.savefig(path_init + '\\Images\\Results1\\Correlation Methods\\' + data_type + '_' + correlation_type + '_' + str(width) + '_' + str(event_id) + '.png', format='png', dpi=300, bbox_inches='tight')
                plt.show()
                plt.close(fig=fig)

config = Configuration()
path_init = config.path

window_sizes(path_init)
#leakage_sizes(path_init)
#correlation_methods(path_init)





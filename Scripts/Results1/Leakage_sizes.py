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

def get_df(path_init, data_type, width, correlation_type):
    
    if data_type == 'all':
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_' + correlation_type.lower() +'_' + str(width) + '.csv'
    else:
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_2\\dataset_'+ data_type + '_' + correlation_type.lower() +'_' + str(width) + '.csv'
    
    df = pd.read_csv(path, index_col=0)
    columns = df.columns
    return df, columns

def leakage_sizes_hm(path_init):

    width = 40
    correlation_type = 'dcca'
    data_types = ['all']
    
    for data_type in data_types:
    
        sensors, n = get_sensors_n(data_type)
        df, columns = get_df(path_init, data_type, width, correlation_type)
        
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

def leakage_sizes(path_init):

    width = 40
    correlation_type = 'dcca'
    data_types = ['all']
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    
    for data_type in data_types:
    
        sensors, n = get_sensors_n(data_type)
        df, columns = get_df(path_init, data_type, width, correlation_type)
        
        event_id_init = 1393
        event_id_final = event_id_init+9
        events_id = list(range(event_id_init-1, event_id_final+1, 2))
        
        row_names = ['1-4', '22-25', '1-25']
        
        ytexts_array = [[12,12,-18,-18,12,-18],[12,12,-18,-18,12,-18],[15,-16,12,12,-18,12]]
        has_array = [['left','left','right','center','center','right'],
                   ['left','left','right','center','center','right'],
                   ['left','left','right','center','center','right']]
        
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10.5,5), sharey=True, sharex=True)
        i = 0
        for ax in axs.flat:
            
            if i == 0:
                ax.set_ylabel('DCCA (n=1)')
                
            if i == 1:
                ax.set_xlabel('Leakage Coefficient')
        
            row_name = row_names[i]
            ytexts = ytexts_array[i]
            has = has_array[i]
            
            coefficients = ['0.05', '0.1', '0.5', '1.0', '1.5', '2.0']
            df_column = df[row_name]
            
            ax.axhline(y=df_column[event_id_init], color=color1, linestyle='--', linewidth=1.25, label="Without leakage")
            y = []
            for event_id in events_id:
                y.append(df_column[event_id])
                
            ax.plot(coefficients, y, color=color2, marker='o', markersize=4)
            
            for x,y,ytext,ha in zip(coefficients,y, ytexts, has):

                label = "{:.2f}".format(y)
            
                ax.annotate(label, # this is the text
                             (x,y), # this is the point to label
                             textcoords="offset points", # how to position the text
                             xytext=(0,ytext), # distance from text to points (x,y)
                             ha=ha, bbox=dict(facecolor="w",edgecolor=color2,alpha=0.4,boxstyle="round")
                             ) # horizontal alignment can be left, right or center
            
            min_xlim, max_xlim = ax.get_xlim()
            
            ax.grid(True, axis='y', alpha=0.3, which='both')
            
            row_name_split = row_name.split('-')
            title = 'Sensors ' + row_name_split[0] + ' & ' + row_name_split[1]
            
            if i==1:
                ax.text(max_xlim-0.1, df_column[event_id_init]*1.1+0.01,'w/o leakage = {:.2f}'.format(df_column[event_id_init]), ha="right", va="center", color=color1)
                title += '\n(two flowrate sensors)'
            elif i == 0:
                ax.text(max_xlim-0.1, df_column[event_id_init]*1.1-0.015,'w/o leakage = {:.2f}'.format(df_column[event_id_init]), ha="right", va="center", color=color1)
                title += '\n(two pressure sensors)'
            else:
                ax.text(max_xlim-0.1, df_column[event_id_init]*0.92,'w/o leakage = {:.2f}'.format(df_column[event_id_init]), ha="right", va="center", color=color1)
                title += '\n(one sensor of each type)'
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.set_title(title) 
            i += 1
        
        plt.ylim(-1,1)
        
        fig.tight_layout()
        plt.savefig(path_init + '\\Images\\Results1\\Leakage Sizes\\leakage_sizes_lc.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig=fig)           

config = Configuration()
path_init = config.path

leakage_sizes(path_init)
#leakage_sizes_hm(path_init)


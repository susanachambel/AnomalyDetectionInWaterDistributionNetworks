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
from statsmodels.tsa.seasonal import seasonal_decompose

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
    if data_type == 'r':
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_r_' + correlation_type.lower() +'_' + str(width) + '.csv'
        df = pd.read_csv(path, index_col=0)
        df['init'] = pd.to_datetime(df['init'], format='%Y/%m/%d %H:%M:%S')
        df['final'] = pd.to_datetime(df['final'], format='%Y/%m/%d %H:%M:%S')
        df.index = df['init']
        del df['init']
    else:
        path = path_init + '\\Data\\infraquinta\\events\\Organized_Data_4\\dataset_' + correlation_type.lower() +'_' + str(width) + '.csv'
        df = pd.read_csv(path, index_col=0)
    
    columns = df.columns
    return df, columns

def get_df_events(path_init):
    df_events = pd.read_csv(path_init + "\\Data\\events_ruturas_infraquinta_2017.csv", sep=';')
    df_events['date_executed'] = pd.to_datetime(df_events['date_executed'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_detected'] = pd.to_datetime(df_events['date_detected'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_water_closed'] = pd.to_datetime(df_events['date_water_closed'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_water_opened'] = pd.to_datetime(df_events['date_water_opened'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_possible'] = pd.to_datetime(df_events['date_possible'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_start'] = pd.to_datetime(df_events['date_start'], format='%Y/%m/%d %H:%M:%S')
    df_events['date_end'] = pd.to_datetime(df_events['date_end'], format='%Y/%m/%d %H:%M:%S')
    df_events = df_events[df_events['read'] == 'y']
    return df_events

def decompose(df):
    res = seasonal_decompose(df, model='additive', period=4)
    residual = res.resid
    seasonal = res.seasonal 
    trend = res.trend
    return trend

def correlation_evolution_s(path_init):

    width = 40
    correlation_type = 'dcca'
    data_type = 'all'
    
    row_names = ['1-4', '22-25', '1-25']
    
    df, columns = get_df(path_init, data_type, width, correlation_type)
    df1, columns1 = get_df(path_init, data_type, width, 'pearson') 
    
    #print(df[df['y']>0])
        
    event_id_init = 0
    event_id_final = 105
    events_id = list(range(event_id_init, event_id_final+1, 1))
    
    titles = row_names
    
    x = []
    x1 = []
    for event_id in events_id:
        x.append(event_id*600)
        x1.append(event_id*600)
    
    fig, axs = plt.subplots(3, 1, figsize=(11.4, 6.4),sharey=True, sharex=True)
    
    i=0
    for ax in axs.flat:
        
        title = titles[i]
        y = df[title].to_numpy()
        y1 = df1[title].to_numpy()

        ax.plot(x, y,color='tab:blue', label='DCCA (n=1)')
        ax.plot(x1, y1,color='tab:orange', label='PCC')
            
        
        ax.grid(True, axis='y', alpha=0.3,which='both')
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(8*600))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(4*600))
        
        ax.axvspan(48600, 63000, color='tab:red', alpha=0.1, label="Positive Instances (coef=2.0)")
        
        #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        title_split = title.split('-')
        title = 'Sensors ' + title_split[0] + ' & ' + title_split[1]        
        if i == 1:
            ax.set_ylabel('Correlation')
            title += ' (two flowrate sensors)'
            
        elif i == 2:
            ax.set_xlabel('            Time Point', labelpad = 8.5)
            title += ' (one sensor of each type)'
            ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0.007, -0.15))
        else:
            title += ' (two pressure sensors)'
        
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.set_title(title) 
        i+=1
       
    plt.ylim(-1,1)
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results1\\Correlation Evolution\\correlation_evolution_lc.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig=fig)
        
def correlation_evolution_s_ea(path_init):

    width = 40
    correlation_type = 'dcca'
    data_type = 'all'
    
    row_name = '1-25'
    
    df, columns = get_df(path_init, data_type, width, correlation_type)
    df1, columns1 = get_df(path_init, data_type, width, 'pearson') 
    
    #print(df[df['y']>0])
        
    event_id_init = 0
    event_id_final = 105
    events_id = list(range(event_id_init, event_id_final+1, 1))
    
    x = []
    x1 = []
    for event_id in events_id:
        x.append(event_id*600)
        x1.append(event_id*600)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 2.8),sharey=True, sharex=True)
    
    y = df[row_name].to_numpy()
    y1 = df1[row_name].to_numpy()

    ax.plot(x, y,color='tab:blue', label='DCCA (n=1)')
    ax.plot(x1, y1,color='tab:orange', label='PCC')
        
    
    ax.grid(True, axis='y', alpha=0.3,which='both')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8*600))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(4*600))
    
    ax.axvspan(48600, 63000, color='tab:red', alpha=0.1, label="Positive Instances (coef=2.0)")
    
    #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    title_split = row_name.split('-')
    #title = 'Sensors ' + title_split[0] + ' & ' + title_split[1] + ' (one sensor of each type)'      
    #ax.set_title(title, fontsize=14)
    ax.set_ylabel('Correlation', fontsize=14)
    ax.set_xlabel('Time Point', fontsize=14)
    ax.legend(ncol=2, loc='upper left', fontsize=14)
    
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
     
    plt.ylim(-1,1)
    fig.tight_layout()
    plt.savefig(path_init + '\\Images\\Results1\\Correlation Evolution\\correlation_evolution_lc_ea.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig=fig)       

config = Configuration()
path_init = config.path

#correlation_evolution_s(path_init)
correlation_evolution_s_ea(path_init)

"""
color1 = 'tab:blue'
color2 = 'tab:red'


df_events = get_df_events(path_init)

flow = [1, 2, 6, 10, 12, 14] #9
pressure = [3, 7, 8, 11, 13, 15]
sensors = flow
#sensors.extend(pressure)
correlation_type = 'pearson'
combos = list(combinations(sensors, 2))

data_type = 'r'
width = 60
correlation_type = 'dcca'
df, columns = get_df(path_init, data_type, width, correlation_type)


#df_events.iloc[0]
event = df_events.iloc[0,:]

init = event['date_start']
end = event['date_end']

mask = ((df.index >= init) & (df.index <= end))
df = df[mask]

print(df)
locator = mdates.HourLocator(interval=2)
formatter = mdates.ConciseDateFormatter(locator)
locator_min = mdates.HourLocator(interval=2)


m = len(combos)
fig, axs = plt.subplots(m, 1, figsize=(11.4, 3.8*m),sharey=True, sharex=True)

i=0
for ax in axs.flat:
        
    combo = get_combo_name(combos[i])
    df_aux = df.loc[:,combo]
    
    df_aux = decompose(df_aux)
        
    ax.plot(df_aux, color=color1)
    ax.axvline(x=event['date_detected'], color=color2, linestyle='--', linewidth=1.25, label="Report of Leakage")
    
    combo_split = combo.split('-')
    title = 'Sensors ' + combo_split[0] + ' & ' + combo_split[1]
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(locator_min)
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    i+=1

#plt.ylim(-1,1)

fig.tight_layout()
plt.show()
plt.close(fig=fig)

"""



















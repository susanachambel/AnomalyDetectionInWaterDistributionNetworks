# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan


For each event, we save an image with two plots:
    1) The histogram of the correlation differences of the respective month. 
    This plot also shows the mean of the differences.
    2) The plot of the correlation differences from 2 days prior to the event 
    until the water is opened. It also shows when the event was detected and
    when the water was closed (if applied).

"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *
from correlation_analysis import *
from correlation import *
import matplotlib.pyplot as plt
from itertools import combinations
import calendar
import matplotlib.dates as mdates
import geopandas
import numpy as np


config = Configuration() 
path_init = config.path
    
sensors = { 2:'PB2 caudal caixa 1', 6:'RSV R5 Caudal Caixa', 12:'RPR Caudal Pre',}
sensors_aux = {'aMC409150114':2, 'aMC406150114':6, 'aMC404150114':12}
df_events = pd.read_csv(path_init + "\\Data\\events_ruturas_infraquinta_2017_2.csv", sep=';')
df_events = df_events[df_events['read'] == 'y']

df_events['date_detected'] = pd.to_datetime(df_events['date_detected'], format='%Y/%m/%d %H:%M:%S')
df_events['date_water_closed'] = pd.to_datetime(df_events['date_water_closed'], format='%Y/%m/%d %H:%M:%S')
df_events['date_possible'] = pd.to_datetime(df_events['date_possible'], format='%Y/%m/%d %H:%M:%S')


corr_array = ["pearson","kullback-leibler", "dcca"]
corr_array = ["pearson"]
granularity = ['0',1]
chunk_granularity = ['1',1]
k = 2

path_shp = 'zip://C:/Users/susan/Documents/IST/Tese/QGIS/'
tubagens = geopandas.read_file(path_shp + 'shp.zip!shp/tubagens.shp')
medidores_caudal = geopandas.read_file(path_shp + 'shp.zip!shp/medidores_caudal.shp')
medidores_caudal_target = ['aMC404150114','aMC409150114','aMC406150114']
medidores_caudal = medidores_caudal[medidores_caudal.identidade.isin(medidores_caudal_target)]

for index_event, event in df_events.iterrows():
    
    date_detected = event['date_detected']
    date_water_closed = event['date_water_closed'] 
    date_possible = event['date_possible']  
    
    month = event['month']
    
    date_min = '2017-' + str(month) + '-' + '1' + ' 00:00:00'
    date_max = '2017-' + str(month) + '-' + str(calendar.monthrange(2017, month)[1]-1) + ' 23:59:59'
    
    dates, chunk_limits = get_dates_chunk_limits(date_min, date_max, granularity, chunk_granularity)
    dates_event, chunk_limits_event = get_dates_chunk_limits(event['date_start'], event['date_end'], granularity, chunk_granularity)
        
    dfs = {}
    dfs_event = {}
    for sensor in sensors:
        dfs[sensor] = select_data(path_init, "infraquinta", "interpolated", sensor, date_min, date_max)
        dfs_event[sensor] = select_data(path_init, "infraquinta", "interpolated", sensor, event['date_start'], event['date_end'])
            
    combos = combinations(sensors.keys(), 2)
    
    m = 3
    n = 2
    #fig, axs = plt.subplots(m, n, figsize=(8*n,4*m), constrained_layout=True)
    
    widths = [8,8,8]
    heights = [4*3]
    
    fig = plt.figure(figsize=(10*3,6*3))
    gs0 = fig.add_gridspec(ncols=3, nrows=1)
    
    
    gs1 = gs0[0].subgridspec(1, 1)
    ax1 = fig.add_subplot(gs1[0])
    
    if(pd.notna(event['infrastructure'])):
        
        intervencao = tubagens.copy()
        intervencao = intervencao[intervencao['identidade'] == event['infrastructure']]
        
        if (not intervencao.empty):
            intervencao['centroid_column'] = intervencao.centroid
            intervencao = intervencao.set_geometry('centroid_column')
            intervencao_aux = intervencao[:1]
            label = "Interventioned Infrastructure [" + intervencao.iloc[0]['identidade'] + "]"
            intervencao_aux.plot(ax=ax1, label=label, color='red',zorder=2, marker='x')
            
            for x, y, identidade in zip(intervencao_aux.geometry.x, intervencao_aux.geometry.y, intervencao_aux.identidade):
                ax1.annotate(identidade, xy=(x, y), xytext=(3, 3), textcoords="offset points")
        
    medidores_caudal.plot(ax=ax1, label='Flow Sensor', color='purple',zorder=1)
    tubagens.plot(ax=ax1,zorder=0)
    ax1.set(xlabel='Coordinate x', ylabel='Coordinate y', title="Infraquinta's WDN Map")
    
    for x, y, identidade in zip(medidores_caudal.geometry.x, medidores_caudal.geometry.y, medidores_caudal.identidade):
        text = sensors_aux[identidade]
        ax1.annotate(text, xy=(x, y), xytext=(3, 3), textcoords="offset points")
        
    ax1.legend(loc='upper left')
    
    gs2 = gs0[1].subgridspec(3, 1, hspace=.3)
    gs3 = gs0[2].subgridspec(3, 1, hspace=.3)
    
    i = 0 
    for combo in combos:
        
        ax2 = fig.add_subplot(gs2[i])
        ax3 = fig.add_subplot(gs3[i])
        
        results = calculate_correlation_line(dfs[combo[0]], dfs[combo[1]], corr_array, dates, chunk_granularity, k)      
        results_diff = calculate_correlation_diff(chunk_limits, results)
        
        x = results_diff['pearson']['diff'].values
        
        
        bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        n, bins, patches = ax2.hist(x, bins=bin_edges, color='darkturquoise', edgecolor='k')  # arguments are passed to np.histogram
              
        mean = x.mean()
        
        ax2.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        min_ylim, max_ylim = ax2.get_ylim()
        ax2.text(mean*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(mean), bbox=dict(facecolor="w",alpha=0.5))
        
        title = calendar.month_name[month] + "\n[" + str(combo[0]) + "] " + sensors[combo[0]] + " & " + "[" + str(combo[1]) + "] " + sensors[combo[1]]    
        ax2.set(xlabel='Correlation difference [0-2]', ylabel='Number of observations', title=title)
        
        results_event = calculate_correlation_line(dfs_event[combo[0]], dfs_event[combo[1]], corr_array, dates_event, chunk_granularity, k)        
        results_diff_event = calculate_correlation_diff(chunk_limits_event, results_event)
        
        results_diff_event['pearson']['date_init'] = pd.to_datetime(results_diff_event['pearson']['date_init'], format='%Y/%m/%d %H:%M:%S')
        
        x = results_diff_event['pearson']['date_init']
        y = results_diff_event['pearson']['diff']
        
        
        width = np.min(np.diff(mdates.date2num(x)))
        ax3.bar(x, y, width, edgecolor='k')
        
        title = event['date_start'] + " to " + event['date_end']
        ax3.set(xlabel='', ylabel='Correlation difference [0-2]', title=title)
        
        locator = mdates.HourLocator(interval=4)
        formatter = mdates.ConciseDateFormatter(locator)
        locator_min = mdates.HourLocator(interval=1)
        
        ax3.xaxis.set_major_locator(locator)
        ax3.xaxis.set_major_formatter(formatter)   
        ax3.xaxis.set_minor_locator(locator_min)
        
        ax3.axhline(y=mean, color='k', linestyle='dashed', linewidth=1)
        
        min_xlim, max_xlim = ax3.get_xlim()
        ax3.text(0.15, mean*1.2, 'Mean: {:.2f}'.format(mean), ha="right", va="center", 
                 bbox=dict(facecolor="w",alpha=0.5), transform=ax3.get_yaxis_transform())
        
        if not pd.isnull(date_detected):
            label = "Event Detected [" + date_detected.strftime("%H:%M") + "]"
            ax3.axvline(x=date_detected, color='red', linestyle='dashed', linewidth=1, label=label)
            ax3.legend(loc='upper left')
            
        if not pd.isnull(date_water_closed):
            label = "Water Closed [" + date_water_closed.strftime("%H:%M") + "]"
            ax3.axvline(x=date_water_closed, color='darkturquoise', linestyle='dashed', linewidth=1, label=label)
            ax3.legend(loc='upper left')
        #if not pd.isnull(date_possible):
           # ax.axvline(x=date_possible, color='red', linestyle='dashed', linewidth=1)
        
        
        plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')
        
        i += 1
    
    #plt.savefig(path_init + "\\Reports\\Events Correlation Map\\" + str(event['id']) + "_" + event['date_end'].split(" ")[0] + '_event_flow.png', format='png', dpi=300, bbox_inches='tight')    
    plt.show()    
        
        
        
        
        
        
        
        
       
                        
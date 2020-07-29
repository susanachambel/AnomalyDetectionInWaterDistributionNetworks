# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan

@about: For each event, we save an image with the plots of each sensor from 2 
days prior to the event until the water is opened. It also shows when the event 
was detected and when the water was closed (if applied).
    
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


config = Configuration()
path_init = config.path


df_events = pd.read_csv(path_init + "\\Data\\events_ruturas_infraquinta_2017.csv", sep=';')
df_events['date_executed'] = pd.to_datetime(df_events['date_executed'], format='%Y/%m/%d %H:%M:%S')
df_events['date_detected'] = pd.to_datetime(df_events['date_detected'], format='%Y/%m/%d %H:%M:%S')
df_events['date_water_closed'] = pd.to_datetime(df_events['date_water_closed'], format='%Y/%m/%d %H:%M:%S')
df_events['date_water_opened'] = pd.to_datetime(df_events['date_water_opened'], format='%Y/%m/%d %H:%M:%S')
df_events['date_possible'] = pd.to_datetime(df_events['date_possible'], format='%Y/%m/%d %H:%M:%S')
df_events = df_events[df_events['read'] == 'y']

df_sensors = pd.read_csv(path_init + "\\Data\\events_ruturas_infraquinta_2017_sensors.csv", sep=';')
df_sensors = df_sensors[df_sensors['read'] == 'y']

m = len(df_sensors)
n = 1


j = 0
for index_event, event in df_events.iterrows():
    
    fig, axs = plt.subplots(m, n, figsize=(8*n,4*m), constrained_layout=True)
    
    dfs = []
    
    for index_sensor, sensor in df_sensors.iterrows():
        dfs.append(select_data(path_init, "infraquinta", "interpolated", sensor['id'], event['date_start'], event['date_end']))        
    
    ylabel = "Water flow [m3/h]"
    color = 'steelblue'
    
    locator = mdates.HourLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    locator_min = mdates.HourLocator(interval=1)
    
    date_detected = event['date_detected']
    date_water_closed = event['date_water_closed'] 
    date_possible = event['date_possible']  
          
    i = 0
    for df in dfs:
        
        title_aux = df_sensors.iloc[i,:]
        title = "[" + str(title_aux[0]) + "] " + title_aux[1] 
        
        df = df.resample('30min').mean()
        axs[i].plot(df.index,df['value'], color=color)
        axs[i].set(xlabel='', ylabel=ylabel, title=title)
        
        axs[i].xaxis.set_major_locator(locator)
        axs[i].xaxis.set_major_formatter(formatter)
        
        axs[i].xaxis.set_minor_locator(locator_min)
        
        
        if not pd.isnull(date_detected):
            axs[i].axvline(x=date_detected, color='red', linestyle='--')
        
        
        if not pd.isnull(date_water_closed):
            axs[i].axvline(x=date_water_closed, color='darkturquoise', linestyle='--')
        
        if not pd.isnull(date_possible):
            axs[i].axvline(x=date_possible, color='purple', linestyle='--')
        
        
        plt.setp(axs[i].get_xticklabels(), rotation=30, ha='right')
        
        i += 1
         
    
    #plt.savefig(path_init + "\\Reports\\Events\\" + str(j) + "_" + event['date_end'].split(" ")[0] + '_event_flow.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

    j += 1
                        
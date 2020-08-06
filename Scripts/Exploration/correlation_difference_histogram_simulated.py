# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan


For each event, we save an image with 4 columns:
    1) The map of Infraquinta's WDN with the location of the Flow Sensors and 
    of the Interventioned Infrastructure (if applied).
    2) The histogram of the correlation differences of the respective month. 
    This plot also shows the mean of the differences.
    3) The plot of the correlation differences from 2 days prior to the event 
    until the water is opened. It also shows when the event was detected and
    when the water was closed (if applied).
    4) The time series of the two sensors correlated.

"""

import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *
from correlation_analysis import *
from correlation import *

from itertools import combinations
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas
import calendar



"""
    Date format: 2017-04-21 23:59:59
    Returns: df with data from the events
"""    
def event_selection(path_init, wme, sensors_id, event_id, date_min, date_max):
    
    path = path_init + '\\Data\\' + wme + '\\events\\Original\\Rotura_Q\\Rotura_Q_Medidores' + str(event_id) + '.txt'
    
    df = pd.read_csv(path, delimiter='\s+', header=None, names=[9, 6, 12, 1, 10, 14, 2])
    
    df = df.apply(abs)
    
    dates = pd.date_range(date_min, date_max, freq='10min')
    len_dates = len(dates)
    
    df = df[:len_dates]
    df.index = dates
    
    
    return df
    

    
    
config = Configuration() 
path_init = config.path
mydb = config.create_db_connection()
date_min = '2017-06-01 00:00:00'
date_max = '2017-06-02 00:00:00'

#sensors_id = [1, 9, 6, 2, 14, 12, 10]
sensors_id = [1, 2, 6, 9, 10, 12, 14]
sensors_id_bd = [7248, 8375, 6724, 5389, 7251, 7246, 7699]



#sensors = { 8375:'PB2 caudal caixa 1', 6724:'RSV R5 Caudal Caixa', 7246:'RPR Caudal Pre'}
#sensors_aux = {'aMC409150114':8375, 'aMC406150114':6, 'aMC404150114':12}

df_events = event_selection(path_init, 'infraquinta', sensors_id, 150, date_min, date_max)




date_min_aux = '2017-06-01 00:00:00'
date_max_aux = '2017-06-08 00:00:00'

df_sensors = {}
for sensor in sensors_id_bd: 
     df_aux = select_data_db(mydb, 'infraquinta', 'simulated', sensor, date_min_aux, date_max_aux)
     df_sensors[sensor] = df_aux.loc[date_min:date_max]
     



for i in range(0,len(sensors_id)):
    
    locator = mdates.HourLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    locator_min = mdates.HourLocator(interval=1)
    
    sensor_id_aux = sensors_id[i]
    sensor_id_bd_aux = sensors_id_bd[i]
    
    fig = plt.figure(figsize=(8*2,4*1))
    gs0 = fig.add_gridspec(ncols=2, nrows=1)
    
    ax1 = fig.add_subplot(gs0[0])
    ax1.plot(df_sensors[sensor_id_bd_aux].index,df_sensors[sensor_id_bd_aux]['value'], color="purple")       
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_minor_locator(locator_min)
    
    title = sensor_id_aux
    ax1.set(xlabel='', ylabel="Water flow [m3/h]", title=title)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
    
    ax2 = fig.add_subplot(gs0[1])
    ax2.plot(df_events.index,df_events[sensor_id_aux], color="purple")       
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_minor_locator(locator_min)
    
    title = sensor_id_aux
    ax2.set(xlabel='', ylabel="Water flow [m3/h]", title=title)     
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')




"""
    
for k, v in dfs.items():
    
    locator = mdates.HourLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    locator_min = mdates.HourLocator(interval=1)
    
    #print(v)
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.plot(v.index,v['value'], label="hey", color="purple")
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(locator_min)
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.show()


"""

"""
path_import = path_init + "\\Data\\infraquinta\\simulated\\link_flow_summer.csv"
df = pd.read_csv(path_import, sep=",")

columns = list(df.columns)
df1 = pd.DataFrame(columns, columns=['value'])

sensors_1 = ['aTU1096150205','aTU1093150205','2']

df1 = df1[df1['value'].isin(sensors_1)]

print(df1)      
"""

"""      
columns = list(df.columns)
print(columns)
        
columns_len = len(columns)
print(columns_len)
"""


        
        
        
        
        
        
        
        
       
                        
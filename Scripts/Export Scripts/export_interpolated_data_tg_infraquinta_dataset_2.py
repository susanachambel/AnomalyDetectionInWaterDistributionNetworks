# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:54:48 2020

@author: susan

Creates the dataset and files we are going to use to perform event detection
in the real data.

"""

import sys
sys.path.append('../Functions')
from configuration import *
from event_archive_2 import *
from correlation import *
from itertools import combinations
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import calendar


def calculate_correlation(df, sensors, correlation_type, dcca_k):  
    x1 = df.loc[:,sensors[0]].to_numpy()
    x2 = df.loc[:,sensors[1]].to_numpy()
    if correlation_type == "pearson":
        return stats.pearsonr(x1, x2)[0]
    elif correlation_type == "dcca":
        return calculate_dcca_2(x1, x2, dcca_k)

def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])    

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

def get_df_sensors(path_init):
    df_sensors = pd.read_csv(path_init + "\\Data\\events_ruturas_infraquinta_2017_sensors.csv", sep=';')
    df_sensors = df_sensors[df_sensors['read'] == 'y']
    return df_sensors

def export_dataset(path_init):
    
    d1 = ['2017-01-01 00:00:00', '2017-01-30 23:59:59']
    d2 = ['2017-02-01 00:00:00', '2017-02-27 23:59:59']
    d3 = ['2017-03-01 00:00:00', '2017-03-30 23:59:59']
    d4 = ['2017-04-01 00:00:00', '2017-04-29 23:59:59']
    d5 = ['2017-05-01 00:00:00', '2017-05-30 23:59:59']
    d6 = ['2017-06-01 00:00:00', '2017-06-29 23:59:59']
    d7 = ['2017-07-01 00:00:00', '2017-07-30 23:59:59']
    d8 = ['2017-08-01 00:00:00', '2017-08-30 23:59:59']
    d9 = ['2017-09-01 00:00:00', '2017-09-29 23:59:59']
    d10 = ['2017-10-01 00:00:00', '2017-10-30 23:59:59']
    d11 = ['2017-11-01 00:00:00', '2017-11-29 23:59:59']
    d12 = ['2017-12-01 00:00:00', '2017-12-30 23:59:59']
    dates = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12]
    
    flow = [1, 2, 6, 9, 10, 12, 14]
    pressure = [3, 7, 8, 11, 13, 15]
    sensors = flow
    sensors.extend(pressure)
    correlation_type = 'dcca'
    combos = list(combinations(sensors, 2))
    widths = [60]#[30, 45, 60, 75, 90, 105, 120, 180, 240, 300]
    
    for width in widths:
        
        for dcca_k in range(4,11,2):
    
            df_corr = pd.DataFrame()
            for date in dates:
            
                df = pd.DataFrame()
                for sensor_id in sensors:
                    df = pd.concat([df,select_data(path_init, "infraquinta", "interpolated", str(sensor_id), date[0], date[1]).rename(columns={'value':sensor_id})], axis=1)
            
                init = 0
                final = init + width
                len_df = len(df)
                while (final < len_df):
                    chunk = df.iloc[init:final, :]
                    results = {}
                    init_date = df.index[init]
                    results['init'] = init_date
                    results['final'] = df.index[final]
                    for combo in combos:
                        corr = calculate_correlation(chunk, combo, correlation_type, dcca_k)
                        results[get_combo_name(combo)] = abs(corr)
                    
                    df_corr = df_corr.append(results, ignore_index=True)
                    print(2*'\x1b[2K\r' + "Progress " + str(init_date), flush=True, end="\r")
                    init += 15
                    final = init + width
        
            print(df_corr)
            path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data_3\\dataset_r_' + correlation_type + '_' + str(width) + '_' + str(dcca_k) +  '.csv'
            df_corr.to_csv(index=True, path_or_buf=path_export)
    

config = Configuration()
path_init = config.path

df_events = get_df_events(path_init)
df_sensors = get_df_sensors(path_init)

export_dataset(path_init)
       
        
        
#print(df_events)
#mask = (df.index >= str(event['date_start'])) & (df_corr.index < event['date_end'])
#df_corr[mask] = np.NaN












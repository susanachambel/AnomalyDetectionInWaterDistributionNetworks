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
from scipy import stats
import pandas as pd
import numpy as np
import calendar


def calculate_correlation(df, sensors, correlation_type):  
    x1 = df.loc[:,sensors[0]].to_numpy()
    x2 = df.loc[:,sensors[1]].to_numpy()
    if correlation_type == "pearson":
        return stats.pearsonr(x1, x2)[0]
    elif correlation_type == "dcca":
        return calculate_dcca_2(x1, x2, 2)

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

def get_dataset_wo_event(path_init, df_events, df_sensors, correlation_type):

    date_init = '2017-01-01 00:00:00'
    date_final = '2017-12-30 23:59:59'
    
    dfs = pd.DataFrame()
    
    for index_sensor, sensor in df_sensors.iterrows():
        dfs = pd.concat([dfs,select_data(path_init, "infraquinta", "interpolated", str(sensor['id']), date_init, date_final).rename(columns={'value':sensor['id']})], axis=1)
    
    mask = df_sensors['type'] == 'f'
    df_flow = df_sensors[mask]
    df_pressure = df_sensors[~mask]
    combos_flow = list(combinations(df_flow.loc[:,'id'].to_numpy(), 2))
    combos_pressure = list(combinations(df_pressure.loc[:,'id'].to_numpy(), 2))
    
    df_corr = pd.DataFrame()
    
    for chunk_limit,chunk in dfs.resample('1h'):
        results = {}
        for combos in [combos_flow, combos_pressure]:
            for combo in combos:
                if(len(chunk) > 0):
                    results[get_combo_name(combo)] = calculate_correlation(chunk, combo, correlation_type)
                else:
                    results[get_combo_name(combo)] = np.NaN
        
        
        results['date'] = chunk_limit
        df_corr = df_corr.append(results, ignore_index=True)
    
    df_corr = df_corr.set_index('date')
    
    for index_event, event in df_events.iterrows():
        mask = (df_corr.index >= event['date_start']) & (df_corr.index < event['date_end'])
        df_corr[mask] = np.NaN
    
    df_corr_diff = abs(df_corr.diff()).reset_index(drop=True)
    #print(df_corr_diff.isnull().sum().sort_values(ascending = False))
    #df_corr_diff = df_corr_diff.dropna()
    df_corr_diff["y"] = 0

    return df_corr_diff

def get_dataset_w_event(path_init, df_events, df_sensors, correlation_type):
    
    mask = df_sensors['type'] == 'f'
    df_flow = df_sensors[mask]
    df_pressure = df_sensors[~mask]
    combos_flow = list(combinations(df_flow.loc[:,'id'].to_numpy(), 2))
    combos_pressure = list(combinations(df_pressure.loc[:,'id'].to_numpy(), 2))

    df_corr_diff = pd.DataFrame()
    
    for index_event, event in df_events.iterrows():
        
        #date3 = event['date_detected']
        date3 = event['date_water_closed']
        
        if(isinstance(date3, pd.Timestamp)):
        
            date2 = date3 - timedelta(hours=1, minutes=0)
            date1 = date2 - timedelta(hours=1, minutes=0)
            
            #date2 = date3 - timedelta(hours=0, minutes=30)
            #date3 = date3 + timedelta(hours=0, minutes=30)
            #date1 = date2 - timedelta(hours=1, minutes=0)            
            
            dfs = pd.DataFrame()
            
            for index_sensor, sensor in df_sensors.iterrows():
                dfs = pd.concat([dfs,select_data(path_init, "infraquinta", "interpolated", str(sensor['id']), date1, date3).rename(columns={'value':sensor['id']})], axis=1)
            
            dfs = dfs.iloc[1:, :]
            
            if(not dfs.empty):
                dfs_len= int(len(dfs)/2)
                dfs1 = dfs.iloc[:dfs_len,:]
                dfs2 = dfs.iloc[dfs_len:,:]
                results = {}
                for combos in [combos_flow, combos_pressure]:
                    for combo in combos:
                        corr1 = calculate_correlation(dfs1, combo, correlation_type)
                        corr2 = calculate_correlation(dfs2, combo, correlation_type)
                        results[get_combo_name(combo)] = abs(corr1-corr2)
            
                #results['date'] = date3
                df_corr_diff = df_corr_diff.append(results, ignore_index=True)
        
    df_corr_diff["y"] = 1
    return df_corr_diff

config = Configuration()
path_init = config.path

df_events = get_df_events(path_init)
df_sensors = get_df_sensors(path_init)

for correlation_type in ['dcca', 'pearson']:

    df_corr_diff1 = get_dataset_wo_event(path_init, df_events, df_sensors, correlation_type)
    df_corr_diff2 = get_dataset_w_event(path_init, df_events, df_sensors, correlation_type)
    df_corr_diff = df_corr_diff1.append(df_corr_diff2, ignore_index=True)
    df_corr_diff = df_corr_diff.iloc[1:,:]
    
    print(df_corr_diff)
    
    path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_r_' + correlation_type + '.csv'
    #df_corr_diff.to_csv(index=True, path_or_buf=path_export)


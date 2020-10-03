# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:54:48 2020

@author: susan

Creates the dataset and files we are going to use to perform event detection
in the simulated event data.

"""

import sys
sys.path.append('../Functions')
from configuration import *
from event_archive_2 import *
from correlation import *
from itertools import combinations, product
from scipy import stats
import pandas as pd

def get_data_with_event(df, event_info, width):
    time_init = event_info.time_init
    width_aux = width/2
    
    middle_point = (time_init/600) - width_aux + 1
    final_point = (time_init/600) + width_aux + 1
    init_point = middle_point - width_aux*2
    
    if(init_point < 0 or init_point < 0):
        time_final = event_info.time_final
        init_point = (time_final/600) - width_aux + 1
        middle_point = (time_final/600) + width_aux + 1
        final_point = middle_point + width_aux*2
        
    if (final_point > 144 or init_point < 0):
        return None, None
    else:
        df1 = df.iloc[int(init_point):int(middle_point),:]
        df2 = df.iloc[int(middle_point):int(final_point),:]
        
        return df1, df2

def get_data_without_event(df, event_info, width):
    time_init = event_info.time_init
    width_aux = width/2
    
    final_point = (time_init/600)
    middle_point = final_point - width_aux*2
    init_point = middle_point - width_aux*2
    
    if(init_point < 0):
        time_final = event_info.time_final
        init_point = (time_final/600) + 1
        middle_point = init_point + width_aux*2
        final_point = middle_point + width_aux*2
        
    if (final_point > 144 or init_point < 0):
        return None, None
    else:
        df1 = df.iloc[int(init_point):int(middle_point),:]
        df2 = df.iloc[int(middle_point):int(final_point),:]
        return df1, df2


def get_data_with_event_2(df, event_info, width):
    time_init = event_info.time_init
    width_aux = width/2
    
    init_point = (time_init/600) - width_aux + 1
    final_point = (time_init/600) + width_aux + 1
    
    if(init_point < 0 or init_point < 0):
        time_final = event_info.time_final
        init_point = (time_final/600) - width_aux + 1
        final_point = (time_final/600) + width_aux + 1
        
    if (final_point > 144 or init_point < 0):
        return None, None
    else:
        df1 = df.iloc[int(init_point):int(final_point),:]
        return df1

def get_data_without_event_2(df, event_info, width):
    time_init = event_info.time_init
    width_aux = width/2
    
    final_point = (time_init/600)
    init_point = final_point - width_aux*2
    
    if(init_point < 0):
        time_final = event_info.time_final
        init_point = (time_final/600) + 1
        final_point = init_point + width_aux*2
        
    if (final_point > 144 or init_point < 0):
        return None, None
    else:
        df1 = df.iloc[int(init_point):int(final_point),:]
        return df1


def get_combo_name(combo):
    return str(combo[0]) + "-" + str(combo[1])

def calculate_correlation_difference(df1, df2, sensors, correlation_type):  
    
    x11 = df1.loc[:,sensors[0]].to_numpy()
    x12 = df1.loc[:,sensors[1]].to_numpy()    
    x21 = df2.loc[:,sensors[0]].to_numpy() 
    x22 = df2.loc[:,sensors[1]].to_numpy()
    
    corr1 = 0
    corr2 = 0
    
    if correlation_type == "pearson":
        corr1 = stats.pearsonr(x11, x12)[0]
        corr2 = stats.pearsonr(x21, x22)[0]
    elif correlation_type == "dcca":
        corr1 = calculate_dcca_2(x11, x12, 2)
        corr2 = calculate_dcca_2(x21, x22, 2)
    
    return abs(corr1-corr2)

def update_df_diff(ea, df1, df2, df_diff2, combos, correlation_type, option):
    for combo in combos:
        sensor1 = combo[0]
        sensor2 = combo[1]
        diff = calculate_correlation_difference(df1, df2, [sensor1, sensor2], correlation_type)
        df_diff[get_combo_name(combo)].append(diff)
        
    if(option == "with"):
        y = transform_y(ea.get_event_info(event_id).c)
        df_diff['y'].append(y)
    else:
        df_diff['y'].append(0)
        
    return df_diff

def update_df_diff_2(ea, df1, df_diff, combos, correlation_type, option):
    for combo in combos:
        sensor1 = combo[0]
        sensor2 = combo[1]
        diff = calculate_correlation_difference_2(df1, [sensor1, sensor2], correlation_type)
        df_diff[get_combo_name(combo)].append(diff)
        
    if(option == "with"):
        y = transform_y(ea.get_event_info(event_id).c)
        df_diff['y'].append(y)
    else:
        df_diff['y'].append(0)
        
    return df_diff

def calculate_correlation_difference_2(df1, sensors, correlation_type):
    x1 = df1.loc[:,sensors[0]].to_numpy()
    x2 = df1.loc[:,sensors[1]].to_numpy()    
    corr = 0
    if correlation_type == "pearson":
        corr = stats.pearsonr(x1, x2)[0]
    elif correlation_type == "dcca":
        corr = calculate_dcca_2(x1, x2, 2)
    return corr



def transform_y(c):
    if c == 0.05:
        return 1
    elif c == 0.1:
        return 2
    elif c == 0.5:
        return 3
    elif c == 1.0:
        return 4
    elif c == 1.5:
        return 5
    elif c == 2:
        return 6  
        
config = Configuration()
path_init = config.path
data_type = "p"
ea = EventArchive(path_init, data_type)
correlation_types = ["dcca","pearson"]

event_range_min = 1
event_range_max = 18696

sensors = []
if(data_type == "p"):
    sensors_aux = list(range(1,21, 4))
    for sensor in sensors_aux:
        sensors.append(str(sensor))
else:
    sensors = ['1', '2', '6', '9', '10']

combos = list(combinations(sensors, 2))

"""
for correlation_type in correlation_types:

    for width in range(40, 41, 1):
        
        df_diff = {}
        for combo in combos:
            df_diff[get_combo_name(combo)] = []
        df_diff['y'] = []
        
        for event_id in range(event_range_min, event_range_max+1):
        
            df11, df12 = get_data_with_event(ea.get_event(event_id), ea.get_event_info(event_id), width)
            if(df11 is not None):
                df_diff = update_df_diff(ea, df11, df12, df_diff, combos, correlation_type, "with")
            
            df21, df22 = get_data_without_event(ea.get_event(event_id), ea.get_event_info(event_id), width)
            if(df21 is not None):
                df_diff = update_df_diff(ea, df21, df22, df_diff, combos, correlation_type, "without")
            
            print(2*'\x1b[2K\r' + "Progress " + str(event_id) + "/" + str(event_range_max), flush=True, end="\r")
            
        df_diff = pd.DataFrame(df_diff)
        print(df_diff)
        path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\dataset_' + data_type + '_' + correlation_type + '_' + str(width) + '.csv'
        df_diff.to_csv(index=True, path_or_buf=path_export)
"""


for correlation_type in correlation_types:

    for width in range(40, 41, 1):
        
        df_diff = {}
        for combo in combos:
            df_diff[get_combo_name(combo)] = []
        df_diff['y'] = []
        
        for event_id in range(event_range_min, event_range_max+1):
        
            df1 = get_data_with_event_2(ea.get_event(event_id), ea.get_event_info(event_id), width)
            if(df1 is not None):
                df_diff = update_df_diff_2(ea, df1, df_diff, combos, correlation_type, "with")
            
            df2 = get_data_without_event_2(ea.get_event(event_id), ea.get_event_info(event_id), width)
            if(df2 is not None):
                df_diff = update_df_diff_2(ea, df2, df_diff, combos, correlation_type, "without")
            
            print(2*'\x1b[2K\r' + "Progress " + str(event_id) + "/" + str(event_range_max), flush=True, end="\r")
            
        df_diff = pd.DataFrame(df_diff)
        print(df_diff)
        path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data_2\\dataset_' + data_type + '_' + correlation_type + '_' + str(width) + '.csv'
        df_diff.to_csv(index=True, path_or_buf=path_export)


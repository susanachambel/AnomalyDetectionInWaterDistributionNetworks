# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 20:54:48 2020

@author: susan
"""

import sys
sys.path.append('../Functions')
from configuration import *
from event_archive import *
from itertools import combinations
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


def get_data_with_event(df, event_info):
    
    time_init = event_info.time_init
    width = 20 # talvez 17 seja mais seguro
    
    middle_point = (time_init/600) - width
    final_point = (time_init/600) + width
    init_point = middle_point - width*2
    
    if(init_point < 0 or init_point < 0):
        time_final = event_info.time_final
        init_point = (time_final/600) - width
        middle_point = (time_final/600) + width
        final_point = middle_point + width*2
        
    if (final_point > 144 or init_point < 0):
        return None, None
    else:
        df1 = df.iloc[int(init_point):int(middle_point),:]
        df2 = df.iloc[int(middle_point):int(final_point),:]       
        return df1, df2
    

def get_data_with_event_2(df, event_info):
    
    time_init = event_info.time_init
    width = 20 # idealmente 12
    
    middle_point = (time_init/600)
    init_point = middle_point - width*2
    final_point = middle_point + width*2
    
    if(init_point < 0):
        time_final = event_info.time_final
        middle_point = (time_final/600)
        init_point = middle_point - width*2
        final_point = middle_point + width*2
        
    if (final_point > 144 or init_point < 0):  
        return None, None
    else:
        
        df1 = df.iloc[int(init_point):int(middle_point),:]
        df2 = df.iloc[int(middle_point):int(final_point),:]       
        return df1, df2
    

def get_data_without_event(df, event_info):
    
    time_init = event_info.time_init
    width = 20 # talvez 17 seja mais seguro
    
    final_point = (time_init/600)
    middle_point = final_point - width*2
    init_point = middle_point - width*2
    
    if(init_point < 0):
        time_final = event_info.time_final
        init_point = (time_final/600) + 1
        middle_point = init_point + width*2
        final_point = middle_point + width*2
        
    if (final_point > 144 or init_point < 0):
        return None, None
    else:
        df1 = df.iloc[int(init_point):int(middle_point),:]
        df2 = df.iloc[int(middle_point):int(final_point),:]       
        return df1, df2

def calculate_correlation_difference(df1, df2, sensors):    
    x11 = df1.loc[:,sensors[0]].to_numpy()
    x12 = df1.loc[:,sensors[1]].to_numpy()    
    x21 = df2.loc[:,sensors[0]].to_numpy() 
    x22 = df2.loc[:,sensors[1]].to_numpy()    
    corr1 = stats.pearsonr(x11, x12)[0]
    corr2 = stats.pearsonr(x21, x22)[0]    
    return abs(corr1-corr2)

def plot_sensors(df1, df2, sensor1, sensor2):
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.plot(df1.index,df1.loc[:,sensor1], color='orange')
    ax.plot(df2.index, df2.loc[:,sensor1], color='red')
    ax.plot(df1.index,df1.loc[:,sensor2])
    ax.plot(df2.index, df2.loc[:,sensor2], color='purple')
    plt.show()       

def plot_histogram(x, combo):
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    n, bins, patches = ax.hist(x, bins=bin_edges, color='darkturquoise', edgecolor='k')
    
    mean = np.mean(x)
        
    ax.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    ax.text(mean+0.05, max_ylim*0.9, 'Mean: {:.3f}'.format(mean), bbox=dict(facecolor="w",alpha=0.5,boxstyle="round"))
    ax.set(xticks=bin_edges)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    title = "[ " + str(combo[0]) + " & " + str(combo[1]) + " ] Correlation Difference Histogram"
    ax.set(xlabel='Correlation difference [0-2]', ylabel='Number of observations', title=title)
    
    plt.show()


# Melhorar a performance juntando os dois executes  
def execute_difference_with_event(combos, with_event, event_archive):
    df_diff = {}
    for combo in combos:
        df_diff[combo] = []
    
    event_range_min = 300 #12 ou 30
    event_range_max = 400
    
    events_id = list(range(event_range_min,event_range_max+1))
    
    print(2*'\x1b[2K\r' + "Progress " + str(event_range_min-1) + "/" + str(event_range_max), flush=True, end="\r")
    for event_id in events_id: 
        
        df = event_archive.get_event(event_id)
        event_info = event_archive.get_event_info(event_id)
        
        df1 = None
        df2 = None
        
        if(with_event == 1):
            df1, df2 = get_data_with_event(df, event_info)
        else:
            df1, df2 = get_data_without_event(df, event_info)
        
        if(df1 is not None):
     
            for combo in combos:
                sensor1 = combo[0]
                sensor2 = combo[1]
                diff = calculate_correlation_difference(df1, df2, [sensor1, sensor2])
                df_diff[combo].append(diff)
                #plot_sensors(df1, df2, sensor1, sensor2)
                  
        print(2*'\x1b[2K\r' + "Progress " + str(event_id) + "/" + str(event_range_max), flush=True, end="\r")
    
    print(2*'\x1b[2K\r' + "Completed " + str(event_id) + "/" + str(event_range_max), flush=True)
    
    return df_diff

def print_mean_differences(combos, df_diff):
    print("Mean differences:")
    for combo in combos:
        #plot_histogram(df_diff[combo], combo)
        mean = round(np.mean(df_diff[combo]),3)
        print( "(" + str(combo[0]) + "," + str(combo[1]) + ") = "+ str(mean))       
    print("#Events = " + str(len(df_diff[combos[0]]))) 


config = Configuration()
path_init = config.path
event_archive = EventArchive(path_init, 0)
   
sensors = ['1', '2', '6', '9', '10']
combos = list(combinations(sensors, 2))

df_diff1 = execute_difference_with_event(combos, 1, event_archive)
df_diff2 = execute_difference_with_event(combos, 0, event_archive)

print_mean_differences(combos, df_diff1)
print_mean_differences(combos, df_diff2)
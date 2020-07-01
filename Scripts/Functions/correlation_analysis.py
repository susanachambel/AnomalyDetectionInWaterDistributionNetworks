# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:56:59 2020

@author: susan
"""

import pandas as pd
import numpy as np

from configuration import *
from data_selection import *
from correlation import *
from itertools import combinations
import matplotlib.pyplot as plt

def replace_with_nan_none(results, action):   
    if(action == "nan"):
        for key in results:
            result = [x if x!=999999999 else np.nan for x in results[key]]
            results[key] = result
    else:
        for key in results:
            result = [x for x in results[key] if x!=999999999]
            results[key] = result      
    return results

def difference_correlation(results):
    
    results = replace_with_nan_none(results, "none")
    
    results_diff = {}
    
    for key in results:
        
        results_diff[key] = {}
        data = [round(abs(t - s),4) for s, t in zip(results[key], results[key][1:])]        
        results_diff[key]['data'] = data
    
    return results_diff

def calculate_correlation_diff(chunk_limits, results):   
    results_diff = {}
    for key in results:
        df = pd.DataFrame({"date_init": chunk_limits, "value": results[key]})    
        df = df[df.value != 999999999]    
        df["date_final"] = df["date_init"].shift(-1)
        df["diff"] = abs(df["value"].diff(-1))
        del df['value']
        df = df[:-1]
        results_diff[key] = df
        
    return results_diff

def get_mean_std(results_diff):
    results_diff_mean_std = {}
    for key in results_diff:
        results_diff_mean_std[key] = {}
        df = results_diff[key]
        results_diff_mean_std[key]['mean'] = round(df.loc[:,'diff'].mean(),4)
        results_diff_mean_std[key]['std'] = round(df.loc[:,'diff'].std(),4)
    
    return results_diff_mean_std


def get_outliers_iqr(results_diff):   
    results_diff_outliers = {}
    for key in results_diff:
        df = results_diff[key].copy()
        values = df.loc[:,'diff'].to_numpy()
        quartile_1, quartile_3 = np.percentile(values, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)      
        results_diff_outliers[key] = df[(df['diff'] > upper_bound) | (df['diff'] < lower_bound)]        
    return results_diff_outliers

def get_outliers_z_score(results_diff):
    threshold = 3
    results_diff_outliers = {}
    for key in results_diff:
        df = results_diff[key].copy()
        values = df.loc[:,'diff'].to_numpy()
        mean = np.mean(values)
        std = np.std(values)       
        df["z_score"] = abs((df["diff"]-mean)/std)  
        df_aux = df[(df['z_score'] > threshold)]
        #del df_aux["z_score"]
        results_diff_outliers[key] = df_aux    
    return results_diff_outliers
    
def get_outliers_z_score_modified(results_diff):
    threshold = 3.5
    results_diff_outliers = {}
    for key in results_diff:
        df = results_diff[key].copy()
        values = df.loc[:,'diff'].to_numpy()
        mean = np.mean(values)
        std = np.std(values)        
        median = np.median(values)
        median_absolute_deviation = np.median([np.abs(value - median) for value in values])        
        df["z_score"] = abs((df["diff"]-mean)/std)        
        df["z_score"] = abs((0.6745 * (df["diff"] - median)) / median_absolute_deviation)
        df_aux = df[(df['z_score'] > threshold)]
        #del df_aux["z_score"]
        results_diff_outliers[key] = df_aux 
    return results_diff_outliers

def test_correlation_line_diff_2():
    
    config = Configuration() 
    path_init = config.path
    
    sensors = [1,2,6,9,10,12,14]
    date_min = '2017-02-07 00:00:00'
    date_max = '2017-02-07 23:59:59'
    corr_array = ["pearson","kullback-leibler", "dcca"]
    corr_array = ["pearson"]
    granularity = ['0',1]
    chunk_granularity = ['1',1]
    k = 2
    
    dates, chunk_limits = get_dates_chunk_limits(date_min, date_max, granularity, chunk_granularity)
    
    dfs = {}
    for sensor in sensors:
        dfs[sensor] = select_data(path_init, "infraquinta", "interpolated", sensor, date_min, date_max)
        
    combos = combinations(sensors, 2)
    
    final_results = []
    
    for combo in combos:
            
        results = calculate_correlation_line(dfs[combo[0]], dfs[combo[1]], corr_array, dates, chunk_granularity, k)      
        results_diff = calculate_correlation_diff(chunk_limits, results)
        
        outliers = get_outliers_iqr(results_diff)
                 
        #final_results.append({'sensor1':combo[0], 'sensor2':combo[1], 'corr':outliers})
        
        #print(str(combo[0]) + " --> " + str(combo[1]))
        
        print(results_diff)
        #print(outliers['pearson'])
        print("")
        
    

def test_correlation_line_diff():
    config = Configuration() 
    path_init = config.path
        
    date_min = '2017-01-01 00:00:00'
    date_max = '2017-01-31 23:59:59'  
    corr_array = ["pearson","kullback-leibler", "dcca"]
    corr_array = ["pearson"]
    granularity = ['0',1]
    chunk_granularity = ['1',1]
    k = 2
        
    df1 = select_data(path_init, "infraquinta", "interpolated", 1, date_min, date_max)
    df2 = select_data(path_init, "infraquinta", "interpolated", 6, date_min, date_max)
     
    dates, chunk_limits = get_dates_chunk_limits(date_min, date_max, granularity, chunk_granularity)   
    results = calculate_correlation_line(df1, df2, corr_array, dates, chunk_granularity, k)
      
    results_diff = calculate_correlation_diff(chunk_limits, results)
    #results_diff_mean_std = get_mean_std(results_diff)
    
    print(get_outliers_iqr(results_diff))
    print(get_outliers_z_score(results_diff))
    print(get_outliers_z_score_modified(results_diff))
    
    #print(results_diff_iqr['pearson'])       
    #print(results_diff["pearson"])
    

    
#test_correlation_line_diff()
test_correlation_line_diff_2()
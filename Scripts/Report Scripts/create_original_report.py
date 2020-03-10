# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:48:49 2020

@author: susan

"""

import mysql.connector
import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *

root = read_config()
path_init = get_path(root)
wmes = get_wmes_sensors(root)


print("\nReport initiated\n")


path_export = path_init + "\\Reports\\original_data_report.csv"


df = pd.DataFrame()

for wme_tmp in wmes:

    sensor_count = wme_tmp[1]
    wme = wme_tmp[0]
    
    print(wme + ": " + str(sensor_count) + " sensors found")
    
    sensor_id = 1
    
    for sensor_id in range(1,sensor_count+1):
        
        df_sensor = select_data(path_init, wme, "real", sensor_id, 1, 1)
        df_description = df_sensor.describe()
        
        neg_values = len(df_sensor[df_sensor['value'] < 0].index)   
        max_date = df_sensor.index.max()
        min_date = df_sensor.index.min()
        n_rows = len(df_sensor.index)
        mean = df_description.loc['mean',:].value
        std = df_description.loc['std',:].value
        min = df_description.loc['min',:].value
        p_25 = df_description.loc['25%',:].value
        p_50 =  df_description.loc['50%',:].value
        p_75 =  df_description.loc['75%',:].value
        max =  df_description.loc['max',:].value
        
        
        df = df.append({'wme': wme, 'id': sensor_id,'n_rows': n_rows, 
                        'min_date': min_date, 'max_date': max_date,
                        'mean': mean, 'std': std, 'min': min,
                        '25%': p_25, '50%': p_50, '75%': p_75,
                        'max': max, 'neg_values': neg_values}, ignore_index=True)

              
df.to_csv(index=False, path_or_buf=path_export)
print("\nReport completed")  
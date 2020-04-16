# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:09:00 2020

@author: susan
"""

"""
    Please be aware that the data will contain gaps if the months are not complete.
    If its only a few time points, please garantee that the date_range in 
    create_interpolated_df(df) is equal to the one given when selecting the data.
"""

import sys
sys.path.append('../Functions')
from data_selection import *
from data_interpolation import *
from configuration import *
import pandas as pd
import numpy as np

config = Configuration()
path_init = config.path

wme = ["infraquinta",15]
  
print(wme[0] + ": " + str(wme[1]) + " sensors found")
     
sensor_id = 1
for sensor in range(sensor_id, wme[1]+1):
        
    print(" -> Sensor " + str(sensor_id), end="")
    
    dr_1 = ['2017-01-01 00:00:00', '2017-01-30 23:59:59']
    dr_2 = ['2017-02-01 00:00:00', '2017-02-27 23:59:59']
    dr_3 = ['2017-03-01 00:00:00', '2017-03-30 23:59:59']
    dr_4 = ['2017-04-01 00:00:00', '2017-04-29 23:59:59']
    dr_5 = ['2017-05-01 00:00:00', '2017-05-30 23:59:59']
    dr_6 = ['2017-06-01 00:00:00', '2017-06-29 23:59:59']
    dr_7 = ['2017-07-01 00:00:00', '2017-07-30 23:59:59']
    dr_8 = ['2017-08-01 00:00:00', '2017-08-30 23:59:59']
    dr_9 = ['2017-09-01 00:00:00', '2017-09-29 23:59:59']
    dr_10 = ['2017-10-01 00:00:00', '2017-10-30 23:59:59']
    dr_11 = ['2017-11-01 00:00:00', '2017-11-29 23:59:59']
    dr_12 = ['2017-12-01 00:00:00', '2017-12-30 23:59:59']
    
    drs = [dr_1,dr_2,dr_3,dr_4,dr_5,dr_6,dr_7,dr_8,dr_9,dr_10,dr_11,dr_12]
    
    dfs_drs = []
    
    for dr in drs:
        df_aux = select_data(path_init, wme[0], "real", sensor_id, dr[0], dr[1])
        dfs_drs.append([df_aux,dr[0],dr[1]])
          
    df_total = pd.DataFrame()
    df_id = 1
        
    for df_dr in dfs_drs:
        df = df_dr[0]
        
        df[df < 0] = np.nan
        dates = pd.date_range(df_dr[1], df_dr[2], freq='1min')
        df_aux = pd.DataFrame(np.zeros((len(dates), 1)))
        df_aux[0] = np.nan
        df_aux.columns = ['value']
        df_aux.index = dates
        df = df.combine_first(df_aux)
        df = df.interpolate(method='time', limit_direction='both')
        flagged = df[df.index.second != 0].index
        df = df.drop(flagged)
        
           
        df_total = df_total.append(df)
        print(2*'\x1b[2K\r' + " -> Sensor " + str(sensor_id) + ": " + str(df_id) + "/12" , flush=True, end="\r")
        df_id += 1
            
    path = path_init + "\\Data\\" + wme[0] + "\\interpolated\\sensor_" + str(sensor_id) + ".csv"
    df_total.index.name = "date"
    df_total.to_csv(index=True, path_or_buf=path)
        
    print(2*'\x1b[2K\r' + " -> Sensor " + str(sensor_id) + ": " + str(df_total.shape[0]) + " rows")
        
    sensor_id += 1
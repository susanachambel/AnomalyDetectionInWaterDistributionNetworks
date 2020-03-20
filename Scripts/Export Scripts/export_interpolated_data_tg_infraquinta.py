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

config = Configuration()
path_init = config.path

wme = ["infraquinta",15]
  
print(wme[0] + ": " + str(wme[1]) + " sensors found")
     
sensor_id = 5
for sensor in range(sensor_id, wme[1]+1):
        
    print(" -> Sensor " + str(sensor_id))
    
    dr_1 = ['2017-01-01 00:00:00', '2017-01-31 23:59:59']
    dr_2 = ['2017-02-01 00:00:00', '2017-02-28 23:59:59']
    dr_3 = ['2017-03-01 00:00:00', '2017-03-31 23:59:59']
    dr_4 = ['2017-04-01 00:00:00', '2017-04-30 23:59:59']
    dr_5 = ['2017-05-01 00:00:00', '2017-05-31 23:59:59']
    dr_6 = ['2017-06-01 00:00:00', '2017-06-30 23:59:59']
    dr_7 = ['2017-07-01 00:00:00', '2017-07-31 23:59:59']
    dr_8 = ['2017-08-01 00:00:00', '2017-08-31 23:59:59']
    dr_9 = ['2017-09-01 00:00:00', '2017-09-30 23:59:59']
    dr_10 = ['2017-10-01 00:00:00', '2017-10-31 23:59:59']
    dr_11 = ['2017-11-01 00:00:00', '2017-11-30 23:59:59']
    dr_12 = ['2017-12-01 00:00:00', '2017-12-31 23:59:59']
    
    drs = [dr_1,dr_2,dr_3,dr_4,dr_5,dr_6,dr_7,dr_8,dr_9,dr_10,dr_11,dr_12]
    
    dfs_drs = []
    
    for dr in drs:
        df_aux = select_data(path_init, wme[0], "real", sensor_id, dr[0], dr[1])
        dfs_drs.append([df_aux,dr[0],dr[1]])
          
    df_total = pd.DataFrame()
    df_id = 1
        
    for df_dr in dfs_drs:
        df = df_dr[0]
        print("  -> DF_" + str(df_id) + ": " + str(df.shape[0]) + " rows")
        #df = create_interpolated_df(df, df_dr[1], df_dr[2]) # use this for complete data
        df = create_interpolated_df(df, 1, 1)   
        df_id += 1   
        df_total = df_total.append(df)
        print("  Result: " + str(df.shape[0]) + " rows")
            
    path = path_init + "\\Data\\" + wme[0] + "\\interpolated\\sensor_" + str(sensor_id) + ".csv"
    df_total.to_csv(index=True, path_or_buf=path)
        
    print("  Result: " + str(df_total.shape[0]) + " rows")
        
    sensor_id += 1
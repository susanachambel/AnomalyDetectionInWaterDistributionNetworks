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
from event_archive import *
import pandas as pd

def get_data_with_event(df, event_info):
    
    time_init = event_info.time_init
    width = 20
    
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
        
        df = df.iloc[int(init_point):int(final_point),:]
        return df, middle_point

def get_data_without_event(df, event_info):
    
    time_init = event_info.time_init
    width = 20
    
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
        df = df.iloc[int(init_point):int(final_point),:]    
        return df, middle_point*600

config = Configuration()
path_init = config.path
event_archive = EventArchive(path_init, 0)

event_range_min = 1
event_range_max = 18696

rows_event_archive1 = []
rows_event_archive2 = []

print(2*'\x1b[2K\r' + "Progress " + str(event_range_min-1) + "/" + str(event_range_max), flush=True, end="\r")

for event_id in range(event_range_min, event_range_max+1):
    
    df = event_archive.get_event(event_id)
   
    event_info = event_archive.get_event_info(event_id)
    
    df1, mp1 = get_data_with_event(df, event_info) 
    df2, mp2 = get_data_without_event(df, event_info)
        
    if(df1 is not None):
        dic = {}
        dic['event'] = event_id
        dic['time_middle'] = int(mp1)
        dic['node'] = event_info.node
        dic['node_epanet'] = event_info.node_epanet
        dic['coordinate_x'] = event_info.coordinate_x
        dic['coordinate_y'] = event_info.coordinate_y
        dic['q'] = event_info.q
        dic['c'] = event_info.c
        rows_event_archive1.append(dic)
        path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\with_event\\event_' + str(event_id) + '.csv'
        df1.to_csv(index=True, path_or_buf=path_export)
    
    if(df2 is not None):
        dic = {}
        dic['event'] = event_id
        dic['time_middle'] = int(mp2)
        rows_event_archive2.append(dic)
        path_export = path_init + '\\Data\\infraquinta\\events\\Organized_Data\\without_event\\event_' + str(event_id) + '.csv'
        df2.to_csv(index=True, path_or_buf=path_export)
        
    print(2*'\x1b[2K\r' + "Progress " + str(event_id) + "/" + str(event_range_max), flush=True, end="\r")
        
df_event_archive1 = pd.DataFrame(rows_event_archive1)
df_event_archive2 = pd.DataFrame(rows_event_archive2)

df_event_archive1.index = df_event_archive1['event']
del df_event_archive1['event']

df_event_archive2.index = df_event_archive2['event']
del df_event_archive2['event']
     
path_export = path_init + '\\Data\\infraquinta\\events\\Notes\\with_event_archive.csv'
df_event_archive1.to_csv(index=True, path_or_buf=path_export)
        
path_export = path_init + '\\Data\\infraquinta\\events\\Notes\\without_event_archive.csv'
df_event_archive2.to_csv(index=True, path_or_buf=path_export)  

print(2*'\x1b[2K\r' + "Completed " + str(event_range_max) + "/" + str(event_range_max), flush=True, end="\r")


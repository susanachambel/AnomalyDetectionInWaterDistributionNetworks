# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:54:48 2020

@author: susan
"""

import sys
sys.path.append('../Functions')
from configuration import *
import pandas as pd
import numpy as np

config = Configuration()
path_init = config.path

path = path_init + '\\Data\\infraquinta\\events\\Notes\\event_archive.csv'
df_archive = pd.read_csv(path, delimiter=';', index_col=0, decimal=',')

event_range_min = 1
event_range_max = 1000

print(2*'\x1b[2K\r' + "Progress " + str(event_range_min-1) + "/" + str(event_range_max), flush=True, end="\r")

for event_id in range(event_range_min, event_range_max+1):
    
    path = path_init + '\\Data\\infraquinta\\events\\Original\\Rotura_Q\\Rotura_Q_Medidores' + str(event_id) + '.txt'
    df = pd.read_csv(path, delimiter='\s+', header=None, names=[9, 6, 12, 1, 10, 14, 2])  
    df = df.apply(abs)
    df.index = list(range(0,145*600, 600))
    df.index.name = 'time'
    
    event_archive = df_archive.loc[event_id]
    time_init = event_archive.time_init
    time_final = event_archive.time_final
    conditions = [
        (df.index >= time_init) & (df.index <= time_final),
        (df.index > time_final)]
    choices = [1, 2]
    df['event'] = np.select(conditions, choices, default=0)
    
    path_export = path_init + '\\Data\\infraquinta\\events\\Event_Q\\event_' + str(event_id) + '.csv'
    df.to_csv(index=True, path_or_buf=path_export)
    
    print(2*'\x1b[2K\r' + "Progress " + str(event_id) + "/" + str(event_range_max), flush=True, end="\r")

print(2*'\x1b[2K\r' + "Completed " + str(event_range_max) + "/" + str(event_range_max), flush=True, end="\r")

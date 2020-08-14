# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:54:48 2020

@author: susan

Processes the original simulated event data and:
    1) Turns negative values into positive ones
    2) Adds time column with the correct time point
    3) Adds the names of the sensors (columns)

"""

import sys
sys.path.append('../Functions')
from configuration import *
import pandas as pd

config = Configuration()
path_init = config.path

event_range_min = 1
event_range_max = 18696

print(2*'\x1b[2K\r' + "Progress " + str(event_range_min-1) + "/" + str(event_range_max), flush=True, end="\r")

for event_id in range(event_range_min, event_range_max+1):
    
    path = path_init + '\\Data\\infraquinta\\events\\Original\\Rotura_Q\\Rotura_Q_Medidores' + str(event_id) + '.txt'
    df = pd.read_csv(path, delimiter='\s+', header=None, names=[9, 6, 12, 1, 10, 14, 2])  
    df = df.apply(abs)
    df.index = list(range(0,145*600, 600))
    df.index.name = 'time'
       
    path_export = path_init + '\\Data\\infraquinta\\events\\Event_Q\\event_' + str(event_id) + '.csv'
    df.to_csv(index=True, path_or_buf=path_export)
    
    print(2*'\x1b[2K\r' + "Progress " + str(event_id) + "/" + str(event_range_max), flush=True, end="\r")

print(2*'\x1b[2K\r' + "Completed " + str(event_range_max) + "/" + str(event_range_max), flush=True, end="\r")

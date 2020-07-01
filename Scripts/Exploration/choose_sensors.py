# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from data_selection import *
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


config = Configuration()
path_init = config.path

date_min = '2017-01-06 08:30:00'
date_max = '2017-01-08 14:00:00'
date_rupture_detection = '2017-01-08 08:30:00'
date_water_closure = '2017-01-08 09:00:00'
date_possible_rupture = '2017-01-07 18:00:00'

df1 = select_data(path_init, "infraquinta", "interpolated", 1, date_min, date_max)
df9 = select_data(path_init, "infraquinta", "interpolated", 9, date_min, date_max)
df10 = select_data(path_init, "infraquinta", "interpolated", 10, date_min, date_max)
df12 = select_data(path_init, "infraquinta", "interpolated", 12, date_min, date_max)
df14 = select_data(path_init, "infraquinta", "interpolated", 14, date_min, date_max)
df2 = select_data(path_init, "infraquinta", "interpolated", 2, date_min, date_max)
df6 = select_data(path_init, "infraquinta", "interpolated", 6, date_min, date_max)


dfs = [df1, df9, df10, df12, df14, df2, df6]
dfs_names = ['[1] APA Caudal Actual','[9] QV Caudal','[10] HC Caudal', 
             '[12] RPR Caudal Pre', '[14] RPR Caudal Grv', '[2] PB2 caudal caixa 1', '[6] RSV R5 Caudal Caixa']


m = len(dfs)
n = 1

fig, axs = plt.subplots(m, n, figsize=(8*n,4*m), constrained_layout=True)
ylabel = "Water flow [m3/h]"
color = 'steelblue'


locator = mdates.HourLocator(interval=2)
formatter = mdates.ConciseDateFormatter(locator)
locator_min = mdates.HourLocator(interval=1)
rupture_detection = datetime.strptime(date_rupture_detection, '%Y-%m-%d %H:%M:%S')
water_closure = datetime.strptime(date_water_closure, '%Y-%m-%d %H:%M:%S')
possible_rupture = datetime.strptime(date_possible_rupture, '%Y-%m-%d %H:%M:%S')

i = 0
for df in dfs:
    
    title = dfs_names[i]
    
    df = df.resample('30min').mean()
    axs[i].plot(df.index,df['value'], color=color)
    axs[i].set(xlabel='', ylabel=ylabel, title=title)
    
    axs[i].xaxis.set_major_locator(locator)
    axs[i].xaxis.set_major_formatter(formatter)
    
    axs[i].xaxis.set_minor_locator(locator_min)
    axs[i].axvline(x=rupture_detection, color='red', linestyle='--')
    axs[i].axvline(x=water_closure, color='orange', linestyle='--')
    axs[i].axvline(x=possible_rupture, color='purple', linestyle='--')
    
    plt.setp(axs[i].get_xticklabels(), rotation=30, ha='right')
        
    i += 1


plt.savefig('destination_path.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()